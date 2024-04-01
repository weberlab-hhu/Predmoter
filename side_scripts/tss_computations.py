import numpy as np
import gffpandas.gffpandas as gffpd
import argparse
import os
import pyBigWig


def pear_coeff(replicate, target, is_log=False):
    """Calculate the pearson correlation coefficient (numpy version).

            This function only accepts tensors with a maximum of 2 dimensions. It excludes NaNs
            so that NaN can be used for positions to be masked as masked tensors are at present
            still in development.

                Args:
                    replicate: 'numpy.array', experimental NGS coverage replicate 1
                    target: 'numpy.array', experimental NGS coverage replicate 2/target
                    is_log: 'bool', if True (default) assumes the models' predictions are logarithmic

                Returns:
                    'numpy.array', average pearson correlation coefficient (correlation between
                        the replicates are calculated per chunk/subsequence and then averaged)
    """

    dims = len(replicate.shape)
    assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
    if dims == 2:
        replicate = np.squeeze(replicate.transpose(0, 1))
        target = np.squeeze(target.transpose(0, 1))

    if is_log:
        replicate = np.exp(replicate)

    p = replicate - np.nanmean(replicate, axis=0)
    t = target - np.nanmean(target, axis=0)
    coeff = np.nansum(p * t, axis=0) / (
            np.sqrt(np.nansum(p ** 2, axis=0)) * np.sqrt(np.nansum(t ** 2, axis=0)) + 1e-8)
    # eps avoiding division by 0
    return np.mean(coeff)


def string_table(names, array):
    assert len(array.shape) == 2, "the matrix should be a 2D array"

    header = " ".join(names)
    s = "X " + header
    matrix = array.tolist()
    for i, mat in enumerate(matrix):
        row = names[i] + " " + " ".join([str(n) for n in mat])
        s = s + "\n" + row
    return s


def get_chromosome_length(gff_file):
    with open(gff_file) as file_content:
        lines = [line.rstrip().split() for line in file_content.readlines() if line.startswith("##sequence-region")]
    return {lines[i][1]: int(lines[i][-1]) for i in range(len(lines))}  # chromosomes as key, max length as value


def get_tss_region(gff_file, bl_chroms, upstream_region, downstream_region):
    annotation = gffpd.read_gff3(gff_file)
    genes = annotation.filter_feature_of_type(['gene'])

    # recover arrays of interest
    seqids = genes.df["seq_id"].to_numpy()
    starts = genes.df["start"].to_numpy()
    ends = genes.df["end"].to_numpy()
    strands = genes.df["strand"].to_numpy()

    region_starts = []
    region_ends = []
    if bl_chroms is not None:
        # get the data not in blacklisted chromosomes
        mask = []
        for chrom in bl_chroms:
            mask.append(seqids != chrom)
        mask = np.logical_and.reduce(np.array(mask))  # True for indices to keep
        seqids = seqids[mask]
        starts = starts[mask]
        ends = ends[mask]
        strands = strands[mask]

    for i in range(seqids.shape[0]):
        if strands[i] == "+":
            # region +- 3 kbp around tss
            start = starts[i] - 1 - upstream_region  # gff has 1-based integer coordinates, correct for Python
            end = starts[i] + downstream_region  # end coordinate is the same as in Python indexing
        else:
            start = ends[i] - 1 - downstream_region  # a little unsure here, maybe ask someone
            end = ends[i] + upstream_region
        region_starts.append(start)
        region_ends.append(end)

    return seqids, strands, np.array([region_starts, region_ends]).T


def get_region_coverage(bw, seqid, strand, max_length, region, type_):  # bw either one or multiple bigWig generators
    if region[0] >= 0 and region[1] <= max_length:  # all within chromosome boundaries
        if type_ == "single":
            cov = np.array(bw.values(seqid, region[0], region[1]))
        else:
            cov = [np.array(b.values(seqid, region[0], region[1])) for b in bw]
    elif region[0] < 0 and region[1] <= max_length:  # start region too small/out of chromosome boundaries
        start_filler = np.full((-1 * region[0],), fill_value=np.nan)
        region[0] = 0
        if type_ == "single":
            cov = np.concatenate([start_filler, np.array(bw.values(seqid, region[0], region[1]))])
        else:
            cov = [np.concatenate([start_filler, np.array(b.values(seqid, region[0], region[1]))]) for b in bw]
    elif region[0] >= 0 and region[1] > max_length:  # end region too large/out of chromosome boundaries
        end_filler = np.full((region[1] - max_length,), fill_value=np.nan)
        region[1] = max_length  # set to max chromosome length
        if type_ == "single":
            cov = np.concatenate([np.array(bw.values(seqid, region[0], region[1])), end_filler])
        else:
            cov = [np.concatenate([np.array(b.values(seqid, region[0], region[1])), end_filler]) for b in bw]
    else:  # both out of bounds (very rare for usual usage)
        start_filler = np.full((-1 * region[0],), fill_value=np.nan)
        region[0] = 0
        end_filler = np.full((region[1] - max_length,), fill_value=np.nan)
        region[1] = max_length  # set to max chromosome length
        if type_ == "single":
            cov = np.concatenate([start_filler, np.array(bw.values(seqid, region[0], region[1])), end_filler])
        else:
            cov = [np.concatenate([start_filler, np.array(b.values(seqid, region[0], region[1])), end_filler])
                   for b in bw]
    # flip coverage appropriately
    if strand == "-":
        if type_ == "single":
            return np.flip(cov)
        return np.stack([np.flip(c) for c in cov])  # make an array out of the coverage array list
    else:
        if type_ == "single":
            return cov
        return np.stack(cov)  # make an array out of the coverage array list


def compute_reduced_region_coverage(bw_file, seqids, strands, lengths, tss_regions, reduction_type):
    # -1 just symbolize padding and are excluded due to the way data is recovered from the bw files
    # as they just contain valid/non-negative numbers
    bw = pyBigWig.open(bw_file)
    coverage = []
    for i in range(tss_regions.shape[0]):  # loop through all tss regions
        cov = get_region_coverage(bw, seqids[i], strands[i], max_length=lengths[seqids[i]],
                                  region=tss_regions[i], type_="single")
        coverage.append(cov)

    if reduction_type == "median":
        return np.nanmedian(np.array(coverage, dtype=np.float32), axis=0)
    return np.nanmean(np.array(coverage, dtype=np.float32), axis=0)


def calc_pearson_matrix(bw_files, seqids, strands, lengths, tss_regions):
    matrix = []
    num = len(bw_files)
    bw_generators = [pyBigWig.open(bw_file) for bw_file in bw_files]
    for i in range(tss_regions.shape[0]):  # loop through all tss regions
        # recovers specified region from every file in multiple mode
        array = get_region_coverage(bw_generators, seqids[i], strands[i], max_length=lengths[seqids[i]],
                                    region=tss_regions[i], type_="multiple")
        cols = 0
        pears = np.full(shape=(num, num), fill_value=0., dtype=np.float32)  # sub-matrix
        for j in range(num):
            for k in range(num - cols):
                pears[j, j + k] = pear_coeff(array[j], array[j + k])
            cols += 1

        matrix.append(np.expand_dims(pears, axis=2))

    matrix = np.mean(np.concatenate(matrix, axis=2), axis=2)

    # mirror along diagonal
    matrix_copy = np.array(matrix.T, copy=True)
    np.fill_diagonal(matrix_copy, 0.)
    matrix = matrix + matrix_copy
    return matrix


def check_files(bw_file_list):
    for bw_file in bw_file_list:
        bw = pyBigWig.open(bw_file)
        assert bw.isBigWig(), f"the input file {bw_file} is not a bigWig file"


def main(mode, gff, bl_chroms, upstream_region, downstream_region, infile, infiles, sample_names,
         reduction, outfile):
    if mode == "correlation":
        check_files(infiles)
    else:
        check_files([infile])
    chromosome_lengths = get_chromosome_length(gff)
    if bl_chroms is not None:
        bl_chroms = np.loadtxt(bl_chroms, usecols=0, dtype=str)
        id_ = bl_chroms[0]
        assert id_ in chromosome_lengths.keys(), \
            f"first sequence ID, {bl_chroms[0]}, to exclude from computations doesn't match the " \
            f"any sequence ID in the annotation file {gff}"

    # get tss regions and corresponding seqids/chromosomes
    seqids, strands, tss_regions = get_tss_region(gff, bl_chroms, upstream_region, downstream_region)
    if mode == "coverage":
        np.save(f"{outfile}.npy",
                compute_reduced_region_coverage(infile, seqids, strands, chromosome_lengths, tss_regions, reduction))
    else:  # coverage calculation
        pearson_matrix = calc_pearson_matrix(infiles, seqids, strands, chromosome_lengths, tss_regions)
        table = string_table(sample_names, pearson_matrix)
        with open(f"{outfile}_pearson.csv", "w") as f:
            f.write(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Calculate TSS statistics.",
                                     description="Calculate Pearson's correlation or mean/median coverage in a "
                                                 "specified region around the TSS.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", type=str, default=None, required=True,
                        help="compute either Pearson's correlation (mode: correlation) or mean/median "
                             "coverage (mode: coverage)")
    parser.add_argument("-gff", "--gff-file", type=str, default=None, required=True,
                        help="annotation gff file from which to extract the TSS regions")
    parser.add_argument("--bl-file", type=str, default=None,
                        help=r"blacklist file; a text file containing the sequence/chromosome IDs to exclude in"
                             r"the computation; the IDs need to be line separated (\n) (one ID per line)")
    parser.add_argument("--upstream-region", type=int, default=3000, help="distance upstream of the TSS in bp")
    parser.add_argument("--downstream-region", type=int, default=3000, help="distance downstream of the TSS in bp")
    parser.add_argument("--infile", type=str, default=None, help="single bigWig input file for mode 'coverage'")
    parser.add_argument("--input-files", nargs="+", dest="infiles",
                        help="space separated list of bigWig input files for correlation mode, at least two files")
    parser.add_argument("--sample-names", nargs="+", dest="samples",
                        help="sample names for the correlation computations")
    parser.add_argument("--reduction", type=str, default="mean", help="calulate mean or median coverage around TSS")
    parser.add_argument("--outfile", type=str, default=None,
                        help="output file name, for coverage mode: 'outfile'.npy (a numpy array file), for "
                             "correlation mode: 'outfile'_pearson.csv (a csv file of the correlation matrix")
    args = parser.parse_args()

    assert args.mode in ["correlation", "coverage"],\
        f"invalid mode {args.mode}, valid modes are correlation or coverage"
    assert os.path.isfile(args.gff_file), f"either {args.gff_file} doesn't exist or path is wrong"
    if args.bl_file is not None:
        assert os.path.isfile(args.bl_file), f"either {args.bl_file} doesn't exist or path is wrong"

    if args.mode == "coverage":
        assert args.infile is not None, f"one input bigWig file needs to be provided in coverage mode via --infile"
        assert os.path.isfile(args.infile), f"either {args.infile} doesn't exist or path is wrong"
        assert args.reduction in ["mean", "median"],\
            f"invalid reduction method {args.reduction}, valid reduction methods are mean or median"
    else:
        assert args.infiles is not None, "please provide at least two files in correlation mode via --input-files"
        assert len(args.infiles) > 1, "please provide at least two files in correlation mode via --input-files"
        for file in args.infiles:
            assert os.path.isfile(file), f"either {file} doesn't exist or path is wrong"
        assert args.samples is not None, "please provide sample names in correlation mode via --sample-names"
        assert len(args.samples) == len(args.infiles),\
            "please provide the same number of input files and sample names"

    assert args.outfile is not None, f"please provide a name for the output file via --outfile"

    main(args.mode, args.gff_file, args.bl_file, args.upstream_region, args.downstream_region,
         args.infile, args.infiles, args.samples, args.reduction, args.outfile)
