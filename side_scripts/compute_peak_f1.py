import argparse
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix


def extract_chrom_lengths(chrom_file, bl_chroms):
    """Extract chromosome lengths from a text file."""
    ids = np.loadtxt(chrom_file, usecols=0, dtype=str)
    lengths = np.loadtxt(chrom_file, usecols=1, dtype=np.int32)
    chrom_lengths = {id_: length for id_, length in zip(ids, lengths)}

    if bl_chroms is not None:
        for chrom in bl_chroms:
            chrom_lengths.pop(chrom)

    return chrom_lengths


def get_chrom_lengths(h5df, bl_chroms):
    """Extract chromosome lengths from a h5 file created by Helixer or Predmoter."""
    chrom_lengths = dict.fromkeys(h5df["data/seqids"][:])
    
    if bl_chroms is not None:
        for chrom in bl_chroms:
            chrom_lengths.pop(chrom.encode())
            
    for c in chrom_lengths:
        idxs = np.where(h5df["data/seqids"][:] == c)[0].astype(dtype=np.int32)
        chrom_lengths[c] = np.max(h5df["data/start_ends"][idxs])
    return chrom_lengths


def extract_peaks(bed_file):
    """Create a dictionary out of called peaks."""
    ids = np.loadtxt(bed_file, delimiter="\t", dtype=str, skiprows=1, usecols=0)
    starts = np.loadtxt(bed_file, delimiter="\t", dtype=int, skiprows=1, usecols=1)
    ends = np.loadtxt(bed_file, delimiter="\t", dtype=int, skiprows=1, usecols=2)
    peaks = dict.fromkeys(ids)
    for id_ in peaks:
        peaks[id_] = {"starts": starts[np.where(id_ == ids)[0]], "ends": ends[np.where(id_ == ids)[0]]}
    return peaks


def compute_f1(tp, fp, fn):  # adapted from Helixer
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0  # avoid an error due to division by 0
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0  # avoid an error due to division by 0
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return f1


def create_peak_array(chrom, chrom_length, peak_dict):
    """Create base-wise binary peak array. No peak is denoted with 0 and a peak with 1."""
    array = np.full(chrom_length, fill_value=0)  # array of the same length as the chromosome filled with zeros
    if chrom in peak_dict:
        for i in range(peak_dict[chrom]["starts"].shape[0]):
            # replace the zeros in peak regions (extracted from the bed file) with 1
            array[peak_dict[chrom]["starts"][i]:peak_dict[chrom]["ends"][i]] = 1
    return array


def convert_to_table_line(spacing, contents):
    contents = list(map(str, contents))
    # every string will have the length of spacing
    contents = [f"{i: <{spacing}}" if len(i) <= spacing else f"{i[:spacing - 2]}" + ".." for i in contents]
    return "|".join(contents)


def main(h5_file, chrom_file, predicted_peaks, experimental_peaks, bl_chroms=None):
    if bl_chroms is not None:
        bl_chroms = np.loadtxt(bl_chroms, usecols=0, dtype=str)

    if h5_file is None:
        chromosomes = extract_chrom_lengths(chrom_file, bl_chroms)
    else:
        h5df = h5py.File(h5_file, "r")
        chromosomes = get_chrom_lengths(h5df, bl_chroms)

    pred_peaks = extract_peaks(predicted_peaks)
    exp_peaks = extract_peaks(experimental_peaks)
    
    matrices = []
    for c, length in chromosomes.items():
        # compute a confusion matrix per sequence/chromosome
        if h5_file is not None:
            c = c.decode()  # was bytes string before when extracted out of a h5 file
        pred = create_peak_array(c, length, pred_peaks)
        true = create_peak_array(c, length, exp_peaks)
        matrices.append(confusion_matrix(true, pred, labels=[0, 1]))

    matrix = np.sum(matrices, axis=0)
    tn, fp, fn, tp = matrix.ravel()
    print(f"{convert_to_table_line(20, ['F1', 'TP', 'FP', 'FN'])}\n"
          f"{convert_to_table_line(20, [compute_f1(tp, fp, fn), tp, fp, fn])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Compute peak F1",
                                     description="Computes the F1, true positives (TP), false positives (FP), and "
                                                 "false negatives (FN) between two bed files (e.g., experimental and "
                                                 "predicted) containing called peaks (e.g., ATAC-seq peaks) from one "
                                                 "species. The peaks are assumed to have been called with MACS3.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", "-h5", type=str, default=None,
                        help="h5 input file; this file is used to extract the chromosome lengths of the species;"
                             "it can either be a h5 file created with Helixer or a prediction h5 file (Predmoter "
                             "output, e.g., <species>_predictions.h5)")
    parser.add_argument("--chromosome-lengths", "-cl", type=str, default=None,
                        help="ONLY needed if no h5 file is available; a text file containing the sequence/chromosome "
                             "IDs of the species' genome assembly in the first column and the chromosome lengths in "
                             "the second column; the columns need to be whitespace separated")
    parser.add_argument("--predicted-peaks", "-pp", type=str, default=None, required=True,
                        help="called predicted peak bed file (MACS3 output)")
    parser.add_argument("--experimental-peaks", "-ep", type=str, default=None, required=True,
                        help="called experimental peak bed file (MACS3 output)")
    parser.add_argument("--bl-file", type=str, default=None,
                        help=r"blacklist file; a text file containing the sequence/chromosome IDs to exclude in"
                             r"the computation; the IDs need to be line separated (\n) (one ID per line)")
    args = parser.parse_args()

    if args.h5_file is None and args.chromosome_lengths is None:
        raise OSError("either a h5 file or a chromosome legths text file is required")

    main(args.h5_file, args.chromosome_lengths, args.predicted_peaks, args.experimental_peaks, args.bl_file)
