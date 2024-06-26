import os
import logging
import time
import h5py
import numpy as np
import pyBigWig
import gzip

from predmoter.core.constants import GIT_COMMIT, MAX_VALUES_IN_RAM


log = logging.getLogger("PredmoterLogger")


class Converter:
    def __init__(self, infile, output_dir, outformat, basename, strand, experimental=False,
                 dsets=None, bl_chroms=None, window_size=None):
        super(Converter, self).__init__()
        self.infile = h5py.File(infile, "r")
        self.output_format = "bw" if outformat in ["bw", "bigwig"] else "bg.gz"
        if self.output_format == "bw":
            assert pyBigWig.numpy == 1, \
                "numpy is by default installed before pyBigWig, so that numpy " \
                "arrays can be used in the creation of bigwig files, but this " \
                "doesn't seem to be the case here"
        self.strand = strand
        if experimental:
            self.dsets = [dset for dset in dsets if f"{dset}_coverage" in self.infile["evaluation"].keys()]
            assert len(self.dsets) > 0,\
                f"the input file {infile} doesn't include any of the datasets: {', '.join(self.dsets)}"
        else:
            self.dsets = [dset.decode() for dset in self.infile["prediction/datasets"][:]]
        self.window_size = window_size
        self.outfiles = self.get_output_files(output_dir, basename, bl_chroms, window_size)
        if experimental:
            self.seq_len = self.infile["data/X"][:1].shape[1]
        else:
            self.seq_len = self.infile["prediction/predictions"][:1].shape[1]
        # np.atleast_1d() in case there is just one sequence ID to blacklist which would otherwise
        # result in a 0-dimensional array
        self.blacklisted_chromosomes = bl_chroms if bl_chroms is None \
            else np.atleast_1d(np.loadtxt(bl_chroms, dtype=str, usecols=0))
        self.experimental = experimental
        start = time.time()
        log.info("\n", extra={"simple": True})
        log.info(f"Starting conversion of the file {infile} to {outformat} file(s). "
                 f"The current commit is {GIT_COMMIT}.")
        smooth = False if window_size is None else True
        w_size = "" if window_size is None else f"with window size {window_size}"
        exclude_bl = False if bl_chroms is None else True
        log.info(f"Chosen/processed converter options: datasets to convert: {', '.join(self.dsets)}; "
                 f"convert experimental data: {experimental}; exclude flagged sequences: "
                 f"{exclude_bl}; smooth predictions: {smooth} {w_size}")
        self.chromosome_map = self.chrom_map(self.infile, self.blacklisted_chromosomes)
        self.last_chromosome = list(self.chromosome_map.keys())[-1]
        self.last_start = None
        self.last_end = None
        self.last_value = None
        self.convert()
        log.info(f"Conversion finished. It took {round(((time.time() - start) / 60), ndigits=2)} min.\n")

    def get_output_files(self, output_dir, basename, bl_chroms, window_size):
        """Create list of output files.

        Each dataset will get its own bigwig or bedgraph file. The predictions for the
        positive, negative or the average of both strands can be converted. Already
        existing files won't be overwritten.
        """
        strand = self.strand if self.strand is not None else "avg"
        bl = "" if bl_chroms is None else "_bl"
        smooth = "" if window_size is None else f"_ws={window_size}"
        outfiles = []
        for dset in self.dsets:
            filename = f"{basename}_{dset}_{strand}_strand{bl}{smooth}.{self.output_format}"
            outfile = os.path.join(output_dir, filename)
            if os.path.exists(outfile):
                raise OSError(f"the output file {filename} "
                              f"exists in {output_dir}, please move or delete it")
            outfiles.append(outfile)
        return outfiles

    @staticmethod
    def chrom_map(h5df, bl_chroms):
        """Create a 'map' of the chromosome order in the h5 file.

        Example:
            sequence length: 21384 bp
            length of chromosome 1: 51321 bp

            chunk           0          1              2              3               4           5
            strand          +          +              +              -               -           -
            start/end   0/21384    21384/42768    42768/51321    51321/42768    42768/21384    21384/0

        All strands are represented from 5' to 3', so the negative strand is in descending order.
        For each chromosome the length (end) in bp and the indices of the positive and negative
        strand are listed.
        """
        chromosome_map = dict.fromkeys(h5df["data/seqids"][:])

        if bl_chroms is not None:
            for chrom in bl_chroms:
                chromosome_map.pop(chrom.encode())

        for c in chromosome_map:
            idxs = np.where(h5df["data/seqids"][:] == c)[0].astype(dtype=np.int32)
            # if the first coordinate in start_ends is smaller than the second, it's a chunk
            # from the positive strand, else it's a chunk from the negative strand
            pos = idxs[~np.greater(h5df["data/start_ends"][idxs, 0], h5df["data/start_ends"][idxs, 1])]
            # flip negative strand, because the values need to be added in reverse
            neg = np.flipud(idxs[np.greater(h5df["data/start_ends"][idxs, 0], h5df["data/start_ends"][idxs, 1])])
            chromosome_map[c] = {"end": np.max(h5df["data/start_ends"][idxs]), "+": pos, "-": neg}
        return chromosome_map

    def convert(self):
        """Convert predictions in the h5 file into either bigwig or bedgraph files."""
        n = MAX_VALUES_IN_RAM // self.seq_len  # dynamic step for different sequence lengths
        for i, file in enumerate(self.outfiles):
            if self.output_format == "bw":
                bw = pyBigWig.open(file, "w")
                bw.addHeader([(key.decode(), value_dict["end"]) for key, value_dict in self.chromosome_map.items()])

            # step through per chromosome
            for key, value_dict in self.chromosome_map.items():
                # read/write in chunks for saving memory (RAM) in case of large chromosomes
                # max_length: how many chunks per chromosome (just of 1 strand, doesn't matter which one)
                max_length = len(value_dict["+"])
                for j in range(0, max_length, n):
                    # if it is the last iteration of this chromosome
                    is_last = True if max_length <= (j + n) else False
                    if self.strand == "+":
                        array = self.get_data(value_dict["+"][j:j + n], i)
                    elif self.strand == "-":
                        # since the - strand is in reverse order, the indices have to be sorted for h5py
                        array = self.get_data(np.sort(value_dict["-"][j:j + n]), i, True)
                    else:
                        pos_strand_idxs = value_dict["+"][j:j + n]
                        neg_strand_idxs = np.sort(value_dict["-"][j:j + n])
                        array = np.mean(np.array([self.get_data(pos_strand_idxs, i),
                                                  self.get_data(neg_strand_idxs, i, True)]), axis=0).round(decimals=0)

                    # Coordinates and values
                    # -------------------------
                    if self.window_size is not None:
                        array = self.smooth_predictions(array, window_size=self.window_size)  # smooth preds
                    values = self.unique_duplicates(array)
                    starts, ends = self.get_start_ends(array)

                    # Special cutting cases
                    # -------------------------
                    # don't include the last value, in case the next first value is the same
                    # unless it's the end of the chromosome
                    split_end = None if is_last else -1
                    # reading/writing in chunks means adding the base pair length from before
                    starts = starts + (j * self.seq_len)
                    ends = ends + (j * self.seq_len)
                    if j != 0:  # not the first iteration
                        if self.last_value == values[0]:
                            # if the last value of the iteration before is the same as the first here,
                            # replace the start with the old last start to have a continuous interval
                            starts[0] = self.last_start
                        else:
                            # if the last value of the iteration before isn't the same as the first here,
                            # insert the last values, that were left out, back at the beginning of the arrays
                            starts = np.insert(starts, 0, self.last_start)
                            ends = np.insert(ends, 0, self.last_end)
                            values = np.insert(values, 0, self.last_value)
                    self.last_start, self.last_end, self.last_value = starts[-1], ends[-1], values[-1]

                    if self.output_format == "bw":
                        chrom_count = len(starts) - 1 if not is_last else len(starts)
                        # bigwig datatypes: starts and ends: int, values: float (required datatypes)
                        bw.addEntries([key.decode()] * chrom_count, starts[:split_end], ends=ends[:split_end],
                                      values=values[:split_end])
                    else:
                        content = b"\n".join([key + str.encode(f"\t{start}\t{end}\t{value}") for start, end, value in
                                              zip(starts[:split_end], ends[:split_end], values[:split_end])])
                        if not (key == self.last_chromosome and is_last):
                            content = content + b"\n"
                        with gzip.open(f"{file}", "ab", compresslevel=9) as f:
                            f.write(content)

                # set back to None for the next chromosome
                self.last_start, self.last_end, self.last_value = None, None, None

            if self.output_format == "bw":
                bw.close()

    def get_data(self, strand_idxs, dset_idx, is_negative=False):
        """Return a prediction dataset array.

        Specific indices (i.e. the indices of the requested strand) and the requested dataset of
        the predictions array is returned. The array is also flattened. The negative strand data
        gets flipped vertically, e.g. [8, 2, 1] -> [1, 2, 8]. The -1 filler values are excluded.
        """
        if self.experimental:
            dset = self.dsets[dset_idx]
            array = np.array(self.infile[f"evaluation/{dset}_coverage"][strand_idxs], dtype=np.float32)
            array = np.around(np.mean(array, axis=2), 0).flatten()
        else:
            array = np.array(self.infile["prediction/predictions"][strand_idxs, :, dset_idx],
                             dtype=np.float32).flatten()  # the predictions are ints
        if is_negative:
            array = np.flipud(array)
        return array[array != -1]  # exclude padding

    @staticmethod
    def smooth_predictions(array, window_size):
        """Smooth out the predictions with a 'rolling mean' and a given window size.
        The specification 'mode=same' retains the original size of the array."""
        return np.convolve(array, np.ones(window_size) / window_size, mode="same").round(decimals=0)

    @staticmethod
    def unique_duplicates(dataset_array):
        """Deduplicate an array.
        E.g.: from [1, 1, 2, 2, 2, 1, 3, 3, 4, 4, 4, 2, 2] to [1, 2, 1, 3, 4, 2]
        """
        # -3 is the filler value that should not appear at the end of the array, or the last will be dropped
        return dataset_array[dataset_array != np.append(dataset_array[1:], -3)]

    @staticmethod
    def get_start_ends(dataset_array):  # flat/1d array
        """Extract start and end indices from an array.

        E.g. array: [1, 1, 1, 2, 3, 3, 9, 9, 9, 0, 0], starts: [0, 3, 4, 6, 9], ends: [3, 4, 6, 9, 11]
        The filler value -1, denoting padding, is excluded beforehand in get_data().
        """
        starts = np.array([], dtype=np.int32)  # unlikely to have a chromosome this big else change to float64
        ends = np.array([], dtype=np.int32)

        for n in np.unique(dataset_array):
            idxs = np.where(dataset_array == n)[0]
            starts = np.append(starts, idxs[np.insert(np.flatnonzero(np.diff(idxs) > 1) + 1, 0, 0)])
            ends = np.append(ends, idxs[np.insert(np.flatnonzero(np.diff(idxs) > 1), 0, -1)] + 1)
        return np.sort(starts), np.sort(ends)
