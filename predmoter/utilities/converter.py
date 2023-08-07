import os
import logging
import time
import h5py
import numpy as np
import pyBigWig
import gzip

from predmoter.core.constants import MAX_VALUES_IN_RAM


log = logging.getLogger("PredmoterLogger")


class Converter:
    def __init__(self, infile, output_dir, outformat, basename, strand):
        super(Converter, self).__init__()
        self.infile = h5py.File(infile, "r")
        self.output_format = "bw" if outformat in ["bw", "bigwig"] else "bg.gz"
        if self.output_format == "bw":
            assert pyBigWig.numpy == 1, \
                "numpy is by default installed before pyBigWig, so that numpy " \
                "arrays can be used in the creation of bigwig files, but this " \
                "doesn't seem to be the case here"
        self.strand = strand
        self.dsets = [dset.decode() for dset in self.infile["prediction/datasets"][:]]
        self.outfiles = self.get_output_files(output_dir, basename)
        self.seq_len = self.infile["prediction/predictions"][:1].shape[1]
        self.chromosome_map = self.chrom_map(self.infile)
        self.last_chromosome = list(self.chromosome_map.keys())[-1]
        self.last_start = None
        self.last_end = None
        self.last_value = None
        start = time.time()
        log.info(f"Starting conversion of the file {infile} to {outformat} file(s).")
        self.convert()
        log.info(f"Conversion finished. It took {round(((time.time() - start) / 60), ndigits=2)} min.")

    def get_output_files(self, output_dir, basename):
        """Create list of output files.

        Each dataset will get its own bigwig or bedgraph file. The predictions for the
        positive, negative or the average of both strands can be converted. Already
        existing files won't be overwritten.
        """
        strand = self.strand if self.strand is not None else "avg"
        outfiles = []
        for dset in self.dsets:
            filename = f"{basename}_{dset}_{strand}_strand.{self.output_format}"
            outfile = os.path.join(output_dir, filename)
            if os.path.exists(outfile):
                raise OSError(f"the predictions output file {filename} "
                              f"exists in {output_dir}, please move or delete it")
            outfiles.append(outfile)
        return outfiles

    @staticmethod
    def chrom_map(h5_file):
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
        chromosome_map = dict.fromkeys(h5_file["data/seqids"][:])

        for c in chromosome_map:
            idxs = np.where(h5_file["data/seqids"][:] == c)[0].astype(dtype=np.int32)
            # if the first coordinate in start_ends is smaller than the second, it's a chunk
            # from the positive strand, else it's a chunk from the negative strand
            pos = idxs[~np.greater(h5_file["data/start_ends"][idxs, 0], h5_file["data/start_ends"][idxs, 1])]
            # flip negative strand, because the values need to be added in reverse
            neg = np.flipud(idxs[np.greater(h5_file["data/start_ends"][idxs, 0], h5_file["data/start_ends"][idxs, 1])])
            chromosome_map[c] = {"end": np.max(h5_file["data/start_ends"][idxs]), "+": pos, "-": neg}
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
                        array = self.get_data(np.sort(value_dict["-"][j:j + n]), i, True, is_last)
                    else:
                        pos_strand_idxs = value_dict["+"][j:j + n]
                        neg_strand_idxs = np.sort(value_dict["-"][j:j + n])
                        array = np.mean(np.concatenate(
                            [self.get_data(pos_strand_idxs, i).reshape((self.seq_len * len(pos_strand_idxs), 1)),
                             self.get_data(neg_strand_idxs, i, True, is_last).reshape(
                                 (self.seq_len * len(neg_strand_idxs), 1))],
                            axis=1), axis=1).round(decimals=0)

                    # Coordinates and values
                    # -------------------------
                    values = self.unique_duplicates(array)
                    # only exclude -1 afterwards, otherwise [2, 1, -1, 1] would be fused to [2, 1]
                    # if -1 represents a gap (N) the value after should remain, e.g. [2, 1, 1]
                    values = values[values > -1]
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
                        # bigwig datatypes: starts and ends: int, values: float
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

    def get_data(self, strand_idxs, dset_idx, is_negative=False, is_last=False):
        """Return a prediction dataset array.

        Specific indices (i.e. the indices of the requested strand) and the requested dataset of
        the predictions array is returned. The array is also flattened. The negative strand data
        gets flipped vertically, e.g. [8, 2, 1] -> [1, 2, 8]. If it's the last iteration of the
        chromosome indices (containing the last chunk) of the negative strand, a custom flip is used.
        """
        array = np.array(self.infile["prediction/predictions"][strand_idxs, :, dset_idx], dtype=np.float32).flatten()
        if is_negative:
            if is_last:
                # start padding: e.g. start_ends[strand_idxs[0]] = [102642, 85536], np.diff: -17106
                start_padding = -np.diff(self.infile["data/start_ends"][strand_idxs[0]]).item()
                return self.custom_flip(array, start_padding, self.seq_len)
            return np.flipud(array)
        return array

    @staticmethod
    def unique_duplicates(dataset_array):
        """Deduplicate an array.
        E.g.: from [1, 1, 2, 2, 2, 1, 3, 3, 4, 4, 4, 2, 2] to [1, 2, 1, 3, 4, 2]
        """
        # -3 is the filler value that should not appear at the end of the array, or the last will be dropped
        return dataset_array[dataset_array != np.append(dataset_array[1:], -3)]

    @staticmethod
    def custom_flip(array, start_padding, end_padding):
        """Flip differently for negative strand values containing the last chunk.

        Example:
            (the -1 denote the padding)
            positive strand
                chunk                0                  1                  2
                mini_array    [1, 2, 3, 2, 4]    [8, 9, 2, 2, 8]    [2, 2, 5, -1, -1]
                start/end           0/5                5/10              10/13

            negative strand
                chunk                3                    4                  5
                mini_array    [4, 2, 2, -1, -1]    [8, 2, 1, 8, 8]    [4, 2, 3, 2, 1]
                start/end          13/10                10/5                5/0

        The last array start/end wise of the negative strand in this example is the chunk
        with the index 3. Since the negative strand needs to be flipped vertically, the
        result for the last strand would be, that the padding is at the start, e.g.
        [-1, -1, 2, 2, 4], instead of the desired [2, 2, 4, -1, -1]. Therefore the array is
        rearranged and flipped:
            negative strand array: [4, 2, 2, -1, -1, 8, 2, 1, 8, 8, 4, 2, 3, 2, 1]
            flip part after padding: [1, 2, 3, 2, 4, 8, 8, 1, 2, 8]
            add flipped part before padding: [1, 2, 3, 2, 4, 8, 8, 1, 2, 8, 2, 2, 4]
            add padding back: [1, 2, 3, 2, 4, 8, 8, 1, 2, 8, 2, 2, 4, -1, -1]
        """
        return np.concatenate([np.flipud(array[end_padding:]), np.flipud(array[:start_padding]),
                               array[start_padding:end_padding]])

    @staticmethod
    def get_start_ends(dataset_array):  # flat/1d array
        """Extract start and end indices from an array.

        E.g. array: [1, 1, 1, 2, 3, 3, 9, 9, 9, 0, 0], starts: [0, 3, 4, 6, 9], ends: [3, 4, 6, 9, 11]
        The filler value -1, denoting gaps (N) and padding, is excluded.
        """
        starts = np.array([], dtype=np.int32)  # unlikely to have a chromosome this big else change to float64
        ends = np.array([], dtype=np.int32)

        for n in np.unique(dataset_array):
            if n == -1:
                continue
            else:
                idxs = np.where(dataset_array == n)[0]
                starts = np.append(starts, idxs[np.insert(np.flatnonzero(np.diff(idxs) > 1) + 1, 0, 0)])
                ends = np.append(ends, idxs[np.insert(np.flatnonzero(np.diff(idxs) > 1), 0, -1)] + 1)
        return np.sort(starts), np.sort(ends)

# round doesn't work as expected in numpy (statistician’s rounding),
# so maybe correct it in PredictCallback and Converter
