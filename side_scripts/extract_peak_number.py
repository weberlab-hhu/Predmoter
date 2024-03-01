import argparse
import numpy as np


def extract_peak_numbers(bed_file, bl_file):
    chroms = np.loadtxt(bed_file, delimiter="\t", dtype=str, skiprows=1, usecols=0)
    if bl_file is not None:
        # blacklist reading specific lines
        bl_chroms = np.atleast_1d(np.loadtxt(bl_file, dtype=str, usecols=0))
        mask = np.full(chroms.shape, fill_value=True)
        mask[np.nonzero(bl_chroms[:, None] == chroms)[1]] = False  # the blacklisted chromosomes will be excluded

        # start ends
        chroms = chroms[mask]

    print(chroms.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Extract the peak number.",
                                     description="Computes the number of all peaks inside a given MACS3 peak bed "
                                                 "file. Unwanted chromosomes/sequence IDs can be excluded via a "
                                                 "blacklist file, created with add_blacklist.py, like for "
                                                 "compute_peak_f1.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bed-file", type=str, default=None, required=True, help="h5 input file")
    parser.add_argument("--bl-file", type=str, default=None,
                        help=r"blacklist file; a text file containing the sequence/chromosome IDs to exclude in"
                             r"the computation; the IDs need to be line separated (\n) (one ID per line)")
    args = parser.parse_args()

    extract_peak_numbers(args.bed_file, args.bl_file)
