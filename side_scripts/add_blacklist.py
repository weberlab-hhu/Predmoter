import argparse
import os
import json
import sys
import h5py
import numpy as np


def main(input_json, h5_file):
    print(f"Starting adding mask/blacklist regions to {h5_file}. Please make sure the {input_json} "
          f"file/your genome assembly is from either the NCBI's RefSeq or GenBank database.")
    bl_chroms = get_blacklist_chromosomes(input_json)
    if len(bl_chroms) == 0:
        print(f"Couldn't find any unplaced scaffolds/contigs or non-nuclear sequences. "
              f"No mask will be added to {h5_file}.")

        sys.exit(0)
    h5df = h5py.File(h5_file, "a")
    chunks = h5df["data/X"].shape[0]
    h5df.create_dataset("data/mask", shape=(chunks,), maxshape=(None,), dtype=np.int32,
                        compression="gzip", compression_opts=9, fillvalue=-1)
    n = 5_000
    for i in range(0, chunks, n):
        mask = ~np.in1d(h5df["data/seqids"][i:i + n], bl_chroms)
        # True: chunk will be retained, False: chunk is masked/blacklisted
        h5df["data/mask"][i:i + n] = mask


def get_blacklist_chromosomes(input_json):
    blacklist_chromosomes = []
    accession = ""
    with open(input_json, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            if i == 0:
                accession = "refseqAccession" if "refseqAccession" in line.keys() else "genbankAccession"
            if line["chrName"] == "Un" and i == 0:
                print("Cannot apply masking to a genome assembly only containing unplaced scaffolds/contigs")
                sys.exit(1)
            elif line["chrName"] == "Un" or line["assemblyUnit"] == "non-nuclear":
                blacklist_chromosomes.append(line[accession].encode())
                # h5 seqids are byte strings, so the blacklist ones are too for easy comparison
    return np.array(blacklist_chromosomes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Add mask/blacklist regions",
                                     description="Add 'data/mask' to training/validation and/or test h5 files to"
                                                 "mask non-nuclear and scaffold regions, if the assembly doesn't"
                                                 "only contain scaffolds.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-json", type=str, default=None, required=True,
                        help="input sequence_report.jsonl from NCBI (sequence information about the assembly, "
                             "only works with NCBI formatting)")
    parser.add_argument("-h5", "--output-h5-file", type=str, default=None, required=True,
                        help="h5 file the masking information will be added to")
    args = parser.parse_args()

    if not os.path.exists(args.output_h5_file):
        raise FileNotFoundError(f"file {args.output_h5_file} doesn't exist")

    try:
        h5py.File(args.output_h5_file, "r")
    except OSError as e:
        raise OSError(f"{args.output_h5_file} is not a h5 file, please provide a valid h5 file") from e

    main(args.input_json, args.output_h5_file)
