import argparse
import os
import json
import sys
import h5py
import numpy as np


def main(input_json, h5_file, overwrite, text_file):
    if text_file is None:
        print(f"Starting adding mask/blacklist regions to {h5_file}. Please make sure the {input_json} "
              f"file/your genome assembly is from either the NCBI's RefSeq or GenBank database.")
    else:
        print(f"Writing flagged/blacklisted sequences to the text file {text_file}.")
    bl_chroms = get_blacklist_chromosomes(input_json, text_file)
    if len(bl_chroms) == 0:
        if text_file is None:
            print(f"Couldn't find any unplaced scaffolds/contigs or non-nuclear sequences. "
                  f"No mask will be added to {h5_file}.")
        else:
            print(f"Couldn't find any unplaced scaffolds/contigs or non-nuclear sequences. "
                  f"The file {text_file} will not be written.")

        sys.exit(0)

    if text_file is None:
        h5df = h5py.File(h5_file, "a")
        check_seqids(h5df, bl_chroms)
        chunks = h5df["data/X"].shape[0]
        blacklist_dataset = "data/blacklist"
        if not overwrite or (overwrite and "blacklist" not in h5df["data"].keys()):
            h5df.create_dataset(blacklist_dataset, shape=(chunks,), maxshape=(None,), dtype=np.int32,
                                compression="gzip", compression_opts=9, fillvalue=-1)
        n = 5_000
        for i in range(0, chunks, n):
            mask = ~np.in1d(h5df["data/seqids"][i:i + n], bl_chroms)
            # True: chunk will be retained, False: chunk is masked/blacklisted
            h5df[blacklist_dataset][i:i + n] = mask
        print("Finished adding mask/blacklist regions.")

    else:
        with open(text_file, "w") as f:
            f.write("\n".join(bl_chroms))
        print(f"Finished writing the file {text_file}.")


def get_blacklist_chromosomes(input_json, text_file):
    blacklist_chromosomes = []
    accession = ""
    with open(input_json, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line)
            if i == 0:
                accession = "refseqAccession" if "refseqAccession" in line.keys() else "genbankAccession"
            if (line["chrName"] == "Un" or line["role"] == "unplaced-scaffold" or
                    line["role"] == "unlocalized-scaffold") and i == 0:
                print("Cannot apply masking to a genome assembly only containing unplaced scaffolds/contigs.")
                sys.exit(0)
            elif line["chrName"] == "Un" or line["assemblyUnit"] == "non-nuclear" \
                    or line["role"] == "unplaced-scaffold" or line["role"] == "unlocalized-scaffold":
                blacklist_chromosomes.append(line[accession])
                # h5 seqids are byte strings, so the blacklist ones are too for easy comparison
    if text_file is None:
        blacklist_chromosomes = [c.encode() for c in blacklist_chromosomes]
        return np.array(blacklist_chromosomes)
    return blacklist_chromosomes


def check_seqids(h5df, bl_chroms):
    if bl_chroms[0] not in h5df["data/seqids"][:]:
        raise ValueError(f"One of the chromosomes to blacklist, {bl_chroms[0].item().decode()}, couldn't be found "
                         f"in the input h5 file. Please check if the sequence report is correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Add mask/blacklist regions",
                                     description="Add 'data/blacklist' to training/validation and/or test h5 files to"
                                                 "mask non-nuclear and scaffold regions, if the assembly doesn't"
                                                 "only contain scaffolds/contigs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-json", type=str, default=None, required=True,
                        help="input sequence_report.jsonl from NCBI (sequence information about the assembly, "
                             "only works with NCBI formatting)")
    parser.add_argument("-h5", "--output-h5-file", type=str, default=None,
                        help="h5 file the masking information will be added to")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing blacklist")
    parser.add_argument("--write-text-file", type=str, default=None,
                        help="when adding a text file name, the program will just output this text file with one "
                             "flagged chromosome/sequence ID per line (can be used as blachlist file for "
                             "convert2coverage.py)")
    args = parser.parse_args()

    if args.write_text_file is None:
        if args.output_h5_file is None:
            raise OSError(f"when adding masking to a h5 file, the argument -h5/--output-h5-file is required")

        if not os.path.exists(args.output_h5_file):
            raise FileNotFoundError(f"file {args.output_h5_file} doesn't exist")

        try:
            h5py.File(args.output_h5_file, "r")
        except OSError as e:
            raise OSError(f"{args.output_h5_file} is not a h5 file, please provide a valid h5 file") from e

    main(args.input_json, args.output_h5_file, args.overwrite, args.write_text_file)
