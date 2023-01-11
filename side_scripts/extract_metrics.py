import numpy as np
import h5py
import argparse
import os


def main(files, keys):
    for file in files:
        assert os.path.isfile(file), f"either {file} doesn't exist or path is wrong"

        h5df = h5py.File(file, mode="r")
        for i, key in enumerate(keys):
            assert f"{key}_coverage" in h5df["evaluation"].keys(), f"there is no {key}" \
                                                                   f" dataset in {file.split('/')[-1]}"
            avg = np.array(h5df[f"evaluation/{key}_means"])
            med = np.array(h5df[f"evaluation/{key}_medians"])
            if i == 0:
                msg = f"File: {file.split('/')[-1]}\n{key}_means: {avg}\n{key}_medians: {med}\n"
            else:
                msg = f"{key}_means: {avg}\n{key}_medians: {med}\n"
            with open("metrics.txt", "a") as f:
                f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Extract averages", description="Extract averages from a h5 file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--files", nargs="+", dest="files")
    parser.add_argument("--keys", nargs="+", dest="keys")
    args = parser.parse_args()

    dict_args = vars(args)

    main(**dict_args)
