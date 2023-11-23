import numpy as np
import h5py
import argparse
import os


def main(files, dsets, metrics):
    for file in files:
        assert os.path.isfile(file), f"either {file} doesn't exist or path is wrong"

        h5df = h5py.File(file, mode="r")
        for i, key in enumerate(dsets):
            assert f"{key}_coverage" in h5df["evaluation"].keys(), f"there is no {key}" \
                                                                   f" dataset in {file.split('/')[-1]}"
            values = []
            for metric in metrics:
                if f"{key}_{metric}s" not in h5df["evaluation"].keys():
                    value = None
                else:
                    value = np.array(h5df[f"evaluation/{key}_{metric}s"])
                values.append(value)
            if i == 0:
                msg = f"File: {file.split('/')[-1]}\n" \
                      f"{key}_{metrics[0]}s: {values[0]}\t{key}_{metrics[1]}s: {values[1]}\n"
            else:
                msg = f"{key}_{metrics[0]}s: {values[0]}\t{key}_{metrics[1]}s: {values[1]}\n"
            with open("metrics.txt", "a") as f:
                f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Extract averages",
                                     description="Extract metrics (mean or median) from a h5 file. The metrics have "
                                                 "to have been added with 'add_metrics.py.'",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--files", nargs="+", dest="files")
    parser.add_argument("--dsets", nargs="+", dest="dsets", default=["atacseq", "h3k4me3"],
                        help="list of datasets of which average should be calculated and added to the "
                             "file; (--dsets atacseq h3k4me3 ...)")
    parser.add_argument("--metrics", nargs="+", dest="metrics", default=["mean", "median"],
                        help="list of metrics per dataset to add; (--metrics mean median)")
    args = parser.parse_args()

    dict_args = vars(args)

    main(**dict_args)
