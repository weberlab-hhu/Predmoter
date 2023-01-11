import argparse
import os
import h5py
import numpy as np


def get_metric(h5df, metric, key):
    # A function to calculate the average or median coverage of each h5 file for each key present.

    metrics = []
    n = 1000
    for i in range(0, len(h5df["data/X"]), n):  # len(h5df["data/X"])
        array = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=np.float32)
        array[array < 0] = np.nan  # only non-negatives/no padding filler values
        if metric == "mean":
            met = np.mean(np.nanmean(array, axis=1), axis=0)
        else:
            met = np.median(np.nanmedian(array, axis=1), axis=0)
        metrics.append(met)

    if metric == "mean":
        np.mean(np.array(metrics), axis=0)
    return np.median(np.array(metrics), axis=0)


def main(h5_file, keys, metrics, overwrite):
    assert os.path.isfile(h5_file), f"either {h5_file} doesn't exist or path is wrong"
    h5df = h5py.File(h5_file, "a")
    for metric in metrics:
        for key in keys:
            if f"{key}_{metric}s" in h5df["evaluation"].keys():
                if overwrite:
                    print("{}s of key {} already exists in {} and will be overwritten"
                          .format(metric, key, h5_file.split("/")[-1]))
                    h5df[f"evaluation/{key}_{metric}s"][:] = get_metric(h5df, metric, key)
                else:
                    print("{}s has already been calculated for {} in {}; task will be skipped"
                          .format(metric, key, h5_file.split("/")[-1]))

            else:
                # only add key if there is a coverage dataframe from which to calculate the average
                assert f"{key}_coverage" in h5df["evaluation"].keys(), f"there is no {key}" \
                                                                       f" dataset in {h5_file.split('/')[-1]}"
                print("adding {}s of {} to {}".format(metric, key, h5_file.split("/")[-1]))
                h5df.create_dataset(f"evaluation/{key}_{metric}s", shape=(len(h5df[f"{key}_meta"]["bam_files"]),),
                                    maxshape=(None,), dtype=float, data=get_metric(h5df, metric, key))

    h5df.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Add metrics",
                                     description="Add averages and medians of chosen NGS dataset(s) to the h5 file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", type=str, default=None, required=True, help="h5 input file")
    parser.add_argument("--keys", nargs="+", dest="keys", default=["atacseq", "h3k4me3"],
                        help="list of keys of which average should be calculated and added to the "
                             "file; (--keys atacseq h3k4me3 ...)")
    parser.add_argument("--metrics", nargs="+", dest="metrics", default=["mean", "median"],
                        help="list of metrics per key to add; (--metrics mean median ...)")
    parser.add_argument("--overwrite", action="store_true",
                        help="if added will ignore if an average of a key already exists in the "
                             "file and still calculate the average")
    args = parser.parse_args()

    dict_args = vars(args)

    main(**dict_args)
