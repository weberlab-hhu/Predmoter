import argparse
import os
import h5py
import numpy as np


def get_avg(h5df, key):
    # A function to calculate the average coverage of each h5 file for each key present.

    averages = []
    # only add key if there is a coverage dataframe from which to calculate the average
    n = 1000
    for i in range(0, len(h5df["data/X"]), n):
        array = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=np.float32)
        array[array < 0] = np.nan  # only non-negatives/no padding filler values
        means = np.mean(np.nanmean(array, axis=1), axis=0)
        averages.append(means)

    return np.mean(np.array(averages), axis=0)


def main(h5_file, keys, overwrite):
    h5df = h5py.File(h5_file, "a")

    for key in keys:
        if f"{key}_means" in h5df["evaluation"].keys():
            if overwrite:
                print("Average of key {} already exists in {} and will be overwritten."
                      .format(key, h5_file.split("/")[-1]))
                h5df[f"evaluation/{key}_means"][:] = get_avg(h5df, key)
            else:
                print("Average has already been calculated for {} in {}. Key will be skipped."
                      .format(key, h5_file.split("/")[-1]))

        else:
            print("Adding average of {} to {}.".format(key, h5_file.split("/")[-1]))
            h5df.create_dataset(f"evaluation/{key}_means", shape=(len(h5df[f"{key}_meta"]["bam_files"]),),
                                maxshape=(None,), dtype=float, fillvalue=get_avg(h5df, key))

    h5df.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Avg", description="Add average of chosen NGS dataset to the h5 file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", type=str, default=None, required=True, help="h5 input file")
    parser.add_argument("--keys", type=str, action="append", required=True,
                        help="list of keys of which average should be calculated and added to the "
                             "file (--keys atacseq h3k4me3 ...)")
    parser.add_argument("--overwrite", action="store_true",
                        help="if added will ignore if an average of a key already exists in the "
                             "file and still calculate the average")
    args = parser.parse_args()

    assert os.path.isfile(args.h5_file), f"either {args.h5_file} doesn't exists or path is wrong"
    dict_args = vars(args)

    main(**dict_args)
