import argparse
import os
import h5py
import numpy as np


def get_avg(h5df, key):
    # A function to calculate the average coverage of each h5 file for each key present.

    average = []
    # only add key if there is a coverage dataframe from which to calculate the average
    n = 1000
    for i in range(0, len(h5df["data/X"]), n):  # len(h5df["data/X"])
        # mask = [np.sum(chunk) == X.shape[1] for chunk in X] ?
        array = np.array(h5df[f"evaluation/{key}"][i:i + n], dtype=np.float32)
        average.append(np.mean(array[array >= 0]))  # only non-negatives/no padding filler values

    return np.mean(np.array(average))


def main(h5_file, keys, overwrite):
    h5df = h5py.File(h5_file, 'a')

    for key in keys:
        if f"average_{key}" in h5df["evaluation"].keys():
            if overwrite:
                print("Average of key {} already exists in {} and will be overwritten."
                      .format(key, h5_file.split("/")[-1]))
                h5df[f'evaluation/average_{key}'][:] = get_avg(h5df, key)
            else:
                print("Average has already been calculated for {} in {}. Key will be skipped."
                      .format(key, h5_file.split("/")[-1]))

        else:
            print("Adding average of {} to {}.".format(key, h5_file.split("/")[-1]))
            h5df.create_dataset(f'evaluation/average_{key}', shape=(1,),
                                maxshape=(None,), dtype=float, fillvalue=get_avg(h5df, key))

    h5df.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Avg", description="Add average of chosen NGS dataset to the h5 file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", type=str, default=None, required=True, help="h5 input file")
    parser.add_argument("--keys", type=str, action="append", required=True,
                        help="list of keys of which average should be calculated and added to the "
                             "file (--keys atacseq moaseq ...)")
    parser.add_argument("--overwrite", action="store_true",
                        help="if added will ignore if an average of a key already exists in the "
                             "file and still calculate the average")
    args = parser.parse_args()

    for j in range(len(args.keys)):
        args.keys[j] = args.keys[j] + "_coverage"

    assert os.path.isfile(args.h5_file), f"either {args.h5_file} doesn't exists or path is wrong"
    dict_args = vars(args)

    main(**dict_args)
