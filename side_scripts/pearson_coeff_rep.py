import argparse
import os
import h5py
import numpy as np


def pear_coeff(prediction, target, is_log=False):
    # Function to calculate the pearson correlation (the numpy version)

    # For two dimensions the program needs to transpose the prediction and target arrays, so
    # that it compares each array to it's target individually instead. Example: two arrays of size:
    # (3,100), one is the prediction, the other the target. The function will compare the first values
    # of all three sub-arrays from the prediction with the first values of all three sub-arrays from the
    # target, then second values of all, then the third and so on. The squeeze function helps to calculate
    # the pearson correlation of arrays with size (i, 1).

    dims = len(prediction.shape)
    assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
    if dims == 2:
        prediction = np.squeeze(prediction.transpose(0, 1))
        target = np.squeeze(target.transpose(0, 1))

    if is_log:
        prediction = np.exp(prediction)

    p = prediction - np.mean(prediction, axis=0)
    t = target - np.mean(target, axis=0)
    coeff = np.sum(p * t, axis=0) / (
            np.sqrt(np.sum(p ** 2, axis=0)) * np.sqrt(np.sum(t ** 2, axis=0)) + 1e-8)
    # 1e-8 avoiding division by 0
    return np.mean(coeff)


def string_table(rep_names, array):
    assert len(array.shape) == 2, "the matrix should be a 2D array"

    header = " ".join(rep_names)
    s = "X " + header
    matrix = array.tolist()
    for i, mat in enumerate(matrix):
        row = rep_names[i] + " " + " ".join([str(n) for n in mat])
        s = s + "\n" + row
    return s


def calc_pearson_matrix(h5df, key, num, mode):
    matrix = []
    n = 1000
    for i in range(0, len(h5df["data/X"]), n):  # len(h5df["data/X"])
        array = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=np.float32)
        # array = array[array > 0]  # only non-negatives/no padding filler values are used; needed?
        # means and medians: right now old stuff/not usable
        if mode == "mean":
            array = (array / np.array(h5df[f"evaluation/{key}_{mode}s"], dtype=np.float32))
        elif mode == "median":
            array = (array / np.array(h5df[f"evaluation/{key}_{mode}s"], dtype=np.float32))

        cols = 0
        pears = np.full(shape=(num, num), fill_value=0., dtype=np.float32)  # sub-matrix
        for j in range(num):
            for k in range(num - cols):
                mask = np.where(array[:, :, j] != -1)  # -1 will also not be predicted
                pears[j, j + k] = pear_coeff(array[:, :, j][mask], array[:, :, j + k][mask])

            cols += 1

        matrix.append(np.expand_dims(pears, axis=2))

    matrix = np.mean(np.concatenate(matrix, axis=2), axis=2)

    # mirror along diagonal
    matrix_copy = np.array(matrix.T, copy=True)
    np.fill_diagonal(matrix_copy, 0.)
    matrix = matrix + matrix_copy
    return matrix


def main(h5_file, output_dir, keys, mode):
    h5df = h5py.File(h5_file, mode="r")

    for key in keys:
        if f"{key}_coverage" in h5df["evaluation"].keys():
            # helixer dev branch: meta in evaluation
            reps = [rep.decode().split('/')[-1].split('.')[0]
                    for rep in h5df[f"evaluation/{key}_meta/bam_files"][:]]  # replicates
            pearson_matrix = calc_pearson_matrix(h5df, key, len(reps), mode=mode)
            table = string_table(reps, pearson_matrix)

            with open(f"{output_dir}/{h5_file.split('/')[-1].split('.')[0]}_{key}_pearson.csv", "w") as f:
                f.write(table)

        else:
            print(f"{key} is not in {h5_file.split('/')[-1]}: skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Pearson's R",
                                     description="Calculate pearson correlation for each replicate combi.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", type=str, default=None, required=True, help="h5 input file")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="output directory")
    parser.add_argument("--keys", nargs="+", dest="keys", required=True,
                        help="list of keys of which average should be calculated and added to the "
                             "file (--keys atacseq h3k4me3 ...)")
    parser.add_argument("--mode", type=str, default="none", help="test normal and normalized (mean/median) correlation")
    args = parser.parse_args()

    assert os.path.isfile(args.h5_file), f"either {args.h5_file} doesn't exist or path is wrong"
    assert os.path.exists(args.output_dir), f"the output path {args.output_dir} doesn't exist"
    assert args.mode in ["none", "mean", "median"], f"invalid mode {args.mode}, valid modes are none, median and mean"
    dict_args = vars(args)

    main(**dict_args)
