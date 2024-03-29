import argparse
import os
import h5py
import numpy as np

from predmoter.core.constants import EPS


def pear_coeff(replicate, target, is_log=False):
    """Calculate the pearson correlation coefficient (numpy version).

            This function only accepts tensors with a maximum of 2 dimensions. It excludes NaNs
            so that NaN can be used for positions to be masked as masked tensors are at present
            still in development.

                Args:
                    replicate: 'numpy.array', experimental NGS coverage replicate 1
                    target: 'numpy.array', experimental NGS coverage replicate 2/target
                    is_log: 'bool', if True (default) assumes the models' predictions are logarithmic

                Returns:
                    'numpy.array', average pearson correlation coefficient (correlation between
                        the replicates are calculated per chunk/subsequence and then averaged)
    """

    dims = len(replicate.shape)
    assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
    if dims == 2:
        replicate = np.squeeze(replicate.transpose(0, 1))
        target = np.squeeze(target.transpose(0, 1))

    if is_log:
        replicate = np.exp(replicate)

    p = replicate - np.mean(replicate, axis=0)
    t = target - np.mean(target, axis=0)
    coeff = np.sum(p * t, axis=0) / (
            np.sqrt(np.sum(p ** 2, axis=0)) * np.sqrt(np.sum(t ** 2, axis=0)) + EPS)
    # eps avoiding division by 0
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
    for i in range(0, h5df["data/X"].shape[0], n):
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


def main(h5_file, output_dir, dsets, mode):
    h5df = h5py.File(h5_file, mode="r")

    for key in dsets:
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
    parser = argparse.ArgumentParser(prog="Calculate replicate Pearson's correlation",
                                     description="Calculate Pearson correlation for each experimental NGS data "
                                                 "replicate combination added in a given h5 file.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5-file", type=str, default=None, required=True,
                        help="h5 input file; not a prediction h5 file, but one created by Helixer to train, "
                             "validate or test on")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="output directory")
    parser.add_argument("--dsets", nargs="+", dest="dsets", required=True,
                        help="list of datasets of which average should be calculated and added to the "
                             "file (--dsets atacseq h3k4me3 ...)")
    parser.add_argument("--mode", type=str, default="none", help="test normal and normalized (mean/median) correlation")
    args = parser.parse_args()

    assert os.path.isfile(args.h5_file), f"either {args.h5_file} doesn't exist or path is wrong"
    assert os.path.exists(args.output_dir), f"the output path {args.output_dir} doesn't exist"
    assert args.mode in ["none", "mean", "median"], f"invalid mode {args.mode}, valid modes are none, median and mean"
    dict_args = vars(args)

    main(**dict_args)
