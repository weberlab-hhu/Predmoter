import random
import os
import shutil
import glob
import subprocess
import h5py
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from predmoter.utilities.utils import get_h5_data, prep_predict_data, fasta2h5, get_meta, \
    get_available_datasets, file_stem
from predmoter.prediction.HybridModel import LitHybridNet
from predmoter.utilities.converter import Converter
from predmoter.utilities.dataset import PredmoterSequence, PredmoterSequence2


# Preparation
# ----------------
DEVICES = ["cpu", "gpu"]
TMP_DIR = f"tmp_{random.randint(0, 10_000)}"
BL_FILEPATH = "../side_scripts/add_blacklist.py"


@pytest.fixture(scope="session", autouse=True)
def check_cwd():
    CWD = os.getcwd()
    if not CWD.endswith("Predmoter/predmoter"):
        pytest.exit("Tests need to be run from the Predmoter/predmoter directory.")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(exclude_device, num_workers, num_devices, reproducibility_test):
    # Check arguments
    # -------------------------
    if exclude_device is not None:
        if exclude_device not in ["cpu", "gpu"]:
            pytest.exit("Valid devices to exclude are CPU or GPU.")
        DEVICES.remove(exclude_device)

    if "gpu" in DEVICES:
        if not torch.cuda.is_available:
            pytest.exit("There seems to be no GPU available on your system for PyTorch to use, please set "
                        "'--exclude-device gpu' or check your system.")
        if torch.cuda.device_count() < num_devices:
            pytest.exit(f"The chosen number of GPUs ({num_devices}) is higher than the available amount "
                        f"of GPUs ({torch.cuda.device_count()}), please choose a lower number with '--num-devices'.")

    if "cpu" in DEVICES:
        if os.cpu_count() < num_devices:
            pytest.exit(f"The chosen number of CPU cores ({num_devices}) is larger than the available "
                        f"number of CPU cores ({os.cpu_count()}), please choose a lower number with '--num-devices'")

    if os.cpu_count() < num_workers:
        pytest.exit(f"The chosen number of workers ({num_workers}) is larger than the available number of "
                    f"CPU cores ({os.cpu_count()}), please choose a lower number with '--num-workers'")

    # Setup test directory structure
    # --------------------------------
    # check existing directory structure
    if not os.path.exists(BL_FILEPATH):
        pytest.exit("The file add_blacklist.py, located in the directory '../side_scripts', doesn't exist.")
    if not os.path.exists("testdata"):
        pytest.exit("The testdata directory doesn't exist.")

    # create test directory structure
    if os.path.exists(TMP_DIR):  # should never be the case before testing
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
    os.makedirs(os.path.join(TMP_DIR, "train"))
    os.makedirs(os.path.join(TMP_DIR, "val"))
    os.makedirs(os.path.join(TMP_DIR, "test"))
    # copy files
    files = glob.glob(os.path.join("testdata", "*.h5"))
    for file in files:
        if "train" in file:
            # the train data will be copied to the directories train and test
            shutil.copy2(file, os.path.join(TMP_DIR, "train"))
            shutil.copy2(file, os.path.join(TMP_DIR, "test"))
        elif "val" in file:
            shutil.copy2(file, os.path.join(TMP_DIR, "val"))

    # copy two special files to test add_blacklist.py also to another directory
    shutil.copy2("testdata/train_data_2.h5", TMP_DIR)  # to test adding blacklist information
    shutil.copy2("testdata/train_data_5.h5", TMP_DIR)  # test overwriting blacklist information

    # create prediction directories to test command line predict on all devices
    for device in DEVICES:
        os.makedirs(os.path.join(TMP_DIR, f"{device}_predictions"))

    # run the test code first with yield
    yield
    # TEARDOWN
    # --------------
    # runs AFTER all tests have been run
    shutil.rmtree(TMP_DIR)


@pytest.fixture(scope="session", autouse=True)
def select_device():
    return "gpu" if "gpu" in DEVICES else "cpu"  # at least one device is always selected


# UNIT TESTS
# ---------------------------------------------------------------------
# 1. Test functions in utils.py
# -------------------------------
@pytest.mark.unittest
def test_get_available_datasets():
    file = "testdata/train_data_1.h5"
    test_datasets = [["atacseq", "h3k4me3"], ["atacseq", "moaseq"]]  # datasets the 'model' can hypothetically predict
    # datasets that the model can predict and the h5 file contains as well
    result_datasets = [["atacseq", "h3k4me3"], ["atacseq"]]

    for test_dsets, result_dsets in zip(test_datasets, result_datasets):
        assert result_dsets == get_available_datasets(file, test_dsets)


@pytest.mark.unittest
def test_finding_and_sorting_h5_input_files():
    # out of all training and validation data, train_data_3.h5, train_data_5.h5 and val_data_3.h5
    # don't contain the dataset h3k4me3 and should be sorted out by the get_h5_data() function
    result_h5_data = {
        "train": [os.path.join(TMP_DIR, "train", file) for file in
                  ["train_data_1.h5", "train_data_2.h5", "train_data_4.h5"]],
        "val": [os.path.join(TMP_DIR, "val", file) for file in ["val_data_1.h5", "val_data_2.h5"]]
    }
    assert result_h5_data == get_h5_data(TMP_DIR, mode="train", dsets=["h3k4me3"])


@pytest.mark.unittest
def test_prepping_predict_input_file():
    # prepping for fasta files should contain the key fasta with the value True
    files = ["testdata/pred_data.fa", "testdata/pred_data_1.h5"]
    result_pred_data = [{"predict": [files[0]], "fasta": True},
                        {"predict": [files[1]]}]
    for i in range(2):
        assert result_pred_data[i] == prep_predict_data(files[i])


@pytest.mark.unittest
def test_helixer_hook_fasta2h5_conversion():
    out_file = os.path.join(TMP_DIR, "conversion_test.h5")
    fasta2h5("testdata/pred_data.fa", out_file, 21_384, species="Species_artificialis_9", multiprocess=False)
    # the configuration of fasta2h5 above should lead to the same file as pred_data_1.h5
    # which was also creted by Helixer
    h5df = h5py.File("testdata/pred_data_1.h5", "r")
    out_h5df = h5py.File(out_file, "r")
    assert np.array_equal(h5df["data/X"][:], out_h5df["data/X"][:])
    assert np.array_equal(h5df["data/seqids"][:], out_h5df["data/seqids"][:])
    assert np.array_equal(h5df["data/species"][:], out_h5df["data/species"][:])
    assert np.array_equal(h5df["data/start_ends"][:], out_h5df["data/start_ends"][:])


@pytest.mark.unittest
def test_extracting_h5_file_meta_data():
    # the file train_data_1.h5 has the sequence length 21384 and pred_data_2.h5 has the sequence length 42768
    files = ["testdata/train_data_1.h5", "testdata/pred_data_2.h5"]
    result_meta = [(21_384, 4), (42_768, 4)]

    for i in range(2):
        assert result_meta[i] == get_meta(files[i])


@pytest.mark.unittest
def test_getting_file_basename():
    assert "train_data_1" == file_stem("testdata/train_data_1.h5")


# 2. Test HybridModel custom class functions
# -------------------------------------------
@pytest.mark.unittest
def test_padding_formulas():
    # Note: even kernel sizes are not supported for a step/stride of 1
    seq_len = 21_384  # default sequence length
    mock_x = torch.rand(2, 4, seq_len)  # mock input tensor, 2 subsequences, 4 bases (nucleotide encoding)
    filter_size = 8  # mock filter size
    # check 'all' possible configurations for layers, stride/step, kernel_size, dilation
    test_configs = [
        {"layers": 1, "stride": 1, "kernel_size": 9, "dilation": 1},
        {"layers": 3, "stride": 3, "kernel_size": 9, "dilation": 3},
        {"layers": 3, "stride": 3, "kernel_size": 9, "dilation": 2},
        {"layers": 3, "stride": 3, "kernel_size": 6, "dilation": 1},
        {"layers": 3, "stride": 3, "kernel_size": 8, "dilation": 4},
        {"layers": 3, "stride": 2, "kernel_size": 5, "dilation": 1},
        {"layers": 3, "stride": 2, "kernel_size": 9, "dilation": 2},
        {"layers": 3, "stride": 2, "kernel_size": 4, "dilation": 3},
        {"layers": 3, "stride": 2, "kernel_size": 2, "dilation": 2},
        {"layers": 2, "stride": 3, "kernel_size": 9, "dilation": 1},
        {"layers": 2, "stride": 3, "kernel_size": 5, "dilation": 4},
        {"layers": 2, "stride": 3, "kernel_size": 8, "dilation": 3},
        {"layers": 2, "stride": 3, "kernel_size": 10, "dilation": 6},
        {"layers": 2, "stride": 6, "kernel_size": 5, "dilation": 3},
        {"layers": 2, "stride": 6, "kernel_size": 11, "dilation": 4},
        {"layers": 2, "stride": 6, "kernel_size": 8, "dilation": 1},
        {"layers": 2, "stride": 6, "kernel_size": 4, "dilation": 2}
    ]
    # config tests (u=uneven, e=even); 16 tests and one stride=1 configuration
    # layers                 _________u__________                          _________e__________
    #                       /                    \                        /                    \
    # stride            ___u____              ___e____                ___u____              ___e____
    #                  /        \            /        \              /        \            /        \
    # kernel size     u         e           u         e             u         e           u         e
    #                /  \      /  \        /  \      /  \          /  \      /  \        /  \      /  \
    # dilation      u    e    u    e      u    e    u    e        u    e    u    e      u    e    u    e

    for configs in test_configs:
        x = mock_x
        input_size = 4  # bases/nucleotide encoding
        t_filter_size = 8  # mock filter size for transposed layers
        # test xpadding (padding for a 1D convolutional layers)
        for layer in range(configs["layers"]):
            l_in = seq_len / configs["stride"] ** layer
            l_out = seq_len / configs["stride"] ** (layer + 1)
            # input size: l_in, desired output size: l_out
            xpad = LitHybridNet.xpadding(l_in, l_out, configs["stride"], configs["dilation"], configs["kernel_size"])
            conv = torch.nn.Conv1d(input_size, filter_size, configs["kernel_size"],
                                   stride=configs["stride"], padding=xpad, dilation=configs["dilation"])
            x = conv(x)
            input_size = filter_size
            # check the correct sequence length after every iteration
            assert x.size(2) == l_out

        # test trans_padding (padding for a 1D transposed convolutional layers)
        out_pad = LitHybridNet.compute_output_padding(configs["stride"], configs["dilation"], configs["kernel_size"])

        for layer in range(configs["layers"], 0, -1):  # step through the layers backwards
            l_in = seq_len / configs["stride"] ** layer
            l_out = l_in * configs["stride"]
            tpad = LitHybridNet.trans_padding(l_in, l_out, configs["stride"], configs["dilation"],
                                              configs["kernel_size"], out_pad)
            # input size: l_in, desired output size: l_out
            trans_conv = torch.nn.ConvTranspose1d(input_size, t_filter_size, configs["kernel_size"],
                                                  stride=configs["stride"], padding=tpad,
                                                  dilation=configs["dilation"], output_padding=out_pad)
            x = trans_conv(x)
            input_size = t_filter_size
            # check the correct sequence length after every iteration
            assert x.size(2) == l_out


@pytest.mark.unittest
def test_compute_masking():
    # padded bases (sub-tensors of [0, 0, 0, 0]) should be marked as True,
    # so they will be marked as 'to mask'
    x = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1], [0, 1, 0, 0]],
                      [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 0]],
                      [[0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    # size: 2 subsequences, sequence length 5, 4 bases
    result_masks = [torch.tensor([[[False], [False], [False], [False], [False]],
                                  [[False], [False], [False], [False], [True]],
                                  [[False], [False], [False], [True], [True]]]),
                    torch.tensor([[[False, False], [False, False], [False, False], [False, False], [False, False]],
                                  [[False, False], [False, False], [False, False], [False, False], [True, True]],
                                  [[False, False], [False, False], [False, False], [True, True], [True, True]]])
                    ]
    for i in range(2):
        # test 1 (so no repeats) and 2
        assert torch.equal(result_masks[i], LitHybridNet.compute_mask(x, repeats=(i + 1)))


@pytest.mark.unittest
def test_custom_metrics():
    # the custom poisson negative log likelihood function should yield the same results as the
    # built-in loss function from torch when no NaN values are present
    pred_tensors = [torch.tensor([1, 2, 3, 4, 5]),
                    torch.tensor([[1, 2, 3, 4, 5],
                                  [2, 3, 4, 5, 6]])]
    target_tensors = [torch.tensor([0, 2, 3, 4, 9]),
                      torch.tensor([[0, 2, 3, 4, 9],
                                    [0, 3, 8, 5, 6]])]
    # 1D and 2D tensor test, non-logarithmic predictions
    for i in range(2):
        assert torch.equal(F.poisson_nll_loss(pred_tensors[i], target_tensors[i], log_input=False,
                                              full=False, eps=1e-08, reduction='mean'),
                           LitHybridNet.poisson_nll_loss(pred_tensors[i], target_tensors[i], is_log=False))

    # 1D and 2D tensor test 'with' NaN values (test masking), logarithmic predictions
    nan_pred_tensors = [torch.tensor([1, 2, torch.nan, 3, 4, 5, torch.nan]),
                        torch.tensor([[torch.nan, 1, 2, 3, 4, 5],
                                      [2, 3, 4, torch.nan, 5, 6]])]
    nan_target_tensors = [torch.tensor([0, 2, torch.nan, 3, 4, 9, torch.nan]),
                          torch.tensor([[torch.nan, 0, 2, 3, 4, 9],
                                        [0, 3, 8, torch.nan, 5, 6]])]
    for i in range(2):
        assert torch.equal(F.poisson_nll_loss(pred_tensors[i], target_tensors[i], log_input=True,
                                              full=False, eps=1e-08, reduction='mean'),
                           LitHybridNet.poisson_nll_loss(nan_pred_tensors[i], nan_target_tensors[i], is_log=True))

    # test pearson correlation
    # -------------------------
    # expected result: 1, logarithmic and non-logarithmic predictions
    # comparing the same tensor should result in a Pearson correlation coefficient of 1
    assert 1 == LitHybridNet.pear_coeff(pred_tensors[0].float(), pred_tensors[0].float(), is_log=False)
    assert 1 == LitHybridNet.pear_coeff(torch.log(pred_tensors[0]), pred_tensors[0].float(), is_log=True)
    # expected result: 0
    assert 0 == LitHybridNet.pear_coeff(pred_tensors[0].float(), torch.tensor([0., 0., 0., 0., 0.]), is_log=False)
    # expected result: -1
    assert -1 == LitHybridNet.pear_coeff(pred_tensors[0].float(), torch.tensor([-1., -2., -3., -4., -5.]), is_log=False)


@pytest.mark.unittest
def test_tensor_unpacking():
    in_tensors = [torch.tensor([[[2], [3], [8]],
                                [[9], [0], [2]],
                                [[3], [1], [1]]]),
                  torch.tensor([[[2, 1],
                                 [3, 4],
                                 [0, 9]],
                                [[6, 4],
                                 [8, 2],
                                 [7, 7]]])]
    out_tensors = [torch.tensor([[2, 3, 8],
                                 [9, 0, 2],
                                 [3, 1, 1]]),
                   torch.tensor([[2, 3, 0],
                                 [1, 4, 9],
                                 [6, 8, 7],
                                 [4, 2, 7]])]
    for i in range(2):
        assert torch.equal(out_tensors[i], LitHybridNet.unpack(in_tensors[i]))


# 3. Test Converter's static methods
# ------------------------------------
@pytest.mark.unittest
def test_chromosome_map_creation():
    h5df = h5py.File("testdata/pred_data_1.h5", "r")
    # map without blacklisting (excluding specific chromosomes from the conversion)
    result_map = {b"Chrom1": {"end": 51321, "+": np.array([0, 1, 2]), "-": np.array([5, 4, 3])},
                  b"Chrom2": {"end": 102642, "+": np.array([6, 7, 8, 9, 10]), "-": np.array([15, 14, 13, 12, 11])}}
    test_map = Converter.chrom_map(h5df, bl_chroms=None)
    assert result_map.keys() == test_map.keys()
    for key in result_map:
        assert result_map[key]["end"] == test_map[key]["end"]
        assert np.array_equal(result_map[key]["+"], test_map[key]["+"])
        assert np.array_equal(result_map[key]["-"], test_map[key]["-"])
    # map with blacklisting (exclude chromosome 2, i.e. Chrom2)
    result_map = {b"Chrom1": {"end": 51321, "+": np.array([0, 1, 2]), "-": np.array([5, 4, 3])}}
    test_map = Converter.chrom_map(h5df, bl_chroms=np.array(["Chrom2"]))
    assert result_map.keys() == test_map.keys()
    for key in result_map:
        assert result_map[key]["end"] == test_map[key]["end"]
        assert np.array_equal(result_map[key]["+"], test_map[key]["+"])
        assert np.array_equal(result_map[key]["-"], test_map[key]["-"])


@pytest.mark.unittest
def test_return_unique_duplicates():
    # [1, 1, 2, 2, 2, 1, 3, 3, 4, 4, 4, 2, 2] to [1, 2, 1, 3, 4, 2]
    test_arrays = [np.array([1, 1, 2, 2, 2, 1, 3, 3, 4, 4, 4, 2, 2]),
                   np.array([3, 5, 5, 5, 5, 3, 3, 6, 2, 2, 2, -1, -1, -1])]
    result_arrays = [np.array([1, 2, 1, 3, 4, 2]),
                     np.array([3, 5, 3, 6, 2, -1])]
    for i in range(2):
        assert np.array_equal(result_arrays[i], Converter.unique_duplicates(test_arrays[i]))


@pytest.mark.unittest
def test_extracting_start_and_end_coordinates():
    # the start and end of each 'bin' containing the same number are returned
    result_start_ends = (np.array([0, 1, 4, 7, 11, 12]), np.array([1, 4, 7, 11, 12, 14]))
    test_start_ends = Converter.get_start_ends(np.array([4, 5, 5, 5, 1, 1, 1, 4, 4, 4, 4, 9, 2, 2]))
    for i in range(2):
        assert np.array_equal(result_start_ends[i], test_start_ends[i])  # test each array in tuple


# 4. Test Dataset classes
# ---------------------------
@pytest.mark.unittest
def test_dataset_classes_without_filtering():
    # PredmoterSequence, the RAM intensive dataset class
    # test without filtering/blacklisting
    seq_len = 21384
    datasets = ["atacseq", "h3k4me3"]
    test_dataset = PredmoterSequence(["testdata/train_data_4.h5", "testdata/train_data_5.h5"],
                                     "train", datasets, seq_len, False)
    # train_data_4 contains 14 subsequences of 21384 bp, train_data_5 contains 20 but 1 is a "gap subsequence",
    # since + and - strand are used in the h5 file two subsequences are filtered out during the dataset creation,
    # so the length should be 32
    assert 32 == test_dataset.__len__()
    assert [14, 18] == test_dataset.chunks

    # PredmoterSequence2, the RAM efficient dataset class
    test_dataset2 = PredmoterSequence2(["testdata/train_data_4.h5", "testdata/train_data_5.h5"],
                                       "train", datasets, seq_len, False)

    assert 32 == test_dataset2.__len__()
    # the resulting coordinates contain a gap subsequence (train_data_5.h5), since + and - strand are used in
    # the h5 file two subsequences are excluded, the subsequences with the indices 1 and 8 of the second
    # h5 file (train_data_5.h5)
    result_coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
                              [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13],
                              [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 9], [1, 10],
                              [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19]])
    assert np.array_equal(result_coords, test_dataset2.coords)

    # create dataset items to test on
    # the first subsequence of train_data_4.h5
    dataset_item1 = test_dataset.__getitem__(0)
    dataset2_item1 = test_dataset2.__getitem__(0)
    # the first subsequence of train_data_5.h5
    dataset_item14 = test_dataset.__getitem__(14)
    dataset2_item14 = test_dataset2.__getitem__(14)
    # the resulting item returned should be the same as when using PredmoterSequence2.create_data()
    # the results of both dataset functions should be identical
    for i in range(2):  # the first subsequence of train_data_4.h5
        assert torch.equal(dataset_item1[i], dataset2_item1[i])  # test each tensor, result: tuple of X and Y (target)
    for i in range(2):  # the first subsequence of train_data_5.h5
        assert torch.equal(dataset_item14[i], dataset2_item14[i])
    # since train_data_5 only contains the dataset atacseq, a filler dataset should have been created for the
    # dataset h3k4me3, the second index of the dataset item is the target/label information with the shape
    # (sequence_length, datasets), i.e. (21384, 2), the second item in each row should be the filler value -3
    assert -3 == dataset_item14[1][:, 1][0] == dataset2_item14[1][:, 1][0]
    # check expected sequence length
    assert (seq_len, 4) == dataset_item1[0].size() == dataset2_item1[0].size()
    assert (seq_len, len(datasets)) == dataset_item1[1].size() == dataset2_item1[1].size()


@pytest.mark.unittest
def test_dataset_classes_with_filtering():
    # PredmoterSequence, the RAM intensive dataset class
    # test with filtering/blacklisting
    seq_len = 21384
    datasets = ["atacseq"]
    test_dataset = PredmoterSequence(["testdata/train_data_4.h5", "testdata/train_data_5.h5"],
                                     "train", datasets, seq_len, True)
    # train_data_4 contains 8 subsequences of 21384 bp after filtering blacklisted/flagged sequences, train_data_5
    # contains 14 after filtering blacklisted/flagged sequences and 1 is a "gap subsequence",
    # since + and - strand are used in the h5 file two subsequences are filtered out during the dataset creation,
    # so the length should be 20
    assert 20 == test_dataset.__len__()
    assert [8, 12] == test_dataset.chunks

    # PredmoterSequence2, the RAM efficient dataset class
    test_dataset2 = PredmoterSequence2(["testdata/train_data_4.h5", "testdata/train_data_5.h5"],
                                       "train", datasets, seq_len, True)

    assert 20 == test_dataset2.__len__()
    # the resulting coordinates contain a gap subsequence (train_data_5.h5), since + and - strand are used in
    # the h5 file two subsequences are excluded, the subsequences with the indices 1 and 8 of the second
    # h5 file (train_data_5.h5), the other indices were flagged using add_blacklist.py beforehand and also filtered
    # out since blacklisting was applied
    result_coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                              [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 9],
                              [1, 12], [1, 13], [1, 14], [1, 15]])
    assert np.array_equal(result_coords, test_dataset2.coords)

    # create dataset items to test on
    # the first subsequence of train_data_4.h5
    dataset_item1 = test_dataset.__getitem__(0)
    dataset2_item1 = test_dataset2.__getitem__(0)
    # the first subsequence of train_data_5.h5
    dataset_item8 = test_dataset.__getitem__(8)
    dataset2_item8 = test_dataset2.__getitem__(8)
    # the resulting item returned should be the same as when using PredmoterSequence2.create_data()
    # the results of both dataset functions should be identical
    for i in range(2):  # the first subsequence of train_data_4.h5
        assert torch.equal(dataset_item1[i], dataset2_item1[i])  # test each tensor, result: tuple of X and Y (target)
    for i in range(2):  # the first subsequence of train_data_5.h5
        assert torch.equal(dataset_item8[i], dataset2_item8[i])
    # both h5 files contain the required dataset atacseq, so there are no filler datasets
    # check expected sequence length
    assert (seq_len, 4) == dataset_item1[0].size() == dataset2_item1[0].size()
    # shape: (21384, 1)
    assert (seq_len, len(datasets)) == dataset_item1[1].size() == dataset2_item1[1].size()


@pytest.mark.unittest
def test_dataset_classes_predict_mode():
    # PredmoterSequence, the RAM intensive dataset class
    seq_len = 42768
    test_dataset = PredmoterSequence(["testdata/pred_data_2.h5"], "predict", None, seq_len, False)
    # the predict dataset always contains just one file, no subsequences will be filtered out
    assert 10 == test_dataset.__len__()
    assert [10] == test_dataset.chunks

    # PredmoterSequence2, the RAM efficient dataset class
    test_dataset2 = PredmoterSequence2(["testdata/pred_data_2.h5"], "predict", None, seq_len, False)

    assert 10 == test_dataset2.__len__()
    # the resulting coordinates contain the only h5 file coordinate and all subsequence coordinates
    result_coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9]])
    assert np.array_equal(result_coords, test_dataset2.coords)

    # create dataset items to test on
    # the first subsequence of pred_data_2.h5
    dataset_item1 = test_dataset.__getitem__(0)
    dataset2_item1 = test_dataset2.__getitem__(0)
    # the resulting item returned should be the same as when using PredmoterSequence2.create_data()
    # the results of both dataset functions should be identical
    assert torch.equal(dataset_item1, dataset2_item1)  # the first subsequence of pred_data_2.h5 (no tuple just X)
    # check expected sequence length, the predict dataset ONLY contains the DNA sequence information
    assert (seq_len, 4) == dataset_item1.size() == dataset2_item1.size()


# CMDLINE TESTS
# ---------------------------------------------------------------------
@pytest.mark.cmdlinetest
def test_train_mode(num_workers, num_devices, reproducibility_test):
    for device in DEVICES:
        # test training with mini model that should fit on every device
        # the two default datasets, atacseq and h3k4me3, are trained on
        command = f"Predmoter.py -m train -i {TMP_DIR} -o {TMP_DIR} --prefix {device}_train -e 2 -b 10 " \
                  f"--hidden-size 16 --filter-size 16 --seed 5 --num-devices {num_devices} " \
                  f"--num-workers {num_workers} --device {device}"
        subprocess.run(command.split(" "), shell=False, check=True)
        # test resuming training
        command = f"Predmoter.py -m train -i {TMP_DIR} -o {TMP_DIR} --prefix {device}_train -e 3 -b 10 " \
                  f"--num-devices {num_devices} --num-workers {num_workers} --device {device} " \
                  f"--resume-training"
        subprocess.run(command.split(" "), shell=False, check=True)

        if reproducibility_test:
            # the training run before used a specific seed, trained 2 epochs and resumed training for 1 epoch
            # if the results are reproducible a training run with the same seed running for 3 epochs
            # should lead to the exact same metrics, this will be tested for each device
            # both complete runs have different prefixes assigned to them, the metrics loggs of both are compared
            command = f"Predmoter.py -m train -i {TMP_DIR} -o {TMP_DIR} --prefix {device}_train_repro -e 3 -b 10 " \
                      f"--hidden-size 16 --filter-size 16 --seed 5 --num-devices {num_devices} " \
                      f"--num-workers {num_workers} --device {device}"
            subprocess.run(command.split(" "), shell=False, check=True)
            assert np.array_equal(np.loadtxt(f"{TMP_DIR}/{device}_train_metrics.log", skiprows=1),
                                  np.loadtxt(f"{TMP_DIR}/{device}_train_repro_metrics.log", skiprows=1))


@pytest.mark.cmdlinetest
def test_test_mode():
    # run test with the previously trained model using the two default datasets atacseq and h3k4me3
    command = f"Predmoter.py -m test -i {TMP_DIR} -o {TMP_DIR} -b 10 " \
              f"--model testdata/mini_model.ckpt"
    subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_test_mode_without_ram_efficiency():
    # test with ram efficient false to test the dataset class PredmoterSequence instead of PredmoterSequence2
    # the test data is read in per file and should be small enough to be tested on systems with small RAM
    command = f"Predmoter.py -m test -i {TMP_DIR} -o {TMP_DIR} -b 10 " \
              f"--model testdata/mini_model.ckpt --ram-efficient false"
    subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_test_mode_with_filtering():
    # test with blacklisting option toggled, this should apply to the two files train_data_4 and train_data_5
    # filtering flagged/blacklisted sequences only applies to h5 files containing the dataframe data/blacklist
    # which is added via add_blacklist.py
    command = f"Predmoter.py -m test -i {TMP_DIR} -o {TMP_DIR} -b 10 " \
              f"--model testdata/mini_model.ckpt -bl"
    subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_predict_mode():
    # the models used are the minimum size models trained in test_cmdline_training()
    # test fasta input file, predicting two datasets, output additional bigWig files
    for device in DEVICES:
        command = f"Predmoter.py -m predict -f testdata/pred_data.fa -o {TMP_DIR}/{device}_predictions -b 10 " \
                  f"--model testdata/mini_model.ckpt --device {device} --species Species_artificialis -of bw"
        subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_convert_predictions_2coverage(select_device):
    # test conversion to bedGraph, just the + strand, average strand conversion was tested during prediction tests
    command = f"convert2coverage.py -i testdata/pred_data_1_predictions.h5 " \
              f"-o {TMP_DIR} -of bg --basename pred_data --strand +"
    subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_convert_experimental_data_2coverage():
    # test conversion of experimental data, just the - strand
    command = f"convert2coverage.py -i testdata/train_data_4.h5 -o {TMP_DIR} -of bw " \
              f"--basename train_data_4 --strand + --experimental --datasets atacseq h3k4me3"
    subprocess.run(command.split(" "), shell=False, check=True)


@pytest.mark.cmdlinetest
def test_write_blacklist_text_file():
    command = f"python3 {BL_FILEPATH} -i testdata/train_data_2_sequence_report.jsonl " \
              f"--write-text-file {TMP_DIR}/train_data_2_blacklist.txt"
    subprocess.run(command.split(" "), shell=False, check=True)
    assert np.array_equal(np.array(["Unplaced1", "Unplaced2"]),
                          np.loadtxt(f"{TMP_DIR}/train_data_2_blacklist.txt", usecols=0, dtype=str))


@pytest.mark.cmdlinetest
def test_add_blacklist():
    # test adding flagged/blacklisted sequences to a h5 file not containing this information
    command = f"python3 {BL_FILEPATH} -i testdata/train_data_2_sequence_report.jsonl " \
              f"-h5 {TMP_DIR}/train_data_2.h5"
    subprocess.run(command.split(" "), shell=False, check=True)
    result_data_blacklist = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    h5df = h5py.File(f"{TMP_DIR}/train_data_2.h5", "r")
    assert np.array_equal(result_data_blacklist, h5df["data/blacklist"][:])


@pytest.mark.cmdlinetest
def test_overwrite_blacklist():
    # test overwriting previous information about flagged/blacklisted sequences
    # just tests if the overwrite process is working, does not compare an expected output
    command = f"python3 {BL_FILEPATH} -i testdata/train_data_5_sequence_report.jsonl " \
              f"-h5 {TMP_DIR}/train_data_5.h5 --overwrite"
    subprocess.run(command.split(" "), shell=False, check=True)
