import sys
import os
import logging
import glob
import time
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import log_table


class PredmoterSequence(Dataset):
    compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)  # class variable

    def __init__(self, h5_files, type_, keys):
        super().__init__()
        self.h5_files = h5_files
        self.type_ = type_
        self.chunks = []
        self.total_mem_size = []
        self.keys = keys
        self.X = []
        self.Y = []
        self.create_dataset()
        assert len(self.X) > 0, f"there is no {self.type_} data," \
                                f" the chosen h5-files don't contain the dataset(s): {' '.join(self.keys)}"

    def __getitem__(self, idx):
        if self.type_ == "test":
            return self.X[idx]
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

    def create_dataset(self):
        # The function to create X and Y and fill the total_mem_size and chunks lists.

        # The data is read in per file. The chunks of the file are then read in 1000 at a time, so that when
        # the allocated data is not/only partially freed later, the impact on memory size will be smaller,
        # since that was a big issue requiring way more RAM than was necessary beforehand.
        # If a specific NGS dataset does not exist for a h5 file the array is filled with NaN, unless just one
        # dataset is chosen, then the file is skipped entirely. For test data, just the DNA data (X)
        # is read in, since test data usually doesn't have experimental data (Y). The data is normalized by taking
        # the average coverage of each bam_file (origin of the experimental data) and divide the coverage of the
        # bam_files by the average. The averages of the bam_files have been precalculated using the script
        # "add_average.py". The normalized data is multiplied by 5, since around 0-2 reads per base pair
        # the poisson loss used by the model looses a bit of its accuracy, not punishing differences enough.

        space = 24
        # start logging
        log_table(["H5 files", "Chunks", "NGS datasets", "Mem size (Gb)", "Loading time (min)"], spacing=space,
                  header=True)
        main_start = time.time()

        for idx, h5_file in enumerate(self.h5_files):
            file_start = time.time()
            h5df = h5py.File(h5_file, mode="r")

            if len(self.keys) == 1 and f"{self.keys[0]}_coverage" not in h5df["evaluation"].keys():
                log_table([h5_file.split('/')[-1], f"no {self.keys[0]} data: skipped", "0", "0", "/"], spacing=24)
                continue

            key_count = 0
            n, mem_size, chunks = 1000, 0, 0
            for i in range(0, len(h5df["data/X"]), n):  # for data saving; len(h5df["data/X"])
                X = np.array(h5df["data/X"][i:i + n], dtype=np.int8)
                if self.type_ != "test":
                    Y = []
                    key_count = 0
                    for key in self.keys:
                        if f"{key}_coverage" in h5df["evaluation"].keys():
                            key_count += 1
                            y = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=np.float32)
                            y = (y / np.array(h5df[f"evaluation/{key}_means"])) * 5
                            y = np.mean(y, axis=2)
                            y = np.around(y, 4)
                            assert np.shape(X)[:2] == np.shape(y)[:2], "Size mismatch between input and labels."
                            Y.append(np.reshape(y, (y.shape[0], y.shape[1], 1)))
                        else:
                            Y.append(np.full((X.shape[0], X.shape[1], 1), np.nan, dtype=np.float32))
                    Y = self.encode_one(np.concatenate(Y, axis=2))  # needs to be an array for memory size calculation
                    mem_size += sys.getsizeof(Y)
                    self.Y += self.unbind(Y)
                mem_size += sys.getsizeof(X)
                self.X += self.unbind(X)
                chunks += len(X)
            self.chunks.append(chunks)
            self.total_mem_size.append(mem_size)
            # more logging
            log_table([h5_file.split('/')[-1], str(chunks), str(key_count), f"{mem_size / 1024 ** 3:.4f}",
                       f"{(time.time() - file_start) / 60:.2f}"], spacing=space)

        # even more logging
        if len(self.h5_files) > 1:
            log_table([f"all {len(self.h5_files)} files", str(sum(self.chunks)), "/",
                       f"{sum(self.total_mem_size) / 1024 ** 3:.4f}", f"{(time.time() - main_start) / 60:.2f}"],
                      spacing=space, table_end=True)

        else:
            logging.info("\n", extra={"simple": True})

    def encode_one(self, array):
        # encoding is done per chunk of the array, so shuffling the chunks is possible during training
        # if the entire array of 1000 was encoded, it would need to be decoded as one as well
        array = [self.compressor.encode(arr) for arr in array]
        return np.stack(array, axis=0)

    @staticmethod
    def unbind(array):
        # squeeze converts resulting array shapes from (1, seq_len, i) to (seg_len, i)
        return [np.squeeze(arr) for arr in np.split(array, len(array), axis=0)]


def decode_one(array, dtype, shape):
    array = np.frombuffer(PredmoterSequence.compressor.decode(array), dtype=dtype)
    array = np.reshape(array, shape)
    array = np.array(array)  # array otherwise not writeable
    return torch.from_numpy(array).float()


def collate_fn(batch_list, seq_len, ngs_count):
    # Custom collate function to yield a decoded tuple of tensors.

    # The batch_list is a list of dataset tuples each with length=batch_size
    # (e.g.: [(tuple of X data), (tuple of Y data)]). The X is converte from np.int8 (smaller in memory)
    # to torch.float32 for training. The Y is decoded from bytes to floats.
    # The final shape of X is (batch_size, seq_len, bases), bases are always 4 for nucleotide encoding.
    # The final shape of Y is (batch_size, seq_len, number_of_datasets), dataset number starts
    # with 1 and is theoretically unlimited.

    if len(batch_list[0]) == 2:  # train/val data
        X, Y = zip(*batch_list)
        X = torch.stack([torch.from_numpy(x).float() for x in X])
        Y = torch.stack([decode_one(y, np.float32, (seq_len, ngs_count)) for y in Y])
        return X, Y
    return torch.stack([torch.from_numpy(x).float() for x in batch_list])  # test data/just X


def get_dataloader(input_dir, type_, batch_size, seq_len, datasets):
    # Creates the dataloaders used for training, validating and predicting.

    # This function collects all files of a specific type (train, val or test) into a list.
    # The test data is used for prediction only. The dataloader is a wrapper around the dataset
    # from PredmoterSequence. It handles shuffling the training data and the batch size (should be
    # scaled around how much data fits on the GPU). The collate_fn is the function to get the numpy arrays
    # and byte objects from the dataset and convert them to larger (in RAM) and decoded tensors.
    # The bigger data will just be on the GPU and not overflowing RAM, as reading in tensors directly from
    # the h5 files would do.

    logging.info(f"Loading {type_} data into memory...")
    input_dir = os.path.join(input_dir, type_)
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    assert len(h5_files) >= 1, f"no input files with type {type_} were provided"
    if type_ == "test":
        assert len(h5_files) == 1, "predictions should only be applied to individual files"
    shuffle = True if type_ == "train" else False
    dataloader = DataLoader(PredmoterSequence(h5_files, type_, keys=datasets), batch_size=batch_size,
                            shuffle=shuffle, pin_memory=True, num_workers=0,
                            collate_fn=lambda batch: collate_fn(batch, seq_len, len(datasets)))
    # lambda function makes it possible to add more arguments than just batch to collate_fn
    return dataloader
