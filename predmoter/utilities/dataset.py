import sys
import logging
import time
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from predmoter.core.constants import MAX_VALUES_IN_RAM
from predmoter.utilities.utils import log_table, file_stem, rank_zero_info

log = logging.getLogger("PredmoterLogger")


class PredmoterSequence(Dataset):
    def __init__(self, h5_files, type_, dsets, seq_len, blacklist):
        super().__init__()
        self.compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)
        self.h5_files = h5_files
        self.type_ = type_
        self.chunks = []
        self.total_mem_size = []
        self.dsets = dsets
        self.seq_len = seq_len
        self.blacklist = blacklist
        self.x_dtype = np.float16
        self.y_dtype = np.float32
        self.bases = h5py.File(h5_files[0], mode="r")["data/X"].shape[-1]
        self.X = []
        self.Y = []
        if type_ != "test":
            rank_zero_info(f"Loading {type_} data into memory ...")
        self.create_dataset()

    def __getitem__(self, idx):
        if self.type_ == "predict":
            return self._decode_one(self.X[idx], self.x_dtype, shape=(self.seq_len, self.bases))
        return self._decode_one(self.X[idx], self.x_dtype, shape=(self.seq_len, self.bases)), \
            self._decode_one(self.Y[idx], self.y_dtype, shape=(self.seq_len, len(self.dsets)))

    def __len__(self):
        return sum(self.chunks)

    def create_dataset(self):
        """Fill X, Y, the total_mem_size and chunks lists.

        The data is read in per h5 file. The chunks of the file are then read in 1000 at a time,
        so that the impact on RAM will be smaller when the allocated data is not/only partially
        freed later. If a specific NGS dataset does not exist in a h5 file an array of would
        be size is filled with -3 to ensure that the datasets don't get shifted/that they can be
        concatenated. If just one dataset is chosen, then the file is skipped entirely
        (see utils: get_h5_data). For predictions just the DNA data (X) is read in. The experimental
        data (Y) is averaged (average of bam file tracks/replicates). For train/val/test data
        entire gap chunks (chunks just containing Ns) are filtered out beforehand. If blacklisting is
        applied chunks in data/blacklist marked as False are excluded (see side_scripts/add_blacklist.py).
        """

        # Start logging
        # ------------------
        space = 20
        log_table(log, ["H5 files", "Chunks", "NGS datasets", "Mem size (Gb)", "Loading time (min)"],
                  spacing=space, header=True, rank_zero=True)

        main_start = time.time()

        # Read in data
        # ------------------
        for idx, h5_file in enumerate(self.h5_files):
            file_start = time.time()
            h5df = h5py.File(h5_file, mode="r")
            if self.type_ != "predict":
                key_count = sum([1 for dset in self.dsets if f"{dset}_coverage" in h5df["evaluation"].keys()])
            else:
                key_count = "/"
            if h5df["data/X"].shape[1] != self.seq_len:
                # in case of train/val/test data, just one predict file allowed
                raise ValueError(f"all {self.type_} input files need to have the same sequence "
                                 f"length, here: {self.seq_len}")
            n = MAX_VALUES_IN_RAM // self.seq_len  # dynamic step for different sequence lengths
            mem_size, chunks = 0, 0
            blacklist = self.blacklist and "blacklist" in h5df["data"].keys()  # bool
            for i in range(0, h5df["data/X"].shape[0], n):  # read in chunks for saving memory (RAM)
                X = np.array(h5df["data/X"][i:i + n], dtype=self.x_dtype)
                if self.type_ != "predict":
                    # masking
                    # -----------------
                    if blacklist:
                        mask = np.logical_and(np.max(X[:, :, 0], axis=1) != 0.25,
                                              np.array(h5df["data/blacklist"][i:i + n], dtype=bool))
                    else:
                        mask = np.max(X[:, :, 0], axis=1) != 0.25  # mask entire N chunks
                    X = X[mask]
                    if X.shape[0] == 0:  # very unlikely, but just in case
                        continue
                    # ------------------
                    Y = []
                    for key in self.dsets:
                        if f"{key}_coverage" in h5df["evaluation"].keys():
                            y = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=self.y_dtype)
                            y = y[mask]
                            y = np.around(np.mean(y, axis=2), 2)  # avg of replicates, round to 2 digits
                            Y.append(np.reshape(y, (y.shape[0], y.shape[1], 1)))
                        else:
                            Y.append(np.full((X.shape[0], X.shape[1], 1), -3, dtype=self.y_dtype))
                    Y = self._encode_one(np.concatenate(Y, axis=2))
                    mem_size += sum([sys.getsizeof(y) for y in Y])
                    self.Y += Y
                X = self._encode_one(X)
                mem_size += sum([sys.getsizeof(x) for x in X])
                self.X += X
                chunks += len(X)
            self.chunks.append(chunks)
            self.total_mem_size.append(mem_size)

            # Continue logging
            # ------------------
            log_table(log, [file_stem(h5_file), chunks, key_count, round((mem_size / 1024 ** 3), ndigits=2),
                            round((time.time() - file_start) / 60, ndigits=2)], spacing=space, rank_zero=True)

        # End of logging
        # ------------------
        if len(self.h5_files) > 1:
            log_table(log, [f"all {len(self.h5_files)} files", sum(self.chunks), "/",
                            round((sum(self.total_mem_size) / 1024 ** 3), ndigits=2),
                            round((time.time() - main_start) / 60, ndigits=2)],
                      spacing=space, table_end=True, rank_zero=True)
        else:
            rank_zero_info("\n", simple=True)

    def _encode_one(self, array):
        """Compress array in RAM.

        The encoding is done per chunk of the array, so shuffling the chunks is possible
        during training. If the entire array of 1000 would be encoded, it would need to be
        decoded as one array as well, making shuffling/batching impossible.
        """
        return [self.compressor.encode(arr) for arr in array]

    def _decode_one(self, array, dtype, shape):
        """Decode one compressed array into a pytorch float tensor."""
        array = np.frombuffer(self.compressor.decode(array), dtype=dtype)
        array = np.reshape(array, shape)
        array = np.array(array)  # without this line the array is not writeable
        return torch.from_numpy(array).float()


class PredmoterSequence2(Dataset):
    def __init__(self, h5_files, type_, dsets, seq_len, blacklist):
        super().__init__()
        self.h5_files = h5_files
        self.type_ = type_
        self.dsets = dsets
        self.seq_len = seq_len
        self.blacklist = blacklist
        if type_ != "test":
            rank_zero_info(f"Creating {self.type_} dataset...")
        self.coords = self.get_coords()

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        h5df = h5py.File(self.h5_files[i], "r")
        return self.create_data(h5df, j)

    def __len__(self):
        return self.coords.shape[0]

    def get_coords(self):
        """Get array of file and chunk indices to pick from. For train/val/test data
        entire gap chunks (chunks just containing Ns) are filtered out beforehand. If
        blacklisting is applied chunks in data/blacklist marked as False are excluded
        (see side_scripts/add_blacklist.py). (see PredmoterSequence for more details)

        Returns: numpy array of shape (dataset_lengths, 2) where the first column contains
        the index of the h5 file to read and the second the index of the chunk to load.
        """
        space = 24
        main_start = time.time()
        log_table(log, ["H5 files", "Chunks", "NGS datasets", "Processing time (min)"],
                  spacing=space, header=True, rank_zero=True)  # logging
        coords = np.empty((0, 2), dtype=int)
        for i, h5_file in enumerate(self.h5_files):
            file_start = time.time()
            h5df = h5py.File(h5_file, "r")
            if h5df["data/X"].shape[1] != self.seq_len:
                # in case of train/val/test data, just one predict file allowed
                raise ValueError(f"all {self.type_} input files need to have the same sequence "
                                 f"length, here: {self.seq_len}")

            if self.type_ != "predict":
                n = MAX_VALUES_IN_RAM // self.seq_len
                chunks = 0
                blacklist = self.blacklist and "blacklist" in h5df["data"].keys()  # bool
                for j in range(0, h5df["data/X"].shape[0], n):
                    X = np.array(h5df["data/X"][j:j + n])
                    indices = np.where(np.max(X[:, :, 0], axis=1) != 0.25)[0]
                    if blacklist:
                        mask = np.array(h5df["data/blacklist"][j:j + n], dtype=bool)
                        indices = indices[mask[indices]] + j  # only mask existing indices
                    else:
                        indices = indices + j
                    chunks += indices.shape[0]
                    coords = np.append(coords, np.concatenate([np.full((indices.shape[0], 1), fill_value=i),
                                                               indices.reshape(indices.shape[0], 1)], axis=1), axis=0)
                num_ngs_dsets = sum([f"{dset}_coverage" in h5df["evaluation"].keys() for dset in self.dsets])
                log_table(log, [file_stem(h5_file), chunks, num_ngs_dsets,
                                round((time.time() - file_start) / 60, ndigits=2)], spacing=space, rank_zero=True)
            else:
                coords = np.concatenate([np.full((h5df["data/X"].shape[0], 1), fill_value=i),
                                         np.arange(h5df["data/X"].shape[0]).reshape(h5df["data/X"].shape[0], 1)],
                                        axis=1)
                log_table(log, [file_stem(h5_file), h5df["data/X"].shape[0], "/", "/"], spacing=space, rank_zero=True)

        # logging end
        # -----------
        if len(self.h5_files) > 1:
            log_table(log, [f"all {len(self.h5_files)} files", coords.shape[0], "/",
                            round((time.time() - main_start) / 60, ndigits=2)],
                      spacing=space, table_end=True, rank_zero=True)
        else:
            rank_zero_info("\n", simple=True)
        return coords

    def create_data(self, h5df, idx):
        """Create the correct Y array/tensor the exact same way as in PredmoterSequence."""
        X = torch.from_numpy(h5df["data/X"][idx]).float()
        if self.type_ == "predict":
            return X

        Y = []
        for key in self.dsets:
            if f"{key}_coverage" in h5df["evaluation"].keys():
                y = torch.from_numpy(h5df[f"evaluation/{key}_coverage"][idx]).float()  # shape: (seq_len, bam_files)
                y = torch.mean(y, dim=1).round(decimals=2)  # avg of replicates, round to 2 digits
                Y.append(y)
            else:
                Y.append(torch.full((self.seq_len,), -3).float())
        return X, torch.stack(Y, dim=1)


def get_dataset(h5_files, type_, dsets, seq_len, ram_efficient, blacklist):
    """Choose between two dataset classes.

    There are two dataset classes, that can be used by Predmoter:

        1. PredmoterSequence:
            - argument: ``--ram-efficient false`` (default)
            - compresses all data and stores it in RAM
            - takes time, depending on the dataset size a significant amount of time,
              (the longest tested around 1.5 h), before training to process all the data
            - the data processing time is multiplied by the number of devices used to train on
            - training is faster afterwards, since the data was already processed

        2. PredmoterSequence2:
            - argument: ``--ram-efficient true``
            - reads data directly from the hard-drive/file for each chunk
            - takes less time (the longest tested around 15 min) before the training to process the data
            - slows down training a bit as the data is always reprocessed at each get_item call
            - extremely RAM efficient
            - Warning: Don't move the input data while Predmoter is running.

    (see performance.md in docs for more details)
    """
    # mention test/predict behavior
    if ram_efficient:
        return PredmoterSequence2(h5_files, type_, dsets, seq_len, blacklist)
    return PredmoterSequence(h5_files, type_, dsets, seq_len, blacklist)
