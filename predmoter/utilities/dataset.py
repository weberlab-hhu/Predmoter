import sys
import logging
import time
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from predmoter.core.constants import MAX_VALUES_IN_RAM
from predmoter.utilities.utils import log_table, file_stem

log = logging.getLogger("PredmoterLogger")


class PredmoterSequence(Dataset):
    def __init__(self, h5_files, type_, dsets, seq_len):
        super().__init__()
        self.compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)
        self.h5_files = h5_files
        self.type_ = type_
        self.chunks = []
        self.total_mem_size = []
        self.dsets = dsets
        self.seq_len = seq_len
        self.x_dtype = np.int8
        self.y_dtype = np.float32
        self.X = []
        self.Y = []
        if type_ != "test":
            log.info(f"Loading {type_} data into memory ...")
        self.create_dataset()

    def __getitem__(self, idx):
        if self.type_ == "predict":
            return torch.from_numpy(self.X[idx]).float()
        return torch.from_numpy(self.X[idx]).float(), \
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
        data (Y) is averaged (average of bam file tracks/replicates).
        """

        # Start logging
        # ------------------
        space = 20
        log_table(log, ["H5 files", "Chunks", "NGS datasets", "Mem size (Gb)", "Loading time (min)"],
                  spacing=space, header=True)

        main_start = time.time()

        # Read in data
        # ------------------
        for idx, h5_file in enumerate(self.h5_files):
            file_start = time.time()
            h5df = h5py.File(h5_file, mode="r")
            if self.type_ != "predict":
                key_count = sum([1 for dset in self.dsets if f"{dset}_coverage" in h5df["evaluation"].keys()])
            else:
                key_count = 0
            if h5df["data/X"].shape[1] != self.seq_len:
                # in case of train/val/test data, just one predict file allowed
                raise ValueError(f"all {self.type_} input files need to have the same sequence "
                                 f"length, here: {self.seq_len}")
            n = MAX_VALUES_IN_RAM // self.seq_len  # dynamic step for different sequence lengths
            mem_size, chunks = 0, 0
            for i in range(0, len(h5df["data/X"]), n):  # read in chunks for saving memory (RAM)
                X = np.array(h5df["data/X"][i:i + n], dtype=self.x_dtype)
                if self.type_ != "predict":
                    Y = []
                    for key in self.dsets:
                        if f"{key}_coverage" in h5df["evaluation"].keys():
                            y = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=self.y_dtype)
                            y = np.around(np.mean(y, axis=2), 2)  # avg of replicates, round to 2 digits
                            Y.append(np.reshape(y, (y.shape[0], y.shape[1], 1)))
                        else:
                            Y.append(np.full((X.shape[0], X.shape[1], 1), -3, dtype=self.y_dtype))
                    Y = self._encode_one(np.concatenate(Y, axis=2))
                    mem_size += sum([sys.getsizeof(y) for y in Y])
                    self.Y += Y
                mem_size += sys.getsizeof(X)
                self.X += self._unbind(X, (X.shape[1], X.shape[2]))
                chunks += len(X)
            self.chunks.append(chunks)
            self.total_mem_size.append(mem_size)

            # Continue logging
            # ------------------
            log_table(log, [file_stem(h5_file), chunks, key_count, round((mem_size / 1024 ** 3), ndigits=2),
                            round((time.time() - file_start) / 60, ndigits=2)], spacing=space)

        # End of logging
        # ------------------
        if len(self.h5_files) > 1:
            log_table(log, [f"all {len(self.h5_files)} files", sum(self.chunks), "/",
                            round((sum(self.total_mem_size) / 1024 ** 3), ndigits=2),
                            round((time.time() - main_start) / 60, ndigits=2)],
                      spacing=space, table_end=True)
        else:
            log.info("\n", extra={"simple": True})

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

    @staticmethod
    def _unbind(array, shape):
        """Split array into list of arrays. Reshape converts resulting array shapes from
        (1, seq_len, num_dsets) to (seg_len, num_dsets)."""
        return [np.reshape(arr, shape) for arr in np.split(array, len(array), axis=0)]


class PredmoterSequence2(Dataset):
    def __init__(self, h5_files, type_, dsets, seq_len):
        super().__init__()
        self.h5_files = h5_files
        self.type_ = type_
        self.dsets = dsets
        self.seq_len = seq_len
        self.chunks = self.compute_chunks()  # i.e. file 1: 3 chunks, file 2: 7 chunks, chunks=[3, 10] (add together)
        # create info on chunk array function here

    def __getitem__(self, idx):
        i, j = self.get_coords(idx, self.chunks)
        h5df = h5py.File(self.h5_files[i], "r")
        if self.type_ == "predict":
            return torch.from_numpy(h5df["data/X"][j]).float()
        return torch.from_numpy(h5df["data/X"][j]).float(), self.create_y(h5df, j)

    def __len__(self):
        return self.chunks[-1].item()

    def compute_chunks(self):
        chunks = []
        chunk_count = 0
        for file in self.h5_files:
            h5df = h5py.File(file, "r")
            chunk_count += h5df["data/X"].shape[0]
            chunks.append(chunk_count)
        return torch.tensor(chunks)

    @staticmethod
    def get_coords(idx, chunk_tensor):
        if idx < chunk_tensor[0].item():  # first file
            return 0, idx
        chunk_tensor = chunk_tensor[chunk_tensor <= idx]
        return len(chunk_tensor), (idx - torch.max(chunk_tensor).item())

    def create_y(self, h5df, idx):
        Y = []
        for key in self.dsets:
            if f"{key}_coverage" in h5df["evaluation"].keys():
                y = torch.from_numpy(h5df[f"evaluation/{key}_coverage"][idx]).float()  # shape: (seq_len, bam_files)
                y = torch.mean(y, dim=1).round(decimals=2)  # avg of replicates, round to 2 digits
                Y.append(y)
            else:
                Y.append(torch.full((self.seq_len,), -3).float())
        return torch.stack(Y, dim=1)


def get_dataset(h5_files, type_, dsets, seq_len, ram_efficient):
    if ram_efficient:
        return PredmoterSequence2(h5_files, type_, dsets, seq_len)
    return PredmoterSequence(h5_files, type_, dsets, seq_len)
