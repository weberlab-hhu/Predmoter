import sys
import logging
import time
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from predmoter.utilities.utils import log_table, file_stem

log = logging.getLogger(__name__)


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
        self.create_dataset()

    def __getitem__(self, idx):
        log.info("getitem")  # check when getitem is called
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
        be size is filled with -3 to ensure that the datasets don't get shifted/can be concatenated.
        If just one dataset is chosen, then the file is skipped entirely (see utils: get_h5_data).
        For predictions just the DNA data (X) is read in. The experimental data (Y) is averaged
        (average of bam file tracks/replicates).
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
                raise ValueError(f"all {self.type_} input files need to have the same sequence "
                                 f"length, here: {self.seq_len}")
            n, mem_size, chunks = 1000, 0, 0
            for i in range(0, len(h5df["data/X"]), n):  # for saving memory (RAM)
                X = np.array(h5df["data/X"][i:i + n], dtype=self.x_dtype)
                if self.type_ != "predict":
                    # excluding large assembly gaps
                    mask = ~np.array([x.all() for x in X])  # True if chunk contains just zeros
                    X = X[mask]

                    Y = []
                    for key in self.dsets:
                        if f"{key}_coverage" in h5df["evaluation"].keys():
                            y = np.array(h5df[f"evaluation/{key}_coverage"][i:i + n], dtype=self.y_dtype)
                            y = np.around(np.mean(y, axis=2), 4)  # avg of replicates, round to 4
                            y = y[mask]
                            Y.append(np.reshape(y, (y.shape[0], y.shape[1], 1)))
                        else:
                            Y.append(np.full((X.shape[0], X.shape[1], 1), -3., dtype=self.y_dtype))
                    self.Y += self._encode_one(np.concatenate(Y, axis=2))
                    mem_size += sum([sys.getsizeof(y) for y in Y])
                mem_size += sys.getsizeof(X)  # works?
                self.X += self._unbind(X)
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
            logging.info("\n", extra={"simple": True})

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

    def _unbind(self, array):
        """Split array into list of arrays. Reshape converts resulting array shapes from
        (1, seq_len, num_dsets) to (seg_len, num_dsets)."""
        return [np.reshape(arr, (self.seq_len, len(self.dsets))) for arr in np.split(array, len(array), axis=0)]
