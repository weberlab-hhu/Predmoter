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


class PredmoterSequence(Dataset):
    compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)  # class variable

    def __init__(self, h5_files, type_):
        super().__init__()
        self.h5_files = h5_files
        self.type_ = type_
        self.chunks = []
        self.total_mem_size = []
        self.X = []
        self.Y = []
        self.create_dataset()

    def __getitem__(self, idx):
        if self.type_ == "test":
            return self.X[idx]
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

    def create_dataset(self):
        self.log_table(is_start=True)
        main_start = time.time()

        for idx, h5_file in enumerate(self.h5_files):
            file_start = time.time()
            h5df = h5py.File(h5_file, mode="r")
            avg = h5df["evaluation/average_atacseq_coverage"][0]
            n, mem_size, chunks = 1000, 0, 0  # 1000
            for i in range(0, len(h5df["data/X"]), n):  # for data saving; len(h5df["data/X"])
                X = np.array(h5df["data/X"][i:i + n], dtype=np.int8)
                if self.type_ != "test":
                    Y = np.array(h5df["evaluation/atacseq_coverage"][i:i + n], dtype=np.float32)
                    Y = np.mean(Y, axis=2)
                    Y = (Y / avg) * 4
                    Y = np.around(Y, 4)
                    assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between input and labels."
                    Y = self.encode_one(Y)  # needs to be an array for memory size calculation
                    mem_size += sys.getsizeof(Y)
                    self.Y += self.unbind(Y)
                mem_size += sys.getsizeof(X)
                self.X += self.unbind(X)
                chunks += len(X)
            self.chunks.append(chunks)
            self.total_mem_size.append(mem_size)
            self.log_table(idx + 1, h5_file.split("/")[-1], chunks, mem_size/1024**3, file_start)

        if len(self.h5_files) > 1:
            self.log_table("tot", "/", sum(self.chunks), sum(self.total_mem_size)/1024**3, main_start, is_end=True)

        else:
            logging.info("\n", extra={"simple": True})

    def encode_one(self, array):
        array = [self.compressor.encode(arr) for arr in array]
        return np.stack(array, axis=0)

    @staticmethod
    def unbind(array):
        return [np.squeeze(arr) for arr in np.split(array, len(array), axis=0)]

    @staticmethod
    def log_table(num=None, file=None, chunks=None, mem_size=None,
                  start_time=None, is_start=False, is_end=False):
        if is_start:
            msg = f"\n   | {'File': <25}| {'chunks': <8}| {'Memory size (Gb)': <20}" \
                  f"| {'Loading time (min)': <18}\n{'-' * 84}"

        else:
            duration = (time.time()-start_time)/60
            msg = "{0: <3}| {1: <25}| {2: <8}| {3: <20.4f}| {4:.2f}"\
                .format(num, file, chunks, mem_size, duration)
            if is_end:
                msg = f"{'-' * 84}\n" + msg + "\n"

        logging.info(msg, extra={"simple": True})


def decode_one(array, dtype):
    array = np.frombuffer(PredmoterSequence.compressor.decode(array), dtype=dtype)
    array = np.array(array)  # array otherwise not writeable
    return torch.from_numpy(array).float()


def collate_fn(batch_list):
    # batch_list: list of dataset tuples with length=batch_size
    if len(batch_list[0]) == 2:  # train/val data
        X, Y = zip(*batch_list)
        X = torch.stack([torch.from_numpy(x).float() for x in X])
        Y = torch.stack([decode_one(y, np.float32) for y in Y])
        return X, Y
    return torch.stack([torch.from_numpy(x).float() for x in batch_list])  # test data/just X


def get_dataloader(input_dir, type_, batch_size, num_workers):
    logging.info(f"Loading {type_} data into memory...")
    input_dir = os.path.join(input_dir, type_)
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    assert len(h5_files) >= 1, f"no input files with type {type_} were provided"
    if type_ == "test":
        assert len(h5_files) == 1, "predictions should only be applied to individual files"
    shuffle = True if type_ == "train" else False
    dataloader = DataLoader(PredmoterSequence(h5_files, type_), batch_size=batch_size,
                            shuffle=shuffle, pin_memory=True, num_workers=num_workers,
                            collate_fn=collate_fn)
    # lambda function makes it possible to add more arguments than just batch to collate_fn
    return dataloader
