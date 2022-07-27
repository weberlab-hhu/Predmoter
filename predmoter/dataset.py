import os
import sys
import logging
import glob
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PredmoterSequence(Dataset):
    compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)  # class variable

    def __init__(self, h5_files, type_):
        super().__init__()  # ?
        self.h5_files = h5_files
        self.type_ = type_
        self.X, self.Y = self.create_dataset()  # returns tuple

    def __getitem__(self, idx):
        if self.type_ == "test":
            return self.X[idx]
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

    def create_dataset(self):
        X_final, Y_final = [], []
        for idx, h5_file in enumerate(self.h5_files):
            h5df = h5py.File(h5_file, mode="r")
            X_list, Y_list, n = [], [], 1000  # 1000
            for i in range(0, len(h5df["data/X"]), n):  # for data saving len(h5df["data/X"])
                X = np.array(h5df["data/X"][i:i + n], dtype=np.int8)
                if self.type_ == "test":
                    X_list.append(X)
                else:
                    Y = np.array(h5df["evaluation/atacseq_coverage"][i:i + n], dtype=np.float32)
                    assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between input and labels."
                    Y[Y < 0] = np.nan  # replace negatives/missing information with nan
                    Y = np.nanmean(Y, axis=2)
                    Y = np.around(Y, 4)
                    mask = [sum(np.isnan(y)) == 0 for y in Y]  # exclude non-informative regions
                    X, Y = X[mask], Y[mask]
                    Y = self.encode_one(Y)
                    X_list.append(X)
                    Y_list.append(Y)
            if self.type_ == "test":  # only one file
                return np.concatenate(X_list, axis=0), None
            X_final.append(np.concatenate(X_list, axis=0))
            Y_final.append(np.concatenate(Y_list, axis=0))
            mem_size = (sys.getsizeof(X_final[idx]) + sys.getsizeof(Y_final[idx]))/1024**3
            logging.info("The compressed data of file {} with shape {} is {:.4f} Gb in size.".
                         format(h5_file.split("/")[-1], X_final[idx].shape, mem_size))
        return np.concatenate(X_final, axis=0), np.concatenate(Y_final, axis=0)

    def encode_one(self, array):
        array = [self.compressor.encode(arr) for arr in array]
        return np.stack(array, axis=0)


def decode_one(array, dtype, shape):
    array = np.frombuffer(PredmoterSequence.compressor.decode(array), dtype=dtype)
    array = np.reshape(array, shape)
    array = np.array(array)  # array otherwise not writeable
    return torch.from_numpy(array).float()


def collate_fn(batch_list, seq_len):
    # batch_list: list of dataset tuples with length=batch_size
    if len(batch_list[0]) == 2:  # train/val data
        X, Y = zip(*batch_list)
        X = torch.stack([torch.from_numpy(x).float() for x in X])
        Y = torch.stack([decode_one(y, np.float32, (seq_len,)) for y in Y])
        return X, Y
    return torch.stack([torch.from_numpy(x).float() for x in batch_list])  # test data/just X


def get_dataloader(input_dir, type_, shuffle, batch_size, num_workers, seq_len):
    assert type_ in ["train", "val", "test"], "valid types are train, val or test"  # needed? program defines those
    input_dir = "/".join([input_dir.rstrip("/"), type_])
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    if type_ == "test":
        assert len(h5_files) == 1, "predictions should only be applied to individual files"
    dataloader = DataLoader(PredmoterSequence(h5_files, type_), batch_size=batch_size,
                            shuffle=shuffle, pin_memory=True, num_workers=num_workers,
                            collate_fn=lambda batch: collate_fn(batch, seq_len))
    # lambda function makes it possible to add more arguments than batch to collate_fn
    return dataloader
