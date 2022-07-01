import os
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PredmoterSequence(Dataset):
    compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=9, shuffle=2)  # class variable

    def __init__(self, h5_files, type_):
        self.h5_files = h5_files
        assert len(self.h5_files) >= 1, "no input file/s were provided"
        self.type_ = type_
        self.X, self.Y = [], []
        self.create_dataset()
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0) if self.mode != "test" else None

    def __getitem__(self, idx):
        if self.type_ == "test":
            return self.X[idx]
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

    def create_dataset(self):
        for h5_file in self.h5_files:
            h5df = h5py.File(h5_file, mode="r")
            X = np.array(h5df["data/X"], dtype=np.int8)
            if self.type_ == "test":
                self.X.append(self.encode_one(X))
            else:
                Y = np.array(h5df["evaluation/atacseq_coverage"], dtype=np.float32)
                assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between input and labels."
                Y = np.sum(Y, axis=2)
                mask = [len(y[y < 0]) == 0 for y in Y]  # exclude padded ends
                X = X[mask]
                Y = Y[mask]
                self.X.append(self.encode_one(X))
                self.Y.append(self.encode_one(Y))

    def encode_one(self, array):
        array = [self.compressor.encode(arr) for arr in array]
        return np.stack(array, axis=0)


def decode_one(array, dtype, shape):
    array = np.frombuffer(PredmoterSequence.compressor.decode(array), dtype=dtype)
    array = np.reshape(array, shape)
    array = np.array(array)  # array otherwise not writeable
    return torch.from_numpy(array).float()


def collate_fn(batch_list):
    # batch_list: list of dataset items with length=batch_size
    seq_len = PredmoterSequence.meta["seq_len"]
    bases = PredmoterSequence.meta["bases"]
    if len(batch_list[0]) == 2:  # train/val data
        X, Y = zip(*batch_list)
        X = torch.stack([decode_one(x, np.int8, (seq_len, bases)) for x in X])
        Y = torch.stack([decode_one(y, np.float32, (seq_len,)) for y in Y])
        return X, Y
    else:  # test data
        X = torch.stack([decode_one(x, np.int8, (seq_len, bases)) for x in batch_list])
        return X


def get_dataloader(input_dir, type_, shuffle, batch_size, num_workers):
    assert type_ in ["train", "val", "test"], "valid types are train, val or test"
    input_dir = "/".join([input_dir.rstrip("/"), type_])
    h5_files = ["/".join([input_dir, file]) for file in os.listdir(input_dir)]
    if type_ == "test":
        assert len(h5_files) == 1, "predictions should only be applied to individual files"
    dataloader = DataLoader(PredmoterSequence(h5_files, type_), batch_size=batch_size, shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
