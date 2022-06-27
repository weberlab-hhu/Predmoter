import os
import numcodecs
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class PromoterDataset(Dataset):
    compressor = numcodecs.blosc.Blosc(cname="blosclz", clevel=6, shuffle=2)  # class variable
    sequence_length = 21384  # hardcoded for now
    nucleotides = 4  # hardcoded for now

    def __init__(self, h5_file, mode):
        self.h5_file = h5_file
        self.mode = mode
        self.dataset = self.create_dataset(self.h5_file)

    def __getitem__(self, idx):
        if self.mode == "test":
            return self.decode_one(self.dataset[idx], np.int8, (self.sequence_length, self.nucleotides))
        else:
            return (self.decode_one(self.dataset[idx][0], np.int8, (self.sequence_length, self.nucleotides)),
                    self.decode_one(self.dataset[idx][1], np.float32, (self.sequence_length,)))

    def __len__(self):
        return len(self.dataset)

    def create_dataset(self, file):
        h5df = h5py.File(file, mode="r")
        X = np.array(h5df["data/X"], dtype=np.int8)
        # assert X.shape[1] == self.sequence_length, "H5 file has the wrong formatting."
        if self.mode == "test":
            array = [self.compressor.encode(x) for x in X]
            return np.array(array)

        Y = np.array(h5df["evaluation/atacseq_coverage"], dtype=np.float32)
        assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between input and labels."

        Y = np.sum(Y, axis=2)
        mask = [len(y[y < 0]) == 0 for y in Y]  # exclude padded ends
        X = X[mask]
        Y = Y[mask]
        arrays = tuple((self.compressor.encode(x), self.compressor.encode(y)) for x, y in zip(X, Y))
        return np.array(arrays)  # should prevent num_workers from exceeding memory limit

    def decode_one(self, array, np_dtype, shape):
        array = np.frombuffer(self.compressor.decode(array), dtype=np_dtype)
        array = np.reshape(array, shape)
        array = np.array(array)
        return torch.from_numpy(array).float()


# has to be a class??
class PromoterSequence:
    def __init__(self, type_, input_dir, shuffle, batch_size, num_workers):
        self.type_ = type_
        assert self.type_ in ["train", "val", "test"], "Valid types are train, val or test."
        self.input_dir = "/".join([input_dir.rstrip("/"), self.type_])
        self.h5_files = ["/".join([self.input_dir, file]) for file in os.listdir(self.input_dir)]

        if type_ == "test":
            assert len(self.h5_files) == 1, "Predictions should only be applied to individual files."

        self.dataset = ConcatDataset([PromoterDataset(file, mode=self.type_) for file in self.h5_files])
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                     pin_memory=True, num_workers=self.num_workers)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    # stuff to keep in mind:
    # 1. un-dynamic: PromoterDataset.sequence_length & nucleotides)
    # 2. trainer: train_dataloaders="variable".dataloader
