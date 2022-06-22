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
        self.dataset = self.create_dataset(self.h5_file, self.mode)

    def __getitem__(self, idx):
        return self.dataset[idx]  # get one chunk of the array

    def __len__(self):  # INCORRECT
        return len(self.dataset)

    def create_dataset(self, file, mode):
        h5df = h5py.File(file, mode="r")
        X = np.array(h5df["data/X"], dtype=np.int8)
        # assert X.shape[1] == self.sequence_length, "H5 file has the wrong formatting."
        if mode == "test":
            return [self.compressor.encode(x) for x in X]  # do encode/decode for prediction?

        Y = np.array(h5df["evaluation/atacseq_coverage"], dtype=np.float32)
        assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between input and labels."

        Y = np.sum(Y, axis=2)
        mask = [len(y[y < 0]) == 0 for y in Y]  # exclude padded ends
        X = X[mask]
        Y = Y[mask]
        return tuple((self.compressor.encode(x), self.compressor.encode(y)) for x, y in zip(X, Y))


# has to be a class??
class PromoterSequence:
    def __init__(self, type_, input_dir, shuffle, batch_size):
        self.type_ = type_
        assert self.type_ in ["train", "val", "test"], "Valid types are train, val or test."
        self.input_dir = "/".join([input_dir.rstrip("/"), self.type_])
        self.h5_files = ["/".join([self.input_dir, file]) for file in os.listdir(self.input_dir)]

        if type_ == "test":
            assert len(self.h5_files) == 1, "Predictions should only be applied to individual files."

        self.datasets = [PromoterDataset(file, mode=self.type_) for file in self.h5_files]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataloader = self.create_dataloader(self.datasets, self.shuffle, self.batch_size)

    def __getitem__(self, idx):
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def create_dataloader(datasets, shuffle, batch_size):
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        return dataloader

    # stuff to keep in mind:
    # 1. compressor problem, right now: compressor = PromoterDataset.compressor in decode_one (also for seqlen & nucl.)
    # 2. trainer: train_dataloaders="variable".dataloader
