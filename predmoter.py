# LATER: maybe add python shebang
import os
import sys
import random
import numcodecs
import warnings
import logging
import h5py
import argparse
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

# Reduce stuff pytorch_lightning has to say (for now):
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-directory', type=str, default='.', required=True,
                    help="Path to a directory containing one train, val and test directory with h5-files.")
parser.add_argument('-o', '--output-dir', type=str, default='.')  # current directory
parser.add_argument("-m", "--mode", type=str, default=None, required=True, help="Valid modes: train or predict.")
parser.add_argument("--model", type=str, default=None, help="Path to the model to use for predictions.")
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-b', '--batch-size', type=int, default=25)
parser.add_argument('-tb', '--test-batch-size', type=int, default=10)
parser.add_argument("--example", type=bool, default=False, help="Example pt-files of a predict run.")
parser.add_argument('--cnn-layers', type=int, default=1)
parser.add_argument('--up', type=int, default=2, help="Multiplier used for up-scaling each convolutional layer.")
parser.add_argument('--lstm-layers', type=int, default=1)  # the more lstm layers = more epochs required
parser.add_argument('--lstm-units', type=int, default=16)
parser.add_argument('--kernel-size', type=int, default=5)
parser.add_argument('--filter-size', type=int, default=16)
parser.add_argument('--step', type=int, default=4)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('--device', type=str, default="gpu")
parser.add_argument('--num-devices', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--checkpoint-path', type=str, default=".")
parser.add_argument('--prefix', type=str, default="")
# parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = parser.parse_args()

# 1. Input stuff (own module in the future)
# ------------------------------------------
valid_modes = ["train", "predict"]  # and test??? else it's a little confusing
if args.mode not in valid_modes:
    sys.exit("ERROR: Valid modes are train or predict.")

if args.input_directory.endswith("/"):
    pass
else:
    args.input_directory = args.input_directory + "/"

# maybe put this under train, not needed for test/predict?
if args.seed is None:
    seed = random.randint(1, 1000)
    seed_everything(seed=seed, workers=True)  # seed for reproducibility
    print(f"A seed wasn't provided by the user. The random seed is: {seed}.")
else:
    seed_everything(seed=args.seed, workers=True)

compressor = numcodecs.blosc.Blosc(cname='blosclz', clevel=4, shuffle=2)


class H5Dataset(Dataset):
    def __init__(self, h5_file):
        self.dataset = self.create_dataset(h5_file)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def create_dataset(file):
        h5df = h5py.File(file, mode="r")
        X = np.array(h5df["data/X"], dtype=np.int8)
        Y = np.array(h5df["evaluation/atacseq_coverage"], dtype=np.float32)  # int16 is ignored by numpy
        assert np.shape(X)[:2] == np.shape(Y)[:2], "Size mismatch between arrays"
        Y = np.sum(Y, axis=2)
        mask = [len(y[y < 0]) == 0 for y in Y]  # exclude padded ends
        X = X[mask]
        Y = Y[mask]
        return tuple((compressor.encode(x), compressor.encode(y)) for x, y in zip(X, Y))


def create_dataloader(group, shuffle, batch_size):
    full_path = args.input_directory + group  # group: train, val, test
    datasets = [H5Dataset(full_path + "/" + file) for file in os.listdir(full_path)]
    dataset = ConcatDataset(datasets)
    # del datasets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    # del dataset
    # pin_memory = faster RAM to GPU transfer, only True if device is gpu?
    return dataloader


# 2. The NN
# ------------
sequence_length = 21384  # hardcoded for now
nucleotides = 4  # hardcoded for now


# at some point: try to implement as static methods: ypadding/y_pool (both not needed in the U-net)
def ypadding(l_in, l_out, stride, kernel_size):
    padding = math.ceil(((l_out - 1) * stride - l_in + kernel_size) / 2)
    # math.ceil to  avoid rounding half to even/bankers rounding, only needed for even L_in
    return padding


def y_pool(target):
    for layer in range(args.cnn_layers):
        l_in = sequence_length / args.step ** layer
        l_out = sequence_length / args.step ** (layer + 1)
        ypad = ypadding(l_in, l_out, args.step, args.kernel_size)
        target = F.avg_pool1d(target, args.kernel_size, stride=args.step, padding=ypad)
    return target


class LitHybridNet(pl.LightningModule):
    def __init__(self, seq_len, input_size, cnn_layers, filter_size, kernel_size, step, up,
                 hidden_size, lstm_layers, learning_rate):
        super(LitHybridNet, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.cnn_layers = cnn_layers
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.step = step
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate

        # for model summary in trainer
        # self.example_input_array = torch.zeros(6, seq_len, input_size)

        # for checkpoints
        self.save_hyperparameters()  # ?

        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []

        # CNN part:
        # -------------
        self.cnn_layer_list = nn.ModuleList()

        for layer in range(cnn_layers):
            l_in = seq_len / step ** layer
            l_out = seq_len / step ** (layer + 1)
            xpad = self.xpadding(l_in, l_out, step, 1, kernel_size)  # dilation=1

            self.cnn_layer_list.append(nn.Conv1d(input_size, filter_size, kernel_size, stride=step,
                                                 padding=xpad, dilation=1))
            input_size = filter_size
            filter_size = filter_size * up

        # LSTM part:
        # --------------
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)  # input_size=math.ceil(filter_size/up)
        # input: last dimension of tensor, output:(batch,new_seq_len,hidden_size)

        # Linear part:
        # --------------
        self.out = nn.Linear(hidden_size, 1)  # output_size hardcoded

    def forward(self, x):
        # CNN part:
        # -------------
        x = x.transpose(1, 2)  # convolution over the base pairs, change position of 2. and 3. dimension

        for layer in self.cnn_layer_list:
            x = F.relu(layer(x))

        x = x.transpose(1, 2)

        # LSTM part:
        # --------------
        # size of hidden and cell state: (num_layers,batch_size,hidden_size)
        x, _ = self.lstm(x)  # _=(hn, cn)

        # Linear part:
        # --------------
        x = self.out(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def training_step(self, batch, batch_idx):
        X = self.decode_one(batch[0], np.int8, shape=(sequence_length, nucleotides))
        Y = self.decode_one(batch[1], np.float32, shape=(sequence_length,))
        Y = y_pool(Y)
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return {"loss": loss, "acc": acc.detach()}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([out["loss"] for out in training_step_outputs]).mean()
        avg_acc = torch.stack([out["acc"] for out in training_step_outputs]).mean()
        self.train_losses.append(avg_loss.item())
        self.train_accuracy.append(avg_acc.item())
        self.log("avg_train_accuracy", avg_acc.item(), logger=False)

    def validation_step(self, batch, batch_idx):
        X = self.decode_one(batch[0], np.int8, shape=(sequence_length, nucleotides))
        Y = self.decode_one(batch[1], np.float32, shape=(sequence_length,))
        Y = y_pool(Y)
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return {"loss": loss, "acc": acc.detach()}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([out["loss"] for out in validation_step_outputs]).mean()
        avg_acc = torch.stack([out["acc"] for out in validation_step_outputs]).mean()
        self.val_losses.append(avg_loss.item())
        self.val_accuracy.append(avg_acc.item())

    def test_step(self, batch, batch_idx):
        X = self.decode_one(batch[0], np.int8, shape=(sequence_length, nucleotides))
        Y = self.decode_one(batch[1], np.float32, shape=(sequence_length,))
        Y = y_pool(Y)
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        metrics = {"loss": loss.item(), "acc": acc.item()}
        self.log_dict(metrics, logger=False)
        return metrics

    def predict_step(self, batch, batch_idx, **kwargs):  # kwargs to make PyCharm happy
        X = self.decode_one(batch[0], np.int8, shape=(sequence_length, nucleotides))
        Y = self.decode_one(batch[1], np.float32, shape=(sequence_length,))
        Y = y_pool(Y)
        pred = self(X)
        return {"prediction": pred, "Y": Y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def decode_one(arrays, np_datatype, shape):
        array_list = []
        for array in arrays:
            array = np.array(array)
            array = np.frombuffer(compressor.decode(array), dtype=np_datatype)
            array = np.reshape(array, shape)
            array_list.append(array)
        return torch.from_numpy(np.stack(array_list)).to(torch.float32)

    @staticmethod
    def xpadding(l_in, l_out, stride, dilation, kernel_size):
        padding = math.ceil(((l_out - 1) * stride - l_in + dilation * (kernel_size - 1) + 1) / 2)
        # math.ceil to  avoid rounding half to even/bankers rounding, only needed for even L_in
        return padding

    @staticmethod
    def pear_coeff(prediction, target, is_log=True):
        if is_log:
            prediction = torch.exp(prediction)
        p = prediction - torch.mean(prediction)
        t = target - torch.mean(target)
        coeff = torch.sum(p * t) / (torch.sqrt(torch.sum(p ** 2)) * torch.sqrt(torch.sum(t ** 2)) + 1e-8)
        # 1e-8 avoiding division by 0
        return coeff


hybrid_model = LitHybridNet(seq_len=sequence_length, input_size=nucleotides, cnn_layers=args.cnn_layers,
                            filter_size=args.filter_size, kernel_size=args.kernel_size, step=args.step,
                            up=args.up, hidden_size=args.lstm_units, lstm_layers=args.lstm_layers,
                            learning_rate=args.learning_rate)

# 3. Training
# -------------------
# 3.1 Define callbacks
# -------------------------------
checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="avg_train_accuracy", mode="max",
                                      dirpath=args.checkpoint_path,
                                      filename="hybrid_model_{epoch}_{avg_train_accuracy:.2f}",
                                      save_on_train_epoch_end=True)

callbacks = [checkpoint_callback]

# 3.2 Train with the trainer
# -------------------------------
trainer = pl.Trainer(callbacks=callbacks, devices=args.num_devices, accelerator=args.device, max_epochs=args.epochs,
                     logger=False, enable_model_summary=False, enable_progress_bar=False, deterministic=True)

if args.mode == "train":
    train_loader = create_dataloader("train", shuffle=True, batch_size=args.batch_size)
    val_loader = create_dataloader("val", shuffle=False, batch_size=args.batch_size)
    trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Create the loss and accuracy files
    # first loss and accuracy for validation is a sanity check, before training begins (this is left out)

    with open(f"{args.output_dir}/{args.prefix}_loss.txt", "w", encoding="utf-8") as f:
        for epoch, (train, val) in enumerate(zip(hybrid_model.train_losses, hybrid_model.val_losses[1:])):
            if epoch == 0:
                f.write("epoch training_losses validation_losses\n")
            f.write(f"{epoch + 1} {train} {val}\n")

    with open(f"{args.output_dir}/{args.prefix}_accuracy.txt", "w", encoding="utf-8") as g:
        for epoch, (train, val) in enumerate(zip(hybrid_model.train_accuracy, hybrid_model.val_accuracy[1:])):
            if epoch == 0:
                g.write("epoch training_accuracy validation_accuracy\n")
            g.write(f"{epoch + 1} {train} {val}\n")
else:
    test_loader = create_dataloader("test", shuffle=False, batch_size=args.test_batch_size)
    trainer.predict(model=hybrid_model, dataloaders=test_loader, ckpt_path=args.model)

# not a great way/method; it will need refinement
if args.example and args.mode != "predict":
    test_loader = create_dataloader("test", shuffle=False, batch_size=args.test_batch_size)
    best_hybrid_model = hybrid_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_hybrid_model.to("cuda")  # cuda hardcoded!
    with torch.no_grad():
        best_hybrid_model.eval()
        for inputs, coverage in test_loader:
            preds = best_hybrid_model.forward(inputs.to("cuda"))
            coverage = y_pool(coverage)
            break
    torch.save(preds, f"{args.output_dir}/test_prediction.pt")
    torch.save(coverage, f"{args.output_dir}/test_coverage.pt")
