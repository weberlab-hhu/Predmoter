import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# add way more selves
class LitHybridNet(pl.LightningModule):
    def __init__(self, cnn_layers, filter_size, kernel_size, step, up,
                 hidden_size, lstm_layers, learning_rate, seq_len, input_size):
        super(LitHybridNet, self).__init__()
        self.seq_len = seq_len  # keep or replace?
        self.input_size = input_size  # self.input_size is not effected by input_size = filter_size
        self.cnn_layers = cnn_layers
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.step = step
        self.up = up
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate

        # for model summary in trainer
        self.example_input_array = torch.zeros(2, self.seq_len, self.input_size)

        # for checkpoints
        self.save_hyperparameters()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []

        assert self.cnn_layers > 0, "at least one convolutional layer is required"

        assert (self.seq_len % self.cnn_layers ** self.step) == 0, \
            f"sequence length is not divisible by {self.cnn_layers} to the power of {self.step}"

        # CNN part:
        # --------------------
        self.cnn_layer_list = nn.ModuleList()  # down part of the U-net
        self.filter_list = []

        for layer in range(self.cnn_layers):
            l_in = self.seq_len / self.step ** layer
            l_out = self.seq_len / self.step ** (layer + 1)
            xpad = self.xpadding(l_in, l_out, self.step, 1, self.kernel_size)  # dilation=1

            self.cnn_layer_list.append(nn.Conv1d(input_size, filter_size, self.kernel_size, stride=self.step,
                                                 padding=xpad, dilation=1))
            self.filter_list.append(input_size)  # example: [4, 64, 128]; the last input_size is not needed
            input_size = filter_size
            filter_size = filter_size * self.up

        self.filter_list = list(reversed(self.filter_list))  # example: [128, 64, 4]

        # LSTM part:
        # --------------------
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers, batch_first=True)
        # input: last dimension of tensor, output:(batch,new_seq_len,hidden_size)

        # Transposed CNN part
        # --------------------
        self.up_layer_list = nn.ModuleList()  # up part of the U-net

        for layer in range(cnn_layers):
            l_in = l_out  # sequence length is not affected by the LSTM
            l_out = l_in * self.step
            tpad = self.trans_padding(l_in, l_out, self.step, 1, self.kernel_size, 1)  # dilation & output_padding = 1
            filter_size = self.filter_list[layer]
            self.up_layer_list.append(nn.ConvTranspose1d(hidden_size, filter_size, self.kernel_size, stride=self.step,
                                                         padding=tpad, dilation=1, output_padding=1))
            hidden_size = filter_size

        # Linear part:
        # --------------------
        self.out = nn.Linear(hidden_size, 1)  # output_size hardcoded

    def forward(self, x):
        # CNN part:
        # --------------------
        x = x.transpose(1, 2)  # convolution over the base pairs, change position of 2. and 3. dimension

        for layer in self.cnn_layer_list:
            x = F.relu(layer(x))

        x = x.transpose(1, 2)

        # LSTM part:
        # --------------------
        # size of hidden and cell state: (num_layers,batch_size,hidden_size)
        x, _ = self.lstm(x)  # _=(hn, cn)

        # Transposed CNN part
        # --------------------
        x = x.transpose(1, 2)

        for layer in self.up_layer_list:
            x = F.relu(layer(x))

        x = x.transpose(1, 2)

        # Linear part:
        # --------------------
        x = self.out(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def training_step(self, batch, batch_idx):
        X, Y = batch
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return {"loss": loss, "acc": acc.detach()}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([out["loss"] for out in training_step_outputs]).mean().item()
        avg_acc = torch.stack([out["acc"] for out in training_step_outputs]).mean().item()
        self.train_losses.append(avg_loss)
        self.train_accuracy.append(avg_acc)
        self.log("avg_train_accuracy", avg_acc, logger=False)  # logging for ModelCheckpoint callback

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return {"loss": loss, "acc": acc.detach()}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([out["loss"] for out in validation_step_outputs]).mean().item()
        avg_acc = torch.stack([out["acc"] for out in validation_step_outputs]).mean().item()
        self.val_losses.append(avg_loss)
        self.val_accuracy.append(avg_acc)

    def test_step(self, batch, batch_idx):  # not needed right now
        X, Y = batch
        pred = self(X)
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        metrics = {"loss": loss.item(), "acc": acc.item()}
        self.log_dict(metrics, logger=False)
        return metrics

    def predict_step(self, batch, batch_idx, **kwargs):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def xpadding(l_in, l_out, stride, dilation, kernel_size):
        padding = math.ceil(((l_out - 1) * stride - l_in + dilation * (kernel_size - 1) + 1) / 2)
        # math.ceil to  avoid rounding half to even/bankers rounding, only needed for even l_in
        return padding

    @staticmethod
    def trans_padding(l_in, l_out, stride, dilation, kernel_size, output_pad):
        padding = math.ceil(((l_in - 1) * stride - l_out + dilation * (kernel_size - 1) + output_pad + 1) / 2)
        # math.ceil to  avoid rounding half to even/bankers rounding
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Model_arguments")
        group.add_argument("--cnn-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("--filter-size", type=int, default=64, help="(default: %(default)d)")
        group.add_argument("--kernel-size", type=int, default=9, help="(default: %(default)d)")
        group.add_argument("--step", type=int, default=2, help="equals stride")
        group.add_argument("--up", type=int, default=2,
                           help="multiplier used for up-scaling each convolutional layer")
        group.add_argument("--hidden_size", type=int, default=128, help="LSTM units per layer")
        group.add_argument("--lstm-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="(default: %(default)f)")
        return parent_parser
