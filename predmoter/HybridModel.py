import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# add way more selves
class LitHybridNet(pl.LightningModule):
    def __init__(self, model_type, cnn_layers, filter_size, kernel_size, step, up,
                 hidden_size, lstm_layers, learning_rate, seq_len, input_size):
        super(LitHybridNet, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.model_type = model_type
        self.cnn_layers = cnn_layers
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.step = step
        self.up = up
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.learning_rate = learning_rate

        # example input tensor for model summary
        self.example_input_array = torch.zeros(2, self.seq_len, self.input_size)

        # for checkpoints
        self.save_hyperparameters()

        assert model_type in ["cnn", "hybrid", "bi-hybrid"], \
            f"valid model types are cnn, hybrid and bi-hybrid not {model_type}"

        assert self.cnn_layers > 0, "at least one convolutional layer is required"

        assert (self.seq_len % self.step ** self.cnn_layers) == 0, \
            f"sequence length {self.seq_len} is not divisible by a step of {self.step} " \
            f"to the power of {self.cnn_layers} cnn layers"

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

        if model_type != "cnn":
            # LSTM part:
            # --------------------
            bidirectional = True if model_type == "bi-hybrid" else False
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                                num_layers=self.lstm_layers, batch_first=True, bidirectional=bidirectional)
            # input: last dimension of tensor, output:(batch,new_seq_len,hidden_size)

        # Transposed CNN part
        # --------------------
        self.up_layer_list = nn.ModuleList()  # up part of the U-net
        out_pad = 1 if self.step % 2 == 0 else 0  # uneven step doesn't need output_padding

        if model_type == "cnn":
            hidden_size = input_size  # was set to the last filter_size in the cnn part

        if model_type == "bi-hybrid":
            hidden_size = hidden_size * 2  # bidirectional doubles hidden size

        for layer in range(cnn_layers, 0, -1):  # count layers backwards
            l_in = self.seq_len / self.step ** layer
            l_out = l_in * self.step
            tpad = self.trans_padding(l_in, l_out, self.step, 1, self.kernel_size, out_pad)  # dilation=1
            filter_size = self.filter_list[layer-1]  # iterate backwards over filter_list
            self.up_layer_list.append(nn.ConvTranspose1d(hidden_size, filter_size, self.kernel_size, stride=self.step,
                                                         padding=tpad, dilation=1, output_padding=out_pad))
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

        if self.model_type != "cnn":
            x = x.transpose(1, 2)

            # LSTM part:
            # --------------------
            # size of hidden and cell state: (num_layers,batch_size,hidden_size)
            x, _ = self.lstm(x)  # _=(hn, cn)

            x = x.transpose(1, 2)

        # Transposed CNN part
        # --------------------
        for layer in self.up_layer_list:
            x = F.relu(layer(x))

        x = x.transpose(1, 2)

        # Linear part:
        # --------------------
        x = self.out(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def training_step(self, batch, batch_idx):
        loss, acc = self.step_fn(batch)
        metrics = {"avg_train_loss": loss, "avg_train_accuracy": acc}
        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False, reduce_fx="mean")  # log for checkpoints
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step_fn(batch)
        metrics = {"avg_val_loss": loss, "avg_val_accuracy": acc}
        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False, reduce_fx="mean")
        return loss

    def step_fn(self, batch):
        X, Y = batch
        pred = self(X)
        mask = torch.sum(X, dim=2)  # zero for any predictions based on padding or Ns
        pred = pred * mask
        Y = Y * mask
        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return loss, acc

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
        padding = ((l_in - 1) * stride - l_out + dilation * (kernel_size - 1) + output_pad + 1) / 2
        return int(padding)

    @staticmethod
    def pear_coeff(prediction, target, is_log=True):
        dims = len(prediction.size())
        assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
        if dims == 2:
            prediction = prediction.transpose(0, 1)
            target = target.transpose(0, 1)

        if is_log:
            prediction = torch.exp(prediction)

        p = prediction - torch.mean(prediction, dim=0)
        t = target - torch.mean(target, dim=0)
        coeff = torch.sum(p * t, dim=0) / (
                    torch.sqrt(torch.sum(p ** 2, dim=0)) * torch.sqrt(torch.sum(t ** 2, dim=0)) + 1e-8)
        # 1e-8 avoiding division by 0
        return torch.mean(coeff)

    @staticmethod
    def unpack():
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Model_arguments")
        group.add_argument("--model-type", type=str, default="hybrid",
                           help="The type of model to train. Valid types are cnn, hybrid (CNN + LSTM) and "
                                "bi-hybrid (CNN + BiLSTM). (default: %(default)d)")
        group.add_argument("--cnn-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("--filter-size", type=int, default=64, help="(default: %(default)d)")
        group.add_argument("--kernel-size", type=int, default=9, help="(default: %(default)d)")
        group.add_argument("--step", type=int, default=2, help="equals stride")
        group.add_argument("--up", type=int, default=2,
                           help="multiplier used for up-scaling each convolutional layer")
        group.add_argument("--hidden-size", type=int, default=128, help="LSTM units per layer")
        group.add_argument("--lstm-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="(default: %(default)f)")
        return parent_parser
