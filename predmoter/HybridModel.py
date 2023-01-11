import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# add way more selves
class LitHybridNet(pl.LightningModule):
    def __init__(self, model_type, cnn_layers, filter_size, kernel_size, step, up, hidden_size,
                 lstm_layers, dropout, learning_rate, seq_len, input_size, output_size, datasets):
        super(LitHybridNet, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.model_type = model_type
        self.cnn_layers = cnn_layers
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.step = step
        self.up = up
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.datasets = datasets

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
        self.down_layer_list = nn.ModuleList()  # down part of the U-net
        self.filter_list = []

        for layer in range(self.cnn_layers):
            l_in = self.seq_len / self.step ** layer
            l_out = self.seq_len / self.step ** (layer + 1)
            xpad = self.xpadding(l_in, l_out, self.step, 1, self.kernel_size)  # dilation=1

            self.down_layer_list.append(nn.Conv1d(input_size, filter_size, self.kernel_size, stride=self.step,
                                                  padding=xpad, dilation=1))
            self.down_layer_list.append(nn.BatchNorm1d(filter_size))
            self.filter_list.append(input_size)  # example: [4, 64, 128]; the last input_size is not needed
            input_size = filter_size
            filter_size = filter_size * self.up

        if model_type == "cnn":
            hidden_size = input_size  # was set to the last filter_size in the cnn part

        if model_type == "bi-hybrid":
            hidden_size = hidden_size * 2  # bidirectional doubles hidden size

        # LSTM part:
        # --------------------
        if model_type != "cnn":
            bidirectional = True if model_type == "bi-hybrid" else False
            # input: last dimension of tensor, output:(batch,new_seq_len,hidden_size)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                                num_layers=self.lstm_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.dropout)

            self.bnorm = nn.BatchNorm1d(hidden_size)

        # Transposed CNN part
        # --------------------
        self.up_layer_list = nn.ModuleList()  # up part of the U-net
        out_pad = 1 if self.step % 2 == 0 else 0  # uneven strides doesn't need output padding

        for layer in range(cnn_layers, 0, -1):  # count layers backwards
            l_in = self.seq_len / self.step ** layer
            l_out = l_in * self.step
            tpad = self.trans_padding(l_in, l_out, self.step, 1, self.kernel_size)  # dilation=1
            filter_size = self.filter_list[layer-1]  # iterate backwards over filter_list
            self.up_layer_list.append(nn.ConvTranspose1d(hidden_size, filter_size, self.kernel_size, stride=self.step,
                                                         padding=tpad, dilation=1, output_padding=out_pad))
            self.up_layer_list.append(nn.BatchNorm1d(filter_size))
            hidden_size = filter_size

        # Linear part:
        # --------------------
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        # CNN part:
        # --------------------
        x = x.transpose(1, 2)  # convolution over the base pairs, change position of 2. and 3. dimension

        for i, layer in enumerate(self.down_layer_list):
            if i % 2 == 0:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.model_type != "cnn":
            x = x.transpose(1, 2)

            # LSTM part:
            # --------------------
            # size of hidden and cell state: (num_layers,batch_size,hidden_size)
            x, _ = self.lstm(x)  # _=(hn, cn)

            x = x.transpose(1, 2)

            x = self.bnorm(x)

        # Transposed CNN part
        # --------------------
        for i, layer in enumerate(self.up_layer_list):
            if i % 2 == 0:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        x = x.transpose(1, 2)

        # Linear part:
        # --------------------
        x = self.out(x)
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

        # mask padding/chromosome ends
        # ------------------------------
        mask = torch.sum(X, dim=2)  # zero for bases that are padding/Ns/chromosome ends
        mask = mask.reshape(mask.size(0), mask.size(1), 1)
        pred = pred * mask
        Y = Y * mask
        # no data in Y is denoted -1 instead of NaN; if it is replaced with NaN,
        # masking would need to be: Y[torch.where(torch.isnan(Y)==True)]= 0.
        pred, Y = self.unpack(pred), self.unpack(Y)
        # mask NaNs if there is more than 1 ngs dataset
        # -------------------------------------------------
        if self.output_size > 1:
            mask = [not torch.isnan(y).any() for y in Y]  # all True where no NaNs are
            # mask missing atacseq/h3k4me3 data (not every file has both datasets)
            pred, Y = pred[mask], Y[mask]

        loss = F.poisson_nll_loss(pred, Y, log_input=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return loss, acc

    def test_step(self, batch, batch_idx):
        X, Y = batch
        pred = self(X)

        # mask padding/chromosome ends
        # ------------------------------
        mask = torch.sum(X, dim=2)  # zero for bases that are padding (Ns or chromosome ends)
        mask = mask.reshape(mask.size(0), mask.size(1), 1)
        pred = pred * mask
        Y = Y * mask

        # just one test dataloader available; keys = dataset(s) used to load test dataset/dataloader;
        # useful if you for example trained the model on 3 datasets but your h5 file just has 1, it is
        # a waste of memory to add 2 datasets filled with NaN (see PredmoterSequence in dataset.py), since
        # this technique is used so data of multiple files with different datasets available doesn't
        # get shifted around (better explanation in PredmoterSequence --> reference that then)
        avail_datasets = self.trainer.test_dataloaders[0].dataset.keys

        # calculate individual and total metrics
        # -------------------------------------------------
        metrics = {}
        if self.output_size > 1:
            indices = [self.datasets.index(a) for a in avail_datasets]
            prefix = "total_avg_val_"

            if self.datasets == avail_datasets:
                metrics[f"{prefix}loss"] = F.poisson_nll_loss(self.unpack(pred), self.unpack(Y), log_input=True)
                metrics[f"{prefix}accuracy"] = self.pear_coeff(self.unpack(pred), self.unpack(Y), is_log=True)
            else:
                idxs = [0] if len(avail_datasets) == 1 else indices
                metrics[f"{prefix}loss"] = F.poisson_nll_loss(self.unpack(pred[:, :, indices]),
                                                              self.unpack(Y[:, :, idxs]), log_input=True)
                metrics[f"{prefix}accuracy"] = self.pear_coeff(self.unpack(pred[:, :, indices]),
                                                               self.unpack(Y[:, :, idxs]), is_log=True)

            for i in range(self.output_size):
                prefix = f"{self.datasets[i]}_avg_val_"
                if i in indices:
                    if len(avail_datasets) == 1:
                        metrics[f"{prefix}loss"] = metrics["total_avg_val_loss"]
                        metrics[f"{prefix}accuracy"] = metrics["total_avg_val_accuracy"]
                    else:
                        metrics[f"{prefix}loss"] = F.poisson_nll_loss(pred[:, :, i], Y[:, :, i], log_input=True)
                        metrics[f"{prefix}accuracy"] = self.pear_coeff(pred[:, :, i], Y[:, :, i], is_log=True)
                else:
                    metrics[f"{prefix}loss"] = torch.tensor([torch.nan])
                    metrics[f"{prefix}accuracy"] = torch.tensor([torch.nan])

        else:
            prefix = f"{self.datasets[0]}_avg_val_"
            metrics[f"{prefix}loss"] = F.poisson_nll_loss(self.unpack(pred), self.unpack(Y), log_input=True)
            metrics[f"{prefix}accuracy"] = self.pear_coeff(self.unpack(pred), self.unpack(Y), is_log=True)

        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False, reduce_fx="mean")

    def predict_step(self, batch, batch_idx, **kwargs):
        # since the network's predictions are logarithmic, torch.exp() is needed
        return torch.exp(self(batch))  # sigmoid?

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def xpadding(l_in, l_out, stride, dilation, kernel_size):
        # The padding formula for Conv1d

        # The formula is adapted from the formula for calculating the output sequence length (L_out) from
        # the Pytorch documentation. The desired Lout is: L_out = L_in/stride.

        padding = math.ceil(((l_out - 1) * stride - l_in + dilation * (kernel_size - 1) + 1) / 2)
        # math.ceil to  avoid rounding half to even/bankers rounding, only needed for even L_in
        return padding

    @staticmethod
    def trans_padding(l_in, l_out, stride, dilation, kernel_size):
        # The padding formula for TransposeConv1d

        # The formula is adapted from the formula for calculating the output sequence length (L_out) from
        # the Pytorch documentation. The desired Lout is: L_out = L_in * stride. The output padding equals
        # zero, if the stride is uneven, and one otherwise.

        output_pad = 1 if stride % 2 == 0 else 0
        padding = ((l_in - 1) * stride - l_out + dilation * (kernel_size - 1) + output_pad + 1) / 2
        return int(padding)

    @staticmethod
    def pear_coeff(prediction, target, is_log=True):
        # Function to calculate the pearson correlation

        # For two dimensions the program needs to transpose the prediction and target tensors, so
        # that it compares each tensor to it's target individually instead. Example: two tensors of size:
        # (3,100), one is the prediction, the other the target. The function will compare the first values
        # of all three sub-tensors from the prediction with the first values of all three sub-tensors from the
        # target, then second values of all, then the third and so on. The squeeze function helps to calculate
        # the pearson correlation of tensors with size (i, 1).

        dims = len(prediction.size())
        assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
        if dims == 2:
            prediction = torch.squeeze(prediction.transpose(0, 1))
            target = torch.squeeze(target.transpose(0, 1))

        if is_log:
            prediction = torch.exp(prediction)

        p = prediction - torch.mean(prediction, dim=0)
        t = target - torch.mean(target, dim=0)
        coeff = torch.sum(p * t, dim=0) / (
                    torch.sqrt(torch.sum(p ** 2, dim=0)) * torch.sqrt(torch.sum(t ** 2, dim=0)) + 1e-8)
        # 1e-8 avoiding division by 0
        return torch.mean(coeff)

    @staticmethod
    def unpack(tensor):
        # unpacks three-dimensional tensors into two dimensions; example: (3,21384,2) into (6,21384)
        # works like torch.squeeze() for shapes with dimensions: (i, j, 1)
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(tensor.size(0) * tensor.size(1), tensor.size(2))
        return tensor

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Model_arguments")
        group.add_argument("--model-type", type=str, default="hybrid",
                           help="The type of model to train. Valid types are cnn, hybrid (CNN + LSTM) and "
                                "bi-hybrid (CNN + BiLSTM).")
        group.add_argument("--cnn-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("--filter-size", type=int, default=64, help="(default: %(default)d)")
        group.add_argument("--kernel-size", type=int, default=9, help="(default: %(default)d)")
        group.add_argument("--step", type=int, default=2, help="equals stride")
        group.add_argument("--up", type=int, default=2,
                           help="multiplier used for up-scaling each convolutional layer")
        group.add_argument("--hidden-size", type=int, default=128, help="LSTM units per layer")
        group.add_argument("--lstm-layers", type=int, default=1, help="(default: %(default)d)")
        group.add_argument("--dropout", type=float, default=0.,
                           help="adds a dropout layer with the specified dropout value after each "
                                "LSTM layer except the last; if it is 0. no dropout layers are added;"
                                "if there is just one LSTM layer specifying dropout will do nothing")
        group.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="(default: %(default)f)")
        return parent_parser
