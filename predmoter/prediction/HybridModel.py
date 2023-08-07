import math
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from predmoter.core.constants import EPS


class LitHybridNet(pl.LightningModule):
    def __init__(self, model_type, cnn_layers, filter_size, kernel_size, step, up, dilation, lstm_layers,
                 hidden_size, bnorm, dropout, learning_rate, seq_len, input_size, output_size, datasets):
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
        self.dilation = dilation
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.bnorm = bnorm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.datasets = datasets

        # example input tensor for model summary
        self.example_input_array = torch.zeros(2, self.seq_len, self.input_size)

        # for checkpoints
        # this saves the args from __init__() under the names used there, not the class attributes
        self.save_hyperparameters()

        # Check model config
        # ----------------------
        if (self.seq_len % self.step ** self.cnn_layers) != 0:
            raise ValueError(f"sequence length {self.seq_len} is not divisible by a step of {self.step} "
                             f"to the power of {self.cnn_layers} cnn layers")

        if self.step == 1 and kernel_size % 2 == 0:
            raise ValueError("Even kernel_sizes are not supported for a step/stride of 1 due to Pytorch's "
                             "rule of configuring ConvTranspose1d() layers. Outputting the same sequence "
                             "length in that case would require setting output_padding=1, which is not allowed.")

        # CNN part:
        # ----------------------
        self.down_layer_list = nn.ModuleList()  # down part of the U-net
        self.filter_list = []

        for layer in range(self.cnn_layers):
            l_in = self.seq_len / self.step ** layer
            l_out = self.seq_len / self.step ** (layer + 1)
            xpad = self.xpadding(l_in, l_out, self.step, self.dilation, self.kernel_size)

            self.down_layer_list.append(nn.Conv1d(input_size, filter_size, self.kernel_size,
                                                  stride=self.step, padding=xpad, dilation=self.dilation))
            if self.bnorm:
                self.down_layer_list.append(nn.BatchNorm1d(filter_size))
            self.filter_list.append(input_size)  # example: [4, 64, 128]; the last input_size is not needed
            input_size = filter_size
            filter_size = filter_size * self.up

        if model_type == "cnn":
            hidden_size = input_size  # was set to the last filter_size in the cnn part

        if model_type == "bi-hybrid":
            hidden_size = hidden_size * 2  # bidirectional doubles hidden size

        # LSTM part:
        # ----------------------
        if model_type != "cnn":
            bidirectional = True if model_type == "bi-hybrid" else False
            # workaround: built-in lstm dropout not reproducible when resuming training/setting seed state
            self.lstm_layer_list = nn.ModuleList()
            for layer in range(self.lstm_layers):
                # input: last dimension of tensor, output: (batch, new_seq_len, hidden_size)
                self.lstm_layer_list.append(nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                                                    num_layers=1, batch_first=True, bidirectional=bidirectional))
                if self.dropout > 0 and (layer + 1) != self.lstm_layers:  # add dropout to every layer except the last
                    self.lstm_layer_list.append(nn.Dropout(self.dropout))
                input_size = hidden_size

        # Transposed CNN part:
        # ----------------------
        self.up_layer_list = nn.ModuleList()  # up part of the U-net
        out_pad = 0 if (self.kernel_size + self.step) % 2 == 0 else 1

        for layer in range(cnn_layers, 0, -1):  # count layers backwards
            l_in = self.seq_len / self.step ** layer
            l_out = l_in * self.step
            tpad = self.trans_padding(l_in, l_out, self.step, self.dilation, self.kernel_size, out_pad)
            filter_size = self.filter_list[layer-1]  # iterate backwards over filter_list
            self.up_layer_list.append(nn.ConvTranspose1d(hidden_size, filter_size, self.kernel_size,
                                                         stride=self.step, padding=tpad, dilation=self.dilation,
                                                         output_padding=out_pad))
            if self.bnorm:
                self.up_layer_list.append(nn.BatchNorm1d(filter_size))
            hidden_size = filter_size

        # Linear part:
        # ----------------------
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        # CNN part:
        # ----------------------
        x = x.transpose(1, 2).contiguous()  # convolution over the base pairs, change position of 2. and 3. dimension

        for i, layer in enumerate(self.down_layer_list):
            if self.bnorm:
                if i % 2 == 0:
                    x = F.relu(layer(x))
                else:
                    x = layer(x)
            else:
                x = F.relu(layer(x))

        if self.model_type != "cnn":
            x = x.transpose(1, 2).contiguous()

            # LSTM part:
            # ----------------------
            # size of hidden and cell state: (num_layers, batch_size, hidden_size)
            for i, layer in enumerate(self.lstm_layer_list):
                if self.dropout > 0:
                    if i % 2 == 0:
                        x, _ = layer(x)  # _ = (hn, cn)
                    else:
                        x = layer(x)
                else:
                    x, _ = layer(x)  # _ = (hn, cn)

            x = x.transpose(1, 2).contiguous()

        # Transposed CNN part:
        # ----------------------
        for i, layer in enumerate(self.up_layer_list):
            if self.bnorm:
                if i % 2 == 0:
                    x = F.relu(layer(x))
                else:
                    x = layer(x)
            else:
                x = F.relu(layer(x))

        x = x.transpose(1, 2).contiguous()

        # Linear part:
        # ----------------------
        return self.out(x)

    def training_step(self, batch, batch_idx):
        loss, acc = self.step_fn(batch)
        metrics = {"avg_train_loss": loss, "avg_train_accuracy": acc}
        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False,
                      reduce_fx="mean", sync_dist=True)  # log for callbacks
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step_fn(batch)
        metrics = {"avg_val_loss": loss, "avg_val_accuracy": acc}
        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False, reduce_fx="mean", sync_dist=True)
        return loss

    def step_fn(self, batch):
        X, Y = batch
        # exclude gaps spanning an entire chunk (here, because it should work for both dataset classes)
        mask = torch.max(X[:, :, 0], dim=1)[0] != 0.25  # mask entire N chunks
        X, Y = X[mask], Y[mask]
        pred = self(X)

        # mask padding/chromosome ends
        # ------------------------------
        mask = self.compute_mask(X, self.output_size)
        pred[mask] = torch.nan
        Y[mask] = torch.nan

        pred, Y = self.unpack(pred), self.unpack(Y)
        # mask filler if there is more than 1 ngs dataset
        # -------------------------------------------------
        if self.output_size > 1:
            mask = torch.where(Y[:, 0] == -3, False, True)  # False if -3 fill value in tensor
            # mask missing ngs data (not every file has every dataset)
            pred, Y = pred[mask], Y[mask]

        loss = self.poisson_nll_loss(pred, Y, is_log=True)
        acc = self.pear_coeff(pred, Y, is_log=True)
        return loss, acc

    def test_step(self, batch, batch_idx):
        X, Y = batch

        # mask entire chunk gaps
        # ------------------------------
        mask = torch.max(X[:, :, 0], dim=1)[0] != 0.25  # mask entire N chunks
        X, Y = X[mask], Y[mask]
        if len(X) == 0:
            return

        pred = self(X)

        # retrieve the dataset's dataset(s) from the one test dataloader
        # helpful if e.g. the model is trained on 3 datasets, but the test set has just 1
        avail_datasets = self.trainer.test_dataloaders.dataset.dsets

        # mask padding/chromosome ends
        # ------------------------------
        mask = self.compute_mask(X, self.output_size)
        pred[mask] = torch.nan

        mask = self.compute_mask(X, len(avail_datasets))
        Y[mask] = torch.nan

        # calculate individual and total metrics
        # -------------------------------------------------
        metrics = {}

        indices = [self.datasets.index(a) for a in avail_datasets]
        idxs = [i for i in range(len(avail_datasets))]
        prefix = "total_avg_val_" if self.output_size > 1 else f"{self.datasets[0]}_avg_val_"

        metrics[f"{prefix}loss"] = self.poisson_nll_loss(self.unpack(pred[:, :, indices]),
                                                         self.unpack(Y[:, :, idxs]), is_log=True)
        metrics[f"{prefix}accuracy"] = self.pear_coeff(self.unpack(pred[:, :, indices]),
                                                       self.unpack(Y[:, :, idxs]), is_log=True)

        if self.output_size > 1:
            for i in range(self.output_size):
                prefix = f"{self.datasets[i]}_avg_val_"
                if i in indices:
                    if len(avail_datasets) == 1:
                        metrics[f"{prefix}loss"] = metrics["total_avg_val_loss"]
                        metrics[f"{prefix}accuracy"] = metrics["total_avg_val_accuracy"]
                    else:
                        j = avail_datasets.index(self.datasets[i])
                        metrics[f"{prefix}loss"] = self.poisson_nll_loss(pred[:, :, i], Y[:, :, j], is_log=True)
                        metrics[f"{prefix}accuracy"] = self.pear_coeff(pred[:, :, i], Y[:, :, j], is_log=True)
                else:
                    metrics[f"{prefix}loss"] = torch.tensor([torch.nan])
                    metrics[f"{prefix}accuracy"] = torch.tensor([torch.nan])

        self.log_dict(metrics, logger=False, on_epoch=True, on_step=False, reduce_fx="mean")

    def predict_step(self, batch, batch_idx, **kwargs):
        mask = self.compute_mask(batch, self.output_size)

        # since the network's predictions are logarithmic, torch.exp() is needed
        preds = torch.exp(self(batch))
        preds[mask] = -1.  # -1 as filler for padding/Ns

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def xpadding(l_in, l_out, stride, dilation, kernel_size):
        """Padding formula for Conv1d.

        The formula is adapted the formula for calculating the output sequence length (L_out)
        from the Pytorch documentation. The desired Lout is: L_out = L_in/stride.

            Args:
                l_in: 'int', input sequence length
                l_out: 'int', output sequence length
                stride: 'int', strides/steps of kernel points
                dilation: 'int', spacing between kernel points
                kernel_size: 'int'

            Returns:
                padding: 'int', padding needed to achieve division without remainder
                    of sequence length, e.g. input seq_len: 10, stride: 2, expected/
                    necessary output seq_len: 5
        """
        padding = math.ceil(((l_out - 1) * stride - l_in + dilation * (kernel_size - 1) + 1) / 2)
        # math.ceil to  avoid rounding half to even/bankers rounding, only needed for even L_in
        return padding

    @staticmethod
    def trans_padding(l_in, l_out, stride, dilation, kernel_size, output_pad):
        """Padding formula for TransposeConv1d.

        The formula is adapted the formula for calculating the output sequence length (L_out)
        from the Pytorch documentation. The desired Lout is: L_out = L_in * stride. The output
        padding equals zero, if the stride is uneven, and one otherwise.

            Args:
                l_in: 'int', input sequence length
                l_out: 'int', output sequence length
                stride: 'int', strides/steps of kernel points
                dilation: 'int', spacing between kernel points
                kernel_size: 'int'
                output_pad: 'int', additional size added to one side of the output shape

            Returns:
                padding: 'int', padding needed to achieve multiplication "without remainder"
                    of sequence length, e.g. input seq_len: 5, stride: 2, expected/
                    necessary output seq_len: 10
        """
        padding = ((l_in - 1) * stride - l_out + dilation * (kernel_size - 1) + output_pad + 1) / 2
        return int(padding)

    @staticmethod
    def compute_mask(base_tensor, repeats):
        """Compute mask to exclude padding/Ns."""
        mask = ~torch.max(base_tensor, dim=2)[0].bool()  # True for bases that are padding/Ns
        if repeats == 1:
            return mask.reshape(mask.size(0), mask.size(1), 1)
        return torch.stack([mask] * repeats, dim=2)

    @staticmethod
    def poisson_nll_loss(prediction, target, is_log=True):
        """Custom way of calculating Poisson negative log likelihood loss.

        The formula is adapted from Pytorch's Poisson loss function
        (see https://github.com/pytorch/pytorch/blob/main/torch/_refs/nn/functional/__init__.py).
        This is a simpler version, that uses torch.nanmean() as a reduction method to simulate masking,
        as Pytorch MaskedTensors are still in the prototype phase.
        """
        dims = len(prediction.size())
        assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
        if is_log:
            loss = torch.exp(prediction) - target * prediction
        else:
            loss = prediction - target * torch.log(prediction + EPS)

        if dims > 1:
            loss = torch.nanmean(loss, dim=1)
        return torch.nanmean(loss)

    @staticmethod
    def pear_coeff(prediction, target, is_log=True):
        """Calculate the pearson correlation coefficient.

        This function only accepts tensors with a maximum of 2 dimensions. It excludes NaNs
        so that NaN can be used for positions to be masked as masked tensors are at present
        still in development.

            Args:
                prediction: 'torch.tensor', models' predictions
                target: 'torch.tensor', targets/labels
                is_log: 'bool', if True (default) assumes the models' predictions are logarithmic

            Returns:
                'torch.tensor', average pearson correlation coefficient (correlation between
                    predictions and targets are calculated per chunk and then averaged)
        """

        dims = len(prediction.size())
        assert dims <= 2, f"can only calculate pearson's r for tensors with 1 or 2 dimensions, not {dims}"
        if dims == 2:
            prediction = prediction.transpose(0, 1)
            target = target.transpose(0, 1)

        if is_log:
            prediction = torch.exp(prediction)

        p = prediction - torch.nanmean(prediction, dim=0)
        t = target - torch.nanmean(target, dim=0)
        coeff = torch.nansum(p * t, dim=0) / (
                torch.sqrt(torch.nansum(p ** 2, dim=0)) * torch.sqrt(torch.nansum(t ** 2, dim=0)) + EPS)
        # EPS to avoid division by 0
        return torch.mean(coeff)

    @staticmethod
    def unpack(tensor):
        """Reduce dimensionality of tensors.

        3-dimensional tensors are unpacked into 2-dimensional tensors, e.g. (3, 500, 2) gets
        unpacked to (6, 500). It works like torch.squeeze() for tensors with the size (i, j, 1).

            Args:
                tensor: 'torch.tensor', input tensor of 3 dimensions

            Returns:
                tensor: 'torch.tensor', tensor of 2 dimensions
        """

        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(tensor.size(0) * tensor.size(1), tensor.size(2))
        return tensor
