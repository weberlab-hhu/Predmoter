import os
import random
import logging
import glob
import time
import numpy as np
import h5py
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.utilities.seed import seed_everything


class MetricCallback(Callback):
    def __init__(self, output_dir, mode, prefix):
        super().__init__()
        self.mode = mode
        self.filename = f"{prefix}val_metrics.log" if self.mode == "validate" else f"{prefix}metrics.log"
        self.file = os.path.join(output_dir, self.filename)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics  # does not add validation sanity check
        msg = "{} {} {} {} {}".format(epoch, metrics["avg_train_loss"], metrics["avg_val_loss"],
                                      metrics["avg_train_accuracy"], metrics["avg_val_accuracy"])
        if epoch == 0:
            msg = "epoch training_loss validation_loss training_accuracy validation_accuracy\n" + msg
        self.save_metrics(msg)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == "validate":
            metrics = trainer.callback_metrics
            msg = "{} {}".format(metrics["avg_val_loss"], metrics["avg_val_accuracy"])
            if not os.path.exists(self.file):  # if the file is already there, the header is not needed
                msg = "validation_loss validation_accuracy\n" + msg
            self.save_metrics(msg)

    def save_metrics(self, msg):
        with open(self.file, "a") as f:
            f.write(f"{msg}\n")


class Timeit(Callback):
    def __init__(self):
        super().__init__()
        self.start = 0
        self.times_called = 0
        self.durations = []
        self.last_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.last_epoch += 1
        duration = time.time() - self.start
        self.durations.append(duration)

        msg = "   | {0: <6}| {1:.4f} min".format(trainer.current_epoch, duration/60)
        if self.times_called == 0:
            msg = f"\n   | {'Epoch': <6}| {'Duration': <25}\n{'-' * 38}\n" + msg
        logging.info(msg, extra={"simple": True})
        self.times_called += 1

    def on_train_end(self, trainer, pl_module):
        msg = "{0}\n{1: <3}| {2: <6}| {3:.4f} h\n".format("-" * 38, "tot", self.last_epoch, sum(self.durations)/60**2)
        logging.info(msg, extra={"simple": True})


def set_callbacks(output_dir, mode, prefix, checkpoint_path, quantity, patience):
    if mode == "predict":
        return None

    metrics_callback = MetricCallback(output_dir, mode, prefix)
    if mode == "validate":
        return [metrics_callback]

    assert quantity in ["avg_train_loss", "avg_train_accuracy", "avg_val_loss", "avg_val_accuracy"],\
        f"can not monitor invalid quantity: {quantity}"
    method = "min" if "loss" in quantity else "max"
    filename = "predmoter_{epoch}_{" + quantity + ":.4f}"  # f-string would mess up formatting
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor=quantity, mode=method, dirpath=checkpoint_path,
                                          filename=filename, save_on_train_epoch_end=True, save_last=True)
    early_stop = EarlyStopping(monitor=quantity, min_delta=0.0, patience=patience, verbose=False,
                               mode=method, strict=True, check_finite=True, check_on_train_epoch_end=True)
    time_callback = Timeit()
    callbacks = [checkpoint_callback, metrics_callback, early_stop, time_callback]
    return callbacks


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 2000)
        logging.info(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        logging.info(f"The seed provided by the user is: {seed}.")
    seed_everything(seed=seed, workers=True)  # seed for reproducibility


class CustomFormatter(logging.Formatter):
    def format(self, msg):
        if hasattr(msg, "simple") and msg.simple:
            return msg.getMessage()
        else:
            return logging.Formatter.format(self, msg)


def init_logging(output_dir, prefix):
    logging.getLogger("torch").setLevel(logging.WARNING)  # only log info from predmoter
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    filename = os.path.join(output_dir, f"{prefix}predmoter.log")
    handler = logging.FileHandler(filename, mode='a')
    formatter = CustomFormatter('%(asctime)s, %(levelname)s: %(message)s',
                                datefmt='%d.%m.%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])


def check_paths(paths):
    for path in paths:
        assert os.path.exists(path), f"the directory {path} doesn't exist"


def get_meta(input_dir, mode):
    folder = "test" if mode == "predict" else "val"
    input_dir = os.path.join(input_dir, folder)
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    assert len(h5_files) >= 1, f"no input file/s of type {folder} were provided"

    for h5_file in h5_files:
        h5df = h5py.File(h5_file, mode="r")
        X = np.array(h5df["data/X"][:1], dtype=np.int8)
        meta = X.shape[1:]
        assert len(meta) == 2, f"expected all arrays to have the shape (seq_len, bases) found {meta}"
        return meta
