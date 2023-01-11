import os
import random
import logging
import glob
import time
import numpy as np
# noinspection PyUnresolvedReferences
import torch  # for item() in metric callback
import h5py
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.utilities.seed import seed_everything
#  Please use `lightning_lite.utilities.seed.seed_everything` instead.


class MetricCallback(Callback):
    def __init__(self, output_dir, mode, prefix):
        super().__init__()
        self.mode = mode
        self.filename = f"{prefix}val_metrics.log" if self.mode == "validate" else f"{prefix}metrics.log"
        self.file = os.path.join(output_dir, self.filename)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics  # does not add validation sanity check
        msg = " ".join([str(epoch)] + [str(m.item()) for m in list(metrics.values())])
        if epoch == 0:
            msg = " ".join(list(metrics.keys())) + "\n" + msg  # self.metric_names
        self.save_metrics(msg)

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # just one test dataloader available
        input_filename = trainer.test_dataloaders[0].dataset.h5_files[0].split("/")[-1]
        msg = f"{input_filename} " + " ".join([str(m.item()) for m in list(metrics.values())])
        if not os.path.exists(self.file):  # if the file is already there the header is not needed
            msg = "species " + " ".join(list(metrics.keys())) + "\n" + msg
        self.save_metrics(msg)

    def save_metrics(self, msg):
        with open(self.file, "a") as f:
            f.write(f"{msg}\n")


class Timeit(Callback):
    def __init__(self):
        super().__init__()
        self.start = 0
        self.durations = []
        self.last_epoch = 0

    def on_train_start(self, trainer, pl_module):
        log_table(["Epoch", "Duration (min)"], spacing=16, header=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.last_epoch += 1
        duration = time.time() - self.start
        self.durations.append(duration)

        log_table([str(trainer.current_epoch), f"{duration/60:.2f}"], spacing=16)

    def on_train_end(self, trainer, pl_module):
        log_table([str(self.last_epoch), f"{sum(self.durations)/60**2:.2f} h"], spacing=16, table_end=True)


def set_callbacks(output_dir, mode, prefix, checkpoint_path, ckpt_quantity, save_top_k, stop_quantity, patience):
    if mode == "predict":
        return None

    metrics_callback = MetricCallback(output_dir, mode, prefix)
    if mode == "validate":
        return [metrics_callback]

    for quantity in [ckpt_quantity, stop_quantity]:
        assert quantity in ["avg_train_loss", "avg_train_accuracy", "avg_val_loss", "avg_val_accuracy"],\
            f"can not monitor invalid quantity: {quantity}"
    ckpt_method = "min" if "loss" in ckpt_quantity else "max"
    filename = "predmoter_{epoch}_{" + ckpt_quantity + ":.4f}"  # f-string would mess up formatting
    # save_top_k=-1 means every model gets saved
    checkpoint_callback = ModelCheckpoint(save_top_k=save_top_k, monitor=ckpt_quantity, mode=ckpt_method,
                                          dirpath=checkpoint_path, filename=filename, save_last=True,
                                          save_on_train_epoch_end=True)
    stop_method = "min" if "loss" in stop_quantity else "max"
    early_stop = EarlyStopping(monitor=stop_quantity, min_delta=0.0, patience=patience, verbose=False,
                               mode=stop_method, strict=True, check_finite=True,
                               check_on_train_epoch_end=True)
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
            return msg.getMessage()  # if simple exists and is True, print just the message without date & level
        else:
            return logging.Formatter.format(self, msg)


def init_logging(output_dir, prefix):
    logging.getLogger("torch").setLevel(logging.WARNING)  # only log info from predmoter
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)  # exclude warnings

    filename = os.path.join(output_dir, f"{prefix}predmoter.log")
    handler = logging.FileHandler(filename, mode='a')
    formatter = CustomFormatter('%(asctime)s, %(levelname)s: %(message)s',
                                datefmt='%d.%m.%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])


def log_table(contents, spacing, header=False, table_end=False):
    contents = [f"{i: <{spacing}}" if len(i) <= spacing else i[:spacing - 2] + ".." for i in contents]
    # every string will have the length of spacing
    msg = "|".join(contents)

    if header:
        msg = "\n" + msg + "\n" + "-" * len(msg)

    elif table_end:
        msg = "-" * len(msg) + "\n" + msg + "\n"

    logging.info(msg, extra={"simple": True})


def check_paths(paths):
    for path in paths:
        assert os.path.exists(path), f"the directory {path} doesn't exist"


def get_meta(input_dir, mode):
    if mode == "train":
        # tests if files in folder train are provided next in train_loader=get_dataloader()
        folder = "val"
    elif mode == "validate":
        folder = "test"
    else:
        folder = "predict"

    input_dir = os.path.join(input_dir, folder)
    h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    assert len(h5_files) >= 1, f"no input file/s of type {folder} were provided"

    for h5_file in h5_files:
        h5df = h5py.File(h5_file, mode="r")
        X = np.array(h5df["data/X"][:1], dtype=np.int8)
        meta = X.shape[1:]
        assert len(meta) == 2, f"expected all arrays to have the shape (seq_len, bases) found {meta}"
        return meta


def get_available_datasets(h5_file, model_datasets):
    h5df = h5py.File(h5_file, mode="r")
    avail_datasets = [dataset for dataset in model_datasets if f"{dataset}_coverage" in h5df["evaluation"].keys()]
    return avail_datasets
