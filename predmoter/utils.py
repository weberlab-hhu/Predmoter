import os
import random
import logging
import glob
import time
import numpy as np
import torch
import h5py
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.utilities.seed import seed_everything
#  Please use `lightning_lite.utilities.seed.seed_everything` for Pytorch Lightning v2.0.0 and upwards instead.


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
            msg = " ".join(["epoch"] + list(metrics.keys())) + "\n" + msg  # self.metric_names
        self.save_metrics(msg)

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # test_dataloaders[0] --> just one test dataloader is available
        # split() removes the filepath
        input_filename = trainer.test_dataloaders[0].dataset.h5_files[0].split("/")[-1]
        msg = f"{input_filename} " + " ".join([str(m.item()) for m in list(metrics.values())])
        if not os.path.exists(self.file):  # if the file is already there the header is not needed
            msg = " ".join(["species"] + list(metrics.keys())) + "\n" + msg
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


# noinspection PyTypedDict
# for this class only
class SeedCallback(Callback):
    def __init__(self, seed):
        self.state = {"seed": seed, "random_seed_state": None, "numpy_seed_state": None,
                      "torch_seed_state": None, "torch.cuda_seed_state": None}

    def on_train_epoch_end(self, trainer, pl_module):
        self.state["random_seed_state"] = random.getstate()
        self.state["numpy_seed_state"] = np.random.get_state()
        self.state["torch_seed_state"] = torch.get_rng_state()
        self.state["torch.cuda_seed_state"] = torch.cuda.get_rng_state()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


class PredictCallback(Callback):
    def __init__(self, prefix, output_dir, add_additional, model, datasets):
        self.prefix = prefix
        self.output_dir = output_dir
        self.add_additional = add_additional
        self.model = model
        self.datasets = datasets
        self.input_file = None
        self.output_file = None
        self.length = None
        self.pred_key = None
        self.start = 0
        self.stop = None
        self.batch_size = None

    def on_predict_start(self, trainer, pl_module):
        # get input file meta data
        # -------------------------
        # predict_dataloaders[0] --> just one predict dataloader is available
        self.input_file = trainer.predict_dataloaders[0].dataset.h5_files[0]
        # split() removes the filepath and file extension
        input_name = self.input_file.split("/")[-1].split(".")[0]
        in_h5df = h5py.File(self.input_file, "r")
        self.length = in_h5df["data/X"].shape[0]
        chunk_len = in_h5df["data/X"].shape[1]

        # define the h5 output file
        self.output_file = os.path.join(self.output_dir, f"{self.prefix}{input_name}_predictions.h5")
        h5df = h5py.File(self.output_file, "a")

        if self.add_additional is None:
            # data group setup
            # --------------------
            h5df.create_group("data")
            h5df.create_dataset("data/species", shape=(self.length,), maxshape=(None,), dtype="S50",
                                fillvalue=in_h5df["data/species"][0])
            h5df.create_dataset("data/seqids", shape=(self.length,), maxshape=(None,), dtype="S50",
                                data=np.array(in_h5df["data/seqids"]))
            h5df.create_dataset("data/start_ends", shape=(self.length, 2), maxshape=(None, 2), dtype=int,
                                data=np.array(in_h5df["data/start_ends"]))

        # prediction groups
        # ---------------------
        if self.add_additional is not None and "alternative" not in h5df.keys():
            h5df.create_group("alternative")

        self.pred_key = "prediction" if self.add_additional is None else \
            f"alternative/{self.add_additional}_prediction"
        h5df.create_group(self.pred_key)
        h5df.create_dataset(self.pred_key + "/predictions", shape=(self.length, chunk_len, len(self.datasets)),
                            maxshape=(None, chunk_len, None), chunks=(1, chunk_len, 1), dtype=int,
                            compression="gzip", compression_opts=9, fillvalue=-1)

        h5df.create_dataset(self.pred_key + "/prediction_meta/model_name", shape=(1,), maxshape=(None,),
                            dtype="S256", fillvalue=self.model.encode("ASCII"))
        h5df.create_dataset(self.pred_key + "/prediction_meta/datasets", shape=(len(self.datasets),),
                            maxshape=(None,), dtype="S25", data=np.array(self.datasets, dtype="S50"))

        in_h5df.close()
        h5df.close()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.stop is None:
            self.batch_size = outputs.size(0)
            self.stop = self.batch_size

        if self.stop > self.length:  # length = length of input data
            self.stop = self.length

        h5df = h5py.File(self.output_file, "a")
        # round, so the outputs are integers
        h5df[f"{self.pred_key}/predictions"][self.start:self.stop] = np.array(outputs.cpu()).round(decimals=0)

        self.start += self.batch_size
        self.stop += self.batch_size

        h5df.flush()
        h5df.close()


def set_callbacks(output_dir, mode, prefix, seed, checkpoint_path, ckpt_quantity, save_top_k,
                  stop_quantity, patience, add_additional, model, datasets):
    if mode == "predict":
        predict_callback = PredictCallback(prefix, output_dir, add_additional, model, datasets)
        return [predict_callback]

    metrics_callback = MetricCallback(output_dir, mode, prefix)
    if mode == "validate":
        return [metrics_callback]

    seed_callback = SeedCallback(seed)
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
    callbacks = [checkpoint_callback, metrics_callback, seed_callback, early_stop, time_callback]
    return callbacks


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 2000)
        logging.info(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        logging.info(f"The seed provided by the user is: {seed}.")
    seed_everything(seed=seed, workers=True)  # seed for reproducibility


def set_seed_state(model):
    # path to the model checkpoint
    model_dict = torch.load(model)
    seed_dict = model_dict["callbacks"]["SeedCallback"]

    # does basically the same as seed_everything(), but reloads states from checkpoint
    os.environ["PL_GLOBAL_SEED"] = str(seed_dict["seed"])
    random.setstate(seed_dict["random_seed_state"])
    np.random.set_state(seed_dict["numpy_seed_state"])
    torch.set_rng_state(seed_dict["torch_seed_state"])
    torch.cuda.set_rng_state(seed_dict["torch.cuda_seed_state"])
    logging.info(f"The seed provided by the model is: {seed_dict['seed']}.")


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


def check_alternative_prediction(h5_file, pred_name):
    h5df = h5py.File(h5_file, mode="r")
    if "alternative" in h5df.keys():
        assert f"{pred_name}_prediction" not in h5df["alternative"].keys(), \
            f"alternative/{pred_name}_prediction is already in {h5_file}"
