import logging
import os
import random
import time
import numpy as np
import torch
import h5py
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import seed_everything

from predmoter.utilities.utils import log_table
#  Please use `lightning_lite.utilities.seed.seed_everything` for Pytorch Lightning v2.0.0 and upwards instead.

log = logging.getLogger(__name__)


class MetricCallback(Callback):
    def __init__(self, output_dir, mode, prefix):
        super().__init__()
        self.mode = mode
        self.filename = f"{prefix}test_metrics.log" if self.mode == "test" else f"{prefix}metrics.log"
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
            msg = " ".join(["file"] + list(metrics.keys())) + "\n" + msg
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
        log_table(log, ["Epoch", "Duration (min)"], spacing=16, header=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.last_epoch += 1
        duration = time.time() - self.start
        self.durations.append(duration)
        # just log every 5/10 epochs??, how much time total and on avg
        log_table(log, [trainer.current_epoch, round((duration/60), ndigits=2)], spacing=16)

    def on_train_end(self, trainer, pl_module):
        log_table(log, [self.last_epoch, f"{sum(self.durations)/60**2:.2f} h"], spacing=16, table_end=True)


# will num_workers work?? -> test
class SeedCallback(Callback):
    def __init__(self, seed: int, resume: bool, model_path, include_cuda: bool):
        # self.device? -> warning please don't change devices resume training
        self.resume = resume
        self.include_cuda = include_cuda
        if not self.resume:
            self.seed = self.set_seed(seed)
            self.state = self.collect_seed_state(self.include_cuda)
        else:
            # load model checkpoint callback
            seed_callback = torch.load(model_path)["callbacks"]["SeedCallback"]

            # retrieve state key and state dict from model checkpoint
            self.seed, self.state = list(seed_callback.items())[0]
            log.info(f"The seed provided by the model is: {self.seed}.")

    @property
    def state_key(self) -> str:
        return f"{self.seed}"

    def on_fit_start(self, trainer, pl_module):
        if self.resume:
            self.set_seed_state(self.state)  # does this need to happen before the trainer is initialized?

    def on_train_epoch_end(self, trainer, pl_module):
        self.state.update(self.collect_seed_state(self.include_cuda))

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

    @staticmethod
    def set_seed(seed):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min

        if seed is None or not (min_seed_value <= seed <= max_seed_value):
            seed = random.randint(min_seed_value, max_seed_value)
            log.info(f"A seed wasn't provided by the user. The random seed is: {seed}.")
        else:
            log.info(f"The seed provided by the user is: {seed}.")
        seed_everything(seed=seed, workers=False)  # seed for reproducibility
        # adapt to num workers if it works with resume training and is faster
        return seed

    @staticmethod
    def collect_seed_state(include_cuda):
        """ Function adapted from lightning.fabric.utilities.seed._collect_rng_states to collect
        the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python (random)."""
        states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if include_cuda:
            states["torch.cuda"] = torch.cuda.get_rng_state_all()
        return states

    @staticmethod
    def set_seed_state(rng_state_dict):
        """Function adapted from lightning.fabric.utilities.seed._set_rng_states to set the global random
        state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python (random) in the current process."""
        torch.set_rng_state(rng_state_dict["torch"])
        # torch.cuda rng_state is only included since v1.8.
        if "torch.cuda" in rng_state_dict:
            torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
        np.random.set_state(rng_state_dict["numpy"])
        version, state, gauss = rng_state_dict["random"]
        random.setstate((version, tuple(state), gauss))


class PredictCallback(Callback):
    def __init__(self, prefix, output_dir, model, datasets):
        self.prefix = prefix
        self.output_dir = output_dir
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
        h5df = h5py.File(self.output_file, "w")

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
        h5df.create_group("prediction")
        h5df.create_dataset("prediction" + "/predictions", shape=(self.length, chunk_len, len(self.datasets)),
                            maxshape=(None, chunk_len, None), chunks=(1, chunk_len, 1), dtype=int,
                            compression="gzip", compression_opts=9, fillvalue=-1)

        h5df.create_dataset("prediction" + "/model_name", shape=(1,), maxshape=(None,),
                            dtype="S256", fillvalue=self.model.encode("ASCII"))
        h5df.create_dataset("prediction" + "/datasets", shape=(len(self.datasets),),
                            maxshape=(None,), dtype="S25", data=np.array(self.datasets, dtype="S50"))

        in_h5df.close()
        h5df.close()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if self.stop is None:
            self.batch_size = outputs.size(0)
            self.stop = self.batch_size

        if self.stop > self.length:  # length = length of input data
            self.stop = self.length

        h5df = h5py.File(self.output_file, "a")
        # round, so the outputs are integers
        h5df[f"prediction/predictions"][self.start:self.stop] = np.array(outputs.cpu()).round(decimals=0).astype(int)

        self.start += self.batch_size
        self.stop += self.batch_size

        h5df.flush()
        h5df.close()
