import os
import logging
import sys
import glob
import numpy as np
import h5py

log = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    def format(self, msg):
        if hasattr(msg, "simple") and msg.simple:
            return msg.getMessage()  # if simple exists and is True, print just the message without date & level
        else:
            return logging.Formatter.format(self, msg)


# simple missing how to integrate custom formatter?
def get_log_dict(output_dir, prefix):
    return {
        "version": 1,
        "formatters": {
            "()": lambda: CustomFormatter('%(asctime)s, %(levelname)s: %(message)s',
                                          datefmt='%d.%m.%Y %H:%M:%S'),
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler",
                        "formatter": "custom",
                        "level": "WARNING",
                        "stream": sys.stdout},
            "file": {"class": "logging.FileHandler",
                     "formatter": "custom",
                     "level": "DEBUG",
                     "filename": f"{output_dir}/{prefix}predmoter.log", "mode": "a"}
        },
        "loggers": {
            __name__: {"level": "INFO",
                       "handlers": ["console", "file"],
                       "propagate": True},
        }
    }


def log_table(logger, contents, spacing, header=False, table_end=False):
    contents = [f"{i: <{spacing}}" if len(i) <= spacing else f"{i[:spacing - 2]}" + ".." for i in contents]
    # every string will have the length of spacing, also converts to string
    msg = "|".join(contents)

    if header:
        msg = "\n" + msg + "\n" + "-" * len(msg)

    elif table_end:
        msg = "-" * len(msg) + "\n" + msg + "\n"

    logger.info(msg, extra={"simple": True})


def check_datasets_in_h5():
    pass


def check_h5_in_directory():
    pass


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
