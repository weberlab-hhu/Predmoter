import os
import logging
import sys
import glob
import time
import numpy as np
import h5py
from collections import Counter
from lightning.pytorch.utilities import rank_zero_only  # don't log twice while training on multiple GPUs
from lightning.pytorch.utilities.model_summary import ModelSummary
from helixer.export.exporter import HelixerFastaToH5Controller

from predmoter.prediction.HybridModel import LitHybridNet

log = logging.getLogger("PredmoterLogger")


class CustomFormatter(logging.Formatter):
    """Custom formatter to exclude the usual format from specific messages"""
    def format(self, msg):
        if hasattr(msg, "simple") and msg.simple:
            return msg.getMessage()  # if simple exists and is True, print just the message without date & level
        else:
            return logging.Formatter.format(self, msg)


# simple missing how to integrate custom formatter?
def get_log_dict(output_dir, prefix):
    return {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "custom":
                {"()": lambda: CustomFormatter('%(asctime)s, %(levelname)s: %(message)s',
                                               datefmt='%d.%m.%Y %H:%M:%S')}
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
            "PredmoterLogger": {"level": "INFO",
                                "handlers": ["console", "file"],
                                "propagate": True},
        }
    }


@rank_zero_only
def rank_zero_info(msg, simple=False):
    log.info(msg, extra={"simple": simple})


@rank_zero_only
def rank_zero_warn(msg, simple=False):
    log.warning(msg, extra={"simple": simple})


def log_table(logger, contents, spacing, header=False, table_end=False, rank_zero=False):
    """Log table rows incrementally to the log file"""
    contents = list(map(str, contents))  # converts to string
    contents = [f"{i: <{spacing}}" if len(i) <= spacing else f"{i[:spacing - 2]}" + ".." for i in contents]
    # every string will have the length of spacing
    msg = "|".join(contents)

    if header:
        msg = "\n" + msg + "\n" + "-" * len(msg)

    elif table_end:
        msg = "-" * len(msg) + "\n" + msg + "\n"

    if rank_zero:
        rank_zero_info(msg, simple=True)
    else:
        logger.info(msg, extra={"simple": True})


def get_available_datasets(h5_file, datasets):
    """Check available datasets in a h5 file in comparison to given datasets."""
    h5df = h5py.File(h5_file, mode="r")
    avail_datasets = [dset for dset in datasets if f"{dset}_coverage" in h5df["evaluation"].keys()]
    return avail_datasets


def get_h5_data(input_dir, mode, dsets):
    """Collect and check the input h5 files.

    The expected input directories are scanned for the h5 files. Non-valid h5 files will throw an error.
    The files are also scanned for their available datasets in comparison to the given datasets.

        Args:
            input_dir: 'str', input directory to find train, val or test folders in
            mode: 'str', mode of main program (here: train/test)
            dsets: 'list', list of chosen/model defined datasets

        Returns:
            h5_data: 'dict', dictionary of lists containing the h5 files
                per type (train/val/test) depending on the mode
    """
    # for all train/val/test -> predict different solution
    types = ["train", "val"] if mode == "train" else ["test"]
    h5_data = {key: None for key in types}
    for type_ in types:
        h5_files = glob.glob(os.path.join(input_dir, type_, "*.h5"))
        assert len(h5_files) >= 1, f"no {type_} h5 files were provided, " \
                                   f"h5 files are expected to have the extension .h5"
        # check individual h5 files
        skip_files = []
        dset_counter = Counter()
        for file in h5_files:
            try:
                h5py.File(file, "r")
            except OSError as e:
                raise OSError(f"{file} is not a h5 file") from e
            available_datasets = get_available_datasets(file, dsets)
            dset_counter.update(available_datasets)
            if len(available_datasets) == 0:
                skip_files.append(file_stem(file))
                h5_files.remove(file)
        # after excluding files not having the specified datasets (either from a model
        # checkpoint or user defined), make sure there are files left
        assert len(h5_files) >= 1, f"no {type_} data available: none of the {type_} h5 files contain " \
                                   f"coverage data from the chosen/model's datasets {', '.join(dsets)}"
        if mode == "train":
            # the PredmoterSequence code will work if a dataset is not available in all files,
            # but it is nonsensical and a waste of memory to let it slide in train/val data
            # in the case of test data: only available datasets will be loaded into memory anyway
            for dset in dsets:
                if dset not in dset_counter:
                    raise ValueError(f"the dataset {dset} was not found in any {type_} h5 files, "
                                     f"please only choose datasets that are available in your {type_} set")

        if len(skip_files) >= 1:
            log.info(f"The h5 file(s) {', '.join(skip_files)} don't contain the chosen/model's "
                     f"datasets {', '.join(dsets)} and will be skipped.")
        # add files to data if no errors occurred
        h5_files.sort()  # sort alphabetically -> keep file order for reproducibility
        h5_data[type_] = h5_files
    return h5_data


def prep_predict_data(filepath):
    """Check the two valid predict file formats: h5 and fasta."""
    if filepath.lower().endswith((".fasta", ".fna", ".ffn", ".faa", ".frn", ".fa")):
        return {"predict": [filepath], "fasta": True}  # list for consistency reasons
    try:
        h5py.File(filepath, "r")
    except OSError as e:
        raise OSError(f"{filepath} is not a h5 file") from e
    return {"predict": [filepath]}  # list for consistency reasons


def fasta2h5(filepath, h5_output_path, subseq_len, multiprocess):
    """Convert fasta file to h5 file using Helixer."""
    start = time.time()
    rank_zero_info("Converting fasta input file to h5 file.")
    controller = HelixerFastaToH5Controller(filepath, h5_output_path)
    controller.export_fasta_to_h5(chunk_size=subseq_len, compression="gzip",
                                  multiprocess=multiprocess, species="Lorem_ipsum")
    rank_zero_info(f"Conversion to h5 file finished. It took {round(((time.time() - start) / 60), ndigits=2)} min.")
    # filler species, as this h5 file will be deleted anyway


def get_meta(h5_file):
    """Get metadata about the sequence length and the input size (bases).

    It is possible to use a different sequence length for testing or predicting than training.
    The bases are checked to make sure that the model is compatible/define the model's input size.
    """
    h5df = h5py.File(h5_file, mode="r")
    X = np.array(h5df["data/X"][:1], dtype=np.int8)
    meta = X.shape[1:]
    assert len(meta) == 2, f"expected all arrays to have the shape (seq_len, bases) found {meta}"
    return meta


def file_stem(path):
    """Returns the file name without extension. Adapted from Helixer."""
    return os.path.basename(path).split('.')[0]


def init_model(args, seq_len, bases, load_from_checkpoint: bool):  # args = Predmoter args
    """Initialize the model."""
    if load_from_checkpoint:
        # if something ever changes in the nucleotide encoding,
        # bases of the input files == model input_size need to be checked here
        rank_zero_info(f"Chosen model checkpoint: {args.model}")
        model = LitHybridNet.load_from_checkpoint(args.model, seq_len=seq_len)
        rank_zero_info(f"Model's dataset(s): {', '.join(args.datasets)}.")
    else:
        rank_zero_info(f"Chosen dataset(s): {', '.join(args.datasets)}.")
        model = LitHybridNet(args.model_type, args.cnn_layers, args.filter_size, args.kernel_size,
                             args.step, args.up, args.dilation, args.lstm_layers, args.hidden_size,
                             args.bnorm, args.dropout, args.learning_rate, seq_len, input_size=bases,
                             output_size=len(args.datasets), datasets=args.datasets)

    rank_zero_info(f"\n\nModel summary (model type: {model.model_type}, dropout: {model.dropout})"
                   f"\n{ModelSummary(model=model, max_depth=-1)}\n")
    return model
