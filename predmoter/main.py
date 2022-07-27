import sys
import os
# import time
import glob
import random
import logging
import argparse
import numpy as np
import h5py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
from dataset import get_dataloader
from HybridModel import LitHybridNet


class MetricCallback(Callback):
    def __init__(self, output_dir, prefix):
        super().__init__()
        self.file = "/".join([output_dir.rstrip("/"), f"{prefix}metrics.txt"])

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics  # does not add validation sanity check
        msg = "{} {} {} {} {}".format(epoch, metrics["avg_train_loss"], metrics["avg_val_loss"],
                                      metrics["avg_train_accuracy"], metrics["avg_val_accuracy"])
        if epoch == 0:
            msg = "epoch training_loss validation_loss training_accuracy validation_accuracy\n" + msg
        self.save_metrics(msg)

    def save_metrics(self, msg):
        with open(self.file, "a") as f:
            f.write(f"{msg}\n")


def set_callbacks(output_dir, prefix, checkpoint_path, quantity, patience):
    assert quantity in ["avg_train_loss", "avg_train_accuracy", "avg_val_loss", "avg_val_accuracy"],\
        f"can not monitor invalid quantity: {quantity}"
    mode = "min" if "loss" in quantity else "max"
    filename = "predmoter_{epoch}_{" + quantity + ":.4f}"  # f-string would mess up formatting
    metrics_callback = MetricCallback(output_dir, prefix)
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor=quantity, mode=mode, dirpath=checkpoint_path,
                                          filename=filename, save_on_train_epoch_end=True, save_last=True)
    early_stop = EarlyStopping(monitor=quantity, min_delta=0.0, patience=patience, verbose=False,
                               mode=mode, strict=True, check_finite=True, check_on_train_epoch_end=True)
    callbacks = [checkpoint_callback, metrics_callback, early_stop]
    return callbacks


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 2000)
        seed_everything(seed=seed, workers=True)  # seed for reproducibility
        logging.info(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        seed_everything(seed=seed, workers=True)


def init_logging(output_dir, prefix):
    logging.getLogger("torch").setLevel(logging.WARNING)  # only log info from main.py
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    file_name = "/".join([output_dir.rstrip("/"), f"{prefix}predmoter.log"])

    logging.basicConfig(filename=file_name, filemode="a",
                        format="%(asctime)s, %(name)s %(levelname)s: %(message)s",
                        datefmt="%d.%m.%Y %H:%M:%S", level=logging.DEBUG)


def check_paths(path_list):
    for path in path_list:
        assert os.path.exists(path), f"the directory {path} doesn't exist"


def get_meta(input_dir, mode):
    input_dir = "/".join([input_dir.rstrip("/"), "test"]) if mode == "predict" \
        else "/".join([input_dir.rstrip("/"), "train"])
    h5_files = glob.glob(os.path.join(input_dir, '*.h5'))
    assert len(h5_files) >= 1, "no input file/s were provided"
    for h5_file in h5_files:
        h5df = h5py.File(h5_file, mode="r")
        X = np.array(h5df["data/X"][:1], dtype=np.int8)
        return X.shape[1:]


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, seed,
         checkpoint_path, quantity, patience, batch_size, test_batch_size, num_workers, prefix, device,
         num_devices, epochs, limit_predict_batches):
    assert mode in ["train", "predict"], f"valid modes are train or predict, not {mode}"
    check_paths([input_directory, output_directory, checkpoint_path])
    init_logging(output_directory, prefix)
    logging.info(f"Predmoter is starting in {mode} mode.")
    set_seed(seed)
    meta = get_meta(input_directory, mode)
    assert len(meta) == 2, f"expected all arrays to have the shape (seq_len, bases) found {meta}"
    if resume_training or mode == "predict":
        model = "/".join([checkpoint_path.rstrip("/"), model])
        hybrid_model = LitHybridNet.load_from_checkpoint(model, seq_len=meta[0])
    else:
        hybrid_model = LitHybridNet(**model_arguments, seq_len=meta[0], input_size=meta[1])
    logging.info(f"\n\nModel summary:\n{ModelSummary(model=hybrid_model, max_depth=-1)}\n")
    callbacks = set_callbacks(output_directory, prefix, checkpoint_path, quantity, patience)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False,
                         deterministic=True, limit_predict_batches=limit_predict_batches)
    if mode == "train":
        logging.info("Loading training and validation data into memory.")
        train_loader = get_dataloader(input_dir=input_directory, type_="train", shuffle=True,
                                      batch_size=batch_size, num_workers=num_workers, seq_len=meta[0])
        val_loader = get_dataloader(input_dir=input_directory, type_="val", shuffle=False,
                                    batch_size=batch_size, num_workers=num_workers, seq_len=meta[0])
        mem_size = (sys.getsizeof(train_loader.dataset.X) + sys.getsizeof(train_loader.dataset.Y) +
                    sys.getsizeof(val_loader.dataset.X) + sys.getsizeof(val_loader.dataset.Y))/1024**3
        logging.info(f"Finished loading data into memory. The compressed data is {mem_size:.4f} Gb in size.")

        logging.info(f"Training started. Resuming training: {resume_training}.")
        if resume_training:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader,
                        val_dataloaders=val_loader, ckpt_path=model)  # restores all previous states
        else:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logging.info("Training ended.")

    elif mode == "predict":
        logging.info("Loading test data into memory.")
        test_loader = get_dataloader(input_dir=input_directory, type_="test", shuffle=False,
                                     batch_size=test_batch_size, num_workers=num_workers, seq_len=meta[0])
        mem_size = sys.getsizeof(test_loader.dataset.X)/1024 ** 3
        logging.info(f"Finished loading data into memory. The compressed data is {mem_size:.4f} Gb in size.")
        logging.info("Predicting started.")
        predictions = trainer.predict(model=hybrid_model, dataloaders=test_loader)
        logging.info("Predicting ended.")
        # need trainer or use loop?? (limit_predict_batches not possible then)
        predictions = torch.cat(predictions, dim=0)  # unify list of preds to tensor
        torch.save(predictions, "/".join([output_directory.rstrip("/"), f"{prefix}predictions.pt"]))
    logging.info("Predmoter finished.\n")


# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Predmoter", description="Predict promoter regions in the DNA.",
                                     add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = LitHybridNet.add_model_specific_args(parser)  # add model specific args
    model_args = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")  # not included in args
    parser.add_argument("--version", action="version", version="%(prog)s 0.8")  # not included in args
    parser.add_argument("-i", "--input-directory", type=str, default=".",
                        help="path to a directory containing one train, val and test directory with h5-files")
    parser.add_argument("-o", "--output-directory", type=str, default=".",
                        help="directory to save log-files and predictions to")
    parser.add_argument("-m", "--mode", type=str, default=None, required=True, help="valid modes: train or predict")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--model", type=str, default="last.ckpt",
                        help="name of the model used for predictions or resuming training")  # maybe best_model_path?!
    parser.add_argument("--seed", type=int, default=None, help="if not provided: will be chosen randomly")
    parser.add_argument("--checkpoint-path", type=str, default=".",
                        help="otherwise empty directory to save model checkpoints to")
    parser.add_argument("--quantity", type=str, default="avg_train_accuracy",
                        help="quantity to monitor for checkpoints and early stopping "
                             "(valid: avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)")
    parser.add_argument("--patience", type=int, default=3,
                        help="allowed epochs without training accuracy improvement before stopping training")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="(default: %(default)d)")
    parser.add_argument("-tb", "--test-batch-size", type=int, default=10, help="(default: %(default)d)")
    parser.add_argument("--num-workers", type=int, default=8, help="for multiprocessing during data loading")
    parser.add_argument("--prefix", type=str, default="", help="prefix for loss, accuracy and log files")
    group = parser.add_argument_group("Trainer_arguments")
    group.add_argument("--device", type=str, default="gpu", help="device to train on")
    group.add_argument("--num-devices", type=int, default=1, help="(default: %(default)d)")
    group.add_argument("-e", "--epochs", type=int, default=35, help="(default: %(default)d)")
    group.add_argument("--limit-predict-batches", action="store", dest="limit_predict_batches",
                       help="limiting predict: float = fraction, int = num_batches (default: 1.0)")
    args = parser.parse_args()

    # limit predict
    if args.limit_predict_batches is None:
        args.limit_predict_batches = 1.0

    # prefix
    if args.prefix != "":
        args.prefix = f"{args.prefix}_"

    # dictionary stuff
    dict_model_args = vars(model_args[0])
    dict_args = vars(args)

    for key in dict_model_args:
        dict_args.pop(key)

    # run main program
    main(model_arguments=dict_model_args, **dict_args)
