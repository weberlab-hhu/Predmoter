#! /usr/bin/env python3
import os
import logging
import logging.config
import tempfile
import torch
import datetime
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from predmoter.core.constants import PREDMOTER_VERSION, GIT_COMMIT
from predmoter.core.parser import PredmoterParser
from predmoter.prediction.callbacks import SeedCallback, MetricCallback, Timeit, PredictCallback
from predmoter.utilities.utils import get_log_dict, rank_zero_info, rank_zero_warn, get_h5_data, \
    prep_predict_data, fasta2h5, get_meta, get_available_datasets, file_stem, init_model
from predmoter.utilities.dataset import get_dataset
from predmoter.utilities.converter import Converter


def train(args, input_data, seq_len, bases, pin_mem, strategy):
    # Callbacks and Trainer
    # ----------------------
    include_cuda = True if args.device == "gpu" else False

    ckpt_path = os.path.join(args.output_dir, f"{args.prefix}checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    if not args.resume_training and len(os.listdir(ckpt_path)) > 0:
        rank_zero_warn("Starting training for the first time and the checkpoint directory is not empty, "
                       "if this is intentional you can ignore this message.")
    ckpt_method = "min" if "loss" in args.ckpt_quantity else "max"
    ckpt_filename = f"predmoter_v{PREDMOTER_VERSION}" + "_{epoch}_{" + args.ckpt_quantity + ":.4f}"
    # full f-string would mess up formatting
    if args.save_top_k == 0:
        rank_zero_warn("save-top-k = 0 means no models will be saved, "
                       "if this is intentional you can ignore this message")

    stop_method = "min" if "loss" in args.stop_quantity else "max"

    callbacks = [MetricCallback(args.output_dir, args.mode, args.prefix),
                 SeedCallback(args.seed, args.resume_training, args.model, include_cuda, args.num_workers),
                 ModelCheckpoint(save_top_k=args.save_top_k, monitor=args.ckpt_quantity,
                                 mode=ckpt_method, dirpath=ckpt_path, filename=ckpt_filename,
                                 save_last=True, save_on_train_epoch_end=True),
                 EarlyStopping(monitor=args.stop_quantity, min_delta=0.0, patience=args.patience,
                               verbose=False, mode=stop_method, strict=True, check_finite=True,
                               check_on_train_epoch_end=True),
                 Timeit(args.epochs)]

    trainer = pl.Trainer(callbacks=callbacks, devices=args.num_devices, accelerator=args.device, strategy=strategy,
                         max_epochs=args.epochs, logger=False, enable_progress_bar=False, deterministic=True)

    # Initialize model
    # ----------------------
    # SeedCallback's set_seed needs to be executed when resume_training is False before initializing the
    # model, when resuming the old seed_state needs to be set after initializing the model and before the
    # training starts (see callbacks.py)
    load_from_checkpoint = True if args.resume_training else False
    hybrid_model = init_model(args, seq_len, bases, load_from_checkpoint=load_from_checkpoint)

    # Training
    # ----------------------
    train_loader = DataLoader(get_dataset(input_data["train"], "train", args.datasets, seq_len,
                                          args.ram_efficient, args.blacklist), batch_size=args.batch_size,
                              shuffle=True, pin_memory=pin_mem, num_workers=args.num_workers)
    val_loader = DataLoader(get_dataset(input_data["val"], "val", args.datasets, seq_len,
                                        args.ram_efficient, args.blacklist), batch_size=args.batch_size,
                            shuffle=False, pin_memory=pin_mem, num_workers=args.num_workers)
    rank_zero_info(f"Training started. Training on {args.num_devices} device(s). "
                   f"Resuming training: {args.resume_training}.")
    if args.resume_training:
        # ckpt_path=model restores all previous states like callbacks, optimizer state, etc.
        trainer.fit(model=hybrid_model, ckpt_path=args.model,
                    train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    rank_zero_info("Training ended.")


def test(args, input_data, seq_len, pin_mem, strategy):
    # Initialize model
    # ----------------------
    hybrid_model = init_model(args, seq_len, None, load_from_checkpoint=True)

    # Callbacks and Trainer
    # ----------------------
    callbacks = [MetricCallback(args.output_dir, args.mode, args.prefix),
                 Timeit(None)]
    trainer = pl.Trainer(callbacks=callbacks, devices=args.num_devices, accelerator=args.device, strategy=strategy,
                         max_epochs=args.epochs, logger=False, enable_progress_bar=False)

    # Testing
    # ----------------------
    rank_zero_info(f"Testing started. Each file will be tested individually.")
    for file in input_data:
        # only use available datasets to load the test data, otherwise more RAM/computing time will be used
        avail_datasets = get_available_datasets(file, args.datasets)
        test_loader = DataLoader(get_dataset([file], "test", avail_datasets, seq_len,
                                             args.ram_efficient, args.blacklist), batch_size=args.batch_size,
                                 shuffle=False, pin_memory=pin_mem, num_workers=args.num_workers)
        trainer.test(model=hybrid_model, dataloaders=test_loader, verbose=False)
    rank_zero_info("Testing ended.")


def predict(args, input_data, seq_len, pin_mem, strategy, temp_dir=None):
    # Initialize model
    # ----------------------
    hybrid_model = init_model(args, seq_len, None, load_from_checkpoint=True)

    # Configure output file
    # ----------------------
    outfile = f"{file_stem(input_data[0])}_predictions.h5"
    out_filepath = os.path.join(args.output_dir, outfile)
    if os.path.exists(out_filepath):
        raise OSError(f"the predictions output file {outfile} exists in your output directory,"
                      f" please move or delete it")

    # Convert fasta to h5
    # ----------------------
    if temp_dir is not None:
        h5_output = os.path.join(temp_dir, f"{args.species}.h5")
        fasta2h5(input_data[0], h5_output, seq_len, args.species, args.multiprocess)
        input_data = [h5_output]  # set input_data to converted file

    # Callbacks and Trainer
    # ----------------------
    callbacks = [PredictCallback(out_filepath, input_data[0], args.model, args.datasets),
                 Timeit(None)]
    trainer = pl.Trainer(callbacks=callbacks, devices=args.num_devices, accelerator=args.device, strategy=strategy,
                         max_epochs=args.epochs, logger=False, enable_progress_bar=False)

    # Predicting
    # ----------------------
    predict_loader = DataLoader(get_dataset(input_data, "predict", None, seq_len, args.ram_efficient, args.blacklist),
                                batch_size=args.batch_size, shuffle=False,
                                pin_memory=pin_mem, num_workers=args.num_workers)
    rank_zero_info("Predicting started.")
    trainer.predict(model=hybrid_model, dataloaders=predict_loader, return_predictions=False)

    if args.output_format is not None:
        Converter(os.path.join(args.output_dir, f"{file_stem(args.filepath)}_predictions.h5"),
                  args.output_dir, args.output_format, basename=file_stem(args.filepath), strand=None)


def main():
    pp = PredmoterParser()
    args = pp.get_args()
    if args.mode == "train" and args.resume_training:
        # load onto CPU: torch.load(<something>, map_location=lambda storage, loc: storage)
        epochs_trained = torch.load(args.model, map_location=lambda storage, loc: storage)["epoch"]
        if epochs_trained >= (args.epochs - 1):
            raise ValueError(f"when resuming training, the chosen number of epochs need to be > epochs "
                             f"already trained, the model {args.model} already trained {(epochs_trained + 1)}")
    if args.resume_training or args.mode in ["test", "predict"]:
        args.datasets = torch.load(args.model,
                                   map_location=lambda storage, loc: storage)["hyper_parameters"]["datasets"]

    # Logging
    # ----------------------
    logging.config.dictConfig(get_log_dict(args.output_dir, args.prefix))
    rank_zero_info("\n", simple=True)
    rank_zero_info(f"Predmoter v{PREDMOTER_VERSION} is starting in {args.mode} mode. "
                   f"The current commit is {GIT_COMMIT}.")
    if args.num_devices > 1 and not args.ram_efficient:
        rank_zero_info(f"Hint: Using {args.num_devices} CPUs/GPUs to train on results in the creation of one "
                       f"dataset for each device. The data read in time and RAM consumption will be multiplied "
                       f"by {args.num_devices}. Consider using --num-workers 0 for memory consumption and speed "
                       f"improvements.")

    # Get input data
    # ----------------------
    if args.mode != "predict":
        h5_data = get_h5_data(args.input_dir, args.mode, args.datasets)  # dictionary of h5 files
    else:
        h5_data = prep_predict_data(args.filepath)  # dictionary of one h5 file

    # Get meta data
    # ----------------------
    if "fasta" in h5_data:
        seq_len, bases = args.subsequence_length, None
    else:
        seq_len, bases = get_meta(h5_data[args.mode][0])

    # Predmoter start
    # --------------------------------
    pin_mem = True if args.device == "gpu" else False
    strategy = DDPStrategy(timeout=datetime.timedelta(seconds=10_800)) if args.num_devices > 1 else "auto"

    if args.mode == "train":
        train(args, h5_data, seq_len, bases, pin_mem, strategy)
    elif args.mode == "test":
        test(args, h5_data["test"], seq_len, pin_mem, strategy)
    else:
        if "fasta" in h5_data:
            # execute prediction with a temporary directory, so whenever an error occurs or the code
            # finished successfully, the temporary h5 file and directory are deleted
            with tempfile.TemporaryDirectory(dir=args.output_dir) as tmp_dir:
                predict(args, h5_data["predict"], seq_len, pin_mem, strategy, temp_dir=tmp_dir)
        else:
            predict(args, h5_data["predict"], seq_len, pin_mem, strategy)

    rank_zero_info("Predmoter finished.\n")


if __name__ == "__main__":
    main()
