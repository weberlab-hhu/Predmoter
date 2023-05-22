import os
import logging
import logging.config
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from predmoter.core.constants import PREDMOTER_VERSION
from predmoter.core.parser import PredmoterParser
from predmoter.prediction.callbacks import SeedCallback, MetricCallback, Timeit, PredictCallback
from predmoter.utilities.utils import get_log_dict, get_h5_data, get_meta, get_available_datasets, file_stem
from predmoter.prediction.HybridModel import LitHybridNet
from predmoter.utilities.dataset import PredmoterSequence


def main():
    pp = PredmoterParser()
    args = pp.get_args()
    if args.resume_training or args.mode in ["test", "predict"]:
        args.datasets = torch.load(args.model)["hyper_parameters"]["datasets"]

    # Logging
    # ----------------------
    logging.config.dictConfig(get_log_dict(args.output_dir, args.prefix))
    log = logging.getLogger(__name__)
    log.info("\n", extra={"simple": True})
    log.info(f"Predmoter v{PREDMOTER_VERSION} is starting in {args.mode} mode.")

    # Check configurations
    # ----------------------
    if args.mode != "predict":
        h5_data = get_h5_data(args.input_dir, args.mode, args.datasets)  # dictionary of h5_files
    else:
        h5_data = {"predict": args.filepath}  # at some point: args.filepath and predict file inspection here

    # Callbacks and Trainer
    # ----------------------
    if args.mode == "train":
        # device if resume training --> works if seed callback has torch.cuda?
        include_cuda = True if args.device == "gpu" else False

        ckpt_path = os.path.join(args.output_dir, f"{args.prefix}checkpoints")
        os.makedirs(ckpt_path, exist_ok=True)
        if not args.resume_training and len(os.listdir(ckpt_path)) > 0:
            log.warning("Starting training for the first time and the checkpoint directory is not empty, "
                        "if this is intentional you can ignore this message")
        ckpt_method = "min" if "loss" in args.ckpt_quantity else "max"
        ckpt_filename = "predmoter_v{" + PREDMOTER_VERSION + "}_{epoch}_{" + args.ckpt_quantity + ":.4f}"
        # f-string would mess up formatting
        if args.save_top_k == 0:
            log.warning("save-top-k = 0 means no models will be saved, "
                        "if this is intentional you can ignore this message")

        stop_method = "min" if "loss" in args.stop_quantity else "max"

        callbacks = [MetricCallback(args.output_dir, args.mode, args.prefix),
                     SeedCallback(args.seed, args.resume_training, args.model, include_cuda),
                     ModelCheckpoint(save_top_k=args.save_top_k, monitor=args.ckpt_quantity,
                                     mode=ckpt_method, dirpath=ckpt_path, filename=ckpt_filename,
                                     save_last=True, save_on_train_epoch_end=True),
                     EarlyStopping(monitor=args.stop_quantity, min_delta=0.0, patience=args.patience,
                                   verbose=False, mode=stop_method, strict=True, check_finite=True,
                                   check_on_train_epoch_end=True),
                     Timeit()]
        # if args.resume_training: just SeedCallback, rest will reinit? -> try when test_data was made

    elif args.mode == "test":
        callbacks = [MetricCallback(args.output_dir, args.mode, args.prefix)]

    else:
        callbacks = [PredictCallback(args.prefix, args.output_dir, args.model, args.datasets)]

    trainer = pl.Trainer(callbacks=callbacks, devices=args.num_devices, accelerator=args.device,
                         max_epochs=args.epochs, logger=False, enable_progress_bar=False, deterministic=True)

    # Initialize model
    # ----------------------
    seq_len, bases = get_meta(h5_data[args.mode])

    if args.resume_training or args.mode in ["test", "predict"]:
        log.info(f"Chosen model checkpoint: {args.model}")
        hybrid_model = LitHybridNet.load_from_checkpoint(args.model, seq_len=seq_len)
        log.info(f"Model's dataset(s): {', '.join(args.datasets)}")
        assert hybrid_model.input_size == bases, \
            f"Your chosen model has the input size {hybrid_model.input_size} and your dataset {bases}." \
            f"Please use the same input size."  # rare to impossible, but just in case
    else:
        log.info(f"Chosen dataset(s): {', '.join(args.datasets)}")
        hybrid_model = LitHybridNet(args.model_type, args.cnn_layers, args.filter_size, args.kernel_size,
                                    args.step, args.up, args.dilation, args.lstm_layers, args.hidden_size,
                                    args.bnorm, args.dropout, args.learning_rate, seq_len, input_size=bases,
                                    output_size=len(args.datasets), datasets=args.datasets)

    log.info(f"\n\nModel summary (model type: {hybrid_model.model_type}; dropout: {hybrid_model.dropout}):"
             f"\n{ModelSummary(model=hybrid_model, max_depth=-1)}\n")

    # Predmoter start
    # --------------------------------
    pin_mem = True if args.device == "gpu" else False

    if args.mode == "train":
        log.info(f"Loading training data into memory ...")
        train_loader = DataLoader(PredmoterSequence(h5_data["train"], "train", args.datasets, seq_len),
                                  batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)
        val_loader = DataLoader(PredmoterSequence(h5_data["val"], "val", args.datasets, seq_len),
                                batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)
        log.info(f"Training started. Resuming training: {args.resume_training}.")
        if args.resume_training:
            # ckpt_path=model restores all previous states like callbacks, optimizer state, etc.
            trainer.fit(model=hybrid_model, ckpt_path=args.model,
                        train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        log.info("Training ended.")

    elif args.mode == "test":
        # does the test step loop accumulate in RAM?
        log.info(f"Testing started. Each file will be loaded into memory and tested individually.")
        for file in h5_data["test"]:
            # only use available datasets to load the test data, otherwise more RAM will be used
            avail_datasets = get_available_datasets(file, args.datasets)
            test_loader = DataLoader(PredmoterSequence([file], "test", avail_datasets, seq_len),
                                     batch_size=args.test_batch_size, shuffle=False,
                                     pin_memory=pin_mem, num_workers=0)
            trainer.test(model=hybrid_model, dataloaders=test_loader, verbose=False)
            log.info("Testing ended.")

    else:
        outfile = f"{file_stem(args.filepath)}_predictions.h5"
        out = os.path.join(args.output_dir, outfile)
        if os.path.exists(out):
            raise OSError(f"the predictions output file {outfile} exists in {args.output_dir},"
                          f" please move or delete it")

        log.info(f"Loading predict data into memory ...")
        predict_loader = DataLoader(PredmoterSequence(h5_data["predict"], "predict", None, None),
                                    batch_size=args.predict_batch_size, shuffle=False,
                                    pin_memory=pin_mem, num_workers=0)
        log.info("Predicting started.")
        trainer.predict(model=hybrid_model, dataloaders=predict_loader)
        logging.info("Predicting ended.")

    logging.info("Predmoter finished.\n")


if __name__ == "__main__":
    main()
