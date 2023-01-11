import os
import logging
import argparse
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from dataset import get_dataloader
from utils import check_paths, init_logging, set_seed, get_meta, set_callbacks, get_available_datasets
from HybridModel import LitHybridNet


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, datasets, save_top_k,
         seed, checkpoint_path, ckpt_quantity, stop_quantity, patience, batch_size, test_batch_size,
         predict_batch_size, prefix, device, num_devices, epochs, limit_predict_batches):

    # Argument checks and cleanup
    # --------------------------------
    prefix = f"{prefix}_" if prefix is not None else ""
    if checkpoint_path is None:  # and mode != "validate" ??
        checkpoint_path = os.path.join(output_directory, f"{prefix}checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
    check_paths([input_directory, output_directory, checkpoint_path])
    assert mode in ["train",  "validate", "predict"], f"valid modes are train, validate or predict, not {mode}"
    limit_predict_batches = 1.0 if limit_predict_batches is None else limit_predict_batches

    # Preset initialization
    # --------------------------------
    init_logging(output_directory, prefix)
    logging.info("\n", extra={"simple": True})
    logging.info(f"Predmoter is starting in {mode} mode.")
    if mode == "train":
        set_seed(seed)
    seq_len, bases = get_meta(input_directory, mode)

    #  Model initialization
    # --------------------------------
    if resume_training or mode in ["predict", "validate"]:
        # if only model name is given/the path doesn't exist, assume model is in the checkpoint directory
        if not os.path.exists(model):
            model = os.path.join(checkpoint_path, model)
        check_paths([model])
        logging.info(f"Chosen model checkpoint: {model}")
        hybrid_model = LitHybridNet.load_from_checkpoint(model, seq_len=seq_len)
        datasets = hybrid_model.datasets
        logging.info(f"Model dataset(s): {' '.join(datasets)}")
        assert hybrid_model.input_size == bases,\
            f"Your chosen model has the input size {hybrid_model.input_size} and your dataset {bases}." \
            f"Please use the same input size."  # how often would this actually happen?
    else:
        logging.info(f"Chosen dataset(s): {' '.join(datasets)}")
        hybrid_model = LitHybridNet(**model_arguments, seq_len=seq_len, input_size=bases,
                                    output_size=len(datasets), datasets=datasets)
    logging.info(f"\n\nModel summary (model type: {hybrid_model.model_type}):"
                 f"\n{ModelSummary(model=hybrid_model, max_depth=-1)}\n")

    # Training preset initialization
    # --------------------------------
    callbacks = set_callbacks(output_directory, mode, prefix, checkpoint_path,
                              ckpt_quantity, save_top_k, stop_quantity, patience)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False,
                         deterministic=True, limit_predict_batches=limit_predict_batches)

    # Predmoter start
    # --------------------------------
    if mode == "train":
        train_loader = get_dataloader(input_dir=input_directory, type_="train", batch_size=batch_size,
                                      seq_len=seq_len, datasets=datasets)
        val_loader = get_dataloader(input_dir=input_directory, type_="val", batch_size=batch_size,
                                    seq_len=seq_len, datasets=datasets)
        logging.info(f"Training started. Resuming training: {resume_training}.")  # to print callback at some point
        if resume_training:
            # ckpt_path=model restores all previous states like callbacks, optimizer state, etc.
            trainer.fit(model=hybrid_model, ckpt_path=model,
                        train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logging.info("Training ended.")

    else:
        type_ = "test" if mode == "validate" else "predict"
        path = os.path.join(input_directory, type_)
        files = glob.glob(os.path.join(path, "*.h5"))
        msg = f"Validation started. Each file in {path} will be validated individually." if mode == "validate" \
            else f"Predicting started. Predictions will be done individually for each file in {path}."
        logging.info(msg)

        for file in files:
            if mode == "validate":
                # only use available datasets to load the test data, otherwise more RAM will be used
                avail_datasets = get_available_datasets(file, datasets)
                if len(avail_datasets) < 1:
                    logging.info(f"None of the model's datasets ({' '.join(datasets)}) are available "
                                 f"in the file {file.split('/')[-1]}: skipping...")
                    continue
                val_loader = get_dataloader(input_dir=input_directory, type_=type_,
                                            batch_size=test_batch_size, seq_len=seq_len,
                                            datasets=avail_datasets, file=file)
                logging.info("Validating ...")
                trainer.test(model=hybrid_model, dataloaders=val_loader, verbose=False)

            else:
                predict_loader = get_dataloader(input_dir=input_directory, type_=type_,
                                                batch_size=predict_batch_size, seq_len=seq_len,
                                                datasets=datasets, file=file)
                logging.info("Predicting ...")
                predictions = trainer.predict(model=hybrid_model, dataloaders=predict_loader)
                predictions = torch.cat(predictions, dim=0)  # unify list of predictions to tensor
                filename = f"{prefix}{file.split('/')[-1].split('.')[0]}_predictions.pt"
                torch.save(predictions, os.path.join(output_directory, filename))

        msg = "Validation ended." if mode == "validate" else "Predicting ended."
        logging.info(msg)

    logging.info("Predmoter finished.\n")


# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Predmoter", description="Predict promoter regions in the DNA.",
                                     add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = LitHybridNet.add_model_specific_args(parser)  # add model specific args
    model_args = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")  # not included in args
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")  # not included in args
    parser.add_argument("-i", "--input-directory", type=str, default=".",
                        help="containing one train and val directory for training "
                             "and/or a test directory for predicting (directories must contain h5 files)")
    parser.add_argument("-o", "--output-directory", type=str, default=".",
                        help="receiving log-files and pt-files (predictions); already contains or creates empty "
                             "subdirectory named checkpoints to save model checkpoints to")
    parser.add_argument("-m", "--mode", type=str, default=None, required=True,
                        help="valid modes: train, validate or predict")
    parser.add_argument("--resume-training", action="store_true",
                        help="only add argument if you want to resume training")
    parser.add_argument("--model", type=str, default="last.ckpt",
                        help="model checkpoint file used for predictions or resuming training (if not in "
                             "checkpoint directory, provide full path")
    parser.add_argument("--datasets", type=str, nargs="+", dest="datasets", default=["atacseq", "h3k4me3"],
                        help="the dataset(s) the network will train on; if you resume the training, validate or "
                             "predict the same dataset as before")
    parser.add_argument("--save-top-k", type=int, default=-1, help="saves the top k (e.g. 3) models;"
                                                                   " -1 means every model gets saved")
    parser.add_argument("--seed", type=int, default=None, help="if not provided: will be chosen randomly")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="only specify, if other path than <output_directory>/checkpoints or "
                             "<output_directory>/<prefix>_checkpoints is preferred")
    parser.add_argument("--ckpt_quantity", type=str, default="avg_val_accuracy",
                        help="quantity to monitor for checkpoints; loss: poisson negative log "
                             "likelihood, accuracy: Pearson's r (valid: avg_train_loss, avg_train_accuracy, "
                             "avg_val_loss, avg_val_accuracy)")
    parser.add_argument("--stop_quantity", type=str, default="avg_train_loss",
                        help="quantity to monitor for early stopping; loss: poisson negative log "
                             "likelihood, accuracy: Pearson's r (valid: avg_train_loss, avg_train_accuracy, "
                             "avg_val_loss, avg_val_accuracy)")
    parser.add_argument("--patience", type=int, default=3,
                        help="allowed epochs without the quantity improving before stopping training")
    parser.add_argument("-b", "--batch-size", type=int, default=195, help="batch size for training and validation data")
    parser.add_argument("--test-batch-size", type=int, default=150, help="batch size for test data")
    parser.add_argument("--predict-batch-size", type=int, default=100, help="batch size for prediction data")
    parser.add_argument("--prefix", type=str, default=None,
                        help="prefix for the metric and main program log files as well as prediction "
                             "files and the checkpoint directory")
    group = parser.add_argument_group("Trainer_arguments")
    group.add_argument("--device", type=str, default="gpu", help="device to train on")
    group.add_argument("--num-devices", type=int, default=1, help="number of devices to train on (default recommended)")
    group.add_argument("-e", "--epochs", type=int, default=35, help="number of training runs")
    group.add_argument("--limit-predict-batches", action="store", dest="limit_predict_batches",
                       help="limiting predict: float = fraction, int = num_batches "
                            "(if not specified, will default to 1.0)")
    args = parser.parse_args()

    # model args and other args into separate dictionaries
    dict_model_args = vars(model_args[0])
    dict_args = vars(args)

    for key in dict_model_args:
        dict_args.pop(key)

    # run main program
    main(model_arguments=dict_model_args, **dict_args)
