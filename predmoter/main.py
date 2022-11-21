import os
import logging
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from dataset import get_dataloader
from utils import check_paths, init_logging, set_seed, get_meta, set_callbacks
from HybridModel import LitHybridNet


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, datasets, norm_metric,
         seed, checkpoint_path, quantity, patience, batch_size, test_batch_size, prefix, device,
         num_devices, epochs, limit_predict_batches):

    # Argument checks and cleanup
    # --------------------------------
    prefix = f"{prefix}_" if prefix is not None else ""
    if checkpoint_path is None:  # and mode != "validate" ??
        checkpoint_path = os.path.join(output_directory, f"{prefix}checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
    check_paths([input_directory, output_directory, checkpoint_path])
    assert mode in ["train",  "validate", "predict"], f"valid modes are train, validate or predict, not {mode}"
    assert norm_metric in ["mean", "median", None], f"valid metrics for normalization are mean, median or none"
    limit_predict_batches = 1.0 if limit_predict_batches is None else limit_predict_batches

    # Preset initialization
    # --------------------------------
    init_logging(output_directory, prefix)
    logging.info(f"Predmoter is starting in {mode} mode.")
    logging.info(f"Chosen dataset(s): {' '.join(datasets)}")
    if mode == "train":
        set_seed(seed)
    seq_len, bases = get_meta(input_directory, mode)

    #  Model initialization
    # --------------------------------
    if resume_training or mode in ["predict", "validate"]:
        if not os.path.exists(model):
            model = os.path.join(checkpoint_path, model)
        # if only model name is given/the path doesn't exist, assume model is in the checkpoint directory
        check_paths([model])
        logging.info(f"Chosen model checkpoint: {model}")
        hybrid_model = LitHybridNet.load_from_checkpoint(model, seq_len=seq_len)
        assert len(datasets) == hybrid_model.output_size,\
            f"the length {len(datasets)} of the chosen datasets ({' '.join(datasets)}) doesn't match the " \
            f"output size {hybrid_model.output_size} of the chosen model"
        assert hybrid_model.input_size == bases,\
            f"Your chosen model has the input size {hybrid_model.input_size} and your dataset {bases}." \
            f"Please use the same input size."  # how often would this actually happen?
    else:
        hybrid_model = LitHybridNet(**model_arguments, seq_len=seq_len, input_size=bases, output_size=len(datasets))
    logging.info(f"\n\nModel summary (model type: {hybrid_model.model_type}):"
                 f"\n{ModelSummary(model=hybrid_model, max_depth=-1)}\n")

    # Training preset initialization
    # --------------------------------
    callbacks = set_callbacks(output_directory, mode, prefix, checkpoint_path, quantity, patience)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False,
                         deterministic=True, limit_predict_batches=limit_predict_batches)

    # Actual program start
    # --------------------------------
    if mode == "train":
        train_loader = get_dataloader(input_dir=input_directory, type_="train", batch_size=batch_size,
                                      seq_len=seq_len, datasets=datasets, norm_metric=norm_metric)
        val_loader = get_dataloader(input_dir=input_directory, type_="val", batch_size=batch_size,
                                    seq_len=seq_len, datasets=datasets, norm_metric=norm_metric)
        logging.info(f"Training started. Resuming training: {resume_training}.")  # to print callback at some point
        if resume_training:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader,
                        val_dataloaders=val_loader, ckpt_path=model)
            # restores all previous states like callbacks, optimizer state, etc.
        else:
            trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logging.info("Training ended.")

    elif mode == "validate":
        val_loader = get_dataloader(input_dir=input_directory, type_="val", batch_size=batch_size,
                                    seq_len=seq_len, datasets=datasets, norm_metric=norm_metric)
        logging.info("Validation started.")
        trainer.validate(model=hybrid_model, dataloaders=val_loader)
        logging.info("Validation ended.")

    elif mode == "predict":
        test_loader = get_dataloader(input_dir=input_directory, type_="test", batch_size=test_batch_size,
                                     seq_len=seq_len, datasets=datasets, norm_metric=norm_metric)
        logging.info("Predicting started.")
        predictions = trainer.predict(model=hybrid_model, dataloaders=test_loader)
        logging.info("Predicting ended.")
        predictions = torch.cat(predictions, dim=0)  # unify list of preds to tensor
        torch.save(predictions, os.path.join(output_directory, f"{prefix}predictions.pt"))
    logging.info("Predmoter finished.\n\n")


# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Predmoter", description="Predict promoter regions in the DNA.",
                                     add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = LitHybridNet.add_model_specific_args(parser)  # add model specific args
    model_args = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")  # not included in args
    parser.add_argument("--version", action="version", version="%(prog)s 0.9")  # not included in args
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
                        help="the dataset(s) the network will train on")
    parser.add_argument("--norm-metric", type=str, default=None, help="metric to normalize data upon read in; "
                                                                      "median or mean can be used")
    parser.add_argument("--seed", type=int, default=None, help="if not provided: will be chosen randomly")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="only specify, if other path than output_directory/checkpoints is preferred")
    parser.add_argument("--quantity", type=str, default="avg_train_loss",
                        help="quantity to monitor for checkpoints and early stopping; loss: poisson negative log "
                             "likelihood, accuracy: Pearson's r (valid: avg_train_loss, avg_train_accuracy, "
                             "avg_val_loss, avg_val_accuracy)")
    parser.add_argument("--patience", type=int, default=3,
                        help="allowed epochs without the quantity improving before stopping training")
    parser.add_argument("-b", "--batch-size", type=int, default=195, help="batch size for training and validation data")
    parser.add_argument("--test-batch-size", type=int, default=100, help="batch size for test data")
    parser.add_argument("--prefix", type=str, default=None, help="prefix for metric, log and prediction files")
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
