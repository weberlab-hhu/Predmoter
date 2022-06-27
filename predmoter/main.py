import os
import sys
import random
import logging
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from dataset import PromoterSequence, PromoterDataset
from HybridModel import LitHybridNet

logging.disable(logging.INFO)  # disables info about how many tpus, gpus , etc. were found


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 2000)
        seed_everything(seed=seed, workers=True)  # seed for reproducibility
        print(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        seed_everything(seed=seed, workers=True)


def set_callbacks(checkpoint_path, patience):
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="avg_train_accuracy", mode="max",
                                          dirpath=checkpoint_path,
                                          filename="hybrid_model_{epoch}_{avg_train_accuracy:.2f}",
                                          save_on_train_epoch_end=True)
    early_stop = EarlyStopping(monitor="avg_train_accuracy", min_delta=0.0, patience=patience, verbose=False,
                               mode="max", strict=True, check_finite=True, check_on_train_epoch_end=True)
    callbacks = [ModelSummary(max_depth=-1), checkpoint_callback, early_stop]
    return callbacks


def print_results(resume_training=False):

    if resume_training:
        pass

# add more arguments to main and argparse later and order them


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, seed,
         checkpoint_path, patience, batch_size, test_batch_size, num_workers, prefix, device, num_devices,
         epochs, limit_predict_batches):
    set_seed(seed)
    if resume_training or mode == "predict":
        hybrid_model = LitHybridNet.load_from_checkpoint(model)
    else:
        hybrid_model = LitHybridNet(**model_arguments, seq_len=PromoterDataset.sequence_length,
                                    input_size=PromoterDataset.nucleotides)

    callbacks = set_callbacks(checkpoint_path, patience)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False, deterministic=True)
    assert mode in ["train", "predict"], "Valid modes are train or predict."
    if mode == "train":
        train = PromoterSequence("train", input_dir=input_directory, shuffle=True,
                                 batch_size=batch_size, num_workers=num_workers)
        val = PromoterSequence("val", input_dir=input_directory, shuffle=False,
                               batch_size=batch_size, num_workers=num_workers)

        trainer.fit(model=hybrid_model, train_dataloaders=train.dataloader, val_dataloaders=val.dataloader)

    elif mode == "predict":
        test = PromoterSequence("test", input_dir=input_directory, shuffle=False,
                                batch_size=test_batch_size, num_workers=num_workers)
        predictions = trainer.predict(model=hybrid_model, dataloaders=test.dataloader)
        # are predictions == dict or list?
        # torch.save(predictions, name)


# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Predmoter", description="Predict promoter regions in the DNA.",
                                     add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = LitHybridNet.add_model_specific_args(parser)  # add model specific args
    model_args = parser.parse_known_args()
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")  # not included in args
    parser.add_argument("--version", action="version", version="%(prog)s 0.8")  # not included in args
    parser.add_argument("-i", "--input-directory", type=str, default=".", required=True,
                        help="path to a directory containing one train, val and test directory with h5-files")
    parser.add_argument("-o", "--output-directory", type=str, default=".",
                        help="directory to save log-files and predictions to")
    parser.add_argument("-m", "--mode", type=str, default=None, required=True, help="valid modes: train or predict")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="path to the model used for predictions or resuming training")
    parser.add_argument("--seed", type=int, default=None, help="if not provided: will be chosen randomly")
    parser.add_argument("--checkpoint-path", type=str, default=".",
                        help="otherwise empty directory to save model checkpoints to")
    parser.add_argument("--patience", type=int, default=3,
                        help="allowed epochs without training accuracy improvement before stopping training")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="(default: %(default)d)")
    parser.add_argument("-tb", "--test-batch-size", type=int, default=10, help="(default: %(default)d)")
    parser.add_argument("--num_workers", type=int, default=8, help="for multiprocessing during data loading")
    parser.add_argument("--prefix", type=str, default="model", help="prefix for loss, accuracy and log files")
    group = parser.add_argument_group("Trainer_arguments")
    group.add_argument("--device", type=str, default="gpu", help="supported devices are CPU and GPU")
    group.add_argument("--num-devices", type=int, default=1, help="(default: %(default)d)")
    group.add_argument("-e", "--epochs", type=int, default=35, help="(default: %(default)d)")
    group.add_argument("--limit_predict_batches", type=int, default=None,
                       help="only predicting limited number of batches")
    args = parser.parse_args()

    # dictionary stuff
    dict_model_args = vars(model_args[0])
    dict_args = vars(args)

    for key in dict_model_args:
        dict_args.pop(key)

    # run main program
    main(model_arguments=dict_model_args, **dict_args)
