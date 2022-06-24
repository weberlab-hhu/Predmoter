import os
import sys
import random
import warnings
import logging
import h5py
import argparse
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
from dataset import PromoterSequence, PromoterDataset
from HybridModel import LitHybridNet

warnings.filterwarnings("ignore")  # disables warning about num_workers=0
logging.disable(logging.INFO)  # disables info about how many tpus, gpus , etc. were found


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 1000)
        seed_everything(seed=seed, workers=True)  # seed for reproducibility
        print(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        seed_everything(seed=seed, workers=True)


def set_callbacks(checkpoint_path):
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="avg_train_accuracy", mode="max",
                                          dirpath=checkpoint_path,
                                          filename="hybrid_model_{epoch}_{avg_train_accuracy:.2f}",
                                          save_on_train_epoch_end=True)
    callbacks = [ModelSummary(max_depth=-1), checkpoint_callback]
    return callbacks


def print_results(resume_training=False):

    if resume_training:
        ...

# add more arguments to main and argparse later and order them


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, seed, checkpoint_path,
         batch_size, test_batch_size, prefix, device, num_devices, epochs, limit_predict_batches):
    set_seed(seed)
    if resume_training or mode == "predict":
        hybrid_model = LitHybridNet.load_from_checkpoint(model)
    else:
        hybrid_model = LitHybridNet(**model_arguments, seq_len=PromoterDataset.sequence_length,
                                    input_size=PromoterDataset.nucleotides)

    callbacks = set_callbacks(checkpoint_path)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False, deterministic=True)
    assert mode in ["train", "predict"], "Valid modes are train or predict."
    if mode == "train":
        train = PromoterSequence("train", input_dir=input_directory, shuffle=True, batch_size=batch_size)
        val = PromoterSequence("val", input_dir=input_directory, shuffle=False, batch_size=batch_size)

        trainer.fit(model=hybrid_model, train_dataloaders=train.dataloader, val_dataloaders=val.dataloader)

    elif mode == "predict":
        test = PromoterSequence("test", input_dir=input_directory, shuffle=False, batch_size=test_batch_size)
        predictions = trainer.predict(model=hybrid_model, dataloaders=test.dataloader)
        # are predictions == dict or list?
        # torch.save(predictions, name)


# def main():
if __name__ == __main__:
    # argparse stuff
    parser = argparse.ArgumentParser(prog="predmoter", description="Predict promoter regions in the DNA.")
    parser = LitModel.add_model_specific_args(parser)  # add model specific args
    model_args = parser.parse_known_args()
    parser.add_argument('-i', '--input-directory', type=str, default='.', required=True,
                        help="Path to a directory containing one train, val and test directory with h5-files.")
    parser.add_argument('-o', '--output-directory', type=str, default='.')
    parser.add_argument("-m", "--mode", type=str, default=None, required=True, help="Valid modes: train or predict.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the model used for predictions or resuming training.")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=".")
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-tb', '--test-batch-size', type=int, default=10)
    parser.add_argument('--prefix', type=str, default="")  # find a default
    # parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    # exclude from dict!!, unless it's used for print
    group = parser.add_argument_group("Trainer_arguments")
    group.add_argument('--device', type=str, default="gpu", help="Supported devices are CPU and GPU.")
    group.add_argument('--num-devices', type=int, default=1)
    group.add_argument('-e', '--epochs', type=int, default=35)
    group.add_argument("--limit_predict_batches", type=int, default=None)

    args = parser.parse_args()

    # dictionary stuff
    dict_model_args = vars(model_args[0])
    dict_args = vars(args)

    for key in dict_model_args:
        dict_args.pop(key)

    # run main program
    main(model_arguments=dict_model_args, **dict_args)
