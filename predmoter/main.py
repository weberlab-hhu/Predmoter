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
from dataset import PromoterSequence
# argparse stuff somewhere

warnings.filterwarnings("ignore")  # disables warning about num_workers=0
logging.disable(logging.INFO)  # disables info about how many tpus, gpus , etc. were found


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 1000)
        seed_everything(seed=seed, workers=True)  # seed for reproducibility
        print(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        seed_everything(seed=args.seed, workers=True)


def set_checkpoints():
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="avg_train_accuracy", mode="max",
                                          dirpath=args.checkpoint_path,
                                          filename="hybrid_model_{epoch}_{avg_train_accuracy:.2f}",
                                          save_on_train_epoch_end=True)
    callbacks = [ModelSummary(max_depth=-1), checkpoint_callback]
    return callbacks


def print_results(resume_training=False):

    if resume_training:
        ...


def main():  # argparse stuff
    hybrid_model = LitHybridNet(seq_len=sequence_length, input_size=nucleotides, cnn_layers=args.cnn_layers,
                                filter_size=args.filter_size, kernel_size=args.kernel_size, step=args.step,
                                up=args.up, hidden_size=args.lstm_units, lstm_layers=args.lstm_layers,
                                learning_rate=args.learning_rate)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=args.epochs, logger=False, enable_progress_bar=False, deterministic=True)
    assert mode in ["train", "predict"], "Valid modes are train or predict."
    if mode == "train":
        train = PromoterSequence("train", input_dir=args.input_directory, shuffle=True, batch_size=args.batch_size)
        val = PromoterSequence("val", input_dir=args.input_directory, shuffle=False, batch_size=args.batch_size)

        if resume_training:
            trainer.fit(model=hybrid_model, train_dataloaders=train.dataloader,
                        val_dataloaders=val.dataloader, ckpt_path=args.model_path)
        else:
            trainer.fit(model=hybrid_model, train_dataloaders=train.dataloader, val_dataloaders=val.dataloader)

    elif mode == "predict":
        test = PromoterSequence("test", input_dir=args.input_directory,
                                shuffle=False, batch_size=args.test_batch_size)  # batch = 1???
        predictions = trainer.predict(model=hybrid_model, dataloaders=test.dataloader, ckpt_path=args.model_path)
        # are predictions == dict or list?
        # torch.save(predictions, name)


# def main():
if __name__ == __main__:
    # argparse stuff
    parser = argparse.ArgumentParser()
    # add model specific args
    parser = LitModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main()
