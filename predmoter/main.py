# import sys
import time
import random
import logging
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from dataset import PredmoterSequence, get_dataloader
from HybridModel import LitHybridNet

logging.disable(logging.INFO)  # disables info about how many tpus, gpus , etc. were found


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 2000)
        seed_everything(seed=seed, workers=True)  # seed for reproducibility
        print(f"A seed wasn't provided by the user. The random seed is: {seed}.")
    else:
        seed_everything(seed=seed, workers=True)


def get_meta(input_dir, mode):
    input_dir = "/".join([input_dir.rstrip("/"), "test"]) if mode == "predict" \
        else "/".join([input_dir.rstrip("/"), "train"])
    for h5_file in os.listdir(input_dir):
        h5_file = "/".join([input_dir, h5_file])
        h5df = h5py.File(h5_file, mode="r")
        X = np.array(h5df["data/X"][:1], dtype=np.int8)
        return X.shape[1:]


def set_callbacks(checkpoint_path, patience):
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="avg_train_accuracy", mode="max",
                                          dirpath=checkpoint_path,
                                          filename="hybrid_model_{epoch}_{avg_train_accuracy:.2f}",
                                          save_on_train_epoch_end=True)
    early_stop = EarlyStopping(monitor="avg_train_accuracy", min_delta=0.0, patience=patience, verbose=False,
                               mode="max", strict=True, check_finite=True, check_on_train_epoch_end=True)
    callbacks = [ModelSummary(max_depth=-1), checkpoint_callback, early_stop]
    return callbacks


def save_metrics(prefix, output_dir, resume_training):  # print_mode ?
    prefix = "" if prefix is None else f"{prefix}_"
    # log_file ??
    file_list = ["/".join([output_dir.rstrip("/"), f"{prefix}{metric}.txt"]) for metric in ["loss", "accuracy"]]
    if resume_training:
        with open(loss_file, "r") as f:
            losses = f.readlines()
        last_epoch = int(losses[-1][0]) + 1
    else:
        last_epoch = 1
    for file in file_list:
        with open(file, "a", encoding="utf-8") as f:
            for epoch, (train, val) in enumerate(zip(hybrid_model.train_losses, hybrid_model.val_losses[1:])):  # !!!
                # un-dynamic -> accuracy
                if epoch == 0 and not resume_training:
                    f.write("epoch training_losses validation_losses\n")
                f.write(f"{epoch + last_epoch} {train} {val}\n")

# add more arguments to main and argparse later and order them


def main(model_arguments, input_directory, output_directory, mode, resume_training, model, seed,
         checkpoint_path, patience, batch_size, test_batch_size, num_workers, prefix, device, num_devices,
         epochs, limit_predict_batches):
    set_seed(seed)
    meta = get_meta(input_directory, mode)
    assert len(meta) == 2, f"expected all arrays to have the shape (seq_len, bases) found {meta}"
    if resume_training or mode == "predict":
        hybrid_model = LitHybridNet.load_from_checkpoint(model, seq_len=meta[0])
    else:
        hybrid_model = LitHybridNet(**model_arguments, seq_len=meta[0], input_size=meta[1])

    callbacks = set_callbacks(checkpoint_path, patience)
    trainer = pl.Trainer(callbacks=callbacks, devices=num_devices, accelerator=device,
                         max_epochs=epochs, logger=False, enable_progress_bar=False,
                         deterministic=True, limit_predict_batches=limit_predict_batches)

    assert mode in ["train", "predict"], "Valid modes are train or predict."
    if mode == "train":
        print(f"Loading training and validation data into memory started at {time.asctime()}.")
        train_loader = get_dataloader(input_dir=input_directory, type_="train", shuffle=True,
                                      batch_size=batch_size, num_workers=num_workers)
        val_loader = get_dataloader(input_dir=input_directory, type_="val", shuffle=False,
                                    batch_size=batch_size, num_workers=num_workers)
        mem_size = (sys.getsizeof(train_loader.dataset.X) + sys.getsizeof(train_loader.dataset.Y) +
                    sys.getsizeof(val_loader.dataset.X) + sys.getsizeof(val_loader.dataset.Y))/1024**3
        print(f"Loading training and validation data into memory ended at {time.asctime()}."
              f"The compressed data is {mem_size:.4f} Gb in size.")

        print(f"Training started at {time.asctime()}")
        trainer.fit(model=hybrid_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"Training ended at {time.asctime()}")

        save_metrics(prefix, output_directory, resume_training)

    elif mode == "predict":
        print(f"Loading test data into memory started at {time.asctime()}.")
        test = get_dataloader(input_dir=input_directory, type_="test", shuffle=False,
                              batch_size=test_batch_size, num_workers=num_workers)
        mem_size = sys.getsizeof(test_loader.dataset.X)/1024 ** 3
        print(f"Loading test data into memory ended at {time.asctime()}."
              f"The compressed data is {mem_size:.4f} Gb in size.")

        print(f"Predicting started at {time.asctime()}")
        predictions = trainer.predict(model=hybrid_model, dataloaders=test.dataloader)
        print(f"Predicting ended at {time.asctime()}")
        # need trainer or use loop?? (limit_predict_batches not possible then)
        predictions = torch.cat(predictions, dim=0)  # unify list of preds to tensor
        torch.save(predictions, "/".join([output_directory.rstrip("/"), "predictions.pt"]))


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
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="path to the model used for predictions or resuming training")
    parser.add_argument("--seed", type=int, default=None, help="if not provided: will be chosen randomly")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="otherwise empty directory to save model checkpoints to")
    parser.add_argument("--patience", type=int, default=3,
                        help="allowed epochs without training accuracy improvement before stopping training")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="(default: %(default)d)")
    parser.add_argument("-tb", "--test-batch-size", type=int, default=10, help="(default: %(default)d)")
    parser.add_argument("--num_workers", type=int, default=8, help="for multiprocessing during data loading")
    parser.add_argument("--prefix", type=str, default=None, help="prefix for loss, accuracy and log files")
    group = parser.add_argument_group("Trainer_arguments")
    group.add_argument("--device", type=str, default="gpu", help="supported devices are CPU and GPU")
    group.add_argument("--num-devices", type=int, default=1, help="(default: %(default)d)")
    group.add_argument("-e", "--epochs", type=int, default=35, help="(default: %(default)d)")
    group.add_argument("--limit_predict_batches", action="store", dest="limit_predict_batches",
                       help="limiting predict: float = fraction, int = num_batches (default: 1.0)")
    args = parser.parse_args()

    # limit predict
    if args.limit_predict_batches is None:
        args.limit_predict_batches = 1.0

    # dictionary stuff
    dict_model_args = vars(model_args[0])
    dict_args = vars(args)

    for key in dict_model_args:
        dict_args.pop(key)

    # run main program
    main(model_arguments=dict_model_args, **dict_args)