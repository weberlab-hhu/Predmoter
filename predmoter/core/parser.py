import argparse
import os
import warnings

import torch.cuda

from predmoter.core.constants import PREDMOTER_VERSION


class BaseParser:
    def __init__(self, prog="", description=""):
        super(BaseParser, self).__init__()
        self.parser = argparse.ArgumentParser(prog=prog, description=description, add_help=False,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("-h", "--help", action="help", help="show this help message and exit")

    def check_args(self, args):
        pass

    def get_args(self):
        args = self.parser.parse_args()
        self.check_args(args)
        return args

    @staticmethod
    def str2bool(arg):
        """Use optional boolean value and default true."""
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif arg.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("boolean value expected")


class PredmoterParser(BaseParser):
    def __init__(self):
        super().__init__(prog="Predmoter", description="predict NGS data associated with regulatory DNA regions")
        self.parser.add_argument("--version", action="version", version=f"%(prog)s {PREDMOTER_VERSION}")
        self.parser.add_argument("-m", "--mode", type=str.lower, default=None, required=True,
                                       help="valid modes: train, test or predict")

        self.io_group = self.parser.add_argument_group("Data input/output parameters")
        self.io_group.add_argument("-i", "--input-dir", type=str, default=".",
                                   help="containing one train and one val directory for training and/or "
                                        "a test directory for testing (directories must contain h5 files)")
        self.io_group.add_argument("-o", "--output-dir", type=str, default=".",
                                   help="output: log file(s), checkpoint directory with model checkpoints "
                                        "(if training), predictions (if predicting)")
        self.io_group.add_argument("-f", "--filepath", type=str, default=None,
                                   help="input file to predict on, either h5 or fasta file")
        self.io_group.add_argument("--subsequence-length", type=int, default=21384,
                                   help="size of the chunks each genomic sequence gets cut into; only needed "
                                        "for predicting on fasta files")
        self.io_group.add_argument("--species", type=str, default=None,
                                   help="species name required if the prediction is done on a fasta file")
        self.io_group.add_argument("--multiprocess", action="store_true",
                                   help="add to parallelize the numerification of large sequences, uses half "
                                        "the memory but can be much slower when many CPU cores can be utilized; "
                                        "only needed for predicting on fasta files")
        self.io_group.add_argument("-of", "--output-format", type=str.lower, default=None,
                                   help="output format for predictions, if unspecified will output no additional "
                                        "files besides the h5 file (valid: bigwig (bw), bedgraph (bg)); "
                                        "the file naming convention is: <basename_of_input_file>_dataset_"
                                        "avg_strand.bw/bg.gz, if just + or - strand is preferred, convert the "
                                        "h5 file later on using convert2coverage.py")
        self.io_group.add_argument("--prefix", type=str, default=None,
                                   help="prefix for log files and the checkpoint directory")

        self.config_group = self.parser.add_argument_group("Configuration parameters")
        self.config_group.add_argument("--resume-training", action="store_true",
                                       help="add argument to resume training")
        self.config_group.add_argument("--model", type=str, default=None,
                                       help="model checkpoint file for predicting, testing or resuming"
                                            "training (if not <outdir>/<prefix>_checkpoints/last.ckpt,"
                                            "provide full path")
        # model default: <outdir>/<prefix>_checkpoints/last.ckpt -> just if resume train or resources or None
        self.config_group.add_argument("--datasets", type=str, nargs="+",
                                       dest="datasets", default=["atacseq", "h3k4me3"],
                                       help="the dataset prefix(es) to use; are overwritten by the model "
                                            "checkpoint's dataset prefix(es) if one is chosen (in case of "
                                            "resuming training, testing, predicting)")
        self.config_group.add_argument("--ram-efficient", type=BaseParser.str2bool, default=True,
                                       help="if true will use RAM efficient data class (see docs/performance.md), "
                                            "Warning: Don't move the input data while Predmoter is running.")
        self.config_group.add_argument("-bl", "--blacklist", action="store_true",
                                       help="mask blacklisted regions in data/blacklist (if it exists)")

        self.model_group = self.parser.add_argument_group("Model parameters")
        self.model_group.add_argument("--model-type", type=str.lower, default="bi-hybrid",
                                      help="the type of model to train, valid types are: cnn, "
                                           "hybrid (CNN + LSTM), bi-hybrid (CNN + bidirectional LSTM)")
        self.model_group.add_argument("--cnn-layers", type=int, default=1,
                                      help="number of convolutional layers (e.g. 3 layers: 3 layer "
                                           "convolution and 3 layer deconvolution")
        self.model_group.add_argument("--filter-size", type=int, default=64,
                                      help="filter size for convolution, is scaled up per layer after the first by up")
        self.model_group.add_argument("--kernel-size", type=int, default=9, help="kernel size for convolution")
        self.model_group.add_argument("--step", type=int, default=2, help="stride for convolution")
        self.model_group.add_argument("--up", type=int, default=2,
                                      help="multiplier used for up-scaling the convolutional filter per "
                                           "layer after the first")
        self.model_group.add_argument("--dilation", type=int, default=1,
                                      help="dilation should be kept at the default, as it isn't "
                                           "useful for sequence data")
        self.model_group.add_argument("--lstm-layers", type=int, default=1, help="LSTM layers")
        self.model_group.add_argument("--hidden-size", type=int, default=128, help="LSTM units per layer")
        self.model_group.add_argument("--bnorm", type=BaseParser.str2bool, default=True,
                                      help="add a batch normalization layer after each convolutional "
                                           "and transposed convolutional layer")
        self.model_group.add_argument("--dropout", type=float, default=0.,
                                      help="adds a dropout layer with the specified dropout value (between "
                                           "0. and 1.) after each LSTM layer except the last; if it is 0. "
                                           "no dropout layers are added; if there is just one LSTM layer "
                                           "specifying dropout will do nothing")
        self.model_group.add_argument("-lr", "--learning-rate", type=float, default=0.001,
                                      help="learning rate for training (default recommended)")

        self.trainer_group = self.parser.add_argument_group("Trainer/callback parameters")
        self.trainer_group.add_argument("--seed", type=int, default=None,
                                        help="for reproducibility, if not provided will be chosen randomly")
        self.trainer_group.add_argument("--ckpt_quantity", type=str, default="avg_val_accuracy",
                                        help="quantity to monitor for checkpoints; loss: poisson negative log "
                                             "likelihood, accuracy: Pearson's r (valid: avg_train_loss, "
                                             "avg_train_accuracy, avg_val_loss, avg_val_accuracy)")
        self.trainer_group.add_argument("--save-top-k", type=int, default=-1,
                                        help="saves the top k (e.g. 3) models; -1 means every model gets saved;"
                                             "the last model checkpoint saved is named last.ckpt")
        self.trainer_group.add_argument("--stop_quantity", type=str, default="avg_train_loss",
                                        help="quantity to monitor for early stopping; loss: poisson negative log "
                                             "likelihood, accuracy: Pearson's r (valid: avg_train_loss, "
                                             "avg_train_accuracy, avg_val_loss, avg_val_accuracy)")
        self.trainer_group.add_argument("--patience", type=int, default=5,
                                        help="allowed epochs without the stop quantity improving before "
                                             "stopping training")
        self.trainer_group.add_argument("-b", "--batch-size", type=int, default=120,
                                        help="batch size for training, validation, test or prediction sets")
        self.trainer_group.add_argument("--device", type=str.lower, default="gpu", help="device to train on")
        self.trainer_group.add_argument("--num-devices", type=int, default=1,
                                        help="number of devices to train on (see docs about performance), "
                                             "devices have to be on the same machine (leave at default "
                                             "for test/predict)")
        self.trainer_group.add_argument("--num-workers", type=int, default=0,
                                        help="how many subprocesses to use for data loading (number of CPU cores)")
        self.trainer_group.add_argument("-e", "--epochs", type=int, default=5,
                                        help="number of training runs; Attention: max_epochs, so when training "
                                             "for 2 epochs and then resuming training for 4 additional epochs, "
                                             "you need -e 6")

    def check_args(self, args):
        # Mode
        # ------------
        assert args.mode in ["train",  "test", "predict"], f"valid modes are train, test or predict, not {args.mode}"

        # IO checks
        # -------------
        # default (current path) should always exist
        for path in [args.input_dir, args.output_dir]:
            if not os.path.exists(path):
                raise OSError(f"path {path} doesn't exist")

        for path in [args.input_dir, args.output_dir]:
            if not os.path.isdir(path):
                raise NotADirectoryError(f"path {path} is not a directory")

        if args.mode == "predict":
            if args.filepath is None:
                raise OSError("if mode is predict the argument -f/--filepath is required to find the input file")
            else:
                if not os.path.exists(args.filepath):
                    raise FileNotFoundError(f"file {args.filepath} doesn't exist")
                if not os.path.isfile(args.filepath):  # remove?, check predict file differently
                    raise OSError(f"the chosen file {args.filepath} is not a file")
            if args.filepath.lower().endswith((".fasta", ".fna", ".ffn", ".faa", ".frn", ".fa")):
                assert args.species is not None, "to use a fasta file to predict on, --species must be specified"

        if args.output_format is not None:
            assert args.output_format in ["bigwig", "bedgraph", "bw", "bg"], \
                f"valid additional output formats are bigwig (bw) and bedgraph (bg) not {args.output_format}"

        args.prefix = f"{args.prefix}_" if args.prefix is not None else ""

        # Config checks
        # --------------
        if args.resume_training and args.mode != "train":
            args.resume_training = False
            warnings.warn("can only resume training if mode is train, resume-training will be set to False")

        if args.mode == "train" and not args.resume_training and args.model is not None:
            args.model = None
            warnings.warn("starting training for the first time, if you have a pretrained model "
                          "(even trained on a different dataset) set resume-training, model will be set to None")

        if args.resume_training and args.model is None:
            args.model = os.path.join(args.output_dir, f"{args.prefix}checkpoints/last.ckpt")
            if not os.path.exists(args.model):
                raise OSError(f"did not find default model path ({args.model}) for resuming training, "
                              f"maybe the prefix is wrong?")
            if not os.path.isfile(args.model):
                raise OSError(f"default model ({args.model}) for resuming training is not a file, "
                              f"please don't give directories this name")

        if args.mode != "train":
            if args.model is None:
                raise OSError(f"please specify a model checkpoint if you want to test or predict")
            if not os.path.exists(args.model):
                raise OSError(f"could not find model path: ({args.model})")
            if not os.path.isfile(args.model):
                raise OSError(f"model {args.model} is not a file")

        # Model checks
        # -------------
        assert args.model_type in ["cnn", "hybrid", "bi-hybrid"], \
            f"valid model types are cnn, hybrid and bi-hybrid not {args.model_type}"

        # Trainer checks
        # ---------------
        for quantity in [args.ckpt_quantity, args.stop_quantity]:
            assert quantity in ["avg_train_loss", "avg_train_accuracy", "avg_val_loss", "avg_val_accuracy"], \
                f"can not monitor invalid quantity: {quantity}"

        assert args.device in ["cpu", "gpu"], \
            "valid devices are cpu or gpu, Predmoter is not configured to work on other devices"

        if args.device == "gpu":
            assert torch.cuda.is_available(), "there seems to be no GPU available on your system for PyTorch to " \
                                              "use, please set '--device cpu' or check your system"
            assert torch.cuda.device_count() >= args.num_devices, \
                f"the chosen number of GPUs ({args.num_devices}) is higher than the available amount of GPUs " \
                f"({torch.cuda.device_count()}), please choose a lower number with '--num-devices'"
        else:  # device = CPU
            assert os.cpu_count() >= args.num_devices, \
                f"the chosen number of CPU cores ({args.num_devices}) is larger than the available number of " \
                f"CPU cores ({os.cpu_count()}), please choose a lower number with '--num-devices'"

        assert os.cpu_count() >= args.num_workers, \
            f"the chosen number of workers ({args.num_workers}) is larger than the available number of CPU cores " \
            f"({os.cpu_count()}), please choose a lower number with '--num-workers'"

        if args.mode != "train":
            assert args.num_devices == 1, "testing/predicting should be done on 1 device only to ensure " \
                                          "that each sample/batch gets evaluated/predicted exactly once"


class ConverterParser(BaseParser):
    def __init__(self):
        super().__init__(prog="h5_to_coverage",
                         description="write Predmoter predictions from h5 format to a bigWig or bedGraph file")
        self.parser.add_argument("-i", "--input-file", type=str, default=None, required=True,
                                 help="input h5 predictions file (Predmoter output)")
        self.parser.add_argument("-o", "--output-dir", type=str, default=".",
                                 help="output directory for all converted files")
        self.parser.add_argument("-of", "--output-format", type=str.lower, default="bigwig",
                                 help="output format for predictions (valid: bigwig (bw), bedgraph (bg))")
        self.parser.add_argument("--basename", type=str, default=None, required=True,
                                 help="basename of the output files, naming convention: "
                                      "basename_dataset_strand.bw/bg.gz")
        self.parser.add_argument("--strand", type=str, default=None,
                                 help="None means the average of both strands is used, else + or - can be selected")
        self.parser.add_argument("--prefix", type=str, default=None, help="prefix for log file")
        self.parser.add_argument("--experimental", action="store_true",
                                 help="add to convert the mean experimental coverage of a experimental h5 file "
                                      "instead of predictions")
        self.parser.add_argument("--datasets", type=str, nargs="+", dest="dsets", default=None,
                                 help="the dataset prefix(es) to convert when --experimental is set")
        self.parser.add_argument("--blacklist-file", type=str, dest="bl_chroms", default=None,
                                 help="text file of chromosomes/sequences that will not be included "
                                      "in the conversion, e.g., chloroplast or mitochondrial sequences; "
                                      "one chromosome ID per line in the text file, i.e. add_blacklist test files;"
                                      "the output files get 'bl' added at the end to identify blacklisting")

    def check_args(self, args):
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"file {args.input_file} doesn't exist")

        if not os.path.isfile(args.input_file):  # remove??, check file somewhere
            raise OSError(f"the chosen file {args.input_file} is not a file")

        if not os.path.exists(args.output_dir):
            raise OSError(f"path {args.output_dir} doesn't exist")

        if not os.path.isdir(args.output_dir):
            raise NotADirectoryError(f"path {args.output_dir} is not a directory")

        assert args.output_format in ["bigwig", "bedgraph", "bw", "bg"], \
            f"valid output formats are bigwig (bw) and bedgraph (bg) not {args.output_format}"

        assert args.strand in ["+", "-", None], f"valid strand is either +, - or None not {args.strand}"

        args.prefix = f"{args.prefix}_" if args.prefix is not None else ""

        if args.experimental:
            assert args.dsets is not None, "datasets to convert have to be chosen for converting from" \
                                           " h5 files containing experimental coverage"
