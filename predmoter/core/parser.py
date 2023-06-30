import argparse
import os
import warnings

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
        self.parser.add_argument("-m", "--mode", type=str, default=None, required=True,
                                       help="valid modes: train, test or predict")

        self.io_group = self.parser.add_argument_group("Data input/output parameters")
        self.io_group.add_argument("-i", "--input-dir", type=str, default=".",
                                   help="containing one train and val directory for training and/or "
                                        "a test directory for testing (directories must contain h5 files)")
        self.io_group.add_argument("-o", "--output-dir", type=str, default=".",
                                   help="output: log file(s), checkpoint directory with model checkpoints "
                                        "(if training), predictions (if predicting)")
        self.io_group.add_argument("-f", "--filepath", type=str, default=None,
                                   help="input file to predict on, either h5 or fasta file")
        self.io_group.add_argument("-of", "--output-format", type=str, default=None,
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
                                       help="model checkpoint file for prediction or resuming training (if not "
                                            "<outdir>/<prefix>_checkpoints/last.ckpt, provide full path")
        # model default: <outdir>/<prefix>_checkpoints/last.ckpt -> just if resume train or resources or None
        self.config_group.add_argument("--datasets", type=str, nargs="+",
                                       dest="datasets", default=["atacseq", "h3k4me3"],
                                       help="the dataset prefix(es) to use; are overwritten by the model "
                                            "checkpoint's dataset prefix(es) if one is chosen (in case of "
                                            "resuming training, testing, predicting)")
        self.config_group.add_argument("--ram-efficient", action="store_true",
                                       help="add argument to use RAM efficient data class, "
                                            "this will slow down training (see docs about performance)")

        self.model_group = self.parser.add_argument_group("Model parameters")
        self.model_group.add_argument("--model-type", type=str, default="hybrid",
                                      help="The type of model to train. Valid types are cnn, "
                                           "hybrid (CNN + LSTM) and bi-hybrid (CNN + bidirectional LSTM).")
        self.model_group.add_argument("--cnn-layers", type=int, default=1, help="(default: %(default)d)")
        self.model_group.add_argument("--filter-size", type=int, default=64, help="(default: %(default)d)")
        self.model_group.add_argument("--kernel-size", type=int, default=9, help="(default: %(default)d)")
        self.model_group.add_argument("--step", type=int, default=2, help="equals stride")
        self.model_group.add_argument("--up", type=int, default=2,
                                      help="multiplier used for up-scaling the convolutional filter per layer")
        self.model_group.add_argument("--dilation", type=int, default=1,
                                      help="dilation should be kept at the default, as it isn't that "
                                           "useful for sequence data")
        self.model_group.add_argument("--lstm-layers", type=int, default=1, help="(default: %(default)d)")
        self.model_group.add_argument("--hidden-size", type=int, default=128, help="LSTM units per layer")
        self.model_group.add_argument("--bnorm", type=BaseParser.str2bool, default=True,
                                      help="add a batch normalization layer after each convolutional layer")
        self.model_group.add_argument("--dropout", type=float, default=0.,
                                      help="adds a dropout layer with the specified dropout value after each "
                                           "LSTM layer except the last; if it is 0. no dropout layers are added; "
                                           "if there is just one LSTM layer specifying dropout will do nothing")
        self.model_group.add_argument("-lr", "--learning-rate", type=float, default=0.001,
                                      help="(default: %(default)f)")

        self.trainer_group = self.parser.add_argument_group("Trainer/callback parameters")
        self.trainer_group.add_argument("--seed", type=int, default=None,
                                        help="for reproducibility, if not provided will be chosen randomly")
        self.trainer_group.add_argument("--ckpt_quantity", type=str, default="avg_val_accuracy",
                                        help="quantity to monitor for checkpoints; loss: poisson negative log "
                                             "likelihood, accuracy: Pearson's R (valid: avg_train_loss, "
                                             "avg_train_accuracy, avg_val_loss, avg_val_accuracy)")
        self.trainer_group.add_argument("--save-top-k", type=int, default=-1,
                                        help="saves the top k (e.g. 3) models; -1 means every model gets saved")
        self.trainer_group.add_argument("--stop_quantity", type=str, default="avg_train_loss",
                                        help="quantity to monitor for early stopping; loss: poisson negative log "
                                             "likelihood, accuracy: Pearson's R (valid: avg_train_loss, "
                                             "avg_train_accuracy, avg_val_loss, avg_val_accuracy)")
        self.trainer_group.add_argument("--patience", type=int, default=5,
                                        help="allowed epochs without the quantity improving before stopping training")
        self.trainer_group.add_argument("-b", "--batch-size", type=int, default=120,
                                        help="batch size for training and validation sets")
        self.trainer_group.add_argument("-tb", "--test-batch-size", type=int, default=120,
                                        help="batch size for test set")
        self.trainer_group.add_argument("-pb", "--predict-batch-size", type=int, default=120,
                                        help="batch size for prediction set")
        self.trainer_group.add_argument("--device", type=str, default="gpu", help="device to train on")
        self.trainer_group.add_argument("--num-devices", type=int, default=1,
                                        help="number of devices to train on (see docs about performance)")
        self.trainer_group.add_argument("--num-workers", type=int, default=0,
                                        help="how many subprocesses to use for data loading")
        self.trainer_group.add_argument("-e", "--epochs", type=int, default=5, help="number of training runs")

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
                raise OSError(f"did not find model path: ({args.model})")
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


class ConverterParser(BaseParser):
    def __init__(self):
        super().__init__(prog="H5toBigwig", description="write predictions in h5 format to a bigwig or bedgraph file")
        self.parser.add_argument("-i", "--input-file", type=str, default=None, required=True,
                                 help="input h5 predictions file (Predmoter output)")
        self.parser.add_argument("-o", "--output-dir", type=str, default=".",
                                 help="output directory for all converted files")
        self.parser.add_argument("-of", "--output-format", type=str, default="bigwig",
                                 help="output format for predictions (valid: bigwig (bw), bedgraph (bg))")
        self.parser.add_argument("--basename", type=str, default=None, required=True,
                                 help="basename of the output files, naming convention: "
                                      "basename_dataset_strand.bw/bg.gz")
        self.parser.add_argument("--strand", type=str, default=None,
                                 help="None means the average of both strands is used, else + or - can be selected")

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


class AddAverageParser(BaseParser):
    def __init__(self):
        super().__init__(prog="AddAverage",
                         description="compute average of NGS dataset and add it to the h5 input file")
        self.io_group = self.parser.add_argument_group("Data input/output parameters")
        pass
