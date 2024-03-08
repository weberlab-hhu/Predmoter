#! /usr/bin/env python3
import logging.config

from predmoter.utilities.utils import get_log_dict
from predmoter.core.parser import ConverterParser
from predmoter.utilities.converter import Converter


def main():
    pp = ConverterParser()
    args = pp.get_args()
    logging.config.dictConfig(get_log_dict(args.output_dir, args.prefix))
    Converter(args.input_file, args.output_dir, args.output_format, args.basename,
              args.strand, args.experimental, args.dsets, args.bl_chroms, args.window_size)


if __name__ == "__main__":
    main()
