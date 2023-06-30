from predmoter.core.parser import ConverterParser
from predmoter.utilities.converter import Converter


def main():
    pp = ConverterParser()
    args = pp.get_args()
    print(f"Starting conversion of the file {args.input_file}.")
    Converter(args.input_file, args.output_dir, args.output_format, args.basename, args.strand)
    print(f"Conversion finished.")


if __name__ == "__main__":
    main()
