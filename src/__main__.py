import sys
import argparse
from pathlib import Path
from src.parsing import DataParser


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        type=Path,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        type=Path,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        type=Path,
                        default="data/output/function_calling_results.json")
    return parser.parse_args()


def check_and_prepare_paths(func_def_path: Path,
                            input_path: Path,
                            output_path: Path) -> None:

    for file_path in [func_def_path, input_path]:
        if not file_path.exists():
            print(f"Error : input file '{input_path}' not found.",
                  file=sys.stderr)
            sys.exit(1)

    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error : {output_dir} : {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    args = parse_arguments()
    check_and_prepare_paths(args.functions_definition, args.input, args.output)
    try:
        data_manager = DataParser(
            path_funct_definition=str(args.functions_definition),
            path_funct_calling=str(args.input)
        )
        print(data_manager.functions_definition)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
