import sys
import argparse
from pathlib import Path
from src.parsing import DataParser


def check_and_prepare_paths(args: argparse.Namespace) -> None:
    input_path = args.input
    output_path = args.output

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error : input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    output_file = Path(output_path)
    output_dir = output_file.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error : {output_dir} : {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    args = parser.parse_args()
    check_and_prepare_paths(args)
    try:
        functions_data = DataParser.load_functions(args.functions_definition)
        print(functions_data)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
