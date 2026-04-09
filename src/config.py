import sys
import json
import argparse
from pathlib import Path
from pydantic import BaseModel, ValidationError

from src.model_pydantic import AppConfig, FunctionDefinition, FunctionCalling


def setup_configuration() -> AppConfig:

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument("--output", type=Path,
                        default=Path(
                            "data/output/function_calling_results.json"))
    args = parser.parse_args()

    for file_path in [args.functions_definition, args.input]:
        if not file_path.exists():
            sys.exit(f"Error: File not found: {file_path}")

    functions_def = _load_json_data(
        args.functions_definition, FunctionDefinition)

    calling_tests = _load_json_data(args.input, FunctionCalling)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Could not create directory {args.output.parent}-{e}")

    try:
        return AppConfig(
            output_path=args.output,
            functions_definition=functions_def,
            function_calling_tests=calling_tests
        )
    except ValidationError as e:
        print(f"CRITICAL ERROR: Global configuration invalid.\n{e}",
              file=sys.stderr)
        sys.exit(1)


def _load_json_data(file_path: Path, check_pydantic: type[BaseModel]) -> list:

    try:
        with file_path.open('r') as file:
            raw_data = json.load(file)
        return [check_pydantic.model_validate(item) for item in raw_data]

    except PermissionError:
        print("Error: Does not have right permission", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: '{file_path}' is not valid JSON. {e.msg}",
              file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: Data validation failed '{file_path}'.\nDetails:\n{e}",
              file=sys.stderr)
        sys.exit(1)
