import sys
import json
import argparse
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Any
from src.functions_validator import FunctionDefinition, FunctionCallingTest


def parse_arguments_and_load_data() -> (
            tuple[Path, list[FunctionDefinition],
                  list[FunctionCallingTest]]):

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
            sys.exit(f"Critical Error: File '{file_path}' not found.")

    functions_def = _load_json_data(
        args.functions_definition, FunctionDefinition)
    calling_tests = _load_json_data(args.input, FunctionCallingTest)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Could not create output dir. {args.output.parent}"
                 f"\nDetails: {e}")

    return args.output, functions_def, calling_tests


def _load_json_data(
        file_path: Path, model_class: type[BaseModel]) -> list[Any]:
    """Load a JSON array and validate each item via a Pydantic model.

    Args:
        file_path: Path to the JSON file to read.
        model_class: Pydantic model used for unit validation.

    Returns:
        list[Any]: List of validated objects of type `model_class`.

    Raises:
        SystemExit: In case of a read error, malformed JSON, or data
            that does not conform to the model.
    """
    try:
        with file_path.open('r') as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            sys.exit(f"Error: File '{file_path}' must contain a JSON.")

        return [model_class.model_validate(item) for item in raw_data]

    except (FileNotFoundError, PermissionError):
        sys.exit(f"Error accessing file '{file_path}'.")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: '{file_path}' is not valid JSON. {e.msg}")
    except ValidationError as e:
        sys.exit(f"Error: Data validation failed for '{file_path}'."
                 f"\nDetails:\n{e}")
    except Exception as e:
        sys.exit(f"Unexpected error with '{file_path}': {e}")
