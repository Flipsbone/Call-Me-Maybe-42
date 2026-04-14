import sys
import json
import argparse
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Any
from src.functions_validator import FunctionDefinition


class FunctionCallingTest(BaseModel):
    prompt: str


class AppConfig(BaseModel):
    output_path: Path
    functions_definition: list[FunctionDefinition]
    function_calling_tests: list[FunctionCallingTest]


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
            sys.exit(f"Critical Error: File '{file_path}' not found.")

    functions_def = _load_json_data(
        args.functions_definition, FunctionDefinition)
    calling_tests = _load_json_data(args.input, FunctionCallingTest)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Could not create output dir. {args.output.parent}"
                 f"\nDetails: {e}")

    try:
        return AppConfig(
            output_path=args.output,
            functions_definition=functions_def,
            function_calling_tests=calling_tests
        )
    except ValidationError as e:
        sys.exit(f"CRITICAL ERROR: Invalid global configuration."
                 f"\nDetails:\n{e}")


def _load_json_data(
        file_path: Path, model_class: type[BaseModel]) -> list[Any]:
    try:
        with file_path.open('r') as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            sys.exit(f"Error: File '{file_path}' must contain a JSON.")

        return [model_class.model_validate(item) for item in raw_data]

    except FileNotFoundError:
        sys.exit(f"Error: File '{file_path}' not found.")
    except PermissionError:
        sys.exit(f"Error: Insufficient permissions to read '{file_path}'.")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: '{file_path}' is not valid JSON. {e.msg}")
    except ValidationError as e:
        sys.exit(f"Error: Data validation failed for '{file_path}'."
                 f"\nDetails:\n{e}")
