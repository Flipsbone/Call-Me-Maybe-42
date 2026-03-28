import json
import sys
from src.model_pydantic import FunctionDefinition, FunctionCalling
from pydantic import ValidationError


class DataParser:
    @staticmethod
    def load_functions_definition(file_path: str) -> list:
        try:
            with open(file_path, 'r') as file_functions:
                raw_data = json.load(file_functions)
                return [FunctionDefinition(**item) for item in raw_data]
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.", file=sys.stderr)
            raise
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} is not valid JSON: {e.msg}",
                  file=sys.stderr)
            raise
        except ValidationError as e:
            print(f"Erreur de validation : {e}", file=sys.stderr)
            raise

    @staticmethod
    def load_function_calling_tests(file_path: str) -> list:
        try:
            with open(file_path, 'r') as file_callings_tests:
                raw_data = json.load(file_callings_tests)
                return [FunctionCalling(**item) for item in raw_data]
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.", file=sys.stderr)
            raise
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} is not valid JSON: {e.msg}",
                  file=sys.stderr)
            raise
        except ValidationError as e:
            print(f"Erreur de validation : {e}", file=sys.stderr)
            raise
