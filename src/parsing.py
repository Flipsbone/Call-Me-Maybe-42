import json
import sys
from src.model_pydantic import FunctionDefinition, FunctionCalling
from pydantic import BaseModel, ValidationError


class DataParser:
    def __init__(self, path_funct_definition: str, path_funct_calling: str):
        self.functions_definition: list[FunctionDefinition] = (
            self.load_data(path_funct_definition, FunctionDefinition))
        self.function_calling_tests: list[FunctionCalling] = (
            self.load_data(path_funct_calling, FunctionCalling))

    @staticmethod
    def load_data(file_path: str, model_parse: type[BaseModel]) -> list:
        try:
            with open(file_path, 'r') as file_functions:
                raw_data = json.load(file_functions)
                return [model_parse(**item) for item in raw_data]
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
