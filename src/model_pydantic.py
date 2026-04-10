from pydantic import BaseModel, Field, RootModel
from typing import Any
from pathlib import Path


class FunctionCalling(BaseModel):
    prompt: str


class ParameterModel(BaseModel):
    type: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterModel] = Field(default_factory=dict)
    returns: ParameterModel


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    output_path: Path
    functions_definition: list[FunctionDefinition]
    function_calling_tests: list[FunctionCalling]


class VocabSchema(RootModel[dict[str, int]]):
    pass
