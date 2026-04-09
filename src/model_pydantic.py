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


class VocabularyFilterSchema(BaseModel):
    numeric_tokens: list[int] = Field(default_factory=list)
    string_safe_tokens: list[int] = Field(default_factory=list)
    string_unsafe_tokens: list[int] = Field(default_factory=list)
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)


class VocabularyIndexSchema(BaseModel):
    vocab_path: str = Field(default="")
    vocab: dict[str, int] = Field(default_factory=dict)
    clean_vocab: dict[int, str] = Field(default_factory=dict)
    size: int = Field(default=0)
    filter_schema: VocabularyFilterSchema = Field(
        default_factory=VocabularyFilterSchema)
