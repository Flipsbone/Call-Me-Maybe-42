from pydantic import BaseModel, Field
from typing import Any


class FunctionCallingTest(BaseModel):
    """Single prompt used to validate function-calling generation."""

    prompt: str


class ParameterModel(BaseModel):
    """Schema for a single function parameter definition."""

    type: str


class FunctionDefinition(BaseModel):
    """Schema describing a callable target and its signature."""

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class FunctionCallResult(BaseModel):
    """Validated output produced for a single prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]
