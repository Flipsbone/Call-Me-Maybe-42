from pydantic import BaseModel, Field
from typing import Any


class ParameterModel(BaseModel):
    """Schema for a single function parameter definition."""

    type: str | None = None


class FunctionDefinition(BaseModel):
    """Schema describing a callable target and its signature."""

    name: str
    description: str
    parameters: dict[str, ParameterModel] = Field(default_factory=dict)
    returns: ParameterModel


class FunctionCallResult(BaseModel):
    """Validated output produced for a single prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
