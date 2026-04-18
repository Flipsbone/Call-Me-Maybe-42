from pydantic import BaseModel, Field
from typing import Any


class FunctionCallingTest(BaseModel):
    """Represent a single user prompt used in evaluation.

    Attributes:
        prompt: Raw user request that should map to a function call.
    """

    prompt: str


class ParameterModel(BaseModel):
    """Describe one parameter type in a function schema.

    Attributes:
        type: JSON type name expected for the parameter value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """Represent one callable function and its declared contract.

    Attributes:
        name: Function identifier expected in the generated output.
        description: Natural-language explanation of the function purpose.
        parameters: Mapping of parameter names to their schema definitions.
        returns: Declared return type schema for the function.
    """

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class FunctionCallResult(BaseModel):
    """Store a validated function call produced from one prompt.

    Attributes:
        prompt: Original user prompt.
        name: Selected function name.
        parameters: Extracted arguments prepared for function execution.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
