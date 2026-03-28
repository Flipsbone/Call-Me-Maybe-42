from pydantic import BaseModel


class FunctionCalling(BaseModel):
    prompt: str


class ParameterModel(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel
