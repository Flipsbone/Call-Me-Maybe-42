from pydantic import BaseModel


class ParameterModel(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel
