from pydantic import RootModel


class VocabSchema(RootModel[dict[str, int]]):
    pass
