from enum import Enum, auto
from src.model_pydantic import FunctionDefinition
from src.vocabulary import VocabularyIndex
from pydantic import Model

class GenerationState(Enum):
    EXPECTING_JSON_START = auto()        # Cible: '{"name": "'
    EXPECTING_FUNCTION_NAME = auto()     # Cible: un des noms du catalogue
    EXPECTING_PARAMETERS_START = auto()  # Cible: '", "parameters": {'
    EXPECTING_PARAM_KEY = auto()         # Cible: '"nom_du_param": '
    EXPECTING_PARAM_VALUE = auto()       # Cible: dépend du type (str, int, bool)
    EXPECTING_PARAM_SEPARATOR = auto()   # Cible: ', ' ou '}' (fin)
    DONE = auto()


class SchemaStateMachine:
    def __init__(self,
                 vocab_index: VocabularyIndex,
                 available_functions: list[FunctionDefinition]
                 ):
        self.vocab = vocab_index
        self.functions_catalog: dict[str, FunctionDefinition] = {
            fn.name: fn for fn in available_functions
            }

        self.current_state = GenerationState.EXPECTING_JSON_START
        self.buffer: str = ""

        self.selected_function: FunctionDefinition | None
        self.remaining_parameters: list[str] = []
        self.current_parameter_type: str | None

    def get_allowed_tokens(self) -> set[int]:

        match self.current_state:
            case GenerationState.EXPECTING_JSON_START:
                target = '{"name": "'
                return self._get_prefix_matches(target)

            case GenerationState.EXPECTING_FUNCTION_NAME:
                return self._get_function_name_matches()

            case GenerationState.EXPECTING_PARAMETERS_START:
                target = '", "parameters": {'
                return self._get_prefix_matches(target)

            case GenerationState.EXPECTING_PARAM_KEY:
                if self.remaining_parameters:
                    param_name = self.remaining_parameters[0]
                    target = f'"{param_name}": '
                    return self._get_prefix_matches(target)
                return set()

            case GenerationState.EXPECTING_PARAM_VALUE:
                return self._get_value_matches(self.current_parameter_type)

            case GenerationState.EXPECTING_PARAM_SEPARATOR:
                if not self.remaining_parameters:
                    target = '}'
                else:
                    target = ', '
                return self._get_prefix_matches(target)

            case GenerationState.DONE:
                return set()

    def advance(self, token_id: int):
        token_str = self.vocab.clean_vocab.get(token_id, "")
        self.buffer += token_str
        match self.current_state:
            case GenerationState.EXPECTING_JSON_START:
                if self.buffer == '{"name": "':
                    self.current_state = GenerationState.FUNCTION_NAME
                    self.buffer = ""

            case GenerationState.EXPECTING_FUNCTION_NAME:
                if self.selected_function == (
                                        self.functions_catalog[self.buffer]):
                    param_dict = (
                        self.selected_function.parameters
                        if isinstance(self.selected_function.parameters, dict)
                        else self.selected_function.parameters.model_dump()
                        )

                    self.remaining_parameters = list(param_dict.keys())

                    self.current_state = GenerationState.EXPECTING_PARAMETERS_START
                    self.buffer = ""

            case GenerationState.EXPECTING_PARAMETERS_START:
                if self.buffer == '", "parameters": {':
                    param_data = (
                    self.remaining_parameters = list(params_data.keys())