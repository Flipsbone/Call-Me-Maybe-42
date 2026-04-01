from enum import Enum, auto
from src.model_pydantic import FunctionDefinition
from src.vocabulary import VocabularyIndex


class GenerationState(Enum):
    START = auto()                      # 1. Waiting for opening brace '{' or bracket '['
    EXPECTING_KEY = auto()              # 2. Inside object: waiting for opening quote of a key '"'
    IN_KEY = auto()                     # 3. Writing key name: waiting for closing quote '"'
    EXPECTING_COLON = auto()            # 4. Key completed: waiting for colon ':'
    EXPECTING_VALUE = auto()            # 5. Colon passed: waiting for a value (quote, digit, {, [, true, false, null)
    IN_STRING_VALUE = auto()            # 6. Writing a string value
    IN_NUMBER_VALUE = auto()            # 7. Writing a numeric value
    EXPECTING_COMMA_OR_END = auto()     # 8. Value completed: waiting for comma ',' or closing '}' / ']'
    DONE = auto()                       # 9. Main structure closed: JSON is fully valid


class JSONStateMachine:
    def __init__(self, vocab_index: VocabularyIndex):
        self.voca_build = vocab_index

    def get_allowed_tokens(self, current_state: GenerationState):
        match current_state:
            case GenerationState.START:
                return (
                    self.voca_build.brace_open_ids |
                    self.voca_build.bracket_open_ids |
                    self.voca_build.whitespace_ids
                )
            case GenerationState.EXPECTING_KEY:
                return (
                    self.voca_build.quote_ids |
                    self.voca_build.whitespace_ids
                )

            case GenerationState.IN_KEY | GenerationState.IN_STRING_VALUE:
                return (
                    self.voca_build.quote_ids |
                    self.voca_build.whitespace_ids |
                    self.voca_build.general_text_ids |
                    self.voca_build.number_syntax_ids
                )

            case GenerationState.IN_NUMBER_VALUE:
                return (
                    self.voca_build.number_syntax_ids |
                    self.voca_build.comma_ids |
                    self.voca_build.brace_close_ids |
                    self.voca_build.bracket_close_ids |
                    self.voca_build.whitespace_ids
                )

            case GenerationState.EXPECTING_COLON:
                return (
                    self.voca_build.colon_ids |
                    self.voca_build.whitespace_ids
                )

            case GenerationState.EXPECTING_VALUE:
                return (
                    self.voca_build.quote_ids |
                    self.voca_build.boolean_true_ids |
                    self.voca_build.boolean_false_ids |
                    self.voca_build.null_ids |
                    self.voca_build.number_syntax_ids |
                    self.voca_build.brace_open_ids |
                    self.voca_build.bracket_open_ids |
                    self.voca_build.whitespace_ids
                )

            case GenerationState.EXPECTING_COMMA_OR_END:
                return (
                    self.voca_build.comma_ids |
                    self.voca_build.brace_close_ids |
                    self.voca_build.bracket_close_ids |
                    self.voca_build.whitespace_ids
                )
