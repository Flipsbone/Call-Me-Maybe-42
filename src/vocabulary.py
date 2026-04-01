import sys
import json
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError


class VocabularyIndex:
    def __init__(self) -> None:
        self.model = Small_LLM_Model()
        self.vocab_path = self.model.get_path_to_vocab_file()
        self.vocab = self._load_vocab(self.vocab_path)

        self.brace_open_ids: set[int] = set()
        self.brace_close_ids: set[int] = set()
        self.bracket_open_ids: set[int] = set()
        self.bracket_close_ids: set[int] = set()

        self.dot_ids: set[int] = set()
        self.colon_ids: set[int] = set()
        self.comma_ids: set[int] = set()
        self.quote_ids: set[int] = set()

        self.boolean_true_ids: set[int] = set()
        self.boolean_false_ids: set[int] = set()
        self.null_ids: set[int] = set()

        self.number_syntax_ids: set[int] = set()

        self.whitespace_ids: set[int] = set()

        self.general_text_ids: set[int] = set()

        self._build_index(self.vocab)

    def _build_index(self, vocab: dict[str, int]) -> None:
        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace('Ġ', '')
            if clean_token == '{':
                self.brace_open_ids.add(token_id)
            elif clean_token == '}':
                self.brace_close_ids.add(token_id)
            elif clean_token == '[':
                self.bracket_open_ids.add(token_id)
            elif clean_token == ']':
                self.bracket_close_ids.add(token_id)
            elif clean_token == '.':
                self.dot_ids.add(token_id)
            elif clean_token == ':':
                self.colon_ids.add(token_id)
            elif clean_token == ',':
                self.comma_ids.add(token_id)
            elif clean_token == '"':
                self.quote_ids.add(token_id)
            elif clean_token == 'true':
                self.boolean_true_ids.add(token_id)
            elif clean_token == 'false':
                self.boolean_false_ids.add(token_id)
            elif clean_token == 'null':
                self.null_ids.add(token_id)
            elif clean_token != "" and all(
                    c in "0123456789.-+eE" for c in clean_token):
                self.number_syntax_ids.add(token_id)
            elif clean_token.isspace():
                self.whitespace_ids.add(token_id)
            elif token_str == 'Ġ':
                self.whitespace_ids.add(token_id)
            else:
                self.general_text_ids.add(token_id)

    def _load_vocab(self, file_path: str) -> dict[str, int]:
        try:
            with open(file_path, 'r') as file_vocab:
                raw_data = json.load(file_vocab)
                return raw_data
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.", file=sys.stderr)
            raise
        except ValidationError as e:
            print(f"Erreur de validation : {e}", file=sys.stderr)
            raise
