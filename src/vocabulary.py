import sys
import json
from pydantic import RootModel, ValidationError
from llm_sdk import Small_LLM_Model


class VocabSchema(RootModel[dict[str, int]]):
    pass


class VocabularyIndex:
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model: Small_LLM_Model = model

        self.vocab_path = self.model.get_path_to_vocab_file()
        self.vocab = self._load_vocab(self.vocab_path)
        self.size = len(self.vocab)

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
        structural_chars = {'{', '}', '[', ']', ':', ',', '"'}
        number_chars = set("0123456789.-+eE")

        for token_str, token_id in vocab.items():
            clean_token = token_str.replace('Ġ', '').replace('Ċ', '').strip()

            if not clean_token:
                self.whitespace_ids.add(token_id)
                continue

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
            elif all(c in number_chars for c in clean_token):
                self.number_syntax_ids.add(token_id)
            elif not any(c in structural_chars for c in token_str):
                self.general_text_ids.add(token_id)

    def _load_vocab(self, file_path: str) -> dict[str, int]:
        try:
            with open(file_path, 'r') as file_vocab:
                raw_data = json.load(file_vocab)

                validated_data = VocabSchema.model_validate(raw_data)
                return validated_data.root

        except FileNotFoundError:
            print(f"Error: Vocabulary file '{file_path}' not found.",
                  file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Vocabulary file is not valid JSON. {e}",
                  file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            print(f"Error: Vocabulary file structure is invalid (Pydantic)."
                  f"{e}", file=sys.stderr)
            sys.exit(1)
