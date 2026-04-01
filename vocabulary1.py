import json
import string
from typing import Dict, Set, List
from src.model_pydantic import FunctionDefinition


class VocabularyIndex:
    def __init__(self, vocab_path: str) -> None:
        self.vocab: Dict[str, int] = self._load_vocab(vocab_path)

        self.digit_ids: Set[int] = set()
        self.number_syntax_ids: Set[int] = set()
        self.quote_ids: Set[int] = set()
        self.whitespace_ids: Set[int] = set()
        self.json_syntax_ids: Set[int] = set()

        self._build_index()

    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_index(self) -> None:
        for token_str, token_id in self.vocab.items():
            clean_token = token_str.replace('Ġ', '').strip()

            if all(c in string.digits for c in clean_token) and clean_token != "":
                self.digit_ids.add(token_id)

            if clean_token in ['.', '-', '+', 'e', 'E']:
                self.number_syntax_ids.add(token_id)

            if '"' in token_str:
                self.quote_ids.add(token_id)

            if clean_token in ['{', '}', '[', ']', ':', ',']:
                self.json_syntax_ids.add(token_id)

            if token_str.isspace() or token_str == 'Ġ':
                self.whitespace_ids.add(token_id)

    def get_allowed_function_names(self, definitions: List[FunctionDefinition]) -> Set[int]:
        allowed_ids = set()
        for func in definitions:
            allowed_ids.update(self.get_exact_string_ids(func.name))
        return allowed_ids

    def get_ids_for_parameter_type(self, param_name: str, definition: FunctionDefinition) -> Set[int]:
        param_info = definition.parameters.get(param_name)
        if not param_info:
            return set()

        if param_info.type == "number" or param_info.type == "integer":
            return self.get_allowed_ids_for_number()

        if param_info.type == "string":
            return self.quote_ids | self.whitespace_ids

        return set()

    def get_exact_string_ids(self, target_string: str) -> Set[int]:
        allowed = set()
        for token_str, token_id in self.vocab.items():
            clean_str = token_str.replace('Ġ', '')
            if clean_str in target_string and clean_str != "":
                allowed.add(token_id)
        return allowed
