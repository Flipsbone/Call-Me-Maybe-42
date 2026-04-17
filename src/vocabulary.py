import json
import re
from typing import Any
from pydantic import BaseModel, Field


class StrictVocabFilter(BaseModel):
    numeric_tokens: set[int] = Field(default_factory=set)
    string_content_tokens: set[int] = Field(default_factory=set)
    string_closer_tokens: set[int] = Field(default_factory=set)
    exact_quote_tokens: set[int] = Field(default_factory=set)
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_clean_vocab(
            cls, clean_vocab: dict[int, str]) -> "StrictVocabFilter":

        numeric_tokens_set: set[int] = set()
        string_content_tokens_set: set[int] = set()
        string_closer_tokens_set: set[int] = set()
        exact_quote_tokens_set: set[int] = set()

        for token_id, token_str in clean_vocab.items():
            if token_str == "":
                continue

            if re.match(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$', token_str):
                numeric_tokens_set.add(token_id)

            if token_str == '"':
                exact_quote_tokens_set.add(token_id)

            if '"' in token_str:
                string_closer_tokens_set.add(token_id)
            else:
                string_content_tokens_set.add(token_id)

        return cls(
            numeric_tokens=numeric_tokens_set,
            string_content_tokens=string_content_tokens_set,
            string_closer_tokens=string_closer_tokens_set,
            exact_quote_tokens=exact_quote_tokens_set
        )

    def get_literal_matches(
        self, remainder: str, clean_vocab: dict[int, str]
    ) -> set[int]:
        if remainder not in self.literal_cache:
            matching_tokens: set[int] = set()
            for token_id, token_str in clean_vocab.items():
                if (remainder.startswith(token_str) or
                        token_str.startswith(remainder)):
                    matching_tokens.add(token_id)
            self.literal_cache[remainder] = matching_tokens
        return self.literal_cache[remainder]


class VocabIndex(BaseModel):
    clean_vocab: dict[int, str]
    filter_vocab: StrictVocabFilter

    @classmethod
    def from_model(cls, model: Any) -> "VocabIndex":
        vocabulary_file_path = model.get_path_to_vocab_file()

        with open(vocabulary_file_path, 'r', encoding='utf-8') as file_vocab:
            raw_vocabulary_data = json.load(file_vocab)

        clean_vocabulary_dict: dict[int, str] = {}

        for raw_token_str, token_id in raw_vocabulary_data.items():
            decoded_str = model.decode([token_id])
            clean_vocabulary_dict[token_id] = decoded_str

        return cls(
            clean_vocab=clean_vocabulary_dict,
            filter_vocab=StrictVocabFilter.from_clean_vocab(
                clean_vocabulary_dict)
        )

    def get_literal_matches(self, remainder: str) -> set[int]:
        return self.filter_vocab.get_literal_matches(
            remainder, self.clean_vocab)
