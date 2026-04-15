import json
import re
from typing import Any
from pydantic import BaseModel, Field


class VocabFilter(BaseModel):
    """Token groups and caches used for fast filtering during
        constrained decoding.

    This class categorizes vocabulary tokens into specific sets (numeric,
    safe strings, etc.) to enable rapid filtering by the state machine
    during generation.

    Attributes:
        numeric_tokens (set[int]): IDs of tokens that can form a number.
        string_safe_tokens (set[int]):
        Tokens containing no quotes or escape characters.
        string_unsafe_tokens (set[int]):
        Complex tokens requiring regex validation.
        literal_cache (dict[str, set[int]]): Cache for literal string matches.
    """

    numeric_tokens: set[int] = Field(default_factory=set)
    string_safe_tokens: set[int] = Field(default_factory=set)
    string_unsafe_tokens: set[int] = Field(default_factory=set)
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_clean_vocab(cls, clean_vocab: dict[int, str]) -> "VocabFilter":
        """Classify vocabulary tokens to prepare for constrained decoding.

        Args:
            clean_vocab: Dictionary mapping token IDs to their decoded strings.

        Returns:
            VocabFilter: An instance containing the classified token sets.
        """
        numeric_tokens_set: set[int] = set()
        string_safe_tokens_set: set[int] = set()
        string_unsafe_tokens_set: set[int] = set()

        for token_id, token_str in clean_vocab.items():
            if re.match(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$', token_str):
                numeric_tokens_set.add(token_id)

            if '"' not in token_str and '\\' not in token_str:
                string_safe_tokens_set.add(token_id)
            else:
                string_unsafe_tokens_set.add(token_id)

        return cls(
            numeric_tokens=numeric_tokens_set,
            string_safe_tokens=string_safe_tokens_set,
            string_unsafe_tokens=string_unsafe_tokens_set
        )

    def get_literal_matches(
        self, remainder: str, clean_vocab: dict[int, str]
    ) -> set[int]:
        """Find valid tokens to complete a literal string.

        Args:
            remainder: The remaining text that needs to be matched.
            clean_vocab: The ID-to-string vocabulary dictionary.
        Returns:
            set[int]: A set of valid token IDs.
        """
        if remainder not in self.literal_cache:
            matching_tokens: set[int] = set()

            for token_id, token_str in clean_vocab.items():
                is_partial_match = remainder.startswith(token_str)
                is_complete_match = token_str.startswith(remainder)

                if is_partial_match or is_complete_match:
                    matching_tokens.add(token_id)

            self.literal_cache[remainder] = matching_tokens

        return self.literal_cache[remainder]


class VocabIndex(BaseModel):
    """Complete vocabulary index and search utilities.

    Handles loading the vocabulary file, converting between IDs
    and strings, and delegates complex filtering to VocabFilter.

    Attributes:
        vocab_path (str): Path to the source JSON vocabulary file.
        vocab (dict[str, int]): Raw text-to-ID mapping.
        clean_vocab (dict[int, str]): Decoded ID-to-text mapping.
        token_to_id (dict[str, int]): Optimized reverse text-to-ID mapping.
        size (int): Total number of tokens in the vocabulary.
        filter_vocab (VocabFilter): Pre-calculated filters for generation.
    """

    vocab_path: str
    vocab: dict[str, int]
    clean_vocab: dict[int, str]
    token_to_id: dict[str, int]
    size: int
    filter_vocab: VocabFilter

    @classmethod
    def from_model(cls, model: Any) -> "VocabIndex":
        """Build the index from an LLM model instance.

        Args:
            model: Instance of Small_LLM_Model used for decoding.

        Returns:
            VocabIndex: The fully initialized vocabulary index.
        """
        vocabulary_file_path = model.get_path_to_vocab_file()

        with open(vocabulary_file_path, 'r', encoding='utf-8') as file_vocab:
            raw_vocabulary_data = json.load(file_vocab)

        clean_vocabulary_dict: dict[int, str] = {}
        for raw_token_str, token_id in raw_vocabulary_data.items():
            clean_vocabulary_dict[token_id] = model.decode([token_id])

        reverse_token_mapping: dict[str, int] = {}
        for token_id, decoded_token_str in clean_vocabulary_dict.items():
            reverse_token_mapping[decoded_token_str] = token_id

        return cls(
            vocab_path=vocabulary_file_path,
            vocab=raw_vocabulary_data,
            clean_vocab=clean_vocabulary_dict,
            token_to_id=reverse_token_mapping,
            size=len(raw_vocabulary_data),
            filter_vocab=VocabFilter.from_clean_vocab(clean_vocabulary_dict)
        )

    def get_literal_matches(self, remainder: str) -> set[int]:
        """Delegate literal token lookup to the cached filter.

        Args:
            remainder: The string expected by the current state.

        Returns:
            set[int]: IDs of compatible tokens.
        """
        return self.filter_vocab.get_literal_matches(
            remainder, self.clean_vocab)
