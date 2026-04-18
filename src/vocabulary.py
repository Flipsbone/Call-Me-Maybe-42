import json
import re
import sys
from typing import Any
from pydantic import BaseModel, Field


class StrictVocabFilter(BaseModel):
    """Pre-computed token groups used by constrained JSON parsing.

    Attributes:
        numeric_tokens: Tokens that can appear in JSON number contexts.
        string_content_tokens: Tokens safe inside open JSON strings.
        string_closer_tokens: Tokens containing quote characters.
        exact_quote_tokens: Tokens that are exactly one double-quote.
    """

    numeric_tokens: set[int] = Field(default_factory=set)
    string_content_tokens: set[int] = Field(default_factory=set)
    string_closer_tokens: set[int] = Field(default_factory=set)
    exact_quote_tokens: set[int] = Field(default_factory=set)

    @classmethod
    def from_clean_vocab(
            cls, clean_vocab: dict[int, str]) -> "StrictVocabFilter":
        """Build token filter sets from a decoded token vocabulary.

        Args:
            clean_vocab: Mapping from token IDs to decoded token strings.

        Returns:
            StrictVocabFilter: Filter object with token IDs grouped by usage.
        """

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


class VocabIndex(BaseModel):
    """Store decoded vocabulary and derived lookup accelerators.

    Attributes:
        clean_vocab: Mapping of token ID to decoded text.
        filter_vocab: Pre-computed token groups for parser states.
        literal_cache: Cached literal-prefix token matches by remainder string.
    """

    clean_vocab: dict[int, str]
    filter_vocab: StrictVocabFilter
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_model(cls, model: Any) -> "VocabIndex":
        """Load, decode, and index vocabulary from a model backend.

        Args:
            model: Model object exposing vocab path and decode capabilities.

        Returns:
            VocabIndex: Fully initialized vocabulary index and filter sets.

        Raises:
            SystemExit: If vocabulary loading or processing fails.
        """
        vocabulary_file_path = model.get_path_to_vocab_file()

        try:
            with open(
                    vocabulary_file_path, 'r', encoding='utf-8') as file_vocab:
                raw_vocabulary_data = json.load(file_vocab)
        except (FileNotFoundError, PermissionError) as e:
            sys.exit("Error accessing vocabulary file "
                     f"'{vocabulary_file_path}': {e}")
        except Exception as e:
            sys.exit("Unexpected error loading vocabulary file "
                     f"'{vocabulary_file_path}': {e}")

        try:
            clean_vocabulary_dict: dict[int, str] = {}

            for raw_token_str, token_id in raw_vocabulary_data.items():
                decoded_str = model.decode([token_id])
                clean_vocabulary_dict[token_id] = decoded_str

            return cls(
                clean_vocab=clean_vocabulary_dict,
                filter_vocab=StrictVocabFilter.from_clean_vocab(
                    clean_vocabulary_dict)
            )
        except Exception as e:
            sys.exit(f"Error: Failed to process vocabulary data: {e}")

    def get_literal_matches(self, remainder: str) -> set[int]:
        """Get tokens that match the given remainder string.

        Args:
            remainder: Remaining literal text expected by the state machine.

        Returns:
            set[int]: Token IDs where either the token starts with the
            remainder or the remainder starts with the token.

        Results are cached for performance.
        """
        if remainder not in self.literal_cache:
            matching_tokens: set[int] = set()
            for token_id, token_str in self.clean_vocab.items():
                if (remainder.startswith(token_str) or
                        token_str.startswith(remainder)):
                    matching_tokens.add(token_id)
            self.literal_cache[remainder] = matching_tokens
        return self.literal_cache[remainder]
