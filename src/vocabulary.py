import sys
import json
import re
from pydantic import BaseModel, Field, ValidationError, PrivateAttr
from llm_sdk import Small_LLM_Model


class VocabFilter(BaseModel):

    numeric_tokens: list[int] = Field(default_factory=list)
    string_safe_tokens: list[int] = Field(default_factory=list)
    string_unsafe_tokens: list[int] = Field(default_factory=list)
    literal_cache: dict[str, set[int]] = Field(default_factory=dict)

    @classmethod
    def from_clean_vocab(cls, clean_vocab: dict[int, str]) -> "VocabFilter":

        numeric_tokens = []
        string_safe_tokens = []
        string_unsafe_tokens = []

        is_num_chunk = re.compile(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$')

        for token_id, token_str in clean_vocab.items():
            if is_num_chunk.match(token_str):
                numeric_tokens.append(token_id)

            if '"' not in token_str and '\\' not in token_str:
                string_safe_tokens.append(token_id)
            else:
                string_unsafe_tokens.append(token_id)

        return cls(
            numeric_tokens=numeric_tokens,
            string_safe_tokens=string_safe_tokens,
            string_unsafe_tokens=string_unsafe_tokens,
        )

    def get_literal_matches(
            self, remainder: str, clean_vocab: dict[int, str]) -> set[int]:

        if remainder not in self.literal_cache:
            matching_tokens = set()

            for token_id, token_str in clean_vocab.items():
                is_partial = remainder.startswith(token_str)
                is_complete = token_str.startswith(remainder)

                if is_partial or is_complete:
                    matching_tokens.add(token_id)

            self.literal_cache[remainder] = matching_tokens

        return self.literal_cache[remainder]


class VocabIndex(BaseModel):
    vocab_path: str = Field(default="")
    vocab: dict[str, int] = Field(default_factory=dict)
    clean_vocab: dict[int, str] = Field(default_factory=dict)
    size: int = Field(default=0)
    filter_vocab: VocabFilter = Field(default_factory=VocabFilter)
    _token_to_id: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def token_to_id(self) -> dict[str, int]:
        if not self._token_to_id:
            self._token_to_id = {
                token_str: token_id
                for token_id, token_str in self.clean_vocab.items()
            }
        return self._token_to_id

    @classmethod
    def from_model(cls, model: Small_LLM_Model) -> "VocabIndex":

        vocab_path = model.get_path_to_vocab_file()
        vocab = cls._load_vocab(vocab_path)
        clean_vocab = cls._build_clean_vocab(vocab, model)

        filter_vocab = VocabFilter.from_clean_vocab(clean_vocab)

        return cls(
            vocab_path=vocab_path,
            vocab=vocab,
            clean_vocab=clean_vocab,
            size=len(vocab),
            filter_vocab=filter_vocab
        )

    def get_literal_matches(self, remainder: str) -> set[int]:
        return (
            self.filter_vocab.get_literal_matches(remainder, self.clean_vocab)
        )

    @staticmethod
    def _build_clean_vocab(
            vocab: dict[str, int], model: Small_LLM_Model) -> dict[int, str]:

        clean_dict = {}
        for _, token_id in vocab.items():
            clean_str = model.decode([token_id])
            clean_dict[token_id] = clean_str
        return clean_dict

    @staticmethod
    def _load_vocab(file_path: str) -> dict[str, int]:
        try:
            with open(file_path, 'r') as file_vocab:
                data = json.load(file_vocab)
                if not isinstance(data, dict):
                    sys.exit(f"Error: {file_path} must contain a JSON object.")
                return data

        except PermissionError:
            sys.exit("Error: Does not have right permission")
        except FileNotFoundError:
            sys.exit(f"Error: File '{file_path}' not found.")
        except json.JSONDecodeError as e:
            sys.exit(f"Error: '{file_path}' is not valid JSON. {e.msg}")
        except ValidationError as e:
            sys.exit(f"Error: Data validation failed '{file_path}'"
                     f".\nDetails:\n{e}")
