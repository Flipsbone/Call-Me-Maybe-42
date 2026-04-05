import sys
import json
import re
from typing import Any, cast
from pydantic import BaseModel, Field, ValidationError
from llm_sdk import Small_LLM_Model
from model_pydantic import VocabSchema


class VocabularyPruner(BaseModel):
    numeric_tokens: list[int] = Field(default_factory=list)

    def build(self, clean_vocab: dict[int, str]) -> None:
        is_num_chunk = re.compile(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$')
        for token_id, token_str in clean_vocab.items():
            if is_num_chunk.match(token_str):
                self.numeric_tokens.append(token_id)


class VocabularyIndex(BaseModel):
    model: Small_LLM_Model
    vocab_path: str = Field(default="")
    vocab: dict[str, int] = Field(default_factory=dict)
    clean_vocab: dict[int, str] = Field(default_factory=dict)
    size: int = Field(default=0)
    pruner: VocabularyPruner = Field(default_factory=VocabularyPruner)

    def model_post_init(self, __context: Any) -> None:
        self.vocab_path = self.model.get_path_to_vocab_file()
        self.vocab = self._load_vocab(self.vocab_path)
        self.clean_vocab = self._build_clean_vocab(self.vocab)
        self.size = len(self.vocab)
        self.pruner.build(self.clean_vocab)

    def _build_clean_vocab(self, vocab: dict[str, int]) -> dict[int, str]:
        clean_dict = {}
        for token_str, token_id in vocab.items():
            clean_str = self.model.decode([token_id])
            clean_dict[token_id] = clean_str
        return clean_dict

    def _load_vocab(self, file_path: str) -> dict[str, int]:
        try:
            with open(file_path, 'r') as file_vocab:
                raw_data = json.load(file_vocab)
                validated_data = VocabSchema.model_validate(raw_data)
            return cast(dict[str, int], validated_data.root)

        except FileNotFoundError:
            print(f"Error: Vocabulary file '{file_path}' not found.",
                  file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Vocabulary file is not valid JSON. {e}",
                  file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            print("Error: Vocabulary file structure is invalid"
                  f"(Pydantic). {e}",
                  file=sys.stderr)
            sys.exit(1)
