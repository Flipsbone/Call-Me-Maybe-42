import sys
import json
import re
from pydantic import Field, ValidationError
from llm_sdk import Small_LLM_Model

from src.model_pydantic import (VocabSchema,
                                VocabularyIndexSchema,
                                VocabularyFilterSchema)


class VocabularyFilter(VocabularyFilterSchema):

    def build(self, clean_vocab: dict[int, str]) -> None:
        is_num_chunk = re.compile(r'^[ \n\r\t]*-?[\d.eE+-]*[ \n\r\t,}\]]*$')
        for token_id, token_str in clean_vocab.items():
            if is_num_chunk.match(token_str):
                self.numeric_tokens.append(token_id)
            if '"' not in token_str and '\\' not in token_str:
                self.string_safe_tokens.append(token_id)
            else:
                self.string_unsafe_tokens.append(token_id)

    def get_literal_matches(
                            self,
                            remainder: str,
                            clean_vocab: dict[int, str]
                            ) -> set[int]:

        if remainder not in self.literal_cache:
            self.literal_cache[remainder] = {
                token_id for token_id, token_str in clean_vocab.items()
                if (
                    remainder.startswith(token_str)
                    or token_str.startswith(remainder)
                    )
            }
        return self.literal_cache[remainder]


class VocabularyIndex(VocabularyIndexSchema):
    pruner: VocabularyFilter = Field(default_factory=VocabularyFilter)

    def build_from_model(self, model: "Small_LLM_Model") -> None:
        self.vocab_path = model.get_path_to_vocab_file()
        self.vocab = self._load_vocab(self.vocab_path)
        self.clean_vocab = self._build_clean_vocab(self.vocab, model)
        self.size = len(self.vocab)

        self.pruner.build(self.clean_vocab)

    def _build_clean_vocab(
                            self,
                            vocab: dict[str, int],
                            model: "Small_LLM_Model"
                            ) -> dict[int, str]:

        clean_dict = {}
        for token_str, token_id in vocab.items():
            clean_str = model.decode([token_id])
            clean_dict[token_id] = clean_str
        return clean_dict

    def _load_vocab(self, file_path: str) -> dict[str, int]:
        try:
            with open(file_path, 'r') as file_vocab:
                raw_data = json.load(file_vocab)
                validated_data = VocabSchema.model_validate(raw_data)
            return validated_data.root

        except PermissionError:
            print("Error: Permission denied", file=sys.stderr)
            sys.exit(1)
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
