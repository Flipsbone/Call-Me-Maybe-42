import sys
import json
from pydantic import RootModel, ValidationError
from llm_sdk import Small_LLM_Model


class VocabSchema(RootModel[dict[str, int]]):
    pass


class VocabularyIndex:
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model: Small_LLM_Model = model

        self.vocab_path: str = self.model.get_path_to_vocab_file()
        self.vocab: dict[str, int] = self._load_vocab(self.vocab_path)
        self.clean_vocab: dict[int, str] = self._build_clean_vocab(self.vocab)
        self.size: int = len(self.vocab)

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
