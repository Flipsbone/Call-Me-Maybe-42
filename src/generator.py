import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model


class ConstrainedGenerator:
    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm

    def _ensure_flat_list(self, data: Any) -> list[int]:
        arr = np.array(data).flatten()
        return [int(x) for x in arr.tolist()]

    def _get_last_logits(self, input_ids: list[int]) -> np.ndarray:
        clean_input = self._ensure_flat_list(input_ids)
        logits_raw = self.llm.get_logits_from_input_ids(clean_input)
        logits_np = np.array(logits_raw)
        return logits_np

    def _apply_mask(self,
                    logits: np.ndarray,
                    valid_tokens: set[int]
                    ) -> np.ndarray:

        masked_logits = np.full_like(logits, -np.inf)
        for token_idx in valid_tokens:
            if 0 <= token_idx < len(logits):
                masked_logits[token_idx] = logits[token_idx]
        return masked_logits

    def _generate_with_constraints(
        self,
        input_ids: list[int],
        valid_tokens: set[int],
        stop_tokens: set[int] | None = None
    ) -> tuple[list[int], list[int]]:
        if stop_tokens is None:
            stop_tokens = set()

        current_ids = self._ensure_flat_list(input_ids)
        generated_tokens: list[int] = []

        is_generating = True
        while is_generating:
            last_logits = self._get_last_logits(current_ids)
            allowed = valid_tokens | stop_tokens
            masked_logits = self._apply_mask(last_logits, allowed)
            next_token = int(np.argmax(masked_logits))
            if next_token in stop_tokens:
                is_generating = False
            else:
                generated_tokens.append(next_token)
                current_ids.append(next_token)

        return current_ids, generated_tokens

    def force_sequence(self,
                       input_ids: list[int],
                       text_to_force: str) -> list[int]:

        current_ids = self._ensure_flat_list(input_ids)

        tokens_to_force = self._ensure_flat_list(
            self.llm.encode(text_to_force))

        current_ids.extend(tokens_to_force)
        return current_ids

    def choose_from_list(self,
                         input_ids: list[int],
                         allowed_strings: list[str]
                         ) -> tuple[list[int], str]:

        allowed_sequences = [self._ensure_flat_list(self.llm.encode(s))
                             for s in allowed_strings]

        current_ids = self._ensure_flat_list(input_ids)
        generated_tokens: list[int] = []

        is_active = True
        while is_active:
            valid_next_tokens: set[int] = set()
            for seq in allowed_sequences:
                if seq[:len(generated_tokens)] == generated_tokens:
                    if len(seq) > len(generated_tokens):
                        valid_next_tokens.add(seq[len(generated_tokens)])

            if not valid_next_tokens:
                is_active = False

            last_logits = self._get_last_logits(current_ids)
            masked_logits = self._apply_mask(last_logits, valid_next_tokens)

            next_token = int(np.argmax(masked_logits))
            generated_tokens.append(next_token)
            current_ids.append(next_token)

        chosen_text = (self.llm.decode(generated_tokens)
                       if hasattr(self.llm, 'decode') else "")

        return current_ids, str(chosen_text)

    def generate_value_for_type(self,
                                input_ids: list[int],
                                allowed_set: set[int],
                                stop_chars: list[str] | None = None
                                ) -> tuple[list[int], str]:
        if stop_chars is None:
            stop_chars = [",", "}", "\n", " "]
        stop_tokens = set()
        for char in stop_chars:
            encoded = self._ensure_flat_list(self.llm.encode(char))
            stop_tokens.update(encoded)

        current_ids, generated_tokens = self._generate_with_constraints(
            input_ids,
            valid_tokens=allowed_set,
            stop_tokens=stop_tokens
        )

        final_val = (self.llm.decode(generated_tokens)
                     if hasattr(self.llm, 'decode') else "")

        return current_ids, json.dumps(str(final_val).strip())
