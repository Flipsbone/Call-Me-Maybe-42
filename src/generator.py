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
                         allowed_strings: list[str]) -> tuple[list[int], str]:

        current_ids = self._ensure_flat_list(input_ids)
        allowed_sequences = [
            self._ensure_flat_list(self.llm.encode(s))
            for s in allowed_strings
            ]
        generated_tokens: list[int] = []

        keep_going = True
        while keep_going:
            last_logits = self._get_last_logits(current_ids)
            masked_logits = np.full_like(last_logits, -np.inf)

            valid_next_tokens: set[int] = set()
            for seq in allowed_sequences:
                # Si le début correspond à ce qu'on a déjà généré...
                if seq[:len(generated_tokens)] == generated_tokens:
                    if len(seq) > len(generated_tokens):
                        valid_next_tokens.add(seq[len(generated_tokens)])

            if not valid_next_tokens:
                keep_going = False
            else:
                for t_idx in valid_next_tokens:
                    if t_idx < len(masked_logits):
                        masked_logits[t_idx] = last_logits[t_idx]

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
        """
        Laisse l'IA générer une valeur en respectant les
        tokens du bon type et des caractères d'arrêt dynamiques.
        """
        current_ids = self._ensure_flat_list(input_ids)
        generated_tokens: list[int] = []
        # Si aucun stop_chars n'est fourni, on utilise ceux par défaut
        if stop_chars is None:
            stop_chars = [",", "}", "\n", " "]
        stop_tokens = set()
        for char in stop_chars:
            encoded = self._ensure_flat_list(self.llm.encode(char))
            for t in encoded:
                stop_tokens.add(t)

        is_active = True
        while is_active:
            last_logits = self._get_last_logits(current_ids)
            masked_logits = np.full_like(last_logits, -np.inf)

            all_valid = allowed_set | stop_tokens

            for t_idx in all_valid:
                if 0 <= t_idx < len(last_logits):
                    masked_logits[t_idx] = last_logits[t_idx]

            next_token = int(np.argmax(masked_logits))

            if next_token in stop_tokens:
                is_active = False
            else:
                generated_tokens.append(next_token)
                current_ids.append(next_token)

        final_val = (self.llm.decode(generated_tokens)
                     if hasattr(self.llm, 'decode') else "")

        return current_ids, json.dumps(str(final_val).strip())
