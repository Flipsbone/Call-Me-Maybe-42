import sys
import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabularyIndex
from src.state_machine import JsonStateMachine, StateTerminal


class ConstrainedGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabularyIndex
    machine: JsonStateMachine

    def _ensure_flat_list(self, data: list | np.ndarray) -> list[int]:
        arr = np.array(data).flatten()
        return [int(x) for x in arr.tolist()]

    def _get_last_logits(self, input_ids: list[int]) -> np.ndarray:
        clean_input = self._ensure_flat_list(input_ids)
        logits_raw = self.llm.get_logits_from_input_ids(clean_input)
        return np.array(logits_raw)

    def _apply_mask(self, logits: np.ndarray, valid_tokens: set[int]) -> np.ndarray:
        valid_indices = np.array(list(valid_tokens), dtype=np.int32)
        masked_logits = np.full_like(logits, -np.inf)
        masked_logits[valid_indices] = logits[valid_indices]
        return masked_logits

    def generate(self, prompt: str) -> str:
        input_ids = self._ensure_flat_list(self.llm.encode(prompt))
        generated_text = ""

        while not isinstance(self.machine.current_state, StateTerminal):
            try:
                last_logits = self._get_last_logits(input_ids)
                
                filtered_logits_list = self.machine.apply_constraints(
                    logits=last_logits.tolist(),
                    clean_vocab=self.vocab_index.clean_vocab,
                    pruner=self.vocab_index.pruner
                )
                
                valid_tokens = {
                    i for i, val in enumerate(filtered_logits_list) 
                    if val != float('-inf')
                }
                
                masked_logits = self._apply_mask(last_logits, valid_tokens)
                next_token_id = int(np.argmax(masked_logits))
                
                if masked_logits[next_token_id] == float('-inf'):
                    print("Error : The state machine rejected the entire vocabulary.", file=sys.stderr)
                    self.machine.current_state = StateTerminal()
                    continue
                    
                token_str = self.vocab_index.clean_vocab[next_token_id]
                generated_text += token_str
                input_ids.append(next_token_id)
                
                self.machine.step(token_str)
                
            except Exception as e:
                print(f"Critical error during generation: {e}", file=sys.stderr)
                self.machine.current_state = StateTerminal()

        return generated_text