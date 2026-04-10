import sys
import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import (
                                JsonStateMachine,
                                StateTerminal,
                                StateExpectLiteral
                            )


class ConstrainedGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex
    machine: JsonStateMachine

    def generate(self, prompt: str, max_tokens: int = 150) -> str:

        input_ids = self.llm.encode(prompt)[0].tolist()
        generated_text = ""
        token_count = 0

        while not isinstance(self.machine.current_state, StateTerminal):
            if token_count >= max_tokens:
                print("\n[Warning] Safety stop: generated too many tokens.",
                      file=sys.stderr)
                self.machine.current_state = StateTerminal()
                continue

            if isinstance(self.machine.current_state, StateExpectLiteral):
                expected = self.machine.current_state.expected
                buffer_len = len(self.machine.current_state.buffer)
                literal_to_add = expected[buffer_len:]

                if literal_to_add:
                    generated_text += literal_to_add

                self.machine.current_state = (
                    self.machine.current_state.next_state
                    if self.machine.current_state.next_state
                    else StateTerminal()
                )

                input_ids = (
                    self.llm.encode(prompt + generated_text)[0].tolist()
                    )
                continue

            try:
                valid_tokens = self.machine.get_valid_tokens(
                    self.vocab_index.clean_vocab,
                    self.vocab_index.filter_vocab
                )

                if not valid_tokens:
                    print("Error: entire vocab rejected", file=sys.stderr)
                    self.machine.current_state = StateTerminal()
                    continue

                if len(valid_tokens) == 1:
                    next_token_id = list(valid_tokens)[0]
                else:
                    logits_raw = self.llm.get_logits_from_input_ids(input_ids)
                    logits_np = np.array(logits_raw, dtype=np.float32)

                    valid_arr = np.array(list(valid_tokens), dtype=np.int32)
                    scores = logits_np[valid_arr]
                    next_token_id = int(valid_arr[np.argmax(scores)])

                token_str = self.vocab_index.clean_vocab[next_token_id]
                generated_text += token_str
                input_ids.append(next_token_id)
                self.machine.step(token_str)

                token_count += 1

            except Exception as e:
                print(f"Critical error: {e}", file=sys.stderr)
                self.machine.current_state = StateTerminal()
                continue

        return generated_text
