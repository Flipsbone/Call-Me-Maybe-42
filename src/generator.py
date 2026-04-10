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

        while not self._should_stop(token_count, max_tokens):
            current_state = self.machine.current_state

            if isinstance(current_state, StateExpectLiteral):

                generated_text = self._add_literal(
                    current_state, generated_text)
                input_ids = self.llm.encode(
                    prompt + generated_text)[0].tolist()

            else:
                new_token = self._select_next_token(input_ids)
                if new_token is not None:
                    generated_text += new_token
                    input_ids.append(self._get_token_id(new_token))
                    self.machine.step(new_token)
                    token_count += 1

        if token_count >= max_tokens:
            raise ValueError("Error: too many tokens generated")

        return generated_text

    def _should_stop(self, token_count: int, max_tokens: int) -> bool:
        return (isinstance(self.machine.current_state, StateTerminal) or
                token_count >= max_tokens)

    def _add_literal(
            self, state: StateExpectLiteral, generated_text: str) -> str:

        expected = state.expected
        buffer_len = len(state.buffer)
        literal_to_add = expected[buffer_len:]
        generated_text += literal_to_add
        self.machine.current_state = state.next_state or StateTerminal()

        return generated_text

    def _select_next_token(self, input_ids: list[int]) -> str | None:
        try:
            valid_tokens = self.machine.get_valid_tokens(
                self.vocab_index.clean_vocab,
                self.vocab_index.filter_vocab
            )

            if not valid_tokens:
                sys.exit("Error No token available")

            token_id = self._choose_best_token(valid_tokens, input_ids)
            return self.vocab_index.clean_vocab[token_id]

        except Exception as e:
            sys.exit(f"Error {e}")

    def _choose_best_token(
            self, valid_tokens: set[int], input_ids: list[int]) -> int:

        if len(valid_tokens) == 1:
            return next(iter(valid_tokens))

        logits = np.array(
            self.llm.get_logits_from_input_ids(input_ids), dtype=np.float32)

        valid_ids = np.array(list(valid_tokens), dtype=np.int32)
        scores = logits[valid_ids]

        return int(valid_ids[np.argmax(scores)])

    def _get_token_id(self, token_str: str) -> int:
        for token_id, token in self.vocab_index.clean_vocab.items():
            if token == token_str:
                return token_id
        return -1
