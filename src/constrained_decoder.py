import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import State, StateTerminal, StateExpectLiteral


class ConstrainedDecoder(BaseModel):
    """Engine that forces the LLM to generate
    text matching a specific state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex
    current_state: State

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Produce a completion by guiding the LLM token by token."""
        input_ids = self.llm.encode(prompt)[0].tolist()
        generated_text = ""

        for _ in range(max_tokens):

            if isinstance(self.current_state, StateTerminal):
                return generated_text

            if isinstance(self.current_state, StateExpectLiteral):
                literal_to_add = self.current_state.expected[
                    len(self.current_state.buffer):]
                generated_text += literal_to_add
                new_ids = self.llm.encode(literal_to_add)[0].tolist()
                input_ids.extend(new_ids)

                self.current_state = (
                    self.current_state.next_state or StateTerminal())

            else:
                new_token = self._select_next_token(input_ids)
                generated_text += new_token
                input_ids.append(
                    self.vocab_index.token_to_id.get(new_token, -1))
                self._update_state_machine(new_token)

        raise ValueError("Generation error: Token limit exceeded.")

    def _select_next_token(self, input_ids: list[int]) -> str:
        """Get allowed tokens and let the LLM pick the best one."""
        valid_tokens = self.current_state.get_valid_tokens(
            self.vocab_index.clean_vocab,
            self.vocab_index.filter_vocab
        )

        if not valid_tokens:
            raise ValueError(
                "Generation blocked: No valid tokens available to continue.")

        if len(valid_tokens) == 1:
            best_token_id = next(iter(valid_tokens))

        else:
            logits = np.array(
                self.llm.get_logits_from_input_ids(
                    input_ids), dtype=np.float32)

            valid_ids = np.array(list(valid_tokens), dtype=np.int32)
            scores = logits[valid_ids]
            best_token_id = int(valid_ids[np.argmax(scores)])

        return self.vocab_index.clean_vocab[best_token_id]

    def _update_state_machine(self, token_str: str) -> None:
        """Feed the generated token into
        the machine to move to the next state."""
        overflow = token_str
        while overflow and not isinstance(self.current_state, StateTerminal):
            self.current_state, overflow = (
                self.current_state.transition(overflow))
