import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import State, StateTerminal, StateExpectLiteral


class ConstrainedDecoder(BaseModel):
    """Engine that guides LLM generation using a provided state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex

    def generate(
            self, prompt: str, state: State, max_tokens: int = 150) -> str:
        """Produce a completion by following the rules of the given state.

        Args:
            Prompt: Initial text to prime the generation.
            State: Node state that defines valid token sequences.
            Max_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text that adheres to the constraints
            of the state machine.

        Raises:
            ValueError: If the state machine provides
            no valid tokens to continue.
        """
        input_ids: list[int] = self.llm.encode(prompt)[0].tolist()
        generated_text: str = ""
        current_state: State = state

        for _ in range(max_tokens):
            if isinstance(current_state, StateTerminal):
                return generated_text

            if isinstance(current_state, StateExpectLiteral):
                literal_to_add: str = current_state.expected[
                    len(current_state.buffer):]
                generated_text += literal_to_add

                new_ids: list[int] = (
                        self.llm.encode(literal_to_add)[0].tolist())
                input_ids.extend(new_ids)

                current_state = (
                    current_state.next_state or StateTerminal())
            else:
                token_id, new_token = self._select_next_token(
                    input_ids, current_state)

                current_state, remain_str = self._update_state_machine(
                    current_state, new_token)

                consumed_len = len(new_token) - len(remain_str)
                generated_text += new_token[:consumed_len]

                input_ids.append(token_id)

        return generated_text

    def _select_next_token(
            self, input_ids: list[int], state: State) -> tuple[int, str]:
        """Select the best next token among those allowed by the state.

        Args:
            input_ids: Token IDs already present in the model context.
            state: Current finite-state-machine node driving constraints.

        Returns:
            tuple[int, str]: The chosen token ID and its decoded string.

        Raises:
            ValueError: If the current state exposes no valid next tokens.
        """

        valid_tokens: set[int] = state.get_valid_tokens(self.vocab_index)

        if not valid_tokens:
            raise ValueError(
                "No valid tokens available to continue generation.")

        if len(valid_tokens) == 1:
            best_token_id: int = next(iter(valid_tokens))
        else:
            logits = np.array(
                self.llm.get_logits_from_input_ids(input_ids),
                dtype=np.float32
            )
            valid_ids = np.array(list(valid_tokens), dtype=np.int32)
            scores = logits[valid_ids]
            best_token_id = int(valid_ids[np.argmax(scores)])

        return best_token_id, self.vocab_index.clean_vocab[best_token_id]

    def _update_state_machine(
            self, state: State, new_token: str) -> tuple[State, str]:
        """Transition the state machine based on the generated token string.
        Args:
            state: Current state before consuming the new token.
            new_token: The raw string of the newly generated token.

        Returns:
            tuple[State, str]: The new state after consuming the token and any
            remaining string that was not consumed in the transition.
            """
        current_state: State = state
        remain_str: str = new_token

        while remain_str and not isinstance(current_state, StateTerminal):
            current_state, remain_str = current_state.transition(remain_str)

        return current_state, remain_str
