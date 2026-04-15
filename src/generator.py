import numpy as np
from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from src.vocabulary import VocabIndex
from src.state_machine import (
        State,
        StateTerminal,
        StateExpectLiteral
)


class ConstrainedGenerator(BaseModel):
    """Text generator applying constraints via a state machine.

    Attributes:
        llm (Small_LLM_Model): Instance of the language model used.
        vocab_index (VocabIndex): Vocabulary index and associated filters.
        machine (JsonStateMachine): State machine guiding generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Small_LLM_Model
    vocab_index: VocabIndex
    current_state: State

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Produce a completion respecting the current machine state.

        Args:
            prompt: Input text serving as context for the model.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            str: The generated and validated string.

        Raises:
            ValueError: If no valid token is found or if the token
                budget is exceeded before reaching a terminal state.
        """
        input_ids = self.llm.encode(prompt)[0].tolist()
        generated_text = ""
        token_count = 0

        while not self._should_stop(token_count, max_tokens):

            if isinstance(self.current_state, StateExpectLiteral):
                generated_text = self._add_literal(
                    self.current_state, generated_text)
                input_ids = self.llm.encode(
                    prompt + generated_text)[0].tolist()

            else:
                new_token = self._select_next_token(input_ids)
                if new_token is not None:
                    generated_text += new_token
                    input_ids.append(self._get_token_id(new_token))
                    self._step(new_token)
                    token_count += 1

        if token_count >= max_tokens:
            raise ValueError("Error: too many tokens generated")

        return generated_text

    def _should_stop(self, token_count: int, max_tokens: int) -> bool:
        """Return whether generation should stop."""
        return (isinstance(self.current_state, StateTerminal) or
                token_count >= max_tokens)

    def _add_literal(
            self, state: StateExpectLiteral, generated_text: str) -> str:
        """Add the fixed text fragment required by the current state.

        Args:
            state: The literal state currently being processed.
            generated_text: The text already produced by the generator.

        Returns:
            str: Updated text including the expected literal fragment.
        """

        expected = state.expected
        buffer_len = len(state.buffer)
        literal_to_add = expected[buffer_len:]
        generated_text += literal_to_add
        self.current_state = state.next_state or StateTerminal()

        return generated_text

    def _select_next_token(self, input_ids: list[int]) -> str | None:
        """Choose the next valid token string from the constrained set.

        Args:
            input_ids: Tokenized prompt plus previously generated tokens.

        Returns:
            str | None: The next token string, or `None` if no token can be
            selected.

        Raises:
            ValueError: If no valid token is available or selection fails.
        """
        try:
            valid_tokens = self.current_state.get_valid_tokens(
                self.vocab_index.clean_vocab,
                self.vocab_index.filter_vocab
            )

            if not valid_tokens:
                raise ValueError("No valid tokens available")

            token_id = self._choose_best_token(valid_tokens, input_ids)
            return self.vocab_index.clean_vocab[token_id]

        except Exception as e:
            raise ValueError(f"Error selecting token: {e}") from e

    def _choose_best_token(
            self, valid_tokens: set[int], input_ids: list[int]) -> int:
        """Select the highest-scoring token among valid candidates.

        Args:
            valid_tokens: Candidate token ids allowed by the state machine.
            input_ids: Tokenized prompt plus previously generated tokens.

        Returns:
            int: The chosen token id.
        """

        if len(valid_tokens) == 1:
            return next(iter(valid_tokens))

        logits = np.array(
            self.llm.get_logits_from_input_ids(input_ids), dtype=np.float32)

        valid_ids = np.array(list(valid_tokens), dtype=np.int32)
        scores = logits[valid_ids]

        return int(valid_ids[np.argmax(scores)])

    def _get_token_id(self, token_str: str) -> int:
        """Return the token id for a decoded token string."""
        return self.vocab_index.token_to_id.get(token_str, -1)

    def _step(self, token_str: str) -> None:
        """Advance the machine by consuming a token fragment."""
        overflow = token_str
        while overflow and not isinstance(self.current_state, StateTerminal):
            self.current_state, overflow = (
                self.current_state.transition(overflow))
