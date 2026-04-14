import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabFilter

WS = r'[ \n\r\t]*'

REGEX_PARTIAL_STRING = re.compile(fr'^{WS}"([^"\\]|\\.)*$')
REGEX_PREFIX_STRING = re.compile(fr'^{WS}"([^"\\]|\\.)*"')

REGEX_PARTIAL_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
REGEX_PREFIX_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')


class State(BaseModel, ABC):
    """Abstract base class for constrained decoding states.

    Attributes:
        buffer (str): Temporary text accumulator for the current state.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Return token ids that are valid for the current buffer."""
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume a token fragment and determine the next state.

        Args:
            token_str: Raw text fragment produced by the model.

        Returns:
            tuple[State, str]: A tuple containing the new state
                (or itself) and the overflow text not consumed.
        """
        pass


class StateTerminal(State):
    """Terminal state indicating that generation is complete."""

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        """Terminal states accept no further tokens."""
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Remain terminal and discard any incoming token fragment."""
        return self, ""


class StateExpectLiteral(State):
    """State that requires a fixed literal prefix or suffix."""

    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Return tokens that can continue or complete the expected text."""

        if not self.expected.startswith(self.buffer):
            return set()

        remainder = self.expected[len(self.buffer):]
        if not remainder:
            return set()

        return vocab_filter.get_literal_matches(remainder, clean_vocab)

    def transition(self, token_str: str) -> tuple["State", str]:
        """Advance the literal buffer and return any overflow text."""
        self.buffer += token_str
        if self.buffer.startswith(self.expected):
            overflow = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class StateBranch(State):
    """State that accepts one of several literal choices."""

    choices: dict[str, State] = Field(...)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Return tokens that can extend any currently matching branch."""

        valid_ids: set[int] = set()

        for choice in self.choices.keys():
            if choice.startswith(self.buffer):
                remainder = choice[len(self.buffer):]
                if remainder:
                    valid_ids.update(
                        vocab_filter.get_literal_matches(
                            remainder, clean_vocab))

        return valid_ids

    def transition(self, token_str: str) -> tuple[State, str]:
        """Advance the branch buffer and move to the matching next state."""
        self.buffer += token_str
        for choice, next_s in self.choices.items():
            if self.buffer.startswith(choice):
                overflow = self.buffer[len(choice):]
                return next_s, overflow
        return self, ""


class StateParseNumber(State):
    """State incrementally validating a number in JSON format.

    Attributes:
        next_state (State | None): State to transition to after the
            number is fully parsed.
    """

    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Return token ids that keep the buffered text a valid number."""

        valid_ids: set[int] = set()

        test_fullmatch = REGEX_PARTIAL_NUMBER.fullmatch
        test_prefix = REGEX_PREFIX_NUMBER.match
        buf = self.buffer
        expected_next = getattr(self.next_state, 'expected', '')

        for token_id in vocab_filter.numeric_tokens:
            test_str = buf + clean_vocab[token_id]

            if test_fullmatch(test_str):
                valid_ids.add(token_id)
            else:
                match = test_prefix(test_str)
                if match:
                    overflow = test_str[match.end():]
                    if not overflow:
                        valid_ids.add(token_id)
                    elif expected_next.startswith(overflow):
                        valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume numeric text and advance once a delimiter is found."""
        self.buffer += token_str
        match = REGEX_PREFIX_NUMBER.match(self.buffer)
        if match:
            num_part = match.group()
            whitespace_len = len(self.buffer) - len(self.buffer.lstrip())
            overflow = self.buffer[whitespace_len + len(num_part):]
            if overflow and overflow[0] in ',}]\n':
                next_s = (
                    self.next_state if self.next_state else StateTerminal())
                return next_s, overflow
        return self, ""


class StateParseString(State):
    """State that incrementally validates a JSON string literal."""

    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Return token ids that keep the buffered text a valid string."""

        expected_next = getattr(self.next_state, 'expected', '')
        buffer_match = REGEX_PREFIX_STRING.match(self.buffer)

        if buffer_match:
            return self._handle_closed_string(
                match_end=buffer_match.end(),
                expected_next=expected_next,
                clean_vocab=clean_vocab,
                vocab_filter=vocab_filter
            )

        return self._handle_open_string(
            expected_next=expected_next,
            clean_vocab=clean_vocab,
            vocab_filter=vocab_filter
        )

    def _handle_closed_string(
            self, match_end: int, expected_next: str,
            clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Handle a string that is already closed and may expose overflow."""

        valid_ids: set[int] = set()
        overflow_len = len(self.buffer) - match_end

        if overflow_len > 0 and not expected_next.startswith(
                self.buffer[match_end:]):
            return valid_ids

        remaining_expected = expected_next[overflow_len:]
        if remaining_expected:
            valid_ids.update(vocab_filter.get_literal_matches(
                remaining_expected, clean_vocab))

        return valid_ids

    def _handle_open_string(
            self, expected_next: str, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        """Handle a string that is still open and must remain valid JSON."""

        valid_ids: set[int] = set()

        has_opening_quote = self.buffer.lstrip().startswith('"')
        if has_opening_quote:
            valid_ids.update(vocab_filter.string_safe_tokens)

        test_fullmatch = REGEX_PARTIAL_STRING.fullmatch
        test_prefix = REGEX_PREFIX_STRING.match
        buf = self.buffer

        for token_id in vocab_filter.string_unsafe_tokens:
            test_str = buf + clean_vocab[token_id]

            if test_fullmatch(test_str):
                valid_ids.add(token_id)
            else:
                match = test_prefix(test_str)
                if match:
                    overflow = test_str[match.end():]
                    if not overflow or expected_next.startswith(overflow):
                        valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume string text and advance once a full string is matched."""
        self.buffer += token_str
        match = REGEX_PREFIX_STRING.match(self.buffer)
        if match:
            string_part = match.group()
            overflow = self.buffer[len(string_part):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class JsonStateMachine(BaseModel):
    """Wrapper around the current constrained-decoding state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_state: State

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        """Delegate valid-token discovery to the active state."""

        return self.current_state.get_valid_tokens(clean_vocab, vocab_filter)

    def step(self, token_str: str) -> None:
        """Advance the machine by consuming a token fragment."""
        overflow = token_str
        state = self.current_state
        while overflow and not isinstance(state, StateTerminal):
            state, overflow = state.transition(overflow)
            self.current_state = state
