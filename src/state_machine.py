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
            remain_str = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str

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
                remain_str = self.buffer[len(choice):]
                return next_s, remain_str
        return self, ""


class StateParseNumber(State):
    """State incrementally validating a number in JSON format.

    Attributes:
        next_state (State | None): State to transition to after the
            number is fully parsed.
    """

    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        """Return token ids that keep the buffered text a valid number."""
        valid_ids: set[int] = set()
        expected_next_text = getattr(self.next_state, 'expected', '')

        for token_id in vocab_filter.numeric_tokens:
            simulated_text = self.buffer + clean_vocab[token_id]

            if REGEX_PARTIAL_NUMBER.fullmatch(simulated_text):
                valid_ids.add(token_id)
            else:
                match = REGEX_PREFIX_NUMBER.match(simulated_text)
                if match:
                    matched_text = match.group()
                    remain_str = simulated_text[len(matched_text):]
                    if (not remain_str or
                            expected_next_text.startswith(remain_str)):
                        valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        match = REGEX_PREFIX_NUMBER.match(self.buffer)
        if match:
            matched_text = match.group()
            remain_str = self.buffer[len(matched_text):]
            if remain_str and remain_str[0] in ',}]\n':
                next_s = (
                    self.next_state if self.next_state else StateTerminal())
                return next_s, remain_str
        return self, ""


class StateParseString(State):
    """State that incrementally validates a JSON string literal."""

    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:

        """Return token ids that keep the buffered text a valid string."""
        expected_next_text = getattr(self.next_state, 'expected', '')
        match = REGEX_PREFIX_STRING.match(self.buffer)

        if match:
            matched_text = match.group()
            return self._handle_closed_string(
                matched_text=matched_text,
                expected_next=expected_next_text,
                clean_vocab=clean_vocab,
                vocab_filter=vocab_filter
            )

        return self._handle_open_string(
            expected_next=expected_next_text,
            clean_vocab=clean_vocab,
            vocab_filter=vocab_filter
        )

    def _handle_closed_string(
            self, matched_text: str, expected_next: str,
            clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        """Handle a string that is already closed
            and may expose remaining text."""

        valid_ids: set[int] = set()
        remain_str = self.buffer[len(matched_text):]

        if remain_str and not expected_next.startswith(remain_str):
            return valid_ids

        remaining_expected = expected_next[len(remain_str):]
        if remaining_expected:
            valid_ids.update(vocab_filter.get_literal_matches(
                remaining_expected, clean_vocab))

        return valid_ids

    def _handle_open_string(
            self, expected_next: str, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        """Handle a string that is still open and must remain valid JSON."""
        valid_ids: set[int] = set()

        if self.buffer.lstrip().startswith('"'):
            valid_ids.update(vocab_filter.string_safe_tokens)

        for token_id in vocab_filter.string_unsafe_tokens:
            simulated_text = self.buffer + clean_vocab[token_id]

            if REGEX_PARTIAL_STRING.fullmatch(simulated_text):
                valid_ids.add(token_id)
            else:
                match = REGEX_PREFIX_STRING.match(simulated_text)
                if match:
                    matched_text = match.group()
                    remain_str = simulated_text[len(matched_text):]

                    if not remain_str or expected_next.startswith(remain_str):
                        valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        match = REGEX_PREFIX_STRING.match(self.buffer)
        if match:
            matched_text = match.group()
            remain_str = self.buffer[len(matched_text):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str
        return self, ""
