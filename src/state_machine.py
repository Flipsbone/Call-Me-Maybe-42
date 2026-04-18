import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabIndex

WS = r'[ \n\r\t]*'


class JSONValidator:
    """Utility to validate and extract JSON numbers."""

    REGEX_PARTIAL_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
    REGEX_PREFIX_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')

    @staticmethod
    def is_partial_number(text: str) -> bool:
        """Check if text is an incomplete but valid JSON number."""
        return bool(JSONValidator.REGEX_PARTIAL_NUMBER.fullmatch(text))

    @staticmethod
    def extract_complete_number(text: str) -> tuple[str, str]:
        """Extract a complete number. Returns (number, remainder)."""
        match = JSONValidator.REGEX_PREFIX_NUMBER.match(text)
        if match:
            matched_text = match.group()
            return matched_text, text[len(matched_text):]
        return "", text


class State(BaseModel, ABC):
    """Base state class for the state machine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(
            self, vocab_index: VocabIndex
    ) -> set[int]:
        """Return the set of valid token IDs for this state."""
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        """Transition to the next state. Returns (next_state, remainder)."""
        pass


class StateTerminal(State):
    """Terminal state - generation is complete."""

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""


class StateExpectLiteral(State):
    """Expect an exact literal string to be generated."""

    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str

        if self.buffer.startswith(self.expected):
            remain_str = self.buffer[len(self.expected):]
            next_state = self.next_state or StateTerminal()
            return next_state, remain_str

        return self, ""


class StateBranch(State):
    """Choose between multiple possible branches."""

    choices: dict[str, State] = Field(...)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        valid_ids: set[int] = set()

        # Find choices that can continue from the current buffer
        for choice in self.choices.keys():
            if not choice.startswith(self.buffer):
                continue

            # Add tokens that extend this choice
            remainder = choice[len(self.buffer):]
            if remainder:
                valid_ids.update(vocab_index.get_literal_matches(remainder))

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str

        # Find the first matching choice
        for choice, next_state in self.choices.items():
            if self.buffer.startswith(choice):
                remain_str = self.buffer[len(choice):]
                return next_state, remain_str

        return self, ""


class StateParseNumber(State):
    """Parse a JSON number."""

    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        valid_ids: set[int] = set()

        expected_next_text = getattr(self.next_state, 'expected', '')

        for token_id in vocab_index.filter_vocab.numeric_tokens:
            token_str = vocab_index.clean_vocab[token_id]
            simulated_text = self.buffer + token_str

            # If it's a valid incomplete number, accept it
            if JSONValidator.is_partial_number(simulated_text):
                valid_ids.add(token_id)
                continue

            # Check if we can complete the number
            matched_text, remain_str = JSONValidator.extract_complete_number(
                simulated_text)

            if matched_text and (not remain_str or
                                 expected_next_text.startswith(remain_str)):
                valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        matched_text, remain_str = JSONValidator.extract_complete_number(
            self.buffer)

        # Complete number detected and valid separator follows
        if matched_text and remain_str and remain_str[0] in ',}]\n':
            next_state = self.next_state or StateTerminal()
            return next_state, remain_str

        return self, ""


class StateParseString(State):
    """Parse a JSON string - simplified version."""

    next_state: State | None = Field(default=None)
    has_opened: bool = Field(default=False)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:

        if not self.has_opened:
            return vocab_index.filter_vocab.exact_quote_tokens
        return (vocab_index.filter_vocab.string_content_tokens |
                vocab_index.filter_vocab.string_closer_tokens)

    def transition(self, token_str: str) -> tuple["State", str]:
        if not self.has_opened:
            if '"' in token_str:
                self.has_opened = True
            return self, ""

        if '"' in token_str:
            idx = token_str.find('"')
            next_state = self.next_state or StateTerminal()
            return next_state, token_str[idx + 1:]

        self.buffer += token_str
        return self, ""
