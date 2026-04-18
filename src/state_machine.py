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
        """Check whether text is a syntactically valid partial JSON number.

        Args:
            text: Candidate numeric prefix to validate.

        Returns:
            bool: True when the text can still become a valid JSON number.
        """
        return bool(JSONValidator.REGEX_PARTIAL_NUMBER.fullmatch(text))

    @staticmethod
    def extract_complete_number(text: str) -> tuple[str, str]:
        """Extract the longest complete number prefix from text.

        Args:
            text: Text beginning with potential numeric content.

        Returns:
            tuple[str, str]: Extracted number prefix and remaining suffix.
        """
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
        """Return token IDs that keep generation valid for this state.

        Args:
            vocab_index: Vocabulary metadata used to compute valid IDs.

        Returns:
            set[int]: Allowed next token IDs under this state's constraints.
        """
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume token text and compute the next parser state.

        Args:
            token_str: Generated token text to consume.

        Returns:
            tuple[State, str]: Next state and unconsumed remainder text.
        """
        pass


class StateTerminal(State):
    """Terminal state - generation is complete."""

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return no valid tokens because generation has finished.

        Args:
            vocab_index: Unused vocabulary index.

        Returns:
            set[int]: Always an empty set.
        """
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Stay terminal and discard any incoming token text.

        Args:
            token_str: Unused token text.

        Returns:
            tuple[State, str]: This state and an empty remainder.
        """
        return self, ""


class StateExpectLiteral(State):
    """Expect an exact literal string to be generated."""

    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, vocab_index: VocabIndex) -> set[int]:
        """Return an empty set because literals are short-circuited upstream.

        Args:
            vocab_index: Unused vocabulary index.

        Returns:
            set[int]: Always an empty set.
        """
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume literal text and jump when the expected string is matched.

        Args:
            token_str: Incoming token text appended to the internal buffer.

        Returns:
            tuple[State, str]: Next state with overflow text when literal is
            complete; otherwise this state and an empty remainder.
        """
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
        """Compute tokens that can continue at least one branch choice.

        Args:
            vocab_index: Vocabulary index used for literal token matching.

        Returns:
            set[int]: Token IDs that preserve a valid branch continuation.
        """
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
        """Consume token text and transition when a branch is completed.

        Args:
            token_str: Generated token text to append to branch buffer.

        Returns:
            tuple[State, str]: Matching branch target and remainder, or this
            state with an empty remainder if still incomplete.
        """
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
        """Compute token IDs that keep the number parse valid.

        Args:
            vocab_index: Vocabulary index containing numeric token groups.

        Returns:
            set[int]: Token IDs that preserve a valid partial number or can
            complete a number before valid following literal text.
        """
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
        """Consume token text and exit once a full number is delimited.

        Args:
            token_str: Generated token text appended to the number buffer.

        Returns:
            tuple[State, str]: Next state with delimiter remainder when a
            complete number is found; otherwise this state and empty remainder.
        """
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
        """Return valid token IDs for JSON string opening or content.

        Args:
            vocab_index: Vocabulary index with string-related token filters.

        Returns:
            set[int]: Quote token before opening; content and closing tokens
            once the opening quote has been generated.
        """

        if not self.has_opened:
            return vocab_index.filter_vocab.exact_quote_tokens
        return (vocab_index.filter_vocab.string_content_tokens |
                vocab_index.filter_vocab.string_closer_tokens)

    def transition(self, token_str: str) -> tuple["State", str]:
        """Consume string token text and exit when closing quote appears.

        Args:
            token_str: Generated token text for the string value.

        Returns:
            tuple[State, str]: Next state with overflow after closing quote, or
            this state and empty remainder while string parsing continues.
        """
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
