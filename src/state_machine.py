import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabFilter

WS = r'[ \n\r\t]*'

REGEX_PARTIAL_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*$')
REGEX_PREFIX_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*"')

REGEX_PARTIAL_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
REGEX_PREFIX_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')


class State(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        pass


class StateTerminal(State):
    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""


class StateExpectLiteral(State):
    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:

        if not self.expected.startswith(self.buffer):
            return set()

        remainder = self.expected[len(self.buffer):]
        if not remainder:
            return set()

        return vocab_filter.get_literal_matches(remainder, clean_vocab)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        if self.buffer.startswith(self.expected):
            overflow = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class StateBranch(State):
    choices: dict[str, State] = Field(...)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:

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
        self.buffer += token_str
        for choice, next_s in self.choices.items():
            if self.buffer.startswith(choice):
                overflow = self.buffer[len(choice):]
                return next_s, overflow
        return self, ""


class StateParseNumber(State):
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:

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
        self.buffer += token_str
        match = REGEX_PREFIX_NUMBER.match(self.buffer)
        if match:
            num_part = match.group()
            ws_len = len(self.buffer) - len(self.buffer.lstrip())
            overflow = self.buffer[ws_len + len(num_part):]
            if overflow and overflow[0] in ',}]\n':
                next_s = (
                    self.next_state if self.next_state else StateTerminal())
                return next_s, overflow
        return self, ""


class StateParseString(State):
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:

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
        self.buffer += token_str
        match = REGEX_PREFIX_STRING.match(self.buffer)
        if match:
            string_part = match.group()
            overflow = self.buffer[len(string_part):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class JsonStateMachine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    current_state: State

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: VocabFilter
            ) -> set[int]:

        return self.current_state.get_valid_tokens(clean_vocab, vocab_filter)

    def step(self, token_str: str) -> None:
        overflow = token_str
        state = self.current_state
        while overflow and not isinstance(state, StateTerminal):
            state, overflow = state.transition(overflow)
            self.current_state = state
