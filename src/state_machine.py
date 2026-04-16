import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabFilter

WS = r'[ \n\r\t]*'


class JSONValidator:
    REGEX_PARTIAL_STRING = re.compile(fr'^{WS}"([^"\\]|\\.)*$')
    REGEX_PREFIX_STRING = re.compile(fr'^{WS}"([^"\\]|\\.)*"')
    REGEX_PARTIAL_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
    REGEX_PREFIX_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')

    @staticmethod
    def is_partial_string(text: str) -> bool:
        return bool(JSONValidator.REGEX_PARTIAL_STRING.fullmatch(text))

    @staticmethod
    def is_partial_number(text: str) -> bool:
        return bool(JSONValidator.REGEX_PARTIAL_NUMBER.fullmatch(text))

    @staticmethod
    def extract_complete_string(text: str) -> tuple[str, str]:
        match = JSONValidator.REGEX_PREFIX_STRING.match(text)
        if match:
            matched_text = match.group()
            remain_str = text[len(matched_text):]
            return matched_text, remain_str
        return "", text

    @staticmethod
    def extract_complete_number(text: str) -> tuple[str, str]:
        match = JSONValidator.REGEX_PREFIX_NUMBER.match(text)
        if match:
            matched_text = match.group()
            remain_str = text[len(matched_text):]
            return matched_text, remain_str
        return "", text


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
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        if self.buffer.startswith(self.expected):
            remain_str = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str

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
                remain_str = self.buffer[len(choice):]
                return next_s, remain_str
        return self, ""


class StateParseNumber(State):
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        valid_ids: set[int] = set()
        expected_next_text = getattr(self.next_state, 'expected', '')

        for token_id in vocab_filter.numeric_tokens:
            simulated_text = self.buffer + clean_vocab[token_id]

            if JSONValidator.is_partial_number(simulated_text):
                valid_ids.add(token_id)
            else:
                matched_text, remain_str = (
                    JSONValidator.extract_complete_number(
                        simulated_text))
                if matched_text and (
                        not remain_str or
                        expected_next_text.startswith(remain_str)):
                    valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        matched_text, remain_str = JSONValidator.extract_complete_number(
            self.buffer)

        if matched_text and remain_str and remain_str[0] in ',}]\n':
            next_s = (
                self.next_state if self.next_state else StateTerminal())
            return next_s, remain_str

        return self, ""


class StateParseString(State):
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: VocabFilter) -> set[int]:
        expected_next_text = getattr(self.next_state, 'expected', '')

        matched_text, remain_str = JSONValidator.extract_complete_string(
            self.buffer)

        if matched_text:
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
        valid_ids: set[int] = set()

        if self.buffer.lstrip().startswith('"'):
            valid_ids.update(vocab_filter.string_safe_tokens)

        for token_id in vocab_filter.string_unsafe_tokens:
            simulated_text = self.buffer + clean_vocab[token_id]

            if JSONValidator.is_partial_string(simulated_text):
                valid_ids.add(token_id)
            else:
                matched_text, remain_str = (
                    JSONValidator.extract_complete_string(simulated_text))
                if matched_text and (
                        not remain_str or
                        expected_next.startswith(remain_str)):
                    valid_ids.add(token_id)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        matched_text, remain_str = JSONValidator.extract_complete_string(
            self.buffer)

        if matched_text:
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str

        return self, ""
