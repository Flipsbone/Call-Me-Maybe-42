import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import StrictVocabFilter

WS = r'[ \n\r\t]*'


class JSONValidator:
    REGEX_PARTIAL_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
    REGEX_PREFIX_NUMBER = re.compile(
        fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')

    @staticmethod
    def is_partial_number(text: str) -> bool:
        return bool(JSONValidator.REGEX_PARTIAL_NUMBER.fullmatch(text))

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
            self, clean_vocab: dict[int, str], vocab_filter: StrictVocabFilter
            ) -> set[int]:
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        pass


class StateTerminal(State):
    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: StrictVocabFilter) -> set[int]:
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""


class StateExpectLiteral(State):
    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str], vocab_filter: StrictVocabFilter
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
            self, clean_vocab: dict[int, str], vocab_filter: StrictVocabFilter
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
            vocab_filter: StrictVocabFilter) -> set[int]:
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
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str
        return self, ""


class StateParseString(State):
    next_state: State | None = Field(default=None)
    has_opened: bool = Field(default=False)

    def get_valid_tokens(
            self, clean_vocab: dict[int, str],
            vocab_filter: StrictVocabFilter) -> set[int]:

        valid_ids: set[int] = set()

        if not self.has_opened:
            valid_ids.update(vocab_filter.exact_quote_tokens)
            return valid_ids

        valid_ids.update(vocab_filter.string_content_tokens)
        valid_ids.update(vocab_filter.string_closer_tokens)

        return valid_ids

    def transition(self, token_str: str) -> tuple["State", str]:
        if not self.has_opened:
            if '"' in token_str:
                self.has_opened = True
                quote_idx = token_str.find('"')
                remain_str = token_str[quote_idx + 1:]
                return self, remain_str
            return self, ""

        idx = 0
        while True:
            quote_idx = token_str.find('"', idx)
            if quote_idx == -1:
                break

            is_escaped = False
            if quote_idx > 0 and token_str[quote_idx - 1] == '\\':
                is_escaped = True
            elif quote_idx == 0 and self.buffer.endswith('\\'):
                is_escaped = True

            if is_escaped:
                idx = quote_idx + 1
                continue

            remain_str = token_str[quote_idx + 1:]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, remain_str

        self.buffer += token_str
        return self, ""
