import sys
import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from vocabulary import VocabularyPruner

WS = r'[ \n\r\t]*'

REGEX_PARTIAL_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*$')
REGEX_COMPLETE_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*"{WS}$')

REGEX_PARTIAL_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$'
)
REGEX_COMPLETE_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?{WS}$'
)

REGEX_PREFIX_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*"')
REGEX_PREFIX_NUMBER = re.compile(
    fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
)


class State(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        pass


class StateTerminal(State):
    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            return [float('-inf')] * len(logits)
        except Exception as e:
            print(f"Erreur fatale dans StateTerminal : {e}", file=sys.stderr)
            sys.exit(1)

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""


class StateExpectLiteral(State):
    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            for token_id, token_str in clean_vocab.items():
                test_str = self.buffer + token_str
                if not (self.expected.startswith(test_str) or test_str.startswith(self.expected)):
                    logits[token_id] = float('-inf')
            return logits
        except Exception as e:
            print(f"Error in StateExpectLiteral : {e}", file=sys.stderr)
            return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        if self.buffer.startswith(self.expected):
            overflow = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class StateExpectEnum(State):
    choices: list[str] = Field(...)
    next_state: State | None = Field(default=None)

    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            possible = [c for c in self.choices if c.startswith(self.buffer)]
            for token_id, token_str in clean_vocab.items():
                valid = any(
                    c[len(self.buffer):].startswith(token_str) or
                    token_str.startswith(c[len(self.buffer):])
                    for c in possible
                )
                if not valid:
                    logits[token_id] = float('-inf')
            return logits
        except Exception as e:
            print(f"Erreur in StateExpectEnum : {e}", file=sys.stderr)
            return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        for choice in self.choices:
            if self.buffer.startswith(choice):
                overflow = self.buffer[len(choice):]
                next_s = self.next_state if self.next_state else StateTerminal()
                return next_s, overflow
        return self, ""


class StateParseString(State):
    next_state: State | None = Field(default=None)

    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            for token_id, token_str in clean_vocab.items():
                test_str = self.buffer + token_str
                if (REGEX_PARTIAL_STRING.fullmatch(test_str) or
                        REGEX_COMPLETE_STRING.fullmatch(test_str)):
                    continue
                if REGEX_PREFIX_STRING.match(test_str):
                    continue
                logits[token_id] = float('-inf')
            return logits
        except Exception as e:
            print(f"Erreur in StateParseString : {e}", file=sys.stderr)
            return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        match = REGEX_PREFIX_STRING.match(self.buffer)
        if match:
            string_part = match.group()
            overflow = self.buffer[len(string_part):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class StateParseNumber(State):
    next_state: State | None = Field(default=None)

    def filter_logits(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            new_logits = [float('-inf')] * len(logits)

            for token_id in pruner.numeric_tokens:
                token_str = clean_vocab[token_id]
                test_str = self.buffer + token_str
                
                if (REGEX_PARTIAL_NUMBER.fullmatch(test_str) or
                        REGEX_COMPLETE_NUMBER.fullmatch(test_str)):
                    new_logits[token_id] = logits[token_id] 
                    continue

                match = REGEX_PREFIX_NUMBER.match(test_str)
                if match:
                    overflow = test_str[match.end():]
                    if (overflow and overflow[0] in
                            [',', '}', ']', ' ', '\n', '\r', '\t']):
                        new_logits[token_id] = logits[token_id] 

            return new_logits
        except Exception as e:
            print(f"Erreur in StateParseNumber : {e}", file=sys.stderr)
            return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        match = REGEX_PREFIX_NUMBER.match(self.buffer)

        if match and len(match.group()) < len(self.buffer.lstrip()):
            num_part = match.group()
            ws_len = len(self.buffer) - len(self.buffer.lstrip())
            overflow = self.buffer[ws_len + len(num_part):]

            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow

        return self, ""


class JsonStateMachine(BaseModel):
    current_state: State

    def apply_constraints(
            self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner
            ) -> list[float]:
        try:
            return self.current_state.filter_logits(logits, clean_vocab, pruner)
        except Exception as e:
            print(f"Alerte de filtrage : {e}", file=sys.stderr)
            return [float('-inf')] * len(logits)

    def step(self, token_str: str) -> None:
        try:
            overflow = token_str
            while (overflow and not
                   isinstance(self.current_state, StateTerminal)):

                self.current_state, overflow = (
                    self.current_state.transition(overflow))

        except Exception as e:
            print(f"Error in change state : {e}", file=sys.stderr)
            self.current_state = StateTerminal()
