import sys
import re
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from src.vocabulary import VocabularyPruner

WS = r'[ \n\r\t]*'

REGEX_PARTIAL_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*$')
REGEX_PREFIX_STRING = re.compile(f'^{WS}"([^"\\\\]|\\\\.)*"')

REGEX_PARTIAL_NUMBER = re.compile(fr'^{WS}-?(?:0|[1-9]\d*)?(?:\.\d*)?(?:[eE][+-]?\d*)?$')
REGEX_PREFIX_NUMBER = re.compile(fr'^{WS}-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?')


class State(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    buffer: str = Field(default="")

    @abstractmethod
    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        pass


class StateTerminal(State):
    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        return set()

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""


class StateExpectLiteral(State):
    expected: str = Field(...)
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        if not self.expected.startswith(self.buffer):
            return set()
            
        remainder = self.expected[len(self.buffer):]
        if not remainder:
            return set()

        return pruner.get_literal_matches(remainder, clean_vocab)

    def transition(self, token_str: str) -> tuple["State", str]:
        self.buffer += token_str
        if self.buffer.startswith(self.expected):
            overflow = self.buffer[len(self.expected):]
            next_s = self.next_state if self.next_state else StateTerminal()
            return next_s, overflow
        return self, ""


class StateBranch(State):
    choices: dict[str, State] = Field(...)

    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        valid_ids = set()
        for choice in self.choices.keys():
            if choice.startswith(self.buffer):
                remainder = choice[len(self.buffer):]
                if remainder:
                    valid_ids.update(pruner.get_literal_matches(remainder, clean_vocab))
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

    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        valid_ids = set()
        for token_id in pruner.numeric_tokens:
            token_str = clean_vocab[token_id]
            test_str = self.buffer + token_str
            
            if REGEX_PARTIAL_NUMBER.fullmatch(test_str):
                valid_ids.add(token_id)
            else:
                match = REGEX_PREFIX_NUMBER.match(test_str)
                if match:
                    overflow = test_str[match.end():]
                    if not overflow:
                        valid_ids.add(token_id)
                    elif hasattr(self.next_state, 'expected') and getattr(self.next_state, 'expected').startswith(overflow):
                        valid_ids.add(token_id)
                        
        return valid_ids

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


class StateParseString(State):
    next_state: State | None = Field(default=None)

    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        valid_ids = set(pruner.string_safe_tokens) 
        
        for token_id in pruner.string_unsafe_tokens:
            token_str = clean_vocab[token_id]
            test_str = self.buffer + token_str
            
            if REGEX_PARTIAL_STRING.fullmatch(test_str):
                valid_ids.add(token_id)
            else:
                match = REGEX_PREFIX_STRING.match(test_str)
                if match:
                    overflow = test_str[match.end():]
                    if not overflow:
                        valid_ids.add(token_id)
                    elif hasattr(self.next_state, 'expected') and getattr(self.next_state, 'expected').startswith(overflow):
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

    def get_valid_tokens(self, clean_vocab: dict[int, str], pruner: VocabularyPruner) -> set[int]:
        return self.current_state.get_valid_tokens(clean_vocab, pruner)

    def step(self, token_str: str) -> None:
        overflow = token_str
        while overflow and not isinstance(self.current_state, StateTerminal):
            self.current_state, overflow = self.current_state.transition(overflow)
