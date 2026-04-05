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
    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        pass

    @abstractmethod
    def transition(self, token_str: str) -> tuple["State", str]:
        pass

class StateTerminal(State):
    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        return self, ""

class StateInject(State):
    text_to_inject: str = Field(...)
    next_state: State | None = Field(default=None)

    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        return [float('-inf')] * len(logits)

    def transition(self, token_str: str) -> tuple["State", str]:
        next_s = self.next_state if self.next_state else StateTerminal()
        return next_s, ""

class StateBranch(State):
    choices: dict[str, State] = Field(...)

    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        new_logits = [float('-inf')] * len(logits)
        possible_remainders = [c[len(self.buffer):] for c in self.choices.keys() if c.startswith(self.buffer)]
        
        for token_id, token_str in clean_vocab.items():
            if any(rem.startswith(token_str) or token_str.startswith(rem) for rem in possible_remainders):
                new_logits[token_id] = logits[token_id]
        return new_logits

    def transition(self, token_str: str) -> tuple[State, str]:
        self.buffer += token_str
        for choice, next_s in self.choices.items():
            if self.buffer.startswith(choice):
                overflow = self.buffer[len(choice):]
                return next_s, overflow
        return self, ""

class StateParseNumber(State):
    next_state: State | None = Field(default=None)

    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        new_logits = [float('-inf')] * len(logits)
        for token_id in pruner.numeric_tokens:
            token_str = clean_vocab[token_id]
            test_str = self.buffer + token_str
            
            # 1. Toujours en cours de construction (ex: "-4")
            if REGEX_PARTIAL_NUMBER.fullmatch(test_str):
                new_logits[token_id] = logits[token_id] 
                continue

            # 2. Terminé (ex: "-42" ou "-42,")
            match = REGEX_PREFIX_NUMBER.match(test_str)
            if match:
                overflow = test_str[match.end():]
                # Si pas d'overflow, c'est terminé pile-poil.
                # S'il y a un overflow, on vérifie que le caractère suivant est légal en JSON
                if not overflow or overflow[0] in [',', '}', ']', ' ', '\n', '\r', '\t']:
                    new_logits[token_id] = logits[token_id] 
        return new_logits

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

    def filter_logits(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        new_logits = [float('-inf')] * len(logits)
        for token_id, token_str in clean_vocab.items():
            if '"' not in token_str and '\\' not in token_str:
                new_logits[token_id] = logits[token_id]
                continue
                
            test_str = self.buffer + token_str
            
            # 1. En cours de construction
            if REGEX_PARTIAL_STRING.fullmatch(test_str):
                new_logits[token_id] = logits[token_id]
            # 2. Terminé (avec ou sans overflow)
            elif REGEX_PREFIX_STRING.match(test_str):
                new_logits[token_id] = logits[token_id]
        return new_logits

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

    def apply_constraints(self, logits: list[float], clean_vocab: dict[int, str], pruner: VocabularyPruner) -> list[float]:
        return self.current_state.filter_logits(logits, clean_vocab, pruner)

    def step(self, token_str: str) -> None:
        overflow = token_str
        while overflow and not isinstance(self.current_state, StateTerminal):
            self.current_state, overflow = self.current_state.transition(overflow)
