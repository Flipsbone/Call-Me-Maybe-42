import json
from typing import Any
from pydantic import BaseModel, ConfigDict

from src.constrained_decoder import ConstrainedDecoder
from src.functions_validator import FunctionDefinition
from src.state_machine import (
    State,
    StateTerminal,
    StateExpectLiteral,
    StateBranch,
    StateParseString,
    StateParseNumber
)


class GenerationJsonError(Exception):
    """Raised when generated output cannot be converted to valid JSON."""

    pass


class TwoStepJsonGenerator(BaseModel):
    """Generator for function calls in two constrained steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    generator: ConstrainedDecoder

    def prompt_for_name(self) -> str:
        """Build the prompt used to select the target function name."""
        prompt = ("<|im_start|>system\nChoose the exact function name."
                  "\nFunctions:\n")

        for f in self.functions_definition:
            prompt += f"- {f.name}: {f.description}\n"

        prompt += (f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}"
                   "<|im_end|>\n<|im_start|>assistant\n")

        return prompt

    def machine_for_name(self) -> State:
        """Create a state machine that accepts only valid function names."""
        if not self.functions_definition:
            return StateTerminal()

        choices: dict[str, State] = {
            fn.name: StateTerminal() for fn in self.functions_definition
        }
        return StateBranch(choices=choices)

    def prompt_for_params(self, target_fn: FunctionDefinition) -> str:
        """Build the prompt used to extract parameters for one function."""
        params_info = (", ".join(
            [f"'{p_name}' ({p_model.type})"
             for p_name, p_model in target_fn.parameters.items()]))

        prompt = (
            "<|im_start|>system\n"
            "Extract the specific parameters for the function"
            f" '{target_fn.name}'.\n"
            f"You must find these parameters: {params_info}\n"
            "CRITICAL: Do NOT execute the command."
            "Do NOT calculate or reverse anything."
            "ONLY extract the exact literal values from the text.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def machine_for_params(self, target_fn: FunctionDefinition) -> State:
        """Build a state machine enforcing the JSON schema for parameters."""
        if not target_fn.parameters:
            return (StateExpectLiteral(expected='{}',
                                       next_state=StateTerminal()))

        current_state: State = StateExpectLiteral(
            expected='\n}', next_state=StateTerminal())

        params = list(target_fn.parameters.items())

        for i in reversed(range(len(params))):
            p_name, p_model = params[i]
            val_state: State = (
                StateParseNumber(next_state=current_state)
                if p_model.type in ["number", "integer"]
                else StateParseString(next_state=current_state)
            )
            prefix = (
                f'{{\n  "{p_name}": ' if i == 0 else f',\n  "{p_name}": ')

            current_state = StateExpectLiteral(
                expected=prefix, next_state=val_state)

        return current_state

    def generate(self) -> dict[str, Any]:
        """Execute the two-step generation and validate the resulting data."""
        name_state = self.machine_for_name()
        selected_name = self.generator.generate(
            self.prompt_for_name(), name_state, 15)

        target_fn: FunctionDefinition | None = next(
            (f for f in self.functions_definition if f.name == selected_name),
            None
        )

        if target_fn is None:
            raise GenerationJsonError(f"Unknown function: {selected_name}")

        params_state = self.machine_for_params(target_fn)
        params_str = self.generator.generate(
            self.prompt_for_params(target_fn), params_state, 100)

        try:
            params_dict: dict[str, Any] = (
                json.loads(params_str) if params_str.strip() else {}
            )
        except json.JSONDecodeError:
            raise GenerationJsonError(f"Invalid JSON: {params_str}")

        for p_name, p_model in target_fn.parameters.items():
            if p_name in params_dict and p_model.type == "number":
                params_dict[p_name] = float(params_dict[p_name])

        return {
            "prompt": self.user_prompt,
            "name": selected_name,
            "parameters": params_dict
        }
