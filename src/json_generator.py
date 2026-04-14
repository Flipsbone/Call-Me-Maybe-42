import json
from typing import Any
from pydantic import BaseModel, ConfigDict

from src.generator import ConstrainedGenerator
from src.functions_validator import FunctionDefinition
from src.state_machine import (
    State,
    JsonStateMachine,
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
    """Generator for function calls in two constrained steps.

    Attributes:
        user_prompt (str): The user's natural language request.
        functions_definition (list[FunctionDefinition]): Available tools.
        generator (ConstrainedGenerator): Constrained generation engine.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    generator: ConstrainedGenerator

    def prompt_for_name(self) -> str:
        """Build the prompt used to select the target function name.

        Returns:
            str: Chat-style prompt describing the available functions.
        """
        prompt = ("<|im_start|>system\nChoose the exact function name."
                  "\nFunctions:\n")

        for f in self.functions_definition:
            prompt += f"- {f.name}: {f.description}\n"

        prompt += (f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}"
                   "<|im_end|>\n<|im_start|>assistant\n")

        return prompt

    def machine_for_name(self) -> JsonStateMachine:
        """Create a state machine that accepts only valid function names."""
        if not self.functions_definition:
            return JsonStateMachine(current_state=StateTerminal())

        choices: dict[str, State] = {}

        for fn in self.functions_definition:
            choices[fn.name] = StateTerminal()

        branch_state = StateBranch(choices=choices)
        return JsonStateMachine(current_state=branch_state)

    def prompt_for_params(self, target_fn: FunctionDefinition) -> str:
        """Build the prompt used to extract parameters for one function.

        Args:
            target_fn: Selected function definition whose parameters should
                be extracted.

        Returns:
            str: Chat-style prompt focused on parameter extraction.
        """
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

    def machine_for_params(
            self, target_fn: FunctionDefinition) -> JsonStateMachine:
        """Create a state machine that emits JSON for one function call.

        Args:
            target_fn: Selected function definition used to shape the JSON.

        Returns:
            JsonStateMachine: Machine constrained to the function's
            parameter schema.
        """
        val_state: State

        if not target_fn.parameters:
            empty_state = StateExpectLiteral(
                expected='{}', next_state=StateTerminal())
            return JsonStateMachine(current_state=empty_state)

        tail_state = StateTerminal()
        current_state: State = StateExpectLiteral(
            expected='\n}', next_state=tail_state)

        params = list(target_fn.parameters.items())

        for i in reversed(range(len(params))):
            p_name, p_model = params[i]

            if p_model.type in ["number", "integer"]:
                val_state = StateParseNumber(next_state=current_state)
            else:
                val_state = StateParseString(next_state=current_state)

            is_first = (i == 0)

            prefix = (
                f'{{\n  "{p_name}": ' if is_first else f',\n  "{p_name}": ')

            current_state = StateExpectLiteral(
                expected=prefix, next_state=val_state)

        return JsonStateMachine(current_state=current_state)

    def generate(self) -> dict[str, Any]:
        """Generate the function name and then extract its parameters.

        Returns:
            dict[str, Any]: Dictionary containing the prompt, the chosen
                function name, and its extracted parameters.

        Raises:
            GenerationJsonError: If the model selects a non-existent
                function or if the parameters JSON is malformed.
        """

        self.generator.machine = self.machine_for_name()
        name_prompt = self.prompt_for_name()
        selected_name = self.generator.generate(name_prompt, 15)

        target_fn = None
        for function in self.functions_definition:
            if function.name == selected_name:
                target_fn = function

        if target_fn is None:
            raise GenerationJsonError(f"LLM generated an unknown function: "
                                      f"'{selected_name}'")

        self.generator.machine = self.machine_for_params(target_fn)
        params_prompt = self.prompt_for_params(target_fn)
        params_str = self.generator.generate(params_prompt, 100)

        try:
            params_dict = json.loads(params_str) if params_str.strip() else {}
        except json.JSONDecodeError:
            raise GenerationJsonError("LLM generated invalid JSON:"
                                      f"{params_str}")

        for p_name, p_val in params_dict.items():
            if p_name in target_fn.parameters:
                if target_fn.parameters[p_name].type == "number":
                    params_dict[p_name] = float(p_val)

        return {
            "prompt": self.user_prompt,
            "name": selected_name,
            "parameters": params_dict
        }


def process_single_prompt_optimized(
    user_prompt: str, functions_definition: list[FunctionDefinition],
        generator: ConstrainedGenerator) -> dict[str, Any]:
    """Generate a structured function-call result for one user prompt.

    Args:
        user_prompt: Natural-language user request to process.
        functions_definition: Available functions the model may choose from.
        generator: Constrained generator used to produce the result.

    Returns:
        dict[str, Any]: Generated function-call payload.
    """

    json_gen = TwoStepJsonGenerator(
        user_prompt=user_prompt,
        functions_definition=functions_definition,
        generator=generator
    )
    return json_gen.generate()
