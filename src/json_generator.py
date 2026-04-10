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


class GenerationError(Exception):
    pass


class TwoStepJsonGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    generator: ConstrainedGenerator

    def prompt_for_name(self) -> str:
        prompt = ("<|im_start|>system\nChoose the exact function name."
                  "\nFunctions:\n")

        for f in self.functions_definition:
            prompt += f"- {f.name}: {f.description}\n"

        prompt += (f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}"
                   "<|im_end|>\n<|im_start|>assistant\n")

        return prompt

    def machine_for_name(self) -> JsonStateMachine:
        if not self.functions_definition:
            return JsonStateMachine(current_state=StateTerminal())

        choices: dict[str, State] = {}

        for fn in self.functions_definition:
            choices[fn.name] = StateTerminal()

        branch_state = StateBranch(choices=choices)
        return JsonStateMachine(current_state=branch_state)

    def prompt_for_params(self, target_fn: FunctionDefinition) -> str:

        prompt = f"<|im_start|>system\nExtract params for {target_fn.name}.\n"
        prompt += f"Desc: {target_fn.description}\n<|im_end|>\n"
        prompt += (f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n"
                   "<|im_start|>assistant\n")
        return prompt

    def machine_for_params(
            self, target_fn: FunctionDefinition) -> JsonStateMachine:

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

            if p_model.type == "number":
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
        self.generator.machine = self.machine_for_name()
        name_prompt = self.prompt_for_name()
        selected_name = self.generator.generate(name_prompt, 15)

        target_fn = None
        for function in self.functions_definition:
            if function.name == selected_name:
                target_fn = function

        if target_fn is None:
            raise GenerationError(f"LLM generated an unknown function: "
                                  f"'{selected_name}'")

        self.generator.machine = self.machine_for_params(target_fn)
        params_prompt = self.prompt_for_params(target_fn)
        params_str = self.generator.generate(params_prompt, 100)

        try:
            params_dict = json.loads(params_str) if params_str.strip() else {}
        except json.JSONDecodeError:
            raise GenerationError(f"LLM generated invalid JSON: {params_str}")

        return {
            "prompt": self.user_prompt,
            "name": selected_name,
            "parameters": params_dict
        }


def process_single_prompt_optimized(
    user_prompt: str,
    functions_definition: list[FunctionDefinition],
    generator: ConstrainedGenerator
) -> dict[str, Any]:

    json_gen = TwoStepJsonGenerator(
        user_prompt=user_prompt,
        functions_definition=functions_definition,
        generator=generator
    )
    return json_gen.generate()
