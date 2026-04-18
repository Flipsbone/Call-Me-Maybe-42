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
    pass


class TwoStepJsonGenerator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_prompt: str
    functions_definition: list[FunctionDefinition]
    assistant: ConstrainedDecoder

    def generate(self) -> dict[str, Any]:
        chosen_name = self._generate_function_name()
        target_function = self._find_function_by_name(chosen_name)
        final_parameters = self._generate_function_parameters(target_function)

        return {
            "prompt": self.user_prompt,
            "name": chosen_name,
            "parameters": final_parameters
        }

    def _generate_function_name(self) -> str:
        prompt = self._create_prompt_for_function_selection()
        state = self._create_state_machine_for_function_selection()

        chosen_name = self.assistant.generate(
            prompt=prompt,
            state=state,
            max_tokens=15
        )
        return chosen_name

    def _generate_function_parameters(
            self, target_fn: FunctionDefinition) -> dict[str, Any]:
        prompt = self._create_prompt_for_parameter_extraction(target_fn)
        state = self._create_state_machine_for_parameter_extraction(target_fn)

        generated_params_text = self.assistant.generate(
            prompt=prompt,
            state=state,
            max_tokens=100
        )

        final_parameters = self._parse_and_validate_json(generated_params_text)
        self._convert_number_types(final_parameters, target_fn)

        return final_parameters

    def _create_prompt_for_function_selection(self) -> str:
        prompt = ("<|im_start|>system\nChoose the exact function name."
                  "\nFunctions:\n")

        for func in self.functions_definition:
            prompt += f"- {func.name}: {func.description}\n"

        prompt += (f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}"
                   "<|im_end|>\n<|im_start|>assistant\n")

        return prompt

    def _create_state_machine_for_function_selection(self) -> State:
        choices: dict[str, State] = {
            fn.name: StateTerminal() for fn in self.functions_definition
        }
        return StateBranch(choices=choices)

    def _create_prompt_for_parameter_extraction(
            self, target_fn: FunctionDefinition) -> str:
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
            "For string parameters, preserve the EXACT case from the input.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _create_state_machine_for_parameter_extraction(
            self, target_fn: FunctionDefinition) -> State:
        if not target_fn.parameters:
            return StateExpectLiteral(
                expected='{}',
                next_state=StateTerminal())

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

    def _find_function_by_name(self, function_name: str) -> FunctionDefinition:
        for func in self.functions_definition:
            if func.name == function_name:
                return func
        raise GenerationJsonError(f"Unknown function: {function_name}")

    def _parse_and_validate_json(self, json_text: str) -> dict[str, Any]:
        if json_text.strip() == "":
            return {}

        try:
            result = json.loads(json_text)
            if isinstance(result, dict):
                return result
            else:
                return {}
        except json.JSONDecodeError:
            raise GenerationJsonError(
                f"Invalid JSON: {json_text}")

    def _convert_number_types(
            self, parameters: dict[str, Any],
            target_fn: FunctionDefinition) -> None:
        for param_name, param_details in target_fn.parameters.items():
            if param_name in parameters:
                val = parameters[param_name]

                try:
                    if param_details.type == "number":
                        parameters[param_name] = float(val)
                    elif param_details.type == "integer":
                        parameters[param_name] = int(float(val))
                except (ValueError, TypeError):
                    pass
