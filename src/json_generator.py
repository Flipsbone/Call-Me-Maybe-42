import json
from typing import Any
from pydantic import BaseModel

from src.parsing import DataParser
from src.generator import ConstrainedGenerator
from src.model_pydantic import FunctionDefinition
from src.state_machine import (
    JsonStateMachine, StateTerminal, StateExpectLiteral,
    StateBranch, StateParseString, StateParseNumber
)

class TwoStepJsonGenerator(BaseModel):
    """Génère le JSON en deux étapes pour esquiver l'absence de KV Cache."""
    user_prompt: str
    data_manager: DataParser
    generator: ConstrainedGenerator
    
    class Config:
        arbitrary_types_allowed = True

    # --- ÉTAPE 1 : CHOIX DU NOM ---
    def prompt_for_name(self) -> str:
        prompt = "<|im_start|>system\nChoose the exact function name.\nFunctions:\n"
        for f in self.data_manager.functions_definition:
            prompt += f"- {f.name}: {f.description}\n"
        prompt += f"<|im_end|>\n<|im_start|>user\n{self.user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def machine_for_name(self) -> JsonStateMachine:
        choices = {fn.name: StateTerminal() for fn in self.data_manager.functions_definition}
        return JsonStateMachine(current_state=StateBranch(choices=choices))

    # --- ÉTAPE 2 : EXTRACTION DES PARAMÈTRES ---
    def prompt_for_params(self, target_fn: FunctionDefinition) -> str:
        prompt = f"<|im_start|>system\nExtract params for {target_fn.name}.\n"
        prompt += f"Desc: {target_fn.description}\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def machine_for_params(self, target_fn: FunctionDefinition) -> JsonStateMachine:
        if not target_fn.parameters:
            return JsonStateMachine(current_state=StateExpectLiteral(expected='{}', next_state=StateTerminal()))

        current_state = StateExpectLiteral(expected='\n}', next_state=StateTerminal())
        params = list(target_fn.parameters.items())
        
        for i in reversed(range(len(params))):
            p_name, p_model = params[i]
            val_state = StateParseNumber(next_state=current_state) if p_model.type == "number" else StateParseString(next_state=current_state)

            inject_str = f'{{\n  "{p_name}": ' if i == 0 else f',\n  "{p_name}": '
            current_state = StateExpectLiteral(expected=inject_str, next_state=val_state)

        return JsonStateMachine(current_state=current_state)

    # --- EXÉCUTION ---
    def generate(self) -> dict[str, Any]:
        self.generator.machine = self.machine_for_name()
        selected_name = self.generator.generate(self.prompt_for_name(), max_tokens=15)
        
        target_fn = next((f for f in self.data_manager.functions_definition if f.name == selected_name), None)
        if not target_fn:
            raise RuntimeError(f"Fonction invalide générée: {selected_name}")

        self.generator.machine = self.machine_for_params(target_fn)
        params_str = self.generator.generate(self.prompt_for_params(target_fn), max_tokens=100)
        
        try:
            params_dict = json.loads(params_str) if params_str.strip() else {}
        except json.JSONDecodeError:
            params_dict = {}

        return {
            "prompt": self.user_prompt,
            "name": selected_name,
            "parameters": params_dict
        }


def process_single_prompt_optimized(
    user_prompt: str, 
    data_manager: DataParser, 
    generator: ConstrainedGenerator
) -> dict[str, Any]:
    
    json_gen = TwoStepJsonGenerator(
        user_prompt=user_prompt,
        data_manager=data_manager,
        generator=generator
    )
    return json_gen.generate()