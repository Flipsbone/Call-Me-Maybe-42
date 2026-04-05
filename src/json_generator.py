# src/json_generator.py

from typing import Any
from pydantic import BaseModel, Field

from src.parsing import DataParser
from src.generator import ConstrainedGenerator
from src.state_machine import (
    JsonStateMachine, StateTerminal, StateExpectLiteral,
    StateBranch, StateParseString, StateParseNumber
)


class SimpleJsonGenerator(BaseModel):
    """Génère le JSON complet directement"""
    user_prompt: str
    data_manager: DataParser
    generator: ConstrainedGenerator
    
    class Config:
        arbitrary_types_allowed = True
    
    def build_prompt(self) -> str:
        """Construit le prompt pour générer le JSON"""
        prompt_text = "You are a function calling system. Choose the exact function name and provide the correct parameters.\n\nAvailable functions:\n"
        for function in self.data_manager.functions_definition:
            prompt_text += f"- {function.name}: {function.description}\n"
        prompt_text += f"\nUser request: {self.user_prompt}\nJSON Output:\n"
        return prompt_text
    
    def build_machine(self) -> JsonStateMachine:
        """Construit la machine d'état pour générer le JSON"""
        branch_choices = {}

        for fn in self.data_manager.functions_definition:
            current_state = StateExpectLiteral(expected='\n}', next_state=StateTerminal())

            if not fn.parameters:
                branch_choices[fn.name] = StateExpectLiteral(
                    expected='",\n  "parameters": {}',
                    next_state=current_state
                )
                continue

            current_state = StateExpectLiteral(expected='\n  }', next_state=current_state)

            params = list(fn.parameters.items())
            
            for i in reversed(range(len(params))):
                p_name, p_model = params[i]

                if p_model.type == "number":
                    val_state = StateParseNumber(next_state=current_state)
                else:
                    val_state = StateParseString(next_state=current_state)

                if i == 0:
                    inject_str = f'",\n  "parameters": {{\n    "{p_name}": '
                else:
                    inject_str = f',\n    "{p_name}": '

                current_state = StateExpectLiteral(expected=inject_str, next_state=val_state)

            branch_choices[fn.name] = current_state

        root_state = StateExpectLiteral(
            expected='{\n  "name": "', 
            next_state=StateBranch(choices=branch_choices)
        )

        return JsonStateMachine(current_state=root_state)
    
    def generate(self) -> str:
        """Génère le JSON complet"""
        self.generator.machine = self.build_machine()
        return self.generator.generate(self.build_prompt())


def process_single_prompt_optimized(
    user_prompt: str, 
    data_manager: DataParser, 
    generator: ConstrainedGenerator
) -> dict[str, Any]:
    """Interface simplifiée pour traiter un prompt"""
    import json
    
    json_gen = SimpleJsonGenerator(
        user_prompt=user_prompt,
        data_manager=data_manager,
        generator=generator
    )
    
    generated_json_str = json_gen.generate()
    
    try:
        parsed_data = json.loads(generated_json_str)
        return {
            "prompt": user_prompt,
            "name": parsed_data["name"],
            "parameters": parsed_data.get("parameters", {})
        }
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Le JSON généré est invalide : {e}\nContenu brut:\n{generated_json_str}")
