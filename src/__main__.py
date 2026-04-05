import sys
import json
import argparse
from pathlib import Path
from typing import Any

from src.parsing import DataParser
from src.vocabulary import VocabularyIndex
from src.generator import ConstrainedGenerator
from src.model_pydantic import FunctionCallResult
from llm_sdk import Small_LLM_Model

from src.state_machine import (
    JsonStateMachine, StateTerminal, StateExpectLiteral,
    StateBranch, StateParseString, StateParseNumber
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling")
    parser.add_argument("--functions_definition", type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/output/function_calling_results.json"))
    return parser.parse_args()


def check_and_prepare_paths(func_def_path: Path, input_path: Path, output_path: Path) -> None:
    for file_path in [func_def_path, input_path]:
        if not file_path.exists():
            print(f"Error : input file '{file_path}' not found.", file=sys.stderr)
            sys.exit(1)

    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error : {output_dir} : {e}", file=sys.stderr)
            sys.exit(1)


def build_prompt(user_request: str, functions: list) -> str:
    """
    Pré-prompting clair et direct pour guider le LLM.
    """
    prompt_text = "You are a function calling system. Choose the exact function name and provide the correct parameters.\n\nAvailable functions:\n"
    for function in functions:
        prompt_text += f"- {function.name}: {function.description}\n"
    prompt_text += f"\nUser request: {user_request}\nJSON Output:\n"
    return prompt_text


def build_dynamic_machine(functions: list) -> JsonStateMachine:
    branch_choices = {}

    for fn in functions:
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

def process_single_prompt(user_prompt: str, data_manager: DataParser, generator: ConstrainedGenerator) -> dict[str, Any]:
    
    # 1. Formatage du texte pour le LLM
    full_prompt = build_prompt(user_prompt, data_manager.functions_definition)

    # 2. On fabrique le "moule" dynamique pour cette question spécifique
    generator.machine = build_dynamic_machine(data_manager.functions_definition)

    # 3. Le générateur s'occupe de tout (Injection ultra-rapide + LLM contraint)
    generated_json_str = generator.generate(full_prompt)

    # 4. Vérification et parsing
    try:
        parsed_data = json.loads(generated_json_str)
        return {
            "prompt": user_prompt,
            "name": parsed_data["name"],
            "parameters": parsed_data.get("parameters", {})
        }
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Le JSON généré est invalide : {e}\nContenu brut:\n{generated_json_str}")


def main() -> None:
    args = parse_arguments()
    check_and_prepare_paths(args.functions_definition, args.input, args.output)
    
    try:
        data_manager = DataParser(
            path_funct_definition=str(args.functions_definition),
            path_funct_calling=str(args.input)
        )

        print("Initializing the LLM model and vocabulary...")
        llm = Small_LLM_Model()
        vocab = VocabularyIndex(model=llm)

        # Le générateur est initialisé avec un état inactif, il sera écrasé dans process_single_prompt
        dummy_machine = JsonStateMachine(current_state=StateTerminal())
        generator = ConstrainedGenerator(llm=llm, vocab_index=vocab, machine=dummy_machine)

        results = []

        for test_case in data_manager.function_calling_tests:
            print(f"Processing: '{test_case.prompt}'...")

            try:
                result_dict = process_single_prompt(test_case.prompt, data_manager, generator)
                validated_result = FunctionCallResult.model_validate(result_dict)
                results.append(validated_result.model_dump())
                print(f"  ✓ Success: {result_dict['name']}")
            except Exception as e:
                print(f"  ✗ Error: {e}", file=sys.stderr)
                continue

        with open(args.output, 'w', encoding='utf-8') as file_out:
            json.dump(results, file_out, indent=2, ensure_ascii=False)
            
        print(f"\n✓ All results successfully saved to {args.output}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
