# __main__.py

import sys
import json
import argparse
from pathlib import Path

from src.parsing import DataParser
from src.vocabulary import VocabularyIndex
from src.generator import ConstrainedGenerator
from src.model_pydantic import FunctionCallResult
from src.json_generator import process_single_prompt_optimized
from llm_sdk import Small_LLM_Model

from src.state_machine import JsonStateMachine, StateTerminal


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

        dummy_machine = JsonStateMachine(current_state=StateTerminal())
        generator = ConstrainedGenerator(llm=llm, vocab_index=vocab, machine=dummy_machine)

        results = []

        for test_case in data_manager.function_calling_tests:
            print(f"Processing: '{test_case.prompt}'...")

            try:
                result_dict = process_single_prompt_optimized(
                    test_case.prompt, 
                    data_manager, 
                    generator
                )
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
