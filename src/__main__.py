import sys
import json

from src.config import setup_configuration
from src.functions_validator import FunctionCallResult
from src.vocabulary import VocabIndex
from src.generator import ConstrainedGenerator
from src.json_generator import process_single_prompt_optimized
from src.state_machine import JsonStateMachine, StateTerminal
from llm_sdk import Small_LLM_Model


def main() -> None:
    config = setup_configuration()
    print("Initializing the LLM model and vocabulary...")

    try:
        llm = Small_LLM_Model()
        vocab = VocabIndex.from_model(llm)
    except RuntimeError as e:
        sys.exit(f"Error memory GPU/CPU during the loading : {e}")
    except Exception as e:
        sys.exit(f"CRITICAL ERROR: {e}")

    initial_machine = JsonStateMachine(current_state=StateTerminal())
    generator = ConstrainedGenerator(
        llm=llm,
        vocab_index=vocab,
        machine=initial_machine
    )

    results = []

    for test_case in config.function_calling_tests:
        print(f"Processing: '{test_case.prompt}'...")
        try:
            result_dict = process_single_prompt_optimized(
                test_case.prompt,
                config.functions_definition,
                generator
            )
            validated_result = FunctionCallResult.model_validate(
                result_dict)
            results.append(validated_result.model_dump())
            print(f"  ✓ Success: {result_dict.get('name', 'Unknown')}")

        except Exception as e:
            sys.exit(f"  ✗ Error on prompt '{test_case.prompt}': {e}")
            continue

        if results:
            try:
                with config.output_path.open('w') as file_out:
                    json.dump(results, file_out, indent=2, ensure_ascii=False)
            except OSError as e:
                sys.exit(f"  ✗ Error saving the JSON file: {e}")
        else:
            print("\n No results were generated. File not saved.")

        print(f"\n✓ All results successfully saved to {config.output_path}")


if __name__ == "__main__":
    main()
