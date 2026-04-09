import sys
import json

from src.config import setup_configuration
from src.vocabulary import VocabularyIndex
from src.generator import ConstrainedGenerator
from src.model_pydantic import FunctionCallResult
from src.json_generator import process_single_prompt_optimized
from llm_sdk import Small_LLM_Model
from src.state_machine import JsonStateMachine, StateTerminal


def main() -> None:
    config = setup_configuration()

    try:
        print("Initializing the LLM model and vocabulary...")
        llm = Small_LLM_Model()
        vocab = VocabularyIndex()
        vocab.build_from_model(model=llm)

        initial_machine = JsonStateMachine(current_state=StateTerminal())
        generator = ConstrainedGenerator(
            llm=llm, vocab_index=vocab, machine=initial_machine)

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
                print(f"  ✗ Error on prompt '{test_case.prompt}': {e}",
                      file=sys.stderr)
                continue

        with config.output_path.open('w') as file_out:
            json.dump(results, file_out, indent=2, ensure_ascii=False)

        print(f"\n✓ All results successfully saved to {config.output_path}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
