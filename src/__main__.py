import sys
import json
import time
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model
from src.data_loader import parse_arguments_and_load_data
from src.functions_validator import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionCallingTest)
from src.vocabulary import VocabIndex
from src.constrained_decoder import ConstrainedDecoder
from src.json_generator import TwoStepJsonGenerator, GenerationJsonError


def init_ai() -> ConstrainedDecoder:
    print("Initializing the LLM model and vocabulary...")
    try:
        llm = Small_LLM_Model()
        print("Qwen/Qwen3-0.6B model loaded successfully.")
        vocab = VocabIndex.from_model(llm)
        return ConstrainedDecoder(llm=llm, vocab_index=vocab)
    except Exception as e:
        sys.exit(f"CRITICAL ERROR during initialization: {e}")


def process_all_prompts(
        calling_tests: list[FunctionCallingTest],
        functions_def: list[FunctionDefinition],
        assistant: ConstrainedDecoder) -> list[dict[str, Any]]:

    results: list[dict[str, Any]] = []

    for user_prompt in calling_tests:
        print(f"Processing: '{user_prompt.prompt}'...")
        try:
            json_gen = TwoStepJsonGenerator(
                user_prompt=user_prompt.prompt,
                functions_definition=functions_def,
                assistant=assistant
            )
            result_dict = json_gen.generate()

            validated_result = FunctionCallResult.model_validate(result_dict)
            results.append(validated_result.model_dump())
            print(f"  ✓ Success: {result_dict.get('name')} "
                  f"{result_dict.get('parameters')}")

        except (ValueError, GenerationJsonError) as e:
            print(f"  ✗ Generation error: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        print("\nNo results generated. File not saved.")
        return

    try:
        with output_path.open('w') as file_out:
            json.dump(results, file_out, indent=2, ensure_ascii=False)
        print(f"\n✓ All results successfully saved to {output_path}")
    except OSError as e:
        sys.exit(f"  ✗ Error saving the JSON file: {e}")


def main() -> None:
    start_time: float = time.time()
    output_path, functions_def, tests = parse_arguments_and_load_data()
    assistant: ConstrainedDecoder = init_ai()
    results: list[dict[str, Any]] = process_all_prompts(
        tests, functions_def, assistant)
    save_results(results, output_path)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
