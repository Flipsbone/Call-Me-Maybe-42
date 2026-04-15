import sys
import json
import time

from llm_sdk import Small_LLM_Model
from src.config import setup_configuration
from src.functions_validator import FunctionCallResult
from src.vocabulary import VocabIndex
from src.generator import ConstrainedGenerator
from src.json_generator import (process_single_prompt_optimized,
                                GenerationJsonError)
from src.state_machine import StateTerminal


class Theme:
    """ANSI color codes used for terminal status messages."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def main() -> None:
    """Run the function-calling generation workflow."""
    start_time = time.time()

    (output_path,
     functions_definition,
     function_calling_tests) = setup_configuration()

    print("Initializing the LLM model and vocabulary...")

    try:
        llm = Small_LLM_Model()
        print(f"The LLM used is: {llm._model_name}")
        vocab = VocabIndex.from_model(llm)
    except RuntimeError as e:
        sys.exit(f"Error memory GPU/CPU during the loading : {e}")
    except Exception as e:
        sys.exit(f"CRITICAL ERROR: {e}")

    generator = ConstrainedGenerator(
        llm=llm,
        vocab_index=vocab,
        current_state=StateTerminal()
    )

    results = []
    try:
        for test_case in function_calling_tests:
            print(f"{Theme.CYAN}Processing: '{test_case.prompt}'"
                  f"...{Theme.RESET}")
            try:
                result_dict = process_single_prompt_optimized(
                    test_case.prompt,
                    functions_definition,
                    generator
                )
                validated_result = FunctionCallResult.model_validate(
                    result_dict)
                results.append(validated_result.model_dump())
                print(f" {Theme.GREEN}✓ Success:{Theme.RESET}"
                      f"{result_dict.get('name', 'Unknown')}")

            except ValueError as e:
                print(f"  {Theme.RED}✗ Generation error: {e}{Theme.RESET}")
                continue
            except GenerationJsonError as e:
                print(f"  {Theme.RED}✗ JSON decode error: {e}{Theme.RESET}")
                continue
            except Exception as e:
                print(f"  {Theme.RED}✗ Unexpected error: {e}{Theme.RESET}")
                continue

            if results:
                try:
                    with output_path.open('w') as file_out:
                        json.dump(
                            results, file_out, indent=2, ensure_ascii=False)
                except OSError as e:
                    sys.exit(f"  ✗ Error saving the JSON file: {e}")
            else:
                print("\n No results were generated. File not saved.")

        print(f"\n✓ All results successfully saved to {output_path}")

    finally:
        elapsed_time = time.time() - start_time
        print(f"\n{Theme.CYAN}Total execution time:"
              f"{Theme.GREEN}{elapsed_time:.2f}"
              f"{Theme.CYAN} seconds.{Theme.RESET}")


if __name__ == "__main__":
    main()
