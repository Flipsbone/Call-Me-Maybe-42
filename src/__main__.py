import sys
import json
import time

from pydantic import BaseModel, Field
from llm_sdk import Small_LLM_Model
from src.config import setup_configuration
from src.functions_validator import FunctionCallResult
from src.vocabulary import VocabIndex
from src.generator import ConstrainedGenerator
from src.json_generator import (process_single_prompt_optimized,
                                GenerationJsonError)
from src.state_machine import JsonStateMachine, StateTerminal


class TerminalColor(BaseModel):
    GREEN: str = Field(default="\033[92m", frozen=True)
    RED: str = Field(default="\033[91m", frozen=True)
    CYAN: str = Field(default="\033[96m", frozen=True)
    RESET: str = Field(default="\033[0m", frozen=True)


def main() -> None:
    start_time = time.time()
    theme = TerminalColor()
    config = setup_configuration()
    print("Initializing the LLM model and vocabulary...")

    try:
        llm = Small_LLM_Model()
        print(f"The LLM used is: {llm._model_name}")
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
    try:
        for test_case in config.function_calling_tests:
            print(f"{theme.CYAN}Processing: '{test_case.prompt}'"
                  f"...{theme.RESET}")
            try:
                result_dict = process_single_prompt_optimized(
                    test_case.prompt,
                    config.functions_definition,
                    generator
                )
                validated_result = FunctionCallResult.model_validate(
                    result_dict)
                results.append(validated_result.model_dump())
                print(f"  {theme.GREEN}✓ Success:{theme.RESET}"
                      f"{result_dict.get('name', 'Unknown')}")

            except ValueError as e:
                print(f"  {theme.RED}✗ Generation error: {e}{theme.RESET}")
                continue
            except GenerationJsonError as e:
                print(f"  {theme.RED}✗ JSON decode error: {e}{theme.RESET}")
                continue
            except Exception as e:
                print(f"  {theme.RED}✗ Unexpected error: {e}{theme.RESET}")
                continue

            if results:
                try:
                    with config.output_path.open('w') as file_out:
                        json.dump(
                            results, file_out, indent=2, ensure_ascii=False)
                except OSError as e:
                    sys.exit(f"  ✗ Error saving the JSON file: {e}")
            else:
                print("\n No results were generated. File not saved.")
        print(f"\n✓ All results successfully saved to {config.output_path}")

    except KeyboardInterrupt:
        print(f"\n{theme.RED}✗ User interrupted (Ctrl+C)."
              f"Shutting down...{theme.RESET}")
        print("Saving processed results...")

        if results:
            try:
                with config.output_path.open('w') as file_out:
                    json.dump(results, file_out, indent=2, ensure_ascii=False)
            except OSError as e:
                sys.exit(f"  ✗ Error saving the JSON file: {e}")
        else:
            print("\n No results were generated. File not saved.")
        print(f"\n✓ All results successfully saved to {config.output_path}")

    finally:
        elapsed_time = time.time() - start_time
        print(f"\n{theme.CYAN}Total execution time:"
              f"{theme.GREEN}{elapsed_time:.2f}"
              f"{theme.CYAN} seconds.{theme.RESET}")


if __name__ == "__main__":
    main()
