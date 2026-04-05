import sys
import argparse
from pathlib import Path

from src.parsing import DataParser
from src.vocabulary import VocabularyIndex
from src.state_machine import JsonStateMachine, StateExpectLiteral
from llm_sdk import Small_LLM_Model


def DEBUG_display_list_token(llm : Small_LLM_Model):
        my_tokens = [
            6656, 515, 92163, 21509, 5134, 12306, 70180, 1060, 76325, 87079,
            24616, 12841, 1066, 7213, 30779, 5180, 39484, 13887, 46145, 19011,
            15429, 60998, 45128, 18507, 2559, 58958, 71248, 38484, 28247, 90,
            41056, 71779, 4710, 56940, 5238, 28802, 65668, 97417, 8333, 18574,
            33933, 73363, 13463, 664, 55447, 4257, 1698, 31906, 41636, 89253,
            77993, 22701, 11950, 25773, 688, 26285, 18611, 46771, 9401, 26809,
            90306, 10947, 4293, 198, 95429, 715, 42708, 6360, 27352, 37083,
            220, 41693, 59101, 58591, 36577, 94947, 88804, 6374, 31979, 79083,
            86766, 45807, 3824, 5872, 2290, 8945, 17648, 69877, 86770, 65271,
            1789, 2303, 256, 257, 59649, 260, 1797, 262, 10503, 6926,
            271, 3344, 96017, 786, 51475, 56596, 34583, 37144, 38171, 24348,
            14621, 286, 16159, 74525, 79133, 48426, 86827, 54060, 35117, 3374,
            14642, 1843, 310, 314, 15677, 11070, 18749, 22335, 49987, 81221,
            5959, 66376, 19273, 93004, 80719, 5968, 78672, 338, 34642, 341,
            34135, 25435, 10589, 1383, 33641, 17264, 79226, 51068, 1406, 6526,
            53632, 1920, 7561, 394, 14731, 40337, 414, 23459, 15270, 32678,
            88998, 3502, 26546, 51124, 14265, 39865, 10683, 79871, 47549, 82361,
            1476, 8136, 83913, 26065, 981, 14808, 4569, 77787, 75228, 9699,
            503, 999, 36845, 20974, 52720, 5108, 2549, 4597, 55799, 16885,
            15865, 63477, 1022, 61439]

        for token_id in my_tokens:
            token_str = llm.decode([token_id])
            print(f"{token_id:<10} | {repr(token_str)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test des modules internes (Parsing, Vocabulaire, Machine à états)")
    parser.add_argument("--functions_definition",
                        type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--input",
                        type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    return parser.parse_args()


def check_paths(func_def_path: Path, input_path: Path) -> None:
    for file_path in [func_def_path, input_path]:
        if not file_path.exists():
            print(f"Erreur : le fichier d'entrée '{file_path}' est introuvable.",
                  file=sys.stderr)
            sys.exit(1)


def main() -> None:
    args = parse_arguments()
    check_paths(args.functions_definition, args.input)

    print("--- Début des tests des fichiers joints ---")

    print("\n1. Test DataParser...")
    try:
        data_manager = DataParser(
            path_funct_definition=str(args.functions_definition),
            path_funct_calling=str(args.input)
        )
        print(f"{len(data_manager.functions_definition)} def load.")
        print(f"{len(data_manager.function_calling_tests)} tests load.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


    print("\n2. Test VocabularyIndex...")
    try:
        llm = Small_LLM_Model()
        vocab = VocabularyIndex(model=llm)
        print(f"  vocab load : {vocab.size} tokens.")
        print(f"{vocab.vocab_path}")
    except Exception as e:
        print(f" Error Vocab: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n3. Test JsonStateMachine...")
    try:
        state_machine = JsonStateMachine(
            current_state=StateExpectLiteral(expected="{", next_state=None)
        )
        print(f"Current state : {state_machine.current_state.__class__.__name__}")
    except Exception as e:
        print(f"Error state machine: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n All good")


if __name__ == "__main__":
    main()
