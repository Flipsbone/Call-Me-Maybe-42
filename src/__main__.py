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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling")
    parser.add_argument("--functions_definition",
                        type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--input",
                        type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument("--output",
                        type=Path,
                        default=Path(
                            "data/output/function_calling_results.json"))
    return parser.parse_args()


def check_and_prepare_paths(
        func_def_path: Path,
        input_path: Path,
        output_path: Path) -> None:

    for file_path in [func_def_path, input_path]:
        if not file_path.exists():
            print(f"Error : input file '{file_path}' not found.",
                  file=sys.stderr)
            sys.exit(1)

    output_dir = output_path.parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error : {output_dir} : {e}", file=sys.stderr)
            sys.exit(1)


def build_prompt(user_request: str, functions: list) -> str:
    prompt_text = "Available functions:\n"
    for function in functions:
        prompt_text += f"- {function.name}: {function.description}\n"
    prompt_text += f"\nUser request: {user_request}\nJSON Output:\n"
    return prompt_text


def process_single_prompt(
    user_prompt: str,
    data_manager: DataParser,
    generator: ConstrainedGenerator,
    vocab: VocabularyIndex
) -> dict[str, Any]:

    # Initialisation du contexte
    full_prompt = build_prompt(user_prompt, data_manager.functions_definition)
    input_ids = generator.llm.encode(full_prompt)
    generated_json_str = ""

    # 1. Ouverture et clé 'name'
    seq_start = '{\n  "name": "'
    input_ids = generator.force_sequence(input_ids, seq_start)
    generated_json_str += seq_start

    # 2. Sélection de la fonction
    func_names = [fn.name for fn in data_manager.functions_definition]
    input_ids, chosen_func = generator.choose_from_list(input_ids, func_names)
    generated_json_str += chosen_func

    # On récupère le schéma de la fonction choisie
    selected_fn = None
    for f in data_manager.functions_definition:
        if f.name == chosen_func:
            selected_fn = f

    # 3. Transition vers 'parameters'
    seq_params = '",\n  "parameters": {'
    input_ids = generator.force_sequence(input_ids, seq_params)
    generated_json_str += seq_params

    # 4. Remplissage des paramètres si la fonction en possède
    if selected_fn and selected_fn.parameters:
        param_items = list(selected_fn.parameters.items())
        num_params = len(param_items)

        for i in range(num_params):
            p_name = param_items[i][0]
            p_model = param_items[i][1]

            # Forcer la clé du paramètre
            seq_p_key = f'\n    "{p_name}": '
            input_ids = generator.force_sequence(input_ids, seq_p_key)
            generated_json_str += seq_p_key

            # Récupération des tokens valides selon le type du paramètre
            allowed_tokens = set()
            if p_model.type == "number":
                allowed_tokens = vocab.number_syntax_ids
            elif p_model.type == "string":
                allowed_tokens = vocab.quote_ids | vocab.general_text_ids
            elif p_model.type == "boolean":
                allowed_tokens = (
                    vocab.boolean_true_ids |
                    vocab.boolean_false_ids
                    )
            else:
                allowed_tokens = (
                    vocab.general_text_ids |
                    vocab.number_syntax_ids
                )

            # Génération de la valeur contrainte
            input_ids, val_str = generator.generate_value_for_type(
                input_ids, allowed_tokens
                )
            generated_json_str += val_str

            # Ajout de la virgule si ce n'est pas le dernier paramètre
            if i < num_params - 1:
                input_ids = generator.force_sequence(input_ids, ',')
                generated_json_str += ','

        # Fermeture de l'objet parameters
        seq_end_params = '\n  }'
        input_ids = generator.force_sequence(input_ids, seq_end_params)
        generated_json_str += seq_end_params
    else:
        # Si la fonction n'a aucun paramètre, on ferme l'accolade vide
        seq_end_params = '}'
        input_ids = generator.force_sequence(input_ids, seq_end_params)
        generated_json_str += seq_end_params

    # 5. Clôture du JSON
    seq_end = '\n}'
    input_ids = generator.force_sequence(input_ids, seq_end)
    generated_json_str += seq_end

    # 6. Validation Pydantic et retour
    try:
        parsed_data = json.loads(generated_json_str)
        return {
            "prompt": user_prompt,
            "name": parsed_data["name"],
            "parameters": parsed_data.get("parameters", {})
        }
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Le JSON généré est invalide : {e}\nContenu: "
                           f"{generated_json_str}"
                           )


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
        vocab = VocabularyIndex(llm)
        generator = ConstrainedGenerator(llm)

        results = []

        # Traitement de chaque question
        for test_case in data_manager.function_calling_tests:
            print(f"Processing: '{test_case.prompt}'...")

            # Génération contrainte
            result_dict = process_single_prompt(
                test_case.prompt,
                data_manager,
                generator,
                vocab
                )

            # Validation finale via Pydantic pour s'assurer du format parfait
            validated_result = FunctionCallResult.model_validate(result_dict)
            results.append(validated_result.model_dump())

        # Sauvegarde dans le fichier de sortie
        with open(args.output, 'w') as file_out:
            json.dump(results, file_out, indent=2, ensure_ascii=False)
        print(f"\nSuccess! All results have been saved to {args.output}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
