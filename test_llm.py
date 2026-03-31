from llm_sdk import Small_LLM_Model


class FunctionCallGenerator:
    def __init__(self):
        self.model: llm_sdk = Small_LLM_Model()
        self.vocab_path: llm_sdk = self.get_path_to_vocab_file()
    
    def prompt(self, prompt)--> None:


def main() -> None:
    print("Initialisation du modèle (chargement en mémoire)...")
    model = Small_LLM_Model()

    prompt = "What is the sum of 2 and 3?"
    print(f"\nPrompt original : '{prompt}'")

    # 1. On teste l'encodage
    token_ids = model.encode(prompt)
    print(f"Tokens (Input IDs) : {token_ids}")

    # 2. On teste le décodage
    decoded_text = model.decode(token_ids)
    print(f"Texte décodé : '{decoded_text}'")

    # 3. On récupère le chemin du dictionnaire
    vocab_path = model.get_path_to_vocab_file()
    print(f"Chemin vers le fichier vocabulaire : {vocab_path}")


if __name__ == "__main__":
    main()
