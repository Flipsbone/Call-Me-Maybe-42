from enum import Enum, auto
from src.model_pydantic import FunctionDefinition
from src.vocabulary import VocabularyIndex


class GenerationState(Enum):
    # --- Séquence d'ouverture ---
    EXPECTING_ROOT_BRACE = auto()       # Attend "{"
    EXPECTING_NAME_KEY = auto()         # Attend '"name"'
    EXPECTING_NAME_COLON = auto()       # Attend ":"

    # --- Sélection de la fonction ---
    EXPECTING_FUNCTION_NAME = auto()    # Attend '"fn_add"', '"fn_greet"', etc.

    # --- Transition vers les paramètres ---
    EXPECTING_COMMA_AFTER_NAME = auto()  # Attend ","
    EXPECTING_PARAMS_KEY = auto()        # Attend '"parameters"'
    EXPECTING_PARAMS_COLON = auto()      # Attend ":"
    EXPECTING_PARAMS_BRACE = auto()      # Attend "{"

    # --- Remplissage des paramètres ---
    EXPECTING_PARAM_KEY = auto()        # Attend clé du paramètre (ex: '"a"')
    EXPECTING_PARAM_COLON = auto()      # Attend ":"
    EXPECTING_PARAM_VALUE = auto()      # Attend valeur (type du paramètre)
    EXPECTING_PARAM_SEPARATOR = auto()  # Attend "," (autre param)ou "}" (fini)

    # --- Clôture ---
    EXPECTING_FINAL_BRACE = auto()      # Attend "}"
    DONE = auto()


class SchemaStateMachine:
    def __init__(self,
                 vocab_index: VocabularyIndex,
                 available_functions: list[FunctionDefinition]
                 ):
        self.vocab = vocab_index
        self.functions_catalog: dict[str, FunctionDefinition] = {
            fn.name: fn for fn in available_functions
            }

        # --- Mémoire de l'état ---
        self.current_state = GenerationState.EXPECTING_ROOT_BRACE
        self.selected_function: FunctionDefinition | None
        self.remaining_parameters: list[str] = []
        self.current_parameter_type: str | None

    def get_allowed_tokens(self) -> set[int]:
        """Retourne les tokens autorisés selon l'état actuel du protocole."""

        match self.current_state:
            case GenerationState.EXPECTING_ROOT_BRACE:
                return self.vocab.brace_open_ids | self.vocab.whitespace_ids

            case GenerationState.EXPECTING_NAME_KEY:
                # Ici, la logique idéale est de forcer
                # les tokens exacts pour '"name"'
                return self.vocab.quote_ids | self.vocab.general_text_ids

            case (GenerationState.EXPECTING_NAME_COLON |
                  GenerationState.EXPECTING_PARAMS_COLON |
                  GenerationState.EXPECTING_PARAM_COLON
                  ):
                return self.vocab.colon_ids | self.vocab.whitespace_ids

            case GenerationState.EXPECTING_FUNCTION_NAME:
                # Le vigile bloque tout sauf les noms de fonctions valides.
                return self.vocab.quote_ids | self.vocab.general_text_ids

            case GenerationState.EXPECTING_COMMA_AFTER_NAME:
                return self.vocab.comma_ids | self.vocab.whitespace_ids

            case GenerationState.EXPECTING_PARAMS_KEY:
                # Forcer les tokens pour '"parameters"'
                return self.vocab.quote_ids | self.vocab.general_text_ids

            case GenerationState.EXPECTING_PARAMS_BRACE:
                return self.vocab.brace_open_ids | self.vocab.whitespace_ids

            case GenerationState.EXPECTING_PARAM_KEY:
                # On autorise uniquement les tokens
                # correspondant aux clés dans self.remaining_parameters
                return self.vocab.quote_ids | self.vocab.general_text_ids

            case GenerationState.EXPECTING_PARAM_VALUE:
                # Le vigile regarde le schéma du paramètre actuel
                if self.current_parameter_type == "number":
                    return (self.vocab.number_syntax_ids |
                            self.vocab.whitespace_ids
                            )

                elif self.current_parameter_type == "string":
                    return (self.vocab.quote_ids |
                            self.vocab.general_text_ids |
                            self.vocab.whitespace_ids
                            )

                elif self.current_parameter_type == "boolean":
                    return (self.vocab.boolean_true_ids |
                            self.vocab.boolean_false_ids |
                            self.vocab.whitespace_ids
                            )

                return set()  # Sécurité en cas de type inconnu

            case GenerationState.EXPECTING_PARAM_SEPARATOR:
                # Si list vide -> attend "}", sinon attend ","
                if not self.remaining_parameters:
                    return (self.vocab.brace_close_ids |
                            self.vocab.whitespace_ids
                            )
                else:
                    return self.vocab.comma_ids | self.vocab.whitespace_ids

            case GenerationState.EXPECTING_FINAL_BRACE:
                return self.vocab.brace_close_ids | self.vocab.whitespace_ids

            case GenerationState.DONE:
                return set()  # Fini, plus de tokens autorisés.

    def transition(self, token_str: str) -> None:
        clean_token = token_str.strip()
        if not clean_token:
            return  # Ignorer les espaces

        match self.current_state:
            case GenerationState.EXPECTING_ROOT_BRACE:
                if '{' in clean_token:
                    self.current_state = GenerationState.EXPECTING_NAME_KEY

            case GenerationState.EXPECTING_NAME_KEY:
                if 'name' in clean_token or '"' in clean_token:
                    self.current_state = GenerationState.EXPECTING_NAME_COLON

            case GenerationState.EXPECTING_NAME_COLON:
                if ':' in clean_token:
                    self.current_state = (
                        GenerationState.EXPECTING_FUNCTION_NAME
                        )

            case GenerationState.EXPECTING_FUNCTION_NAME:
                # --- ÉTAPE CRUCIALE ---
                found = False
                for fn_name in self.functions_catalog.keys():
                    if not found and fn_name in clean_token:
                        self.selected_function = (
                            self.functions_catalog[fn_name]
                            )
                        self.remaining_parameters = list(
                            self.selected_function.parameters.keys())
                        self.current_state = (
                            GenerationState.EXPECTING_COMMA_AFTER_NAME
                        )
                        found = True

            case GenerationState.EXPECTING_COMMA_AFTER_NAME:
                if ',' in clean_token:
                    self.current_state = GenerationState.EXPECTING_PARAMS_KEY

            case GenerationState.EXPECTING_PARAMS_KEY:
                if 'parameters' in clean_token or '"' in clean_token:
                    self.current_state = GenerationState.EXPECTING_PARAMS_COLON

            case GenerationState.EXPECTING_PARAMS_COLON:
                if ':' in clean_token:
                    self.current_state = GenerationState.EXPECTING_PARAMS_BRACE

            case GenerationState.EXPECTING_PARAMS_BRACE:
                if '{' in clean_token:
                    # Si la fonction n'a pas de paramètres,
                    # on passe directement au séparateur/fermeture
                    if not self.remaining_parameters:
                        self.current_state = (
                            GenerationState.EXPECTING_PARAM_SEPARATOR
                            )
                    else:
                        self.current_state = (
                            GenerationState.EXPECTING_PARAM_KEY
                            )

            case GenerationState.EXPECTING_PARAM_KEY:
                if self.selected_function:
                    found_param = False
                    # On itère sur une copie pour pouvoir modifier
                    # la liste originale en toute sécurité
                    remaining_copy = list(self.remaining_parameters)

                    for p_name in remaining_copy:
                        if not found_param and p_name in clean_token:
                            # Mise à jour du type attendu pour le Vigile
                            self.current_parameter_type = (
                                self.selected_function.parameters[p_name].type
                                )
                            self.remaining_parameters.remove(p_name)
                            self.current_state = (
                                GenerationState.EXPECTING_PARAM_COLON
                                )
                            found_param = True

            case GenerationState.EXPECTING_PARAM_COLON:
                if ':' in clean_token:
                    self.current_state = GenerationState.EXPECTING_PARAM_VALUE

            case GenerationState.EXPECTING_PARAM_VALUE:
                # Cet état est maintenu jusqu'à ce
                # que la valeur soit entièrement générée.
                # Ton générateur (qui s'arrête aux
                # virgules ou accolades) appellera ensuite
                # une transition manuelle vers le séparateur,
                # ou passera la virgule générée.
                pass

            case GenerationState.EXPECTING_PARAM_SEPARATOR:
                if ',' in clean_token:
                    self.current_state = GenerationState.EXPECTING_PARAM_KEY
                elif '}' in clean_token:
                    self.current_state = GenerationState.EXPECTING_FINAL_BRACE

            case GenerationState.EXPECTING_FINAL_BRACE:
                if '}' in clean_token:
                    self.current_state = GenerationState.DONE
