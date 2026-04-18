*This project has been created as part of the 42 curriculum by Advacher*

# Call Me Maybe - A Beginner's Guide to AI Function Calling

Welcome to **Call Me Maybe**!  If you are new to the world of Artificial Intelligence, you might have noticed that AIs are great at chatting, but sometimes struggle with following strict computer rules. 

This project solves that problem. It takes a compact AI model (**Qwen3-0.6B**) and guides it to output data in a perfect, computer-readable format called **JSON**. This allows the AI to reliably "call functions" (use tools, APIs, or scripts) without ever making a syntax error.

## 1. Description
Normally, if you ask an AI to extract a user's age and name, it might reply: *"Sure! The user's name is Alice and she is 25."* We understands this, but a computer program cannot process it directly. A computer needs something structured, like: `{"name": "Alice", "age": 25}`.

**Call Me Maybe** acts as a "logic guardrail" for the AI. It uses a technique called **Constrained Decoding** to physically prevent the AI from typing anything that doesn't fit a valid JSON format. It guarantees 100% perfect formatting every single time.

## 2. Instructions
To run this project, we use a tool called **uv**. It is a high-performance manager that handles all the setup so you don't have to worry about missing libraries.

### Step 1: Install uv (if you don't have it)
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```


### Step 2: Setup the Project
Navigate to the project folder and prepare the environment:
```bash
uv sync
```

### Step 3: Run the Program
```bash
Make run 
```
Use the command below to start the function-calling process (see section 9 for details).

## 3. Resources

Technical References

* **[Qwen Documentation](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)**: I referenced the official Qwen documentation to achieve greater precision in managing and constraining the model's output format.
* **[JSON Schema Standard](https://www.json.org/json-en.html)**: Documentation on the structure used to validate function arguments.
* **[Logit Masking Theory & LLMs (3Blue1Brown)](https://www.youtube.com/c/3blue1brown)**: Research and visual explanations on guided generation, explain neural nets, and theory behind constrained decoding and how does an llm work.

AI Usage & Assistance


* Architecture Design: Brainstorming the state machine transitions for various JSON data types (Strings, Numbers, Objects).

* Code Refactoring: Optimizing the ConstrainedGenerator loop to minimize latency during the decoding phase.

* Technical Writing: Drafting and refining documentation to meet professional standards.

* Debugging: Generating edge-case unit tests to ensure the regex-based parsing handled escaped characters correctly.

## 4. Algorithm Explanation
4. Algorithm Explanation

My approach to guaranteed JSON generation relies on a **Context-Free Grammar (CFG) Constrained Decoder** powered by a custom **Finite State Machine (FSM)**. Instead of parsing the output *after* generation, I mathematically restrict the model's choices *during* generation.

The process operates in a two-step orchestration (`TwoStepJsonGenerator`):

1. **Phase 1 (Function Routing):** The model is forced to output only a valid function name from the provided schema using a `StateBranch`.

2. **Phase 2 (Parameter Extraction):** The engine dynamically builds a linked chain of states (`StateExpectLiteral`, `StateParseString`, `StateParseNumber`) corresponding exactly to the chosen function's parameters.

**The Constrained Decoding Engine (`ConstrainedGenerator`):**
For every single step of generation, the FSM executes a **Logit Masking** protocol:
* The current state (e.g., `StateParseString`) calculates a strict `set` of allowed Token IDs (the "valid tokens").

* I retrieve the raw probability scores (`logits`) from the Qwen model.

* I apply a mathematical mask: all tokens *not* in the valid set are ignored.

* The model is forced to choose the highest-scoring token exclusively from the mathematically valid subset.


## 5. Design Decisions
* **Qwen3-0.6B Model:** Selected for its minimal memory footprint and high inference speed. Being a 0.6B parameter model, it runs efficiently on standard hardware.
* **Pure Greedy Decoding:** All generation uses absolute greedy decoding (`np.argmax` with Temperature = 0). When extracting structured API parameters, "creativity" is a liability. Greedy decoding ensures deterministic, predictable, and highly accurate data extraction.
* **Literal Short-Circuiting:** In JSON, much of the structure is predictable (e.g., `{\n  "name": `). When the FSM enters a `StateExpectLiteral`, **the LLM is completely bypassed**. The text is injected directly into the buffer. 
* **String Fast-Path:** Inside `StateParseString`,a "fast path" approves over 30,000 "safe tokens" instantly, reserving heavy Regex logic only for quotes or escape characters.

## 6. Performance Analysis
* **Accuracy (100% Syntax Validity):** Due to strict logit masking, "Invalid JSON" exceptions are eliminated. The FSM guarantees that brackets close, quotes match, and commas are correctly placed.
* **Speed (High TPS):** The combination of the lightweight Qwen3-0.6B model and our "Literal Short-Circuiting" optimization means the engine generates structural JSON almost instantaneously. Only the actual parameter values require LLM computation.
* **Reliability:** By splitting the generation into two phases (Function Name -> Parameters), the cognitive load on the LLM is reduced, resulting in highly reliable tool selection even with ambiguous user prompts.

## 7. Challenges Faced
The most significant technical hurdle was handling **BPE Token Boundaries**.

* **The Problem:** Tokens frequently span across two distinct JSON structural boundaries (e.g., a token containing both a number and a comma).

* **The Solution:** I engineered an **Overflow Buffer** system. When a token is selected, the state validates it, consumes necessary characters, and slices the remainder to be passed to the next_state.

## 8. Testing Strategy
To ensure the robustness of the FSM and the logit masking, I employed a multi-layered testing strategy:
* **Token Slicing Tests:** Validating the Overflow Buffer by feeding the FSM awkward, multi-boundary tokens (e.g., testing if `StateParseString` properly halts and passes `,` when fed `Alice", `).
* **Regex Edge Cases:** Testing `StateParseNumber` against edge cases like decimals (`0.5`), ensuring it correctly awaits JSON delimiters (`,`, `}`, `]`, `\n`) before validating the end of a digit.
* **Integration Testing:** Running the `process_single_prompt_optimized` wrapper against a diverse suite of prompts (from simple math queries to complex string replacement requests) to ensure the dynamically generated FSM successfully parses.

## 9. Example Usage
You can run the end-to-end evaluation pipeline using the `uv` package manager. This command will parse the provided prompts, match them to the available functions, and output the perfectly constrained JSON calls.

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## 10. Project Architecture
Below is a visual representation of how the different components of **Call Me Maybe** interact:
```mermaid
graph TD
    classDef main_file fill:#eceff1,stroke:#607d8b,stroke-width:2px,color:#333;
    classDef init_file fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#333;
    classDef core_file fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#333;
    classDef rule_file fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#333;
    classDef valid_file fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#333;

    subgraph Orchestration ["The Orchestration"]
        Main(("<b>__main__.py</b>"))
    end
    class Main main_file;

    subgraph Setup ["1. The Preparation"]
        Data["<b>data_loader.py</b>\nLoads inputs/args"]
        Vocab["<b>vocabulary.py</b>\nIndexes allowed tokens"]
        Main -->|"1. Initializes"| Data
        Main -->|"2. Loads"| Vocab
    end
    class Data,Vocab init_file;

    subgraph Core ["2. The Heart"]
        JsonGen{"<b>json_generator.py</b>\nOrchestrates generation"}
        Gen["<b>constrained_decoder.py</b>\nCalls LLM"]
        State["<b>state_machine.py</b>\nEnforces JSON syntax"]
        
        Main -->|"3. Delegates"| JsonGen
        JsonGen <-->|"Directs flow"| Gen
        Gen <-->|"Validates tokens"| State
    end
    class JsonGen,Gen core_file;
    class State rule_file;

    subgraph Verification ["3. The Quality Check"]
        Valid["<b>functions_validator.py</b>\nValidates final schema"]
        JsonGen -->|"4. Submits work"| Valid
        Valid -->|"5. Saves results"| Main
    end
    class Valid valid_file;