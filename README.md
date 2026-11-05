*This project has been created as part of the 42 curriculum by ariarcos.*

# Call Me Maybe

## Description

**Function calling** turns a natural-language request into a structured call. Instead of
answering *"What is the sum of 2 and 3?"* with `5`, the system produces:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The goal is to do this reliably with a very small model, **Qwen/Qwen3-0.6B**. Asked in plain words
to write JSON, such a model fails often, and this project never relies on it. It uses
**constrained decoding** instead: at every generation step, every token that would break the JSON
schema gets its logit set to `-inf`, so the model can only choose among valid continuations. The
program writes the JSON structure itself, and the model only fills three kinds of "holes": which
function to call, numeric values and string values.

Input files (in `data/input/` by default):

- `functions_definition.json`: the available functions, with their parameters and types.
- `function_calling_tests.json`: the prompts to process.

Output: `data/output/function_calling_results.json`, a JSON array with exactly the keys `prompt`,
`name` and `parameters` for each prompt.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). On the first run, the model
(~1.5 GB) is downloaded into the Hugging Face cache (`~/.cache/huggingface`). If your home
directory is short on space, point `HF_HOME` (and, for `uv sync`, `UV_CACHE_DIR`) to a larger
folder first.

```bash
uv sync                      # or: make install
uv run python -m src         # or: make run
```

Command-line options (all optional):

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Makefile rules:

| Rule | What it does |
|---|---|
| `make install` | `uv sync` |
| `make run` | runs the program with the default paths |
| `make debug` | runs the program under `pdb` |
| `make clean` | removes `__pycache__`, `.mypy_cache` and `.pytest_cache` |
| `make lint` | `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | unit tests (`pytest`); no model needed |
| `make accuracy` | strict accuracy of the last output against `test/expected_results.json` |

Set `CMM_DEBUG=1` to log the exact text the model generated, under the constraints, for each
prompt.

## Algorithm explanation

### 1. Vocabulary

`llm.get_path_to_vocab_file()` points to `vocab.json` (`{"token": id}`). Qwen uses **byte-level
BPE**: each token is a sequence of bytes, written in the JSON through the GPT-2 `bytes_to_unicode`
table (a space appears as `Ġ`, for example). `src/vocabulary.py` inverts that table and builds
`id → bytes` and `bytes → id`. Every mask is computed from those bytes.

### 2. Logit masking

`_pick(ids, allowed)` follows the subject's recipe literally. It gets the logits for the current
sequence, builds a vector full of `-inf`, copies back the logits of the allowed tokens and takes
the argmax (greedy). When only one token is allowed, the model is **not** called at all.

### 3. Forced structure

Every structural token is forced: `{"name": "`, `", "parameters": {`, each parameter key in the
order of the definition (all parameters are required), and the separators. The forced text uses
normal JSON spacing (`"a": ` before a number, `"s":` before a string), so the token sequence the
model sees looks like the JSON in its training data.

### 4. Choosing the function: a trie over token sequences

Each function name followed by `",` is tokenized. At each step, only the tokens that continue a
still-possible name are allowed. Shared prefixes (`fn`) are forced for free, the model decides at
the branching points, and the rest is forced once a single candidate remains. Booleans use the
same mechanism, choosing between ` true` and ` false`.

### 5. Numbers: a JSON number grammar

`number_next(prefix, integer)` returns the characters that may follow a prefix and whether the
prefix is already a complete number, according to `-?(0|[1-9][0-9]*)(\.[0-9]+)?` (with no dot for
`integer`). Qwen tokenizes digits one by one. Once the number is complete, the closing token (`,`
or `}`) is also allowed, so **the model decides when the number ends**. `number` values are emitted
as `float` and `integer` values as `int`.

### 6. Strings: a three-state automaton that understands escapes

The vocabulary is scanned once at start-up, for each state:

- `open`: before the opening quote. Allowed tokens start with ` "`, because Qwen often merges the
  quote with the first characters (`Ġ"/`, `Ġ"[`).
- `normal`: inside the string. Allowed tokens contain no unescaped quote and no control
  character; the closing tokens (`"`, `",`, `"}`...) are also allowed.
- `escaped`: right after an unpaired backslash, where a `\"` token is content, not the end of the
  string.

Bytes are accumulated and decoded only at the end, so multi-byte UTF-8 characters split across
tokens survive. `decode_json_string` interprets `\"`, `\\`, `\/` and `\uXXXX` and keeps any other
backslash sequence literally: in a one-line request, `\d` or `\b` is part of a regex, not a control
character. Length limits (64 tokens per string, 24 per number) guarantee termination.

### 7. Building the output

The values go into a Pydantic `FunctionCallResult` (`extra="forbid"`), which is written with
`json.dump(..., allow_nan=False)`. Constrained decoding guarantees the **schema**: the function
exists, every parameter is present and every type is correct. `json.dump` guarantees the
**syntax**.

## Design decisions

- **Fixed structure, model-filled holes.** Every parameter is required, so the keys can be emitted
  in definition order. The model never chooses braces, commas or keys, which means fewer states,
  fewer model calls and fewer ways to fail.
- **Token level, not string level.** Qwen merges JSON punctuation (`{"`, `":`, `",`). The masks
  work on token ids and their bytes, never by comparing strings.
- **The model chooses the opening quote.** Forcing a lone `Ġ"` produced values with a spurious
  leading space (`' /home/user/data.json'`, `' *'`), because Qwen rarely sees that token followed
  directly by `/` or `*`.
- **The prompt informs; the mask enforces.** The prompt uses Qwen's ChatML template and lists every
  function from the input file (nothing is hard-coded). An empty `<think></think>` block skips
  Qwen3's reasoning mode.
- **Greedy decoding**, so the output is deterministic across runs.
- **Pydantic everywhere.** Inputs are validated at load time (an unsupported type such as `array`
  fails immediately with a clear message). The generator class keeps its pre-computed tables as
  private attributes, and the output model cannot hold an extra key.
- **Fail fast, never crash.** Inputs are validated before the model is loaded. A single `AppError`
  type carries every user-facing message. A failing prompt is logged and skipped, the output file
  is written atomically, and `main` has a final safety net (exit code 1, or 130 on Ctrl+C).
- **Only the public SDK API**: the constructor, `encode`, `get_logits_from_input_ids`,
  `get_path_to_vocab_file` and, in debug mode only, `decode`. No file in `src/` imports `torch` or
  `transformers`.
- **When no function fits**, the most likely one is still chosen, because the output format
  requires a `name` from the list.

## Performance analysis

Measured on a 42 workstation (CPU only, no GPU), grading with the moulinette's own grader:

| Set | Moulinette score | Strict match (`make accuracy`) | Generation time |
|---|---|---|---|
| public (11 prompts) | **11/11 (100 %)** | 8/11 | 54 s |
| private (11 prompts) | **11/11 (100 %)** | 11/11 | 68 s |

- **Accuracy.** The moulinette runs each function with our parameters and compares the result.
  `make accuracy` compares the text of every parameter, so it is harsher: the three public
  "misses" are equivalent regexes (`([0-9]+)` instead of `\d+`, and `cat` instead of `\bcat\b` on
  a sentence where both give the same result).
- **Validity.** 100 % of the outputs parse as JSON and match the schema, by construction.
- **Speed.** A full `uv run python -m src` takes about 65 s, ~10 s of which is loading the model
  from a network drive. The main cost is the full forward pass that `get_logits_from_input_ids`
  runs on every call, since the public API exposes no KV cache. Forced tokens and single-choice
  steps skip the model entirely, so choosing a function usually costs a single forward. Short
  numbers and strings take 2–5 s per prompt; long strings, like an SQL query, take ~11 s.
- **Determinism.** Two consecutive runs produce byte-identical output files (checked with `diff`).

## Challenges faced

- **Strings vs tokens.** The first version was a state machine that compared exact strings. It
  could not handle BPE tokens such as `",` and expected a whole string value in a single token. It
  was replaced by token-level masks built from the vocabulary.
- **Escaped quotes.** The model writes `\"` as a single token. A naive "no quote inside" rule
  either forbids it or mistakes it for the end of the string. The `escaped` state, plus a prompt
  hint, fixed the private test `Say "hello" to {name}`.
- **Regex backslashes.** `json.loads` turns `\b` into a backspace, which broke `\bcat\b`. Hence a
  custom decoder that only interprets unambiguous escapes.
- **Leading spaces in strings**, caused by forcing an unnatural tokenization of the opening quote
  (see the design decisions).
- **Integer precision.** Numbers stay as text until the final `int(...)` or `float(...)`, so large
  integers are never rounded through `float`.
- **Environment.** Qwen3 requires `transformers >= 4.51`. The model and the `torch` wheel do not fit
  in a small home quota (`HF_HOME`, `UV_CACHE_DIR`), and CPU-only `torch` wheels avoid several GB
  of CUDA libraries.

## Testing strategy

- **Unit tests** (`make test`, 22 tests, no model): the number grammar; the string scanner and the
  escape decoder; the byte table and the vocabulary loader; the Pydantic models (extra keys
  rejected, types preserved); every input error (missing file, malformed JSON, empty list,
  unsupported type, duplicate names); and the output writer. The generator itself (`_choose`,
  `_number`, `_string`, `generate`) runs against a **fake LLM** that implements the SDK's public
  interface and follows a scripted preference. This also covers escaped quotes, empty strings and
  opening tokens.
- **Accuracy**: `make accuracy` compares names and parameters exactly, and the moulinette runs the
  real functions.
- **Error cases by hand**: missing input files, broken JSON, an empty function list, an `array`
  parameter, a JSON object instead of a list, a directory instead of a file. Each one gives a
  one-line message without a traceback.
- **Static checks**: `make lint` and `make lint-strict`.

## Usage examples

```text
$ uv run python -m src
INFO: Cargando el modelo...
INFO: [1/11] What is the sum of 2 and 3?
INFO:     -> fn_add_numbers {'a': 2.0, 'b': 3.0}
...
INFO: Generación terminada en 54.3 s
11 prompts procesados -> data/output/function_calling_results.json

$ uv run python -m src --input missing.json
Error: No existe el archivo: missing.json

$ uv run python -m src --functions_definition broken.json
Error: JSON inválido en broken.json (línea 2, columna 1): Expecting value
```

Output excerpt:

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  }
]
```

Grading with the moulinette (from its own folder):

```bash
uv run python -m moulinette prepare_exercises --set private --output data_private
# run the program on data_private/input/... with --output <file>, then:
uv run python -m moulinette grade_student_answers --set private --student_answer_path <file>
```

## Resources

- Vaswani et al., *Attention Is All You Need* (2017).
- Sennrich, Haddow and Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016):
  BPE.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019): byte-level BPE and
  `bytes_to_unicode`.
- Willard and Louf, *Efficient Guided Generation for Large Language Models* (2023): constrained
  decoding with automata (the Outlines library).
- Andrej Karpathy, *Let's build the GPT Tokenizer* (video, 2024).
- Qwen3 model card and chat template: <https://huggingface.co/Qwen/Qwen3-0.6B>.
- RFC 8259, the JSON data interchange format.
- Pydantic v2 documentation: <https://docs.pydantic.dev/>.
- uv documentation: <https://docs.astral.sh/uv/>.
- Study notes for this project (in Spanish) in `docs/`.

**How AI was used.** An AI assistant served as a reviewer and tutor for this project. It explained concepts
(byte-level BPE, logit masking, tries) and helped design and debug the token-level masks,
including the escape-aware string automaton and the opening-quote issue. It also helped write the
unit tests and the study notes in `docs/`. Every piece of code was read, run and tested before it
was kept.
