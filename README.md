# Model Benchmarking

A pipeline for benchmarking instruction-tuned LLMs served through the
[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers)
serverless API. It runs a suite of prompts across one or more models, scores the
answers with deterministic rule-based scorers, and records latency, throughput,
and quality to a local SQLite database. No local models and no GPU — inference
runs on provider hardware behind `huggingface_hub.InferenceClient`.

## Setup

1. **Create a Hugging Face account** at <https://huggingface.co> if you don't
   have one.
2. **Create an access token** with **Inference** permission at
   <https://huggingface.co/settings/tokens> (a fine-grained token with
   "Make calls to Inference Providers" enabled, or a classic read token). Gated
   models also need "Read access to public gated repos".
3. **Install dependencies and provide the token:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# HUGGINGFACEHUB_API_TOKEN is also accepted, as is a local .env file
# containing HF_TOKEN=... or passing --token on the command line.
```

A payment method on your HF account is required to route requests to
pay-as-you-go providers; usage for small prompt sets falls within the free
monthly inference credits.

## Models

The `--model` flag accepts either a built-in key or a raw HF repo id (anything
containing `/`). The built-in keys are all verified live on HF Inference
Providers via **pay-as-you-go** providers (Together / Nscale / Novita / Groq /
Scaleway). Gated models require accepting the license on the model page first.

| key                     | hf_id                                       | size_b | gated | provider |
| ----------------------- | ------------------------------------------- | ------ | ----- | -------- |
| `qwen2.5_7b`            | `Qwen/Qwen2.5-7B-Instruct`                  | 7.6    | no    | Together |
| `llama3.1_8b`           | `meta-llama/Llama-3.1-8B-Instruct`          | 8.0    | yes   | Novita/Nscale |
| `qwen3_8b`              | `Qwen/Qwen3-8B`                             | 8.2    | no    | Nscale   |
| `qwen3_14b`             | `Qwen/Qwen3-14B`                            | 14.8   | no    | Nscale   |
| `gemma3_27b`            | `google/gemma-3-27b-it`                     | 27.0   | yes   | Scaleway |
| `qwen2.5_coder_32b`     | `Qwen/Qwen2.5-Coder-32B-Instruct`           | 32.0   | no    | Nscale   |
| `qwen3_32b`             | `Qwen/Qwen3-32B`                            | 32.8   | no    | Groq/Nscale |
| `deepseek_r1_qwen_32b`  | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`  | 32.0   | no    | Nscale   |
| `llama3.3_70b`          | `meta-llama/Llama-3.3-70B-Instruct`         | 70.0   | yes   | Groq + many |

> **Note:** Featherless AI is a subscription provider and does not work through
> HF's routed pay-as-you-go billing, so models served only by Featherless (most
> sub-7B models) are excluded. Qwen3 and DeepSeek-R1 are reasoning models that
> may emit `<think>` blocks; these are stripped before scoring (see below).

## Usage

Run a whole prompt suite against one model, scoring every result:

```bash
python run_benchmark.py --model qwen2.5_7b --suite
sqlite3 data/results.db \
  "SELECT prompt_id, quality_score, latency_ms FROM runs ORDER BY prompt_id"
```

Run a single prompt:

```bash
python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test
sqlite3 data/results.db "SELECT model_id, latency_ms, ttft_ms, tokens_per_sec, output FROM runs"
```

Run arbitrary prompt text, or target a raw repo id directly:

```bash
python run_benchmark.py --model qwen3_8b --prompt-id sky --prompt-text "Why is the sky blue?"
python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --prompt-id smoke_test
```

`--suite` and `--prompt-id` are mutually exclusive. Results are cached per
`(model, prompt_id)`: a pair already in the database is skipped unless you pass
`--force`. The built-in `smoke_test` prompt uses scoring method `"manual"` and
is left unscored (`quality_score` NULL).

## Prompt suite format

Each file in `prompts/` is **one category**; the filename stem is the category
(e.g. `reasoning.json` → category `"reasoning"`). The top level is a **JSON
array** of prompt objects:

```json
{
  "id": "reasoning_001",          // unique across ALL files (required)
  "category": "reasoning",        // must equal the filename stem (required)
  "input": "What is 6 * 7?",      // the user prompt sent to the model (required)
  "expected_output": "42",        // gold answer / regex / pattern (required)
  "scoring_method": "numeric",    // one of the methods below (required)
  "scoring_args": { "tol": 0.5 }  // optional, method-specific (default {})
}
```

`load_prompts("prompts")` reads every `*.json` file, validates the required
fields, enforces that each prompt's `category` matches its filename stem, and
raises on any duplicate `id` across all files. Unknown extra keys are ignored.
The files shipped in `prompts/` are example prompts meant to be replaced or
expanded with your own.

## Scoring methods

`runner.scorer.score(output, prompt)` dispatches on `scoring_method`. An unknown
method raises `ValueError` (a config bug); every individual scorer is pure and
returns `0.0` rather than raising on bad model output.

| method        | what it does                                                                                 | `scoring_args`                          |
| ------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| `exact_match` | Normalized (strip, collapse whitespace, casefold) equality vs `expected_output` → 1.0/0.0    | `case_sensitive` (bool, default false)  |
| `contains`    | 1.0 if normalized `expected_output` is a substring of normalized output                       | `case_sensitive` (bool, default false)  |
| `regex`       | 1.0 if `re.search(expected_output, output)` matches                                          | `flags` (e.g. `"ignorecase"` or a list) |
| `numeric`     | Extract the **last** number from output (ints/decimals/signs/commas) and compare within `tol` | `tol` (float, default 1e-6)             |
| `json_valid`  | Output (or the first balanced `{...}`/`[...]` block in it) parses as JSON                     | `required_keys` (list; all must exist)  |

For `numeric`, phrase prompts so the model ends with the answer — the scorer
reads the last number in the output.

## Reasoning-model `<think>` stripping

Reasoning models (Qwen3, DeepSeek-R1-Distill) emit chain-of-thought wrapped in
`<think>...</think>` before the real answer. Before scoring, the runner applies
`runner.scorer.strip_reasoning(text)`:

- Removes every balanced `<think>...</think>` span (`DOTALL`, case-insensitive).
- **Unclosed opener:** if a `<think>` has no matching close (e.g. generation was
  truncated by `max_tokens`), everything from the opener to the end is dropped,
  keeping only the text before it.
- A stray orphan `</think>` is removed as noise.
- The result is whitespace-trimmed.

The **raw** model output (including any `<think>` block) is stored in the DB
`output` column. Only the `quality_score` is computed from the cleaned copy, so
no data is lost.

## Metrics

Each run records `latency_ms` (total wall time), `ttft_ms` (time to first token,
on the streaming path), `tokens_per_sec` (throughput), and a `quality_score` in
`[0, 1]`. There is no memory/VRAM metric: inference runs on the provider's
hardware, which exposes no honest memory figure. For memory characteristics,
consult each model's published size (`size_b`) and the provider's documentation.

## Project layout

```
model-benchmarking/
├── prompts/                # JSON prompt sets, one file per category
│   ├── reasoning.json
│   └── instruction_following.json
├── runner/
│   ├── __init__.py
│   ├── db.py               # SQLite schema + upsert helpers
│   ├── executor.py         # HF InferenceClient wrapper + metrics
│   ├── prompts.py          # prompt-suite loader + validation
│   └── scorer.py           # rule-based scoring + <think> stripping
├── data/                   # SQLite DB lives here (gitignored)
├── run_benchmark.py        # CLI entry point (single prompt or --suite)
├── scripts/verify_db.py    # network-free verification harness
├── requirements.txt
└── README.md
```

## Verification

`scripts/verify_db.py` exercises the DB layer, scorer, prompt loader, and CLI
plumbing end-to-end against a stubbed executor and a fake token — useful in CI or
any environment without network access. It checks schema creation, idempotent
upserts, `run_exists`, the full round-trip of the run columns, the
`UNIQUE(model_id, prompt_id)` cached-skip behavior, the missing-token error path,
`strip_reasoning`, every scoring method (a passing and a failing case each),
`load_prompts` (including duplicate-id rejection), and the `--suite` path
end-to-end. It does not replace a real run against the API; it only proves the
persistence, scoring, and CLI layers work.

```bash
python scripts/verify_db.py
```
