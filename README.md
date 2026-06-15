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
> may emit `<think>` blocks; these are stripped before scoring (see below). They
> also carry a larger per-model token budget so the answer survives the
> chain-of-thought — see [Token budget](#token-budget).

### Token budget

`--max-tokens` caps generated tokens per call. Reasoning models (`qwen3_8b`,
`qwen3_14b`, `qwen3_32b`, `deepseek_r1_qwen_32b`) spend most of their budget on a
`<think>` chain-of-thought before answering, so the default 512 is too small: the
final answer never lands (empty `content`) or the `<think>` block is truncated
unclosed and stripped to nothing. These models therefore carry a per-model
`max_tokens` of **4096** in `BUILTIN_MODELS`; the others have no per-model budget
and use the default **512**.

The budget for each call is resolved by **precedence (highest first)**:

1. An explicit `--max-tokens N` on the command line **overrides everything**, for
   every selected model.
2. Otherwise the model's per-model `max_tokens` (reasoning models → 4096).
3. Otherwise the global default (512).

The flag's argparse default is `None` (a sentinel), not the numeric default, so
"was `--max-tokens` passed?" is unambiguous — an *unset* flag never silently
clobbers a per-model budget. Raw `owner/repo` ids have no per-model budget and
fall back to the CLI value or the default.

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

## Running the full matrix

To benchmark **many models over the whole suite** in one invocation, use the
batch selectors instead of `--model`:

| flag                | selects                                                        |
| ------------------- | ------------------------------------------------------------- |
| `--model KEY`       | a single model (Day 1/2 behavior; pairs with `--prompt-id` or `--suite`) |
| `--models a,b,c`    | a comma-separated list of keys and/or raw `owner/repo` ids    |
| `--all-models`      | every key in `BUILTIN_MODELS`                                  |

`--model`, `--models`, and `--all-models` are **mutually exclusive**.
`--models`/`--all-models` always run the **full prompt suite** (`--prompts-dir`,
default `prompts/`) for each selected model — they are batch/suite-only, so
combining them with `--prompt-id` or `--suite` is an error (reported, never
silently ignored). `--prompt-id` is only valid with a single `--model`.

```bash
# Run a chosen subset (e.g. the 7 currently-licensed models):
python run_benchmark.py --models \
  qwen2.5_7b,qwen3_8b,qwen3_14b,gemma3_27b,qwen2.5_coder_32b,qwen3_32b,deepseek_r1_qwen_32b

# Or run everything built in:
python run_benchmark.py --all-models

# Inspect the matrix:
sqlite3 data/results.db \
  "SELECT m.name, COUNT(*) AS runs, ROUND(AVG(r.quality_score),3) AS mean_score
     FROM runs r JOIN models m ON m.id = r.model_id
    GROUP BY m.name ORDER BY mean_score DESC"
```

The loop is **model → prompt**: for each selected model, every prompt in the
suite runs (or is skipped if already cached), gets `<think>`-stripped, scored,
and written. You get a per-model summary line and an overall summary at the end:

```
=== [model 1/7 qwen3_8b] Qwen/Qwen3-8B :: 204 prompts ===
...
[model qwen3_8b] 204 prompts: 200 ok, 4 cached, 0 errors, mean score 0.78
...
[done] matrix: 7 models x 204 prompts: 1396 ok, 28 cached, 4 errors, overall mean score 0.74
```

### Caching / resume

Every cell is cached on `UNIQUE(model_id, prompt_id)`. Re-running the same
command **resumes**: already-stored cells are skipped (counted as `cached`) and
only missing cells are executed. A failed (errored) cell writes no row, so it is
retried on the next run. This makes the matrix safe to interrupt and restart, and
lets you add a model (or new prompts) later without recomputing what you have.
Use `--force` to recompute and **cleanly overwrite** an already-cached cell: the
forced re-run replaces the existing row for that `(model, prompt)` in place
(`INSERT ... ON CONFLICT(model_id, prompt_id) DO UPDATE`), updating the output,
score, metrics, and timestamp rather than erroring on the UNIQUE constraint. The
non-`--force` path is unchanged: cached cells are skipped.

### Retry / back-off policy

Suite and batch runs wrap each inference call in bounded exponential back-off
(`tenacity`: `wait_exponential(multiplier=2, min=2, max=20)`,
`stop_after_attempt(3)` — so 1 try + up to 2 retries). One flaky cell never
aborts the matrix: on final failure the cell is logged to stderr, counted as an
error, and the loop continues.

**Only transient failures are retried. Permanent ones fail fast:**

| classified **transient** (retried)                       | classified **permanent** (no retry) |
| -------------------------------------------------------- | ----------------------------------- |
| HTTP 408 / 429 / 500 / 502 / 503 / 504 (and any other 5xx) | HTTP 400 (bad request)              |
| timeouts and connection/protocol errors                  | 401 / 403 (gated / no permission)   |
| "model is currently loading" cold-start messages         | 402 (billing / out of credits)      |
|                                                          | 404 / 410 (model gone), 405         |

Classification (`runner.retry.is_transient_error`) inspects the exception for an
HTTP status code — `huggingface_hub.HfHubHTTPError` exposes
`.response.status_code` — and decides by status. When no status is available it
falls back to the exception type (timeouts/connection errors are transient) and
otherwise treats the error as **permanent**, so we never silently loop on an
unrecognized failure. Retrying 401/402/403/404/400 would just waste wall-clock
time and, for 402, real inference credits. The single-prompt Day 1 path
(`--model` + `--prompt-id`) runs with **no** retry (one attempt).

### Cost / scale

A full run is `N models × ~204 prompts` inference calls (e.g. 7 models ≈ 1,400
calls). Each call is a pay-as-you-go request against an HF Inference Provider, so
budget accordingly; caching means you only pay for cells you haven't run yet.

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
│   ├── retry.py            # transient-vs-permanent retry/back-off policy
│   └── scorer.py           # rule-based scoring + <think> stripping
├── data/                   # SQLite DB lives here (gitignored)
├── run_benchmark.py        # CLI entry point (single / --suite / --models / --all-models)
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
`load_prompts` (including duplicate-id rejection), the `--suite` path
end-to-end, the multi-model `--models`/`--all-models` matrix (one row per
`(model, prompt)`, populated scores, raw `<think>` retained, fully-cached
re-run), and the retry policy (a transient error is retried then succeeds; a
permanent 403-style error fails fast, writes no row, and does not abort the
batch). It does not replace a real run against the API; it only proves the
persistence, scoring, retry, and CLI layers work.

```bash
python scripts/verify_db.py
```

## Dashboard

`app.py` is an interactive [Gradio](https://www.gradio.app) dashboard over the
collected results in `data/results.db`. It is **read-only** and makes no network
or inference calls — it only aggregates the local database. All numbers are
scoped to the `reasoning` and `math` categories with `quality_score IS NOT NULL`
(the saturated `instruction_following` / `smoke` rows are excluded). Throughput
is an approximate metric (tok/s).

Four views:

- **Leaderboard** — one row per model: size, overall mean quality
  (reasoning + math), per-category quality, average latency, and throughput,
  sorted by overall quality.
- **Latency vs quality** — one point per model (x = avg latency, log scale;
  y = mean quality; point size = params), the headline trade-off chart.
- **Per-category** — a grouped bar chart of mean quality per category by model.
- **Head-to-head** — pick two models and a prompt to compare their raw stored
  outputs and quality scores side by side.

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

The DB path defaults to `data/results.db`; override it with the `RESULTS_DB_PATH`
environment variable.

Verify the data layer and figures offline (no server, no network):

```bash
python scripts/verify_dashboard.py
```

### Deploy to a Hugging Face Space

A Gradio Space is just a repo whose `README.md` starts with a YAML front-matter
header and whose `app_file` constructs and launches the demo. Add a header like
this to the **Space's** README (not required for the local repo):

```yaml
---
title: Model Benchmarking Dashboard
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---
```

Then push `app.py`, `requirements.txt`, and the `runner/` package to the Space.

**Important:** `data/results.db` is **gitignored in this repo**, so it is not
committed and will NOT be present on a fresh Space clone. The dashboard reads
that file at runtime, so you must **upload `data/results.db` to the Space**
yourself (commit it directly in the Space repo, or upload via the Space's Files
tab / `huggingface_hub.upload_file`). Without the DB file present at
`data/results.db` (or at `RESULTS_DB_PATH`), the Space will start but every view
will be empty. Do not edit this repo's `.gitignore` to fix this — the DB is
deliberately untracked here; the Space simply needs its own copy of the file.
