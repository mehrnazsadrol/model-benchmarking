# Model Benchmarking

A small pipeline for benchmarking instruction-tuned LLMs served via the
[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers)
serverless API. Day 1 is a vertical slice: **one model x one prompt**, results
persisted to SQLite. No local models, no GPU — inference runs on provider
hardware behind `huggingface_hub.InferenceClient`.

## Day 1 setup

You run these steps yourself — the agent does not (it has no token).

1. **Create a Hugging Face account** at <https://huggingface.co> if you don't
   have one.
2. **Create an access token** with **Inference** permission at
   <https://huggingface.co/settings/tokens> (a fine-grained token with
   "Make calls to Inference Providers" enabled, or a classic read token).
3. **Install deps and export the token:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# (HUGGINGFACEHUB_API_TOKEN is also accepted.)
```

## Day 1 smoke test

```bash
python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test
sqlite3 data/results.db "SELECT model_id, latency_ms, ttft_ms, tokens_per_sec, output FROM runs"
```

Expected: one populated row with non-null `latency_ms` and `tokens_per_sec`, a
short generated string in `output`, and (on the streaming path) a `ttft_ms`
value. Re-running with the same `(model, prompt-id)` is a no-op (cached) unless
you pass `--force`.

The `--model` flag accepts either a built-in key or a raw HF repo id (anything
containing `/`). The nine built-in keys, all verified live on HF Inference
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

> **Note:** Featherless AI is a subscription provider and does NOT work through
> HF's routed pay-as-you-go billing, so models served only by Featherless
> (most sub-7B models) are excluded. Qwen3 and DeepSeek-R1 are reasoning models
> that may emit `<think>` blocks — relevant for output parsing in Day 2 scoring.

You can also pass arbitrary prompt text:

```bash
python run_benchmark.py --model qwen3_8b --prompt-id sky --prompt-text "Why is the sky blue?"
# or a raw repo id directly:
python run_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt-id smoke_test
```

## No memory metric (serverless)

The old ollama pipeline reported a `memory_mb` figure (the RSS of the local
daemon). That metric is **gone** in the serverless architecture: inference runs
on the provider's hardware, which we have no visibility into, so there is no
honest memory or VRAM number to record. The schema drops `memory_mb` and adds
`ttft_ms` (time-to-first-token) instead, which is measurable from the streaming
response. If you need memory characteristics, consult each model's published
size (`size_b`) and the provider's own documentation.

## Layout

```
model-benchmarking/
├── prompts/                # YAML prompt sets — populated Day 2+
├── runner/
│   ├── __init__.py
│   ├── db.py               # SQLite schema + upsert helpers
│   └── executor.py         # HF InferenceClient wrapper + metrics
├── data/                   # SQLite DB lives here (gitignored on Day 1)
├── run_benchmark.py        # CLI entry point
├── requirements.txt
└── README.md
```

`app.py` (a Gradio dashboard) and a committed `data/results.db` are **future
days** — they do not exist yet. On Day 1 `data/*.db` stays gitignored.

## Verifying without a token / huggingface_hub

`scripts/verify_db.py` exercises the DB layer and CLI plumbing end-to-end
against a stubbed executor and a fake token — useful in CI or any sandbox
without network access. It checks schema creation, idempotent upserts,
`run_exists`, the full round-trip of the new run columns, the
`UNIQUE(model_id, prompt_id)` cached-skip behavior, and the missing-token
error path. See the docstring in that file. This does **not** replace the real
smoke test; it only proves the persistence and CLI layers work.

```bash
python scripts/verify_db.py
```
