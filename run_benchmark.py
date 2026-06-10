"""CLI entry point for a single model x single prompt benchmark run.

Inference is serverless via the Hugging Face Inference Providers API, so a
valid HF token (with Inference permission) is required.

Usage
-----
    python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test
    python run_benchmark.py --model qwen3_8b --prompt-id custom \\
        --prompt-text "Why is the sky blue?"
    # A raw HF repo id works too (anything containing "/"):
    python run_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --prompt-id smoke_test

The token is read from --token, else a local .env file (HF_TOKEN=...), else
$HF_TOKEN, else $HUGGINGFACEHUB_API_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from runner import db as db_module
from runner.executor import run_prompt


# Built-in model registry: friendly key -> (hf_id, size_b in billions). Day 1
# ships the five target models; Day 2+ will load this from models.yaml. A
# ``--model`` value containing "/" is treated as a raw HF repo id instead.
# All verified live on HF Inference Providers via pay-as-you-go providers
# (Together / Nscale / Novita / Groq / Scaleway) — NOT Featherless, which is
# subscription-only and unavailable through HF routed billing.
BUILTIN_MODELS: dict[str, dict[str, object]] = {
    # gated models (accept the license on the HF model page first):
    "llama3.1_8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "size_b": 8.0},
    "gemma3_27b": {"hf_id": "google/gemma-3-27b-it", "size_b": 27.0},
    "llama3.3_70b": {"hf_id": "meta-llama/Llama-3.3-70B-Instruct", "size_b": 70.0},
    # ungated:
    "qwen2.5_7b": {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "size_b": 7.6},
    "qwen3_8b": {"hf_id": "Qwen/Qwen3-8B", "size_b": 8.2},
    "qwen3_14b": {"hf_id": "Qwen/Qwen3-14B", "size_b": 14.8},
    "qwen2.5_coder_32b": {"hf_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "size_b": 32.0},
    "qwen3_32b": {"hf_id": "Qwen/Qwen3-32B", "size_b": 32.8},
    "deepseek_r1_qwen_32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "size_b": 32.0,
    },
}


# Built-in prompts available without --prompt-text. Day 1 keeps this tiny;
# Day 2+ will load a real prompt set from prompts/*.yaml.
BUILTIN_PROMPTS: dict[str, dict[str, str]] = {
    "smoke_test": {
        "category": "smoke",
        "input": "Say hello in one short sentence.",
        "expected_output": "",
        "scoring_method": "manual",
    },
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description=(
            "Run one prompt against one model via the Hugging Face Inference "
            "Providers API and record the result."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Model key (e.g. qwen2.5_1.5b) or a raw HF repo id containing '/' "
            f"(e.g. Qwen/Qwen2.5-1.5B-Instruct). Known keys: {sorted(BUILTIN_MODELS)}."
        ),
    )
    p.add_argument(
        "--prompt-id",
        required=True,
        help="Prompt identifier. Use 'smoke_test' for the built-in Day 1 prompt.",
    )
    p.add_argument(
        "--prompt-text",
        default=None,
        help="Override prompt text. Required if --prompt-id is not a built-in.",
    )
    p.add_argument(
        "--token",
        default=None,
        help=(
            "HF access token (Inference permission). Falls back to $HF_TOKEN "
            "then $HUGGINGFACEHUB_API_TOKEN."
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512).",
    )
    p.add_argument(
        "--db-path",
        default=db_module.DEFAULT_DB_PATH,
        help=f"SQLite DB path (default: {db_module.DEFAULT_DB_PATH}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a row already exists for (model, prompt_id).",
    )
    return p.parse_args(argv)


def _resolve_model(model: str) -> dict[str, object]:
    """Resolve a --model value to {name, hf_id, size_b}.

    A value containing "/" is treated as a raw HF repo id (name == hf_id,
    size unknown). Otherwise it must be a known built-in key.
    """
    if "/" in model:
        return {"name": model, "hf_id": model, "size_b": None}
    builtin = BUILTIN_MODELS.get(model)
    if builtin is None:
        raise SystemExit(
            f"Unknown model {model!r}. Pass a known key {sorted(BUILTIN_MODELS)} "
            f"or a raw HF repo id (containing '/')."
        )
    return {"name": model, "hf_id": builtin["hf_id"], "size_b": builtin["size_b"]}


def _resolve_prompt(prompt_id: str, prompt_text: Optional[str]) -> dict[str, str]:
    """Pick the prompt definition, preferring --prompt-text override."""
    builtin = BUILTIN_PROMPTS.get(prompt_id)
    if prompt_text is not None:
        return {
            "category": (builtin or {}).get("category", "custom"),
            "input": prompt_text,
            "expected_output": (builtin or {}).get("expected_output", ""),
            "scoring_method": (builtin or {}).get("scoring_method", "manual"),
        }
    if builtin is None:
        raise SystemExit(
            f"Unknown prompt-id {prompt_id!r} and no --prompt-text provided. "
            f"Known built-ins: {sorted(BUILTIN_PROMPTS)}"
        )
    return builtin


def _load_dotenv_if_available() -> None:
    """Load a local .env into os.environ if python-dotenv is installed.

    Optional by design: the package may not be present (e.g. in the sandbox
    verify path), so a missing import is silently ignored. Existing
    environment variables are NOT overridden.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv(override=False)


def _resolve_token(token_arg: Optional[str]) -> Optional[str]:
    """Resolve the HF token from --token, then a .env file, then env vars."""
    if token_arg:
        return token_arg
    _load_dotenv_if_available()
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    model_def = _resolve_model(args.model)
    prompt_def = _resolve_prompt(args.prompt_id, args.prompt_text)

    token = _resolve_token(args.token)
    if not token:
        print(
            "[error] No Hugging Face token found. Serverless inference requires "
            "one. Pass --token, or set $HF_TOKEN / $HUGGINGFACEHUB_API_TOKEN. "
            "Create a token with Inference permission at "
            "https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    with db_module.open_db(args.db_path) as conn:
        model_id = db_module.upsert_model(
            conn,
            name=str(model_def["name"]),
            hf_id=str(model_def["hf_id"]),
            size_b=model_def["size_b"],  # type: ignore[arg-type]
        )
        db_module.upsert_prompt(
            conn,
            prompt_id=args.prompt_id,
            category=prompt_def["category"],
            input_text=prompt_def["input"],
            expected_output=prompt_def["expected_output"],
            scoring_method=prompt_def["scoring_method"],
        )

        if not args.force and db_module.run_exists(conn, model_id, args.prompt_id):
            print(
                f"[skip] {args.model} x {args.prompt_id} already in DB. "
                f"Use --force to re-run."
            )
            return 0

        try:
            result = run_prompt(
                str(model_def["hf_id"]),
                prompt_def["input"],
                token=token,
                max_tokens=args.max_tokens,
            )
        except ImportError as exc:
            print(
                "[error] The 'huggingface_hub' package is not installed. "
                "Run: pip install -r requirements.txt",
                file=sys.stderr,
            )
            print(f"        ({exc})", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 — top-level CLI boundary
            print(f"[error] inference call failed: {exc}", file=sys.stderr)
            return 1

        run_id = db_module.insert_run(
            conn,
            model_id=model_id,
            prompt_id=args.prompt_id,
            output=result["output"],
            latency_ms=result["latency_ms"],
            ttft_ms=result["ttft_ms"],
            tokens_per_sec=result["tokens_per_sec"],
            ts=int(time.time()),
            quality_score=None,  # Day 2 will fill this in.
        )

    preview = result["output"].strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    ttft = result["ttft_ms"]
    ttft_str = f"{ttft}ms" if ttft is not None else "n/a"
    print(
        f"[ok] run_id={run_id} model={args.model} prompt={args.prompt_id} "
        f"latency={result['latency_ms']}ms "
        f"ttft={ttft_str} "
        f"tok/s={result['tokens_per_sec']:.2f} :: {preview!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
