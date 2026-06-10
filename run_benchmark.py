from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from runner import db as db_module
from runner import scorer as scorer_module
from runner.executor import run_prompt
from runner.prompts import load_prompts


BUILTIN_MODELS: dict[str, dict[str, object]] = {
    "llama3.1_8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "size_b": 8.0},
    "gemma3_27b": {"hf_id": "google/gemma-3-27b-it", "size_b": 27.0},
    "llama3.3_70b": {"hf_id": "meta-llama/Llama-3.3-70B-Instruct", "size_b": 70.0},
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
        default=None,
        help=(
            "Prompt identifier for a single run. Use 'smoke_test' for the "
            "built-in Day 1 prompt, or an id from a prompts/*.json file. "
            "Mutually exclusive with --suite."
        ),
    )
    p.add_argument(
        "--prompt-text",
        default=None,
        help="Override prompt text. Required if --prompt-id is not a built-in.",
    )
    p.add_argument(
        "--suite",
        action="store_true",
        help=(
            "Run every prompt loaded from --prompts-dir against the model and "
            "score each result. Mutually exclusive with --prompt-id."
        ),
    )
    p.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Directory of prompts/*.json files for --suite (default: prompts).",
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
    if "/" in model:
        return {"name": model, "hf_id": model, "size_b": None}
    builtin = BUILTIN_MODELS.get(model)
    if builtin is None:
        raise SystemExit(
            f"Unknown model {model!r}. Pass a known key {sorted(BUILTIN_MODELS)} "
            f"or a raw HF repo id (containing '/')."
        )
    return {"name": model, "hf_id": builtin["hf_id"], "size_b": builtin["size_b"]}


def _resolve_prompt(
    prompt_id: str,
    prompt_text: Optional[str],
    prompts_dir: str = "prompts",
) -> dict[str, object]:
    builtin = BUILTIN_PROMPTS.get(prompt_id)
    if prompt_text is not None:
        return {
            "id": prompt_id,
            "category": (builtin or {}).get("category", "custom"),
            "input": prompt_text,
            "expected_output": (builtin or {}).get("expected_output", ""),
            "scoring_method": (builtin or {}).get("scoring_method", "manual"),
            "scoring_args": (builtin or {}).get("scoring_args", {}),
        }
    if builtin is not None:
        return {
            "id": prompt_id,
            "scoring_args": {},
            **builtin,
        }

    try:
        loaded = {p["id"]: p for p in load_prompts(prompts_dir)}
    except Exception as exc:
        raise SystemExit(f"Failed to load prompts from {prompts_dir!r}: {exc}")
    if prompt_id in loaded:
        return loaded[prompt_id]

    raise SystemExit(
        f"Unknown prompt-id {prompt_id!r} and no --prompt-text provided. "
        f"Known built-ins: {sorted(BUILTIN_PROMPTS)}; "
        f"loaded ids: {sorted(loaded)}"
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _resolve_token(token_arg: Optional[str]) -> Optional[str]:
    if token_arg:
        return token_arg
    _load_dotenv_if_available()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def _score_run(prompt_def: dict[str, object], raw_output: str) -> Optional[float]:
    method = str(prompt_def.get("scoring_method") or "manual")
    if method in scorer_module.MANUAL_METHODS:
        return None
    cleaned = scorer_module.strip_reasoning(raw_output)
    return scorer_module.score(cleaned, prompt_def)


def _run_one(
    conn: object,
    model_id: int,
    model_def: dict[str, object],
    prompt_def: dict[str, object],
    token: str,
    max_tokens: int,
    force: bool,
) -> tuple[str, str]:
    prompt_id = str(prompt_def["id"])
    db_module.upsert_prompt(
        conn,
        prompt_id=prompt_id,
        category=str(prompt_def.get("category", "")),
        input_text=str(prompt_def.get("input", "")),
        expected_output=str(prompt_def.get("expected_output", "")),
        scoring_method=str(prompt_def.get("scoring_method", "manual")),
    )

    if not force and db_module.run_exists(conn, model_id, prompt_id):
        return "skip", f"{model_def['name']} x {prompt_id} already in DB (use --force)."

    try:
        result = run_prompt(
            str(model_def["hf_id"]),
            str(prompt_def["input"]),
            token=token,
            max_tokens=max_tokens,
        )
    except ImportError:
        raise
    except Exception as exc:
        return "error", f"{prompt_id}: inference call failed: {exc}"

    score_value = _score_run(prompt_def, result["output"])

    run_id = db_module.insert_run(
        conn,
        model_id=model_id,
        prompt_id=prompt_id,
        output=result["output"],
        latency_ms=result["latency_ms"],
        ttft_ms=result["ttft_ms"],
        tokens_per_sec=result["tokens_per_sec"],
        ts=int(time.time()),
        quality_score=score_value,
    )

    preview = result["output"].strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    ttft = result["ttft_ms"]
    ttft_str = f"{ttft}ms" if ttft is not None else "n/a"
    score_str = "n/a" if score_value is None else f"{score_value:.2f}"
    return (
        "ok",
        f"run_id={run_id} model={model_def['name']} prompt={prompt_id} "
        f"score={score_str} latency={result['latency_ms']}ms ttft={ttft_str} "
        f"tok/s={result['tokens_per_sec']:.2f} :: {preview!r}",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.suite and args.prompt_id:
        print(
            "[error] --suite and --prompt-id are mutually exclusive.", file=sys.stderr
        )
        return 1
    if not args.suite and not args.prompt_id:
        print("[error] provide --prompt-id <id> or --suite.", file=sys.stderr)
        return 1

    model_def = _resolve_model(args.model)

    if args.suite:
        try:
            prompt_defs = load_prompts(args.prompts_dir)
        except Exception as exc:
            print(f"[error] failed to load prompts: {exc}", file=sys.stderr)
            return 1
        if not prompt_defs:
            print(f"[error] no prompts found in {args.prompts_dir!r}.", file=sys.stderr)
            return 1
    else:
        prompt_defs = [
            _resolve_prompt(args.prompt_id, args.prompt_text, args.prompts_dir)
        ]

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

    n_ok = n_skip = n_err = 0
    with db_module.open_db(args.db_path) as conn:
        model_id = db_module.upsert_model(
            conn,
            name=str(model_def["name"]),
            hf_id=str(model_def["hf_id"]),
            size_b=model_def["size_b"],
        )

        for prompt_def in prompt_defs:
            try:
                status, message = _run_one(
                    conn,
                    model_id=model_id,
                    model_def=model_def,
                    prompt_def=prompt_def,
                    token=token,
                    max_tokens=args.max_tokens,
                    force=args.force,
                )
            except ImportError as exc:
                print(
                    "[error] The 'huggingface_hub' package is not installed. "
                    "Run: pip install -r requirements.txt",
                    file=sys.stderr,
                )
                print(f"        ({exc})", file=sys.stderr)
                return 2

            if status == "ok":
                n_ok += 1
                print(f"[ok] {message}")
            elif status == "skip":
                n_skip += 1
                print(f"[skip] {message}")
            else:
                n_err += 1
                print(f"[error] {message}", file=sys.stderr)

    if args.suite:
        print(f"[done] suite: {n_ok} ok, {n_skip} skipped, {n_err} errors.")

    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
