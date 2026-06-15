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
from runner.retry import safe_run

SUITE_MAX_ATTEMPTS = 3


DEFAULT_MAX_TOKENS = 512
REASONING_MAX_TOKENS = 4096

BUILTIN_MODELS: dict[str, dict[str, object]] = {
    "llama3.1_8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "size_b": 8.0},
    "gemma3_27b": {"hf_id": "google/gemma-3-27b-it", "size_b": 27.0},
    "llama3.3_70b": {"hf_id": "meta-llama/Llama-3.3-70B-Instruct", "size_b": 70.0},
    "qwen2.5_7b": {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "size_b": 7.6},
    "qwen3_8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "size_b": 8.2,
        "max_tokens": REASONING_MAX_TOKENS,
    },
    "qwen3_14b": {
        "hf_id": "Qwen/Qwen3-14B",
        "size_b": 14.8,
        "max_tokens": REASONING_MAX_TOKENS,
    },
    "qwen2.5_coder_32b": {"hf_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "size_b": 32.0},
    "qwen3_32b": {
        "hf_id": "Qwen/Qwen3-32B",
        "size_b": 32.8,
        "max_tokens": REASONING_MAX_TOKENS,
    },
    "deepseek_r1_qwen_32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "size_b": 32.0,
        "max_tokens": REASONING_MAX_TOKENS,
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
        default=None,
        help=(
            "Single model: a key (e.g. qwen2.5_7b) or a raw HF repo id containing "
            f"'/' (e.g. Qwen/Qwen2.5-7B-Instruct). Known keys: {sorted(BUILTIN_MODELS)}. "
            "Mutually exclusive with --models / --all-models."
        ),
    )
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated list of model keys and/or raw HF repo ids to run the "
            "full prompt suite against (e.g. 'qwen2.5_7b,qwen3_8b,gemma3_27b'). "
            "Implies suite mode. Mutually exclusive with --model / --all-models."
        ),
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Run the full prompt suite against EVERY key in BUILTIN_MODELS. "
            "Implies suite mode. Mutually exclusive with --model / --models."
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
        default=None,
        help=(
            "Maximum tokens to generate. When passed, OVERRIDES the per-model "
            "budget for every selected model. When omitted, each model uses its "
            "own per-model 'max_tokens' if set (reasoning models default to "
            f"{REASONING_MAX_TOKENS}), else the global default ({DEFAULT_MAX_TOKENS}). "
            "The default is None — not the numeric default — precisely so that "
            "\"was --max-tokens passed?\" is unambiguous and an unset flag does "
            "not silently clobber per-model budgets."
        ),
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
        return {"name": model, "hf_id": model, "size_b": None, "max_tokens": None}
    builtin = BUILTIN_MODELS.get(model)
    if builtin is None:
        raise SystemExit(
            f"Unknown model {model!r}. Pass a known key {sorted(BUILTIN_MODELS)} "
            f"or a raw HF repo id (containing '/')."
        )
    return {
        "name": model,
        "hf_id": builtin["hf_id"],
        "size_b": builtin["size_b"],
        "max_tokens": builtin.get("max_tokens"),
    }


def _resolve_models(
    model: Optional[str],
    models_csv: Optional[str],
    all_models: bool,
) -> list[dict[str, object]]:
    if all_models:
        keys: list[str] = sorted(BUILTIN_MODELS)
    elif models_csv is not None:
        keys = [tok.strip() for tok in models_csv.split(",") if tok.strip()]
        if not keys:
            raise SystemExit("--models was empty; pass at least one model key/id.")
    else:
        keys = [str(model)]

    resolved: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in keys:
        model_def = _resolve_model(key)
        name = str(model_def["name"])
        if name in seen:
            continue
        seen.add(name)
        resolved.append(model_def)
    return resolved


def _resolve_max_tokens(
    cli_max_tokens: Optional[int],
    model_def: dict[str, object],
) -> int:
    if cli_max_tokens is not None:
        return cli_max_tokens
    per_model = model_def.get("max_tokens")
    if per_model is not None:
        return int(per_model)
    return DEFAULT_MAX_TOKENS


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


def _last_score(conn: object, model_id: int, prompt_id: str) -> Optional[float]:
    row = conn.execute(
        "SELECT quality_score FROM runs WHERE model_id = ? AND prompt_id = ?",
        (model_id, prompt_id),
    ).fetchone()
    if row is None:
        return None
    value = row["quality_score"]
    return None if value is None else float(value)


def _run_one(
    conn: object,
    model_id: int,
    model_def: dict[str, object],
    prompt_def: dict[str, object],
    token: str,
    cli_max_tokens: Optional[int],
    force: bool,
    max_attempts: int = 1,
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

    max_tokens = _resolve_max_tokens(cli_max_tokens, model_def)

    try:
        result = safe_run(
            run_prompt,
            str(model_def["hf_id"]),
            str(prompt_def["input"]),
            token=token,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
        )
    except ImportError:
        raise
    except Exception as exc:
        return "error", f"{prompt_id}: inference call failed: {exc}"

    score_value = _score_run(prompt_def, result["output"])

    persist = db_module.upsert_run if force else db_module.insert_run
    run_id = persist(
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
    tps = result["tokens_per_sec"]
    tps_str = "n/a" if tps is None else f"{tps:.2f}"
    return (
        "ok",
        f"run_id={run_id} model={model_def['name']} prompt={prompt_id} "
        f"score={score_str} latency={result['latency_ms']}ms ttft={ttft_str} "
        f"tok/s={tps_str} :: {preview!r}",
    )


def _validate_selection(args: argparse.Namespace) -> Optional[str]:
    n_model_selectors = sum(
        bool(x) for x in (args.model, args.models, args.all_models)
    )
    if n_model_selectors == 0:
        return "provide one of --model, --models, or --all-models."
    if n_model_selectors > 1:
        return "--model, --models and --all-models are mutually exclusive."

    is_batch = bool(args.models) or bool(args.all_models)
    if is_batch:
        if args.prompt_id:
            return (
                "--prompt-id is only valid with a single --model; "
                "--models/--all-models always run the full suite."
            )
        if args.suite:
            return (
                "--suite is redundant with --models/--all-models "
                "(batch mode always runs the full suite); drop --suite."
            )
        return None

    if args.suite and args.prompt_id:
        return "--suite and --prompt-id are mutually exclusive."
    if not args.suite and not args.prompt_id:
        return "with a single --model, provide --prompt-id <id> or --suite."
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    err = _validate_selection(args)
    if err:
        print(f"[error] {err}", file=sys.stderr)
        return 1

    is_batch = bool(args.models) or bool(args.all_models)
    suite_mode = is_batch or args.suite

    try:
        model_defs = _resolve_models(args.model, args.models, args.all_models)
    except SystemExit as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if not model_defs:
        print("[error] no models selected.", file=sys.stderr)
        return 1

    if suite_mode:
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

    max_attempts = SUITE_MAX_ATTEMPTS if suite_mode else 1

    total_ok = total_skip = total_err = 0
    score_sum = 0.0
    score_n = 0

    with db_module.open_db(args.db_path) as conn:
        for m_idx, model_def in enumerate(model_defs, start=1):
            model_id = db_module.upsert_model(
                conn,
                name=str(model_def["name"]),
                hf_id=str(model_def["hf_id"]),
                size_b=model_def["size_b"],
            )

            if is_batch:
                print(
                    f"\n=== [model {m_idx}/{len(model_defs)} {model_def['name']}] "
                    f"{model_def['hf_id']} :: {len(prompt_defs)} prompts ==="
                )

            n_ok = n_skip = n_err = 0
            m_score_sum = 0.0
            m_score_n = 0

            for prompt_def in prompt_defs:
                try:
                    status, message = _run_one(
                        conn,
                        model_id=model_id,
                        model_def=model_def,
                        prompt_def=prompt_def,
                        token=token,
                        cli_max_tokens=args.max_tokens,
                        force=args.force,
                        max_attempts=max_attempts,
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
                    score = _last_score(conn, model_id, str(prompt_def["id"]))
                    if score is not None:
                        m_score_sum += score
                        m_score_n += 1
                elif status == "skip":
                    n_skip += 1
                    print(f"[skip] {message}")
                else:
                    n_err += 1
                    print(f"[error] {message}", file=sys.stderr)

            total_ok += n_ok
            total_skip += n_skip
            total_err += n_err
            score_sum += m_score_sum
            score_n += m_score_n

            if suite_mode:
                mean = m_score_sum / m_score_n if m_score_n else 0.0
                mean_str = f"{mean:.2f}" if m_score_n else "n/a"
                print(
                    f"[model {model_def['name']}] {len(prompt_defs)} prompts: "
                    f"{n_ok} ok, {n_skip} cached, {n_err} errors, "
                    f"mean score {mean_str}"
                )

    if is_batch:
        overall_mean = score_sum / score_n if score_n else 0.0
        overall_mean_str = f"{overall_mean:.2f}" if score_n else "n/a"
        print(
            f"\n[done] matrix: {len(model_defs)} models x {len(prompt_defs)} prompts: "
            f"{total_ok} ok, {total_skip} cached, {total_err} errors, "
            f"overall mean score {overall_mean_str}"
        )
    elif args.suite:
        print(
            f"[done] suite: {total_ok} ok, {total_skip} skipped, {total_err} errors."
        )

    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
