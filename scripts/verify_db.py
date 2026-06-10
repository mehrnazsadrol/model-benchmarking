"""Sandbox verification — proves the DB layer + CLI plumbing work without network.

This is NOT a replacement for the real Day 1 smoke test. The user still needs to
run ``python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test`` with a
live Hugging Face token to satisfy the Day 1 deliverable.

What this script verifies:
  1. ``runner.db.connect`` creates the schema in a fresh DB file.
  2. ``upsert_model`` / ``upsert_prompt`` are idempotent and return stable ids.
  3. ``run_exists`` correctly reports presence/absence.
  4. ``insert_run`` round-trips every column (new schema: ttft_ms, no memory_mb).
  5. The CLI's main() runs end-to-end when ``run_prompt`` is monkeypatched to
     return a fake RunResult and a fake --token is supplied — i.e. argparse,
     model/prompt resolution, the token check, DB writes, and the cached-skip
     path all work without huggingface_hub or a network call.

Run it from the repo root:
    python scripts/verify_db.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from runner import db as db_module
import run_benchmark


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok  {msg}")


def test_db_layer(tmp_db: str) -> None:
    print("[1/2] DB layer")
    with db_module.open_db(tmp_db) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        check({"models", "prompts", "runs"}.issubset(tables), "tables created")

        mid1 = db_module.upsert_model(
            conn, "qwen2.5_7b", hf_id="Qwen/Qwen2.5-7B-Instruct", size_b=7.6
        )
        mid2 = db_module.upsert_model(
            conn, "qwen2.5_7b", hf_id="Qwen/Qwen2.5-7B-Instruct", size_b=7.6
        )
        check(mid1 == mid2, f"upsert_model idempotent (got {mid1}, {mid2})")

        mrow = conn.execute(
            "SELECT hf_id, provider, size_b FROM models WHERE id = ?", (mid1,)
        ).fetchone()
        check(mrow["hf_id"] == "Qwen/Qwen2.5-7B-Instruct", "model hf_id round-trip")
        check(abs(mrow["size_b"] - 7.6) < 1e-9, "model size_b round-trip")

        db_module.upsert_prompt(
            conn,
            "smoke_test",
            category="smoke",
            input_text="Say hello.",
            expected_output="",
            scoring_method="manual",
        )

        check(
            not db_module.run_exists(conn, mid1, "smoke_test"),
            "run_exists False before insert",
        )

        rid = db_module.insert_run(
            conn,
            model_id=mid1,
            prompt_id="smoke_test",
            output="Hello!",
            latency_ms=123,
            ttft_ms=45,
            tokens_per_sec=42.5,
            ts=1_700_000_000,
            quality_score=None,
        )
        check(rid > 0, f"insert_run returned rowid={rid}")

        check(
            db_module.run_exists(conn, mid1, "smoke_test"),
            "run_exists True after insert",
        )

        row = conn.execute(
            "SELECT model_id, prompt_id, output, latency_ms, ttft_ms, "
            "tokens_per_sec, ts, quality_score FROM runs WHERE id = ?",
            (rid,),
        ).fetchone()
        check(row["model_id"] == mid1, "model_id round-trip")
        check(row["prompt_id"] == "smoke_test", "prompt_id round-trip")
        check(row["output"] == "Hello!", "output round-trip")
        check(row["latency_ms"] == 123, "latency_ms round-trip")
        check(row["ttft_ms"] == 45, "ttft_ms round-trip")
        check(abs(row["tokens_per_sec"] - 42.5) < 1e-9, "tokens_per_sec round-trip")
        check(row["ts"] == 1_700_000_000, "ts round-trip")
        check(row["quality_score"] is None, "quality_score nullable")

        db_module.upsert_prompt(conn, "smoke_test_2", category="smoke")
        rid2 = db_module.insert_run(
            conn,
            model_id=mid1,
            prompt_id="smoke_test_2",
            output="Hi",
            latency_ms=10,
            ttft_ms=None,
            tokens_per_sec=1.0,
            ts=1,
        )
        nrow = conn.execute("SELECT ttft_ms FROM runs WHERE id = ?", (rid2,)).fetchone()
        check(nrow["ttft_ms"] is None, "ttft_ms nullable")

        try:
            db_module.insert_run(
                conn,
                model_id=mid1,
                prompt_id="smoke_test",
                output="dup",
                latency_ms=1,
                ttft_ms=1,
                tokens_per_sec=1.0,
                ts=1,
            )
            raise AssertionError("expected IntegrityError on duplicate (model, prompt)")
        except sqlite3.IntegrityError:
            print("  ok  UNIQUE(model_id, prompt_id) enforced")


def test_cli_with_stub(tmp_db: str) -> None:
    print("[2/2] CLI end-to-end (stubbed executor)")

    def fake_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        return {
            "output": f"[stub reply to: {prompt_text}]",
            "latency_ms": 250,
            "ttft_ms": 60,
            "tokens_per_sec": 30.0,
        }

    run_benchmark.run_prompt = fake_run_prompt  # type: ignore[assignment]

    rc = run_benchmark.main(
        [
            "--model",
            "qwen2.5_7b",
            "--prompt-id",
            "smoke_test",
            "--db-path",
            tmp_db,
            "--token",
            "fake-token-for-verify",
        ]
    )
    check(rc == 0, f"CLI exit code 0 (got {rc})")

    with db_module.open_db(tmp_db) as conn:
        rows = list(
            conn.execute(
                "SELECT m.name, m.hf_id, r.prompt_id, r.output, r.latency_ms, "
                "r.ttft_ms, r.tokens_per_sec, r.ts "
                "FROM runs r JOIN models m ON m.id = r.model_id"
            )
        )
    check(len(rows) == 1, f"exactly one row written (got {len(rows)})")
    r = rows[0]
    check(r["name"] == "qwen2.5_7b", "model name persisted")
    check(r["hf_id"] == "Qwen/Qwen2.5-7B-Instruct", "hf_id persisted")
    check(r["prompt_id"] == "smoke_test", "prompt_id persisted")
    check("stub reply" in r["output"], "output persisted")
    check(r["latency_ms"] == 250, "latency_ms persisted")
    check(r["ttft_ms"] == 60, "ttft_ms persisted")
    check(r["tokens_per_sec"] == 30.0, "tokens_per_sec persisted")
    check(r["ts"] > 0, "ts populated")

    rc2 = run_benchmark.main(
        [
            "--model",
            "qwen2.5_7b",
            "--prompt-id",
            "smoke_test",
            "--db-path",
            tmp_db,
            "--token",
            "fake-token-for-verify",
        ]
    )
    check(rc2 == 0, "second invocation exits 0 (cached skip)")
    with db_module.open_db(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
    check(count == 1, f"no duplicate row created (count={count})")

    saved_env = {
        k: os.environ.pop(k, None) for k in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    }
    saved_loader = run_benchmark._load_dotenv_if_available
    run_benchmark._load_dotenv_if_available = lambda: None  # type: ignore[assignment]
    try:
        rc3 = run_benchmark.main(
            [
                "--model",
                "qwen2.5_7b",
                "--prompt-id",
                "smoke_test",
                "--db-path",
                tmp_db,
            ]
        )
        check(rc3 == 1, f"missing token exits 1 (got {rc3})")
    finally:
        run_benchmark._load_dotenv_if_available = saved_loader  # type: ignore[assignment]
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        db1 = os.path.join(tmpdir, "layer.db")
        db2 = os.path.join(tmpdir, "cli.db")
        test_db_layer(db1)
        test_cli_with_stub(db2)
    print("\nAll verification checks passed.")
    print("Next step (with a live HF token exported):")
    print("  python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
