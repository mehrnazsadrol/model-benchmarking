from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from runner import db as db_module
from runner import prompts as prompts_module
from runner import scorer as scorer_module
import run_benchmark


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ok  {msg}")


def test_db_layer(tmp_db: str) -> None:
    print("[1/5] DB layer")
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
    print("[2/5] CLI end-to-end (stubbed executor)")

    def fake_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        return {
            "output": f"[stub reply to: {prompt_text}]",
            "latency_ms": 250,
            "ttft_ms": 60,
            "tokens_per_sec": 30.0,
        }

    run_benchmark.run_prompt = fake_run_prompt

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
    run_benchmark._load_dotenv_if_available = lambda: None
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
        run_benchmark._load_dotenv_if_available = saved_loader
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def test_strip_reasoning() -> None:
    print("[3/5] strip_reasoning (chain-of-thought stripping)")
    strip = scorer_module.strip_reasoning

    check(
        strip("<think>let me ponder</think>42") == "42",
        "balanced <think> block removed",
    )
    check(
        strip("before <think>noise</think> after") == "before  after".strip()
        or strip("before <think>noise</think> after") == "before  after",
        "text around block preserved",
    )
    check(
        strip("<think>multi\nline\nthought</think>\nFinal: 7") == "Final: 7",
        "DOTALL multi-line block removed",
    )
    check(
        strip("partial answer <think>unfinished reasoning with no close")
        == "partial answer",
        "unclosed <think> drops from opener onward",
    )
    check(
        strip("<think>only an opener, truncated mid-thought") == "",
        "unclosed opener with no prefix yields empty string",
    )
    check(strip("just a plain answer") == "just a plain answer", "plain text untouched")
    check(strip("  spaced answer  ") == "spaced answer", "whitespace trimmed")
    check(
        strip("<THINK>upper</THINK>done") == "done", "tag matching is case-insensitive"
    )


def test_scoring_methods() -> None:
    print("[4/5] scoring methods (pass + fail cases)")
    s = scorer_module.score

    em = {"scoring_method": "exact_match", "expected_output": "Hello World"}
    check(s("  hello   world ", em) == 1.0, "exact_match: normalized match passes")
    check(s("hello there", em) == 0.0, "exact_match: mismatch fails")
    em_cs = {
        "scoring_method": "exact_match",
        "expected_output": "Yes",
        "scoring_args": {"case_sensitive": True},
    }
    check(s("yes", em_cs) == 0.0, "exact_match: case_sensitive respected")

    co = {"scoring_method": "contains", "expected_output": "Paris"}
    check(
        s("The capital is paris, of course.", co) == 1.0, "contains: substring passes"
    )
    check(s("The capital is Berlin.", co) == 0.0, "contains: absent fails")

    rx = {"scoring_method": "regex", "expected_output": r"\b\d{5}\b"}
    check(s("ZIP: 90210 here", rx) == 1.0, "regex: 5-digit match passes")
    check(s("ZIP: 902 only", rx) == 0.0, "regex: no match fails")
    rx_i = {
        "scoring_method": "regex",
        "expected_output": "yes",
        "scoring_args": {"flags": "ignorecase"},
    }
    check(s("YES indeed", rx_i) == 1.0, "regex: ignorecase flag honored")
    rx_bad = {"scoring_method": "regex", "expected_output": "([unclosed"}
    check(
        s("anything", rx_bad) == 0.0, "regex: malformed pattern returns 0.0, no raise"
    )

    nm = {"scoring_method": "numeric", "expected_output": "1234"}
    check(
        s("After summing, the total is 1,234.", nm) == 1.0,
        "numeric: comma-grouped last number passes",
    )
    check(s("The answer is 99.", nm) == 0.0, "numeric: wrong number fails")
    check(s("no digits here", nm) == 0.0, "numeric: no number returns 0.0")
    nm_tol = {
        "scoring_method": "numeric",
        "expected_output": "40",
        "scoring_args": {"tol": 0.5},
    }
    check(s("speed is 40.3 mph", nm_tol) == 1.0, "numeric: within tolerance passes")
    check(s("speed is 41 mph", nm_tol) == 0.0, "numeric: outside tolerance fails")

    jv = {"scoring_method": "json_valid", "expected_output": ""}
    check(s('{"a": 1}', jv) == 1.0, "json_valid: clean JSON passes")
    check(
        s('Here is the result: {"a": 1}. Done.', jv) == 1.0,
        "json_valid: JSON embedded in prose extracted",
    )
    check(s("not json at all", jv) == 0.0, "json_valid: non-JSON fails")
    jv_keys = {
        "scoring_method": "json_valid",
        "expected_output": "",
        "scoring_args": {"required_keys": ["name", "age"]},
    }
    check(
        s('{"name": "A", "age": 5}', jv_keys) == 1.0,
        "json_valid: required_keys present passes",
    )
    check(s('{"name": "A"}', jv_keys) == 0.0, "json_valid: missing required key fails")

    try:
        s("x", {"scoring_method": "bogus", "expected_output": ""})
        raise AssertionError("expected ValueError on unknown scoring_method")
    except ValueError:
        print("  ok  unknown scoring_method raises ValueError")


def test_load_prompts() -> None:
    print("[5/5] load_prompts (seed files + duplicate rejection)")

    seeds = prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    check(len(seeds) >= 6, f"seed prompts loaded (got {len(seeds)})")
    ids = {p["id"] for p in seeds}
    check(len(ids) == len(seeds), "seed prompt ids are unique")
    methods = {p["scoring_method"] for p in seeds}
    check(
        {"numeric", "exact_match", "contains", "regex", "json_valid"}.issubset(methods),
        "seed prompts exercise all scoring methods",
    )
    check(
        all(isinstance(p.get("scoring_args"), dict) for p in seeds),
        "every loaded prompt has a dict scoring_args",
    )

    with tempfile.TemporaryDirectory() as tdir:
        good = [
            {
                "id": "dup_1",
                "category": "cat_a",
                "input": "q",
                "expected_output": "1",
                "scoring_method": "numeric",
            }
        ]
        dupe = [
            {
                "id": "dup_1",
                "category": "cat_b",
                "input": "q2",
                "expected_output": "2",
                "scoring_method": "numeric",
            }
        ]
        with open(os.path.join(tdir, "cat_a.json"), "w", encoding="utf-8") as fh:
            json.dump(good, fh)
        with open(os.path.join(tdir, "cat_b.json"), "w", encoding="utf-8") as fh:
            json.dump(dupe, fh)
        try:
            prompts_module.load_prompts(tdir)
            raise AssertionError("expected PromptError on duplicate id")
        except prompts_module.PromptError:
            print("  ok  duplicate prompt id rejected")

    with tempfile.TemporaryDirectory() as tdir:
        bad = [
            {
                "id": "x_1",
                "category": "wrong",
                "input": "q",
                "expected_output": "1",
                "scoring_method": "numeric",
            }
        ]
        with open(os.path.join(tdir, "right.json"), "w", encoding="utf-8") as fh:
            json.dump(bad, fh)
        try:
            prompts_module.load_prompts(tdir)
            raise AssertionError("expected PromptError on category/stem mismatch")
        except prompts_module.PromptError:
            print("  ok  category-vs-filename mismatch rejected")


def test_suite_cli_with_stub(tmp_db: str) -> None:
    print("[bonus] --suite CLI end-to-end (stubbed executor, scored rows)")

    _by_input = {
        p["input"]: p
        for p in prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    }

    def fake_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        p = _by_input.get(prompt_text)
        reply = _stub_answer(p) if p else "fallback"
        return {
            "output": reply,
            "latency_ms": 100,
            "ttft_ms": 20,
            "tokens_per_sec": 10.0,
        }

    saved = run_benchmark.run_prompt
    run_benchmark.run_prompt = fake_run_prompt
    try:
        rc = run_benchmark.main(
            [
                "--model",
                "qwen3_8b",
                "--suite",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
    finally:
        run_benchmark.run_prompt = saved
    check(rc == 0, f"suite CLI exit code 0 (got {rc})")

    n_seeds = len(prompts_module.load_prompts(os.path.join(ROOT, "prompts")))
    with db_module.open_db(tmp_db) as conn:
        rows = list(
            conn.execute(
                "SELECT prompt_id, output, quality_score FROM runs ORDER BY prompt_id"
            )
        )
    check(len(rows) == n_seeds, f"one scored row per seed prompt (got {len(rows)})")
    check(
        all(r["quality_score"] is not None for r in rows),
        "every suite row has a populated quality_score",
    )
    check(
        all(r["quality_score"] == 1.0 for r in rows),
        "tailored stub answers all score 1.0 (think-stripped before scoring)",
    )
    check(
        all("<think>" in r["output"] for r in rows),
        "RAW output (with <think>) stored in DB, not the cleaned copy",
    )

    run_benchmark.run_prompt = fake_run_prompt
    try:
        rc2 = run_benchmark.main(
            [
                "--model",
                "qwen3_8b",
                "--suite",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
    finally:
        run_benchmark.run_prompt = saved
    check(rc2 == 0, "suite re-run exits 0 (all cached)")
    with db_module.open_db(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
    check(count == n_seeds, f"no duplicate rows on cached re-run (count={count})")


def _stub_answer(p: dict) -> str:
    method = p.get("scoring_method")
    exp = p.get("expected_output", "")
    args = p.get("scoring_args", {}) or {}
    if method == "numeric":
        body = f"The answer is {exp}."
    elif method == "exact_match":
        body = exp
    elif method == "contains":
        body = f"The answer is {exp}."
    elif method == "regex":
        body = "Example value: 12345 abcde."
    elif method == "json_valid":
        keys = args.get("required_keys") or ["ok"]
        body = json.dumps({k: (0 if k == "age" else "x") for k in keys})
    else:
        body = exp
    return f"<think>reasoning</think>{body}"


def test_batch_multi_model(tmp_db: str) -> None:
    print("[bonus] --models multi-model batch (stubbed executor, scored matrix)")

    prompts = prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    n_prompts = len(prompts)
    by_input = {p["input"]: p for p in prompts}

    def fake_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        p = by_input.get(prompt_text)
        reply = _stub_answer(p) if p else "fallback"
        return {
            "output": reply,
            "latency_ms": 100,
            "ttft_ms": 20,
            "tokens_per_sec": 10.0,
        }

    selected = ["qwen2.5_7b", "qwen3_8b", "gemma3_27b"]
    saved = run_benchmark.run_prompt
    run_benchmark.run_prompt = fake_run_prompt
    try:
        rc = run_benchmark.main(
            [
                "--models",
                ",".join(selected),
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
        check(rc == 0, f"batch CLI exit code 0 (got {rc})")

        with db_module.open_db(tmp_db) as conn:
            model_rows = list(
                conn.execute("SELECT id, name FROM models ORDER BY name")
            )
            names = {m["name"] for m in model_rows}
            check(
                names == set(selected),
                f"exactly the selected models persisted (got {sorted(names)})",
            )
            total = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
            check(
                total == n_prompts * len(selected),
                f"one row per (model, prompt): expected "
                f"{n_prompts * len(selected)}, got {total}",
            )
            for m in model_rows:
                rows = list(
                    conn.execute(
                        "SELECT output, quality_score FROM runs WHERE model_id = ?",
                        (m["id"],),
                    )
                )
                check(
                    len(rows) == n_prompts,
                    f"{m['name']}: {n_prompts} rows (got {len(rows)})",
                )
                check(
                    all(r["quality_score"] is not None for r in rows),
                    f"{m['name']}: every row has a populated quality_score",
                )
                check(
                    all(r["quality_score"] == 1.0 for r in rows),
                    f"{m['name']}: tailored stub answers all score 1.0",
                )
                check(
                    all("<think>" in r["output"] for r in rows),
                    f"{m['name']}: RAW output (with <think>) stored, not cleaned",
                )

        rc2 = run_benchmark.main(
            [
                "--all-models",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
        check(rc2 == 0, "--all-models re-run exits 0")
        with db_module.open_db(tmp_db) as conn:
            total2 = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
            n_all = len(run_benchmark.BUILTIN_MODELS)
            check(
                total2 == n_prompts * n_all,
                f"--all-models fills matrix without dups: expected "
                f"{n_prompts * n_all}, got {total2}",
            )
            n_models = conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()["c"]
            check(n_models == n_all, f"all {n_all} builtin models persisted")

        rc3 = run_benchmark.main(
            [
                "--models",
                ",".join(selected),
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
        check(rc3 == 0, "subset re-run exits 0 (all cached)")
        with db_module.open_db(tmp_db) as conn:
            total3 = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
        check(total3 == total2, f"cached subset re-run adds no rows (count={total3})")
    finally:
        run_benchmark.run_prompt = saved


class _FakeHTTPError(Exception):

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)

        class _Resp:
            pass

        resp = _Resp()
        resp.status_code = status_code
        self.response = resp


def test_retry_transient_then_success(tmp_db: str) -> None:
    print("[bonus] retry: transient error is retried then succeeds")

    prompts = prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    target_input = prompts[0]["input"]

    calls = {"n": 0}

    def flaky_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        if prompt_text == target_input:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeHTTPError("model is currently loading", 503)
        p = {q["input"]: q for q in prompts}.get(prompt_text)
        return {
            "output": _stub_answer(p) if p else "fallback",
            "latency_ms": 100,
            "ttft_ms": 20,
            "tokens_per_sec": 10.0,
        }

    import runner.retry as retry_module

    saved_safe_run = run_benchmark.safe_run

    def fast_safe_run(fn, *args, **kwargs):
        kwargs["wait_min"] = 0.0
        kwargs["wait_max"] = 0.0
        kwargs["wait_multiplier"] = 0.0
        return saved_safe_run(fn, *args, **kwargs)

    saved = run_benchmark.run_prompt
    run_benchmark.run_prompt = flaky_run_prompt
    run_benchmark.safe_run = fast_safe_run
    try:
        rc = run_benchmark.main(
            [
                "--model",
                "qwen3_8b",
                "--suite",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
    finally:
        run_benchmark.run_prompt = saved
        run_benchmark.safe_run = saved_safe_run

    check(rc == 0, f"suite with one transient failure still exits 0 (got {rc})")
    check(calls["n"] >= 2, f"transient cell was retried (call count={calls['n']})")
    with db_module.open_db(tmp_db) as conn:
        row = conn.execute(
            "SELECT 1 FROM runs r JOIN prompts p ON p.id = r.prompt_id "
            "WHERE p.input = ?",
            (target_input,),
        ).fetchone()
    check(row is not None, "retried cell ultimately wrote a row")

    check(
        retry_module.is_transient_error(_FakeHTTPError("loading", 503)) is True,
        "is_transient_error: 503 classified transient",
    )
    check(
        retry_module.is_transient_error(TimeoutError("timed out")) is True,
        "is_transient_error: TimeoutError classified transient",
    )


def test_retry_permanent_fails_fast(tmp_db: str) -> None:
    print("[bonus] retry: permanent error is NOT retried, batch continues")

    prompts = prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    target_input = prompts[0]["input"]
    by_input = {p["input"]: p for p in prompts}

    calls = {"n": 0}

    def perm_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        if prompt_text == target_input:
            calls["n"] += 1
            raise _FakeHTTPError("403 Forbidden: gated repo", 403)
        p = by_input.get(prompt_text)
        return {
            "output": _stub_answer(p) if p else "fallback",
            "latency_ms": 100,
            "ttft_ms": 20,
            "tokens_per_sec": 10.0,
        }

    saved = run_benchmark.run_prompt
    run_benchmark.run_prompt = perm_run_prompt
    try:
        rc = run_benchmark.main(
            [
                "--model",
                "qwen3_8b",
                "--suite",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
    finally:
        run_benchmark.run_prompt = saved

    n_prompts = len(prompts)
    check(rc == 1, f"a permanent error makes the run exit non-zero (got {rc})")
    check(calls["n"] == 1, f"permanent cell tried exactly once, no retry (got {calls['n']})")
    with db_module.open_db(tmp_db) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
        failed_row = conn.execute(
            "SELECT 1 FROM runs r JOIN prompts p ON p.id = r.prompt_id "
            "WHERE p.input = ?",
            (target_input,),
        ).fetchone()
    check(
        total == n_prompts - 1,
        f"batch continued past the failed cell: {n_prompts - 1} rows (got {total})",
    )
    check(failed_row is None, "failed (permanent) cell wrote no row")

    import runner.retry as retry_module

    check(
        retry_module.is_transient_error(_FakeHTTPError("forbidden", 403)) is False,
        "is_transient_error: 403 classified permanent",
    )
    check(
        retry_module.is_transient_error(_FakeHTTPError("billing", 402)) is False,
        "is_transient_error: 402 classified permanent",
    )
    check(
        retry_module.is_transient_error(ValueError("bad request")) is False,
        "is_transient_error: unknown error type classified permanent",
    )


def test_per_model_max_tokens(tmp_db: str) -> None:
    print("[bonus] per-model max_tokens budget + --max-tokens override precedence")

    prompts = prompts_module.load_prompts(os.path.join(ROOT, "prompts"))
    by_input = {p["input"]: p for p in prompts}

    seen: dict[str, list[int]] = {}

    def capturing_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        seen.setdefault(hf_id, []).append(max_tokens)
        p = by_input.get(prompt_text)
        return {
            "output": _stub_answer(p) if p else "fallback",
            "latency_ms": 100,
            "ttft_ms": 20,
            "tokens_per_sec": 10.0,
        }

    reasoning_hf = run_benchmark.BUILTIN_MODELS["qwen3_8b"]["hf_id"]
    plain_hf = run_benchmark.BUILTIN_MODELS["qwen2.5_7b"]["hf_id"]

    saved = run_benchmark.run_prompt
    run_benchmark.run_prompt = capturing_run_prompt
    try:
        rc = run_benchmark.main(
            [
                "--models",
                "qwen3_8b,qwen2.5_7b",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db,
                "--token",
                "fake-token-for-verify",
            ]
        )
        check(rc == 0, f"per-model budget run exits 0 (got {rc})")
        check(
            seen.get(reasoning_hf) and all(v == 4096 for v in seen[reasoning_hf]),
            f"reasoning model (qwen3_8b) gets per-model 4096 "
            f"(got {sorted(set(seen.get(reasoning_hf, [])))})",
        )
        check(
            seen.get(plain_hf) and all(v == 512 for v in seen[plain_hf]),
            f"non-reasoning model (qwen2.5_7b) gets default 512 "
            f"(got {sorted(set(seen.get(plain_hf, [])))})",
        )

        seen.clear()
        rc2 = run_benchmark.main(
            [
                "--models",
                "qwen3_8b,qwen2.5_7b",
                "--max-tokens",
                "256",
                "--prompts-dir",
                os.path.join(ROOT, "prompts"),
                "--db-path",
                tmp_db + ".override",
                "--token",
                "fake-token-for-verify",
            ]
        )
        check(rc2 == 0, f"--max-tokens override run exits 0 (got {rc2})")
        check(
            seen.get(reasoning_hf) and all(v == 256 for v in seen[reasoning_hf]),
            f"--max-tokens 256 overrides per-model 4096 "
            f"(got {sorted(set(seen.get(reasoning_hf, [])))})",
        )
        check(
            seen.get(plain_hf) and all(v == 256 for v in seen[plain_hf]),
            f"--max-tokens 256 overrides default 512 "
            f"(got {sorted(set(seen.get(plain_hf, [])))})",
        )
    finally:
        run_benchmark.run_prompt = saved


def test_force_overwrite(tmp_db: str) -> None:
    print("[bonus] --force cleanly overwrites the existing (model, prompt) row")

    prompt_id = "smoke_test"

    def first_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        return {
            "output": "v1",
            "latency_ms": 111,
            "ttft_ms": 11,
            "tokens_per_sec": 1.0,
        }

    def second_run_prompt(hf_id, prompt_text, token, max_tokens=512, stream=True):
        return {
            "output": "v2",
            "latency_ms": 222,
            "ttft_ms": 22,
            "tokens_per_sec": 2.0,
        }

    base_args = [
        "--model",
        "qwen2.5_7b",
        "--prompt-id",
        prompt_id,
        "--prompt-text",
        "echo back v1",
        "--db-path",
        tmp_db,
        "--token",
        "fake-token-for-verify",
    ]

    saved = run_benchmark.run_prompt
    try:
        run_benchmark.run_prompt = first_run_prompt
        rc = run_benchmark.main(base_args)
        check(rc == 0, f"initial run exits 0 (got {rc})")

        with db_module.open_db(tmp_db) as conn:
            rows = list(
                conn.execute(
                    "SELECT id, output, latency_ms, ttft_ms, tokens_per_sec "
                    "FROM runs"
                )
            )
        check(len(rows) == 1, f"one row after initial run (got {len(rows)})")
        orig_id = rows[0]["id"]
        check(rows[0]["output"] == "v1", "initial output is v1")

        run_benchmark.run_prompt = second_run_prompt
        rc_cached = run_benchmark.main(base_args)
        check(rc_cached == 0, f"non-force re-run exits 0 (got {rc_cached})")
        with db_module.open_db(tmp_db) as conn:
            rows = list(
                conn.execute("SELECT id, output FROM runs")
            )
        check(len(rows) == 1, f"non-force re-run adds no row (got {len(rows)})")
        check(rows[0]["output"] == "v1", "non-force re-run leaves original v1")

        try:
            rc_force = run_benchmark.main(base_args + ["--force"])
        except sqlite3.IntegrityError as exc:
            raise AssertionError(f"--force raised IntegrityError: {exc}")
        check(rc_force == 0, f"--force re-run exits 0 (got {rc_force})")

        with db_module.open_db(tmp_db) as conn:
            rows = list(
                conn.execute(
                    "SELECT id, output, latency_ms, ttft_ms, tokens_per_sec, "
                    "quality_score FROM runs"
                )
            )
        check(len(rows) == 1, f"--force keeps row count at one (got {len(rows)})")
        r = rows[0]
        check(r["output"] == "v2", "--force updated output to v2")
        check(r["latency_ms"] == 222, "--force updated latency_ms")
        check(r["ttft_ms"] == 22, "--force updated ttft_ms")
        check(abs(r["tokens_per_sec"] - 2.0) < 1e-9, "--force updated tokens_per_sec")
        check(r["quality_score"] is None, "--force preserves manual NULL score")
    finally:
        run_benchmark.run_prompt = saved


def test_throughput_metric() -> None:
    print("[bonus] throughput metric: shared estimate + executor cap + backfill")

    from runner import metrics as metrics_module
    from runner import executor as executor_module
    import scripts.fix_throughput as fix_module

    est = metrics_module.estimate_tokens_per_sec

    check(est("x" * 400, 1000) == 100.0, "estimate: 400 chars / 1000ms ≈ 100 tok/s")
    check(est("", 1000) is None, "estimate: empty output → None")
    check(est(None, 1000) is None, "estimate: None output → None")
    check(est("hello world", 0) is None, "estimate: latency 0 → None")
    check(est("hello world", -5) is None, "estimate: negative latency → None")
    normal = est("a normal model answer of moderate length here.", 800)
    check(
        normal is not None and 0 < normal <= metrics_module.MAX_TOKENS_PER_SEC,
        f"estimate: normal case is sane (<= cap), got {normal}",
    )
    big = est("z" * 4000, 1000)
    check(big == 1000.0, f"estimate: 4000 chars / 1000ms = 1000 tok/s (got {big})")

    from_count = metrics_module.tokens_per_sec_from_count
    exploded = from_count(
        500, 1e-9, output="some output text here that is short", latency_ms=300
    )
    check(
        exploded is None or exploded <= metrics_module.MAX_TOKENS_PER_SEC,
        f"from_count: collapsed window does not explode (got {exploded})",
    )
    clean = from_count(50, 1.0, output="x" * 200, latency_ms=1000)
    check(clean == 50.0, f"from_count: 50 tokens / 1.0s = 50 tok/s (got {clean})")
    fellback = from_count(None, 0.0, output="x" * 400, latency_ms=1000)
    check(fellback == 100.0, f"from_count: unknown count falls back to estimate (got {fellback})")

    class _FakeBurstStream:

        def __iter__(self):
            yield {"choices": [{"delta": {"content": "a one-shot burst answer."}}]}

    class _FakeClient:
        def chat_completion(self, messages, max_tokens, stream=False):
            return _FakeBurstStream()

    res = executor_module._run_streaming(_FakeClient(), [{"role": "user", "content": "hi"}], 64)
    tps = res["tokens_per_sec"]
    check(
        tps is None or tps <= metrics_module.MAX_TOKENS_PER_SEC,
        f"executor: burst stream does not emit absurd tok/s (got {tps})",
    )

    with tempfile.TemporaryDirectory() as tdir:
        bdb = os.path.join(tdir, "backfill.db")
        with db_module.open_db(bdb) as conn:
            keep_id = db_module.upsert_model(conn, "keep_me", hf_id="org/keep")
            db_module.upsert_model(conn, "stale_no_runs", hf_id="org/stale")
            db_module.upsert_prompt(conn, "p1", category="c", input_text="q")
            db_module.insert_run(
                conn,
                model_id=keep_id,
                prompt_id="p1",
                output="x" * 400,
                latency_ms=1000,
                ttft_ms=None,
                tokens_per_sec=6_995_748.0,
                ts=1,
                quality_score=None,
            )

        rc = fix_module.main(["--db-path", bdb])
        check(rc == 0, f"fix_throughput.main exits 0 (got {rc})")

        with db_module.open_db(bdb) as conn:
            row = conn.execute(
                "SELECT tokens_per_sec FROM runs WHERE prompt_id = 'p1'"
            ).fetchone()
            check(
                abs(row["tokens_per_sec"] - 100.0) < 1e-9,
                f"backfill: absurd value replaced by sane estimate (got {row['tokens_per_sec']})",
            )
            names = {m["name"] for m in conn.execute("SELECT name FROM models")}
            check("stale_no_runs" not in names, "backfill: stale no-run model removed")
            check("keep_me" in names, "backfill: model with runs kept")

        rc2 = fix_module.main(["--db-path", bdb])
        check(rc2 == 0, "fix_throughput re-run exits 0 (idempotent)")
        with db_module.open_db(bdb) as conn:
            row = conn.execute(
                "SELECT tokens_per_sec FROM runs WHERE prompt_id = 'p1'"
            ).fetchone()
            check(
                abs(row["tokens_per_sec"] - 100.0) < 1e-9,
                "backfill: idempotent re-run keeps the sane estimate",
            )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        db1 = os.path.join(tmpdir, "layer.db")
        db2 = os.path.join(tmpdir, "cli.db")
        db3 = os.path.join(tmpdir, "suite.db")
        db4 = os.path.join(tmpdir, "batch.db")
        db5 = os.path.join(tmpdir, "retry_transient.db")
        db6 = os.path.join(tmpdir, "retry_permanent.db")
        db7 = os.path.join(tmpdir, "max_tokens.db")
        db8 = os.path.join(tmpdir, "force_overwrite.db")
        test_db_layer(db1)
        test_cli_with_stub(db2)
        test_strip_reasoning()
        test_scoring_methods()
        test_load_prompts()
        test_suite_cli_with_stub(db3)
        test_batch_multi_model(db4)
        test_retry_transient_then_success(db5)
        test_retry_permanent_fails_fast(db6)
        test_per_model_max_tokens(db7)
        test_force_overwrite(db8)
        test_throughput_metric()
    print("\nAll verification checks passed.")
    print("Next step (with a live HF token exported):")
    print("  python run_benchmark.py --model qwen2.5_7b --prompt-id smoke_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
