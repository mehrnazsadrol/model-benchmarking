from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app


def _direct_overall(conn: sqlite3.Connection, model: str) -> float:
    row = conn.execute(
        """
        SELECT AVG(r.quality_score)
        FROM runs r
        JOIN prompts p ON p.id = r.prompt_id
        JOIN models m ON m.id = r.model_id
        WHERE r.quality_score IS NOT NULL
          AND p.category IN ('reasoning','math')
          AND m.name = ?
        """,
        (model,),
    ).fetchone()
    return float(row[0])


def main() -> int:
    lb = app.load_leaderboard()
    assert len(lb) == 9, f"expected 9 model rows, got {len(lb)}"

    for col in ("quality", "reasoning_quality", "math_quality"):
        vals = lb[col].dropna()
        assert ((vals >= 0.0) & (vals <= 1.0)).all(), f"{col} out of [0,1]"

    conn = sqlite3.connect(app.DB_PATH)
    cats = {
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT p.category
            FROM runs r JOIN prompts p ON p.id = r.prompt_id
            WHERE r.quality_score IS NOT NULL AND p.category IN ('reasoning','math')
            """
        )
    }
    assert cats == {"reasoning", "math"}, f"unexpected categories fed in: {cats}"

    full = {
        r[0]
        for r in conn.execute("SELECT DISTINCT category FROM prompts").fetchall()
    }
    assert "instruction_following" in full, "fixture sanity: if missing, exclusion is untested"

    pc = app.load_per_category()
    assert not pc.empty, "per-category frame empty"
    assert set(pc["category"].unique()) == {"reasoning", "math"}

    prompts = app.load_prompt_choices()
    assert not prompts.empty, "prompt choices empty"

    names = app.load_model_names()
    assert len(names) == 9, f"expected 9 models, got {len(names)}"

    pid = prompts.iloc[0]["prompt_id"]
    h2h = app.load_head_to_head(names[0], pid)
    assert h2h["output"], "head-to-head returned empty output for known (model,prompt)"

    missing = app.load_head_to_head("does_not_exist", pid)
    assert missing == {"output": "", "quality_score": None}, "missing model not handled"

    disp = app.leaderboard_display()
    assert list(disp["model"])[0] == lb.iloc[0]["model"], "display order mismatch"

    gemma = _direct_overall(conn, "gemma3_27b")
    qwen = _direct_overall(conn, "qwen2.5_7b")
    assert abs(gemma - 0.93) < 0.02, f"gemma3_27b {gemma:.4f} not ~0.93"
    assert abs(qwen - 0.74) < 0.02, f"qwen2.5_7b {qwen:.4f} not ~0.74"

    lb_gemma = float(lb[lb["model"] == "gemma3_27b"]["quality"].iloc[0])
    assert abs(lb_gemma - gemma) < 1e-9, "leaderboard != direct SQL for gemma3_27b"

    conn.close()

    f1 = app.fig_latency_quality()
    f2 = app.fig_per_category()
    assert f1.data, "latency-quality figure has no traces"
    assert f2.data, "per-category figure has no traces"

    print("OK leaderboard rows:", len(lb))
    print("OK gemma3_27b overall:", round(gemma, 4))
    print("OK qwen2.5_7b overall:", round(qwen, 4))
    print("OK figures built:", len(f1.data), "and", len(f2.data), "traces")
    print("ALL DASHBOARD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
