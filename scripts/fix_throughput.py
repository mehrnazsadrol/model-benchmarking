
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runner import db as db_module
from runner.metrics import estimate_tokens_per_sec


def _summary(conn: sqlite3.Connection) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {}
    rows = conn.execute(
        "SELECT m.name AS name, r.tokens_per_sec AS tps "
        "FROM runs r JOIN models m ON m.id = r.model_id "
        "ORDER BY m.name"
    ).fetchall()
    by_model: dict[str, list[float]] = {}
    nulls: dict[str, int] = {}
    for row in rows:
        name = row["name"]
        by_model.setdefault(name, [])
        nulls.setdefault(name, 0)
        if row["tps"] is None:
            nulls[name] += 1
        else:
            by_model[name].append(float(row["tps"]))
    for name, vals in by_model.items():
        out[name] = {
            "min": min(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n": len(vals) + nulls[name],
            "nulls": nulls[name],
        }
    return out


def _print_summary(title: str, summary: dict[str, dict[str, Optional[float]]]) -> None:
    print(f"\n{title}")
    print(f"  {'model':<24} {'n':>4} {'nulls':>6} {'min':>12} {'median':>12} {'max':>14}")
    for name in sorted(summary):
        s = summary[name]

        def fmt(v: Optional[float]) -> str:
            return "—" if v is None else f"{v:,.1f}"

        print(
            f"  {name:<24} {s['n']:>4} {s['nulls']:>6} "
            f"{fmt(s['min']):>12} {fmt(s['median']):>12} {fmt(s['max']):>14}"
        )


def recompute_throughput(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id, output, latency_ms, tokens_per_sec FROM runs"
    ).fetchall()
    changed = 0
    for row in rows:
        new_tps = estimate_tokens_per_sec(row["output"], row["latency_ms"])
        old_tps = row["tokens_per_sec"]
        if old_tps is None and new_tps is None:
            continue
        if (
            old_tps is not None
            and new_tps is not None
            and abs(float(old_tps) - float(new_tps)) < 1e-9
        ):
            continue
        conn.execute(
            "UPDATE runs SET tokens_per_sec = ? WHERE id = ?",
            (new_tps, row["id"]),
        )
        changed += 1
    return changed


def remove_stale_models(conn: sqlite3.Connection) -> list[str]:
    stale = conn.execute(
        "SELECT m.id AS id, m.name AS name FROM models m "
        "WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.model_id = m.id) "
        "ORDER BY m.name"
    ).fetchall()
    removed: list[str] = []
    for row in stale:
        conn.execute("DELETE FROM models WHERE id = ?", (row["id"],))
        removed.append(row["name"])
    return removed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fix_throughput",
        description=(
            "Recompute tokens_per_sec in place from stored columns (chars/4 "
            "estimate) and remove stale models with no runs. Network-free, "
            "idempotent, single transaction."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=db_module.DEFAULT_DB_PATH,
        help=f"SQLite DB path (default: {db_module.DEFAULT_DB_PATH}).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.db_path):
        print(f"[error] DB not found: {args.db_path}", file=sys.stderr)
        return 1

    with db_module.open_db(args.db_path) as conn:
        before = _summary(conn)
        _print_summary("BEFORE (per-model tokens_per_sec):", before)

        try:
            conn.execute("BEGIN")
            changed = recompute_throughput(conn)
            removed = remove_stale_models(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        after = _summary(conn)
        _print_summary("AFTER (per-model tokens_per_sec):", after)

        print(f"\n[recompute] updated {changed} run row(s).")
        if removed:
            print(f"[cleanup] removed stale model(s) with zero runs: {removed}")
        else:
            print("[cleanup] no stale models to remove.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
