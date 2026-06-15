from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_DB_PATH = "data/results.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  hf_id TEXT,
  provider TEXT,
  size_b REAL
);

CREATE TABLE IF NOT EXISTS prompts (
  id TEXT PRIMARY KEY,
  category TEXT,
  input TEXT,
  expected_output TEXT,
  scoring_method TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  model_id INTEGER REFERENCES models(id),
  prompt_id TEXT REFERENCES prompts(id),
  output TEXT,
  quality_score REAL,
  latency_ms INTEGER,
  ttft_ms INTEGER,
  tokens_per_sec REAL,
  ts INTEGER,
  UNIQUE(model_id, prompt_id)
);
"""


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


@contextmanager
def open_db(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def upsert_model(
    conn: sqlite3.Connection,
    name: str,
    hf_id: Optional[str] = None,
    provider: Optional[str] = None,
    size_b: Optional[float] = None,
) -> int:

    if hf_id is None:
        hf_id = name

    conn.execute(
        """
        INSERT INTO models (name, hf_id, provider, size_b)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            hf_id = excluded.hf_id,
            provider = COALESCE(excluded.provider, models.provider),
            size_b = COALESCE(excluded.size_b, models.size_b)
        """,
        (name, hf_id, provider, size_b),
    )
    conn.commit()

    row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert model {name!r}")
    return int(row["id"])


def upsert_prompt(
    conn: sqlite3.Connection,
    prompt_id: str,
    category: Optional[str] = None,
    input_text: Optional[str] = None,
    expected_output: Optional[str] = None,
    scoring_method: Optional[str] = None,
) -> str:
    conn.execute(
        """
        INSERT INTO prompts (id, category, input, expected_output, scoring_method)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            category = COALESCE(excluded.category, prompts.category),
            input = COALESCE(excluded.input, prompts.input),
            expected_output = COALESCE(excluded.expected_output, prompts.expected_output),
            scoring_method = COALESCE(excluded.scoring_method, prompts.scoring_method)
        """,
        (prompt_id, category, input_text, expected_output, scoring_method),
    )
    conn.commit()
    return prompt_id


def run_exists(conn: sqlite3.Connection, model_id: int, prompt_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM runs WHERE model_id = ? AND prompt_id = ? LIMIT 1",
        (model_id, prompt_id),
    ).fetchone()
    return row is not None


def insert_run(
    conn: sqlite3.Connection,
    model_id: int,
    prompt_id: str,
    output: str,
    latency_ms: int,
    tokens_per_sec: float,
    ts: int,
    ttft_ms: Optional[int] = None,
    quality_score: Optional[float] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (
            model_id, prompt_id, output, quality_score,
            latency_ms, ttft_ms, tokens_per_sec, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            prompt_id,
            output,
            quality_score,
            latency_ms,
            ttft_ms,
            tokens_per_sec,
            ts,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_run(
    conn: sqlite3.Connection,
    model_id: int,
    prompt_id: str,
    output: str,
    latency_ms: int,
    tokens_per_sec: float,
    ts: int,
    ttft_ms: Optional[int] = None,
    quality_score: Optional[float] = None,
) -> int:
    conn.execute(
        """
        INSERT INTO runs (
            model_id, prompt_id, output, quality_score,
            latency_ms, ttft_ms, tokens_per_sec, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_id, prompt_id) DO UPDATE SET
            output = excluded.output,
            quality_score = excluded.quality_score,
            latency_ms = excluded.latency_ms,
            ttft_ms = excluded.ttft_ms,
            tokens_per_sec = excluded.tokens_per_sec,
            ts = excluded.ts
        """,
        (
            model_id,
            prompt_id,
            output,
            quality_score,
            latency_ms,
            ttft_ms,
            tokens_per_sec,
            ts,
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT id FROM runs WHERE model_id = ? AND prompt_id = ?",
        (model_id, prompt_id),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Failed to upsert run for (model_id={model_id}, prompt_id={prompt_id!r})"
        )
    return int(row["id"])
