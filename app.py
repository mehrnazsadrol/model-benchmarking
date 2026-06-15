from __future__ import annotations

import os
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = os.environ.get("RESULTS_DB_PATH", "data/results.db")

CATEGORIES = ("reasoning", "math")

_CATEGORY_FILTER = "p.category IN ('reasoning','math')"
_SCORE_FILTER = "r.quality_score IS NOT NULL"

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "$", "right": "$", "display": False},
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_leaderboard() -> pd.DataFrame:
    overall = _query(
        f"""
        SELECT
            m.id AS model_id,
            m.name AS model,
            m.size_b AS size_b,
            AVG(r.quality_score) AS quality,
            AVG(r.latency_ms) AS latency_ms,
            AVG(r.tokens_per_sec) AS tokens_per_sec
        FROM runs r
        JOIN prompts p ON p.id = r.prompt_id
        JOIN models m ON m.id = r.model_id
        WHERE {_SCORE_FILTER} AND {_CATEGORY_FILTER}
        GROUP BY m.id, m.name, m.size_b
        """
    )

    per_cat = _query(
        f"""
        SELECT
            m.id AS model_id,
            p.category AS category,
            AVG(r.quality_score) AS quality
        FROM runs r
        JOIN prompts p ON p.id = r.prompt_id
        JOIN models m ON m.id = r.model_id
        WHERE {_SCORE_FILTER} AND {_CATEGORY_FILTER}
        GROUP BY m.id, p.category
        """
    )

    if overall.empty:
        return overall

    wide = per_cat.pivot(index="model_id", columns="category", values="quality")
    for cat in CATEGORIES:
        if cat not in wide.columns:
            wide[cat] = pd.NA

    df = overall.merge(wide, on="model_id", how="left")
    df = df.rename(
        columns={
            "reasoning": "reasoning_quality",
            "math": "math_quality",
        }
    )
    df = df.sort_values("quality", ascending=False).reset_index(drop=True)
    return df


def load_per_category() -> pd.DataFrame:
    return _query(
        f"""
        SELECT
            m.name AS model,
            p.category AS category,
            AVG(r.quality_score) AS quality
        FROM runs r
        JOIN prompts p ON p.id = r.prompt_id
        JOIN models m ON m.id = r.model_id
        WHERE {_SCORE_FILTER} AND {_CATEGORY_FILTER}
        GROUP BY m.name, p.category
        ORDER BY m.name, p.category
        """
    )


def load_model_names() -> list[str]:
    df = _query("SELECT DISTINCT name FROM models ORDER BY name")
    return df["name"].tolist()


def load_prompt_choices() -> pd.DataFrame:
    return _query(
        f"""
        SELECT DISTINCT p.id AS prompt_id, p.category AS category, p.input AS input
        FROM prompts p
        JOIN runs r ON r.prompt_id = p.id
        WHERE {_SCORE_FILTER} AND {_CATEGORY_FILTER}
        ORDER BY p.id
        """
    )


def load_head_to_head(model_name: str, prompt_id: str) -> dict:
    df = _query(
        """
        SELECT r.output AS output, r.quality_score AS quality_score
        FROM runs r
        JOIN models m ON m.id = r.model_id
        WHERE m.name = ? AND r.prompt_id = ?
        LIMIT 1
        """,
        (model_name, prompt_id),
    )
    if df.empty:
        return {"output": "", "quality_score": None}
    row = df.iloc[0]
    output = row["output"]
    return {
        "output": "" if output is None else str(output),
        "quality_score": None if pd.isna(row["quality_score"]) else float(row["quality_score"]),
    }


def leaderboard_display() -> pd.DataFrame:
    df = load_leaderboard()
    if df.empty:
        return df
    out = pd.DataFrame()
    out["model"] = df["model"]
    out["size_b"] = df["size_b"]
    out["overall quality"] = df["quality"].round(3)
    out["reasoning quality"] = df["reasoning_quality"].round(3)
    out["math quality"] = df["math_quality"].round(3)
    out["avg latency (s)"] = (df["latency_ms"] / 1000.0).round(2)
    out["throughput (approx tok/s)"] = df["tokens_per_sec"].round(1)
    return out


def fig_latency_quality() -> go.Figure:
    df = load_leaderboard()
    if df.empty:
        return go.Figure()
    df = df.copy()
    df["latency_s"] = df["latency_ms"] / 1000.0
    fig = px.scatter(
        df,
        x="latency_s",
        y="quality",
        text="model",
        size="size_b",
        color="model",
        log_x=True,
        labels={
            "latency_s": "avg latency (s, log scale)",
            "quality": "mean quality (reasoning + math)",
        },
        title="Latency vs quality (point size = params in B)",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False, height=560)
    return fig


def fig_per_category() -> go.Figure:
    df = load_per_category()
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df,
        x="model",
        y="quality",
        color="category",
        barmode="group",
        labels={"quality": "mean quality", "model": "model", "category": "category"},
        title="Mean quality per category by model",
    )
    fig.update_layout(height=560, xaxis_tickangle=-30, yaxis_range=[0, 1])
    return fig


def _format_score(score) -> str:
    return "n/a" if score is None else f"{score:.3f}"


def render_head_to_head(model_a: str, model_b: str, prompt_id: str):
    if not prompt_id:
        msg = "Select a prompt."
        return msg, [], msg, []
    prompts = load_prompt_choices()
    match = prompts[prompts["prompt_id"] == prompt_id]
    prompt_text = "" if match.empty else str(match.iloc[0]["input"])

    a = load_head_to_head(model_a, prompt_id)
    b = load_head_to_head(model_b, prompt_id)

    a_out = a["output"] or "(no output recorded)"
    b_out = b["output"] or "(no output recorded)"
    a_label = f"### {model_a} — quality {_format_score(a['quality_score'])}"
    b_label = f"### {model_b} — quality {_format_score(b['quality_score'])}"

    chat_a = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": a_out},
    ]
    chat_b = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": b_out},
    ]
    return a_label, chat_a, b_label, chat_b


def build_demo():
    import gradio as gr

    model_names = load_model_names()
    prompts_df = load_prompt_choices()
    prompt_ids = prompts_df["prompt_id"].tolist()
    default_a = model_names[0] if model_names else None
    default_b = model_names[1] if len(model_names) > 1 else default_a
    default_prompt = prompt_ids[0] if prompt_ids else None

    with gr.Blocks(title="Model Benchmarking Dashboard") as demo:
        gr.Markdown(
            "# Model Benchmarking Dashboard\n"
            "Quality, latency, and throughput across reasoning + math prompts. "
            "Throughput is an approximate metric (tok/s)."
        )

        with gr.Tab("Leaderboard"):
            gr.Markdown(
                "One row per model, sorted by overall mean quality "
                "(reasoning + math)."
            )
            gr.Dataframe(value=leaderboard_display(), interactive=False)

        with gr.Tab("Latency vs quality"):
            gr.Plot(value=fig_latency_quality())

        with gr.Tab("Per-category"):
            gr.Plot(value=fig_per_category())

        with gr.Tab("Head-to-head"):
            with gr.Row():
                model_a = gr.Dropdown(model_names, value=default_a, label="Model A")
                model_b = gr.Dropdown(model_names, value=default_b, label="Model B")
                prompt = gr.Dropdown(prompt_ids, value=default_prompt, label="Prompt")
            with gr.Row():
                with gr.Column():
                    label_a = gr.Markdown()
                    out_a = gr.Chatbot(
                        label="Model A",
                        height=480,
                        latex_delimiters=LATEX_DELIMITERS,
                    )
                with gr.Column():
                    label_b = gr.Markdown()
                    out_b = gr.Chatbot(
                        label="Model B",
                        height=480,
                        latex_delimiters=LATEX_DELIMITERS,
                    )

            inputs = [model_a, model_b, prompt]
            outputs = [label_a, out_a, label_b, out_b]
            for control in inputs:
                control.change(render_head_to_head, inputs=inputs, outputs=outputs)

            demo.load(render_head_to_head, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
