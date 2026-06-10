"""Single-prompt executor: calls the Hugging Face Inference Providers API.

This is the serverless replacement for the old local-ollama executor. There is
no daemon to probe and no local model to load, so the memory/VRAM metric is
gone (see README — "No memory metric (serverless)"). We measure three things:

  - ``latency_ms``     — wall-clock time for the whole request/response.
  - ``ttft_ms``        — time-to-first-token. Only meaningful on the streaming
                         path, where we timestamp the first chunk that carries
                         content. ``None`` when we can't measure it.
  - ``tokens_per_sec`` — generation throughput. We prefer the provider-reported
                         ``usage.completion_tokens`` count; on the streaming
                         path we fall back to counting chunks. Throughput is
                         divided by *generation* time (from first token to last)
                         when streaming, else by total wall time.

Defensiveness mirrors the old code: token counts and the ``usage`` object vary
by ``huggingface_hub`` version and by provider, so every field access is guarded
and we return ``0.0`` / ``None`` rather than crashing on missing data.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypedDict


class RunResult(TypedDict):
    """Return shape of :func:`run_prompt`."""

    output: str
    latency_ms: int
    ttft_ms: Optional[int]
    tokens_per_sec: float


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` whether ``obj`` is a dict or an attribute-style object.

    ``huggingface_hub`` returns dataclass-like objects in newer versions and
    plain dicts in some provider/version combinations, so we normalize both.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _completion_tokens(response: Any) -> Optional[int]:
    """Extract ``usage.completion_tokens`` from a chat-completion response.

    Returns None when usage is absent or the count is missing — callers treat
    that as "unknown throughput" (0.0) rather than failing.
    """
    usage = _safe_get(response, "usage")
    count = _safe_get(usage, "completion_tokens")
    if count is None:
        return None
    try:
        return int(count)
    except (TypeError, ValueError):
        return None


def _tokens_per_sec(token_count: Optional[int], gen_seconds: float) -> float:
    """Compute tokens/sec, returning 0.0 on missing data or non-positive time.

    Bench data with a 0.0 throughput is more useful than a crash.
    """
    if not token_count or gen_seconds <= 0:
        return 0.0
    return float(token_count) / gen_seconds


def run_prompt(
    hf_id: str,
    prompt_text: str,
    token: str,
    max_tokens: int = 512,
    stream: bool = True,
) -> RunResult:
    """Run a single prompt against the HF Inference Providers chat API.

    Parameters
    ----------
    hf_id:
        Hugging Face model repo id (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
    prompt_text:
        The user prompt. No system prompt is set in the Day 1 path; we want
        to measure default behavior.
    token:
        Hugging Face access token with Inference permission. Required —
        serverless inference is authenticated.
    max_tokens:
        Cap on generated tokens (``max_tokens`` in the chat API).
    stream:
        When True (default) we use the streaming path so we can measure
        ``ttft_ms`` (time-to-first-token). If streaming yields no usable
        token count we still return the accumulated text with a chunk-counted
        throughput. When False we make a single blocking call and ``ttft_ms``
        is ``None`` (no per-token timing is available).

    Returns
    -------
    RunResult dict with output text and the three timing metrics.

    Raises
    ------
    ImportError
        If the ``huggingface_hub`` package isn't installed.
    Exception
        Anything raised by the Inference API (auth failure, model not served
        by any provider, rate limit, etc.) propagates — the CLI layer decides
        what to do.
    """
    from huggingface_hub import InferenceClient  # type: ignore

    client = InferenceClient(model=hf_id, token=token)
    messages = [{"role": "user", "content": prompt_text}]

    if stream:
        return _run_streaming(client, messages, max_tokens)
    return _run_blocking(client, messages, max_tokens)


def _run_blocking(client: Any, messages: list, max_tokens: int) -> RunResult:
    """Single blocking chat_completion call. No per-token timing → ttft None."""
    start = time.perf_counter()
    response = client.chat_completion(messages=messages, max_tokens=max_tokens)
    elapsed_s = time.perf_counter() - start

    choices = _safe_get(response, "choices") or []
    output = ""
    if choices:
        message = _safe_get(choices[0], "message")
        output = str(_safe_get(message, "content", "") or "")

    token_count = _completion_tokens(response)
    return RunResult(
        output=output,
        latency_ms=int(elapsed_s * 1000),
        ttft_ms=None,
        tokens_per_sec=_tokens_per_sec(token_count, elapsed_s),
    )


def _run_streaming(client: Any, messages: list, max_tokens: int) -> RunResult:
    start = time.perf_counter()
    first_token_perf: Optional[float] = None
    last_perf = start
    output_parts: list[str] = []
    chunk_token_count = 0
    usage_token_count: Optional[int] = None

    stream = client.chat_completion(
        messages=messages, max_tokens=max_tokens, stream=True
    )
    for chunk in stream:
        now = time.perf_counter()
        last_perf = now

        choices = _safe_get(chunk, "choices") or []
        if choices:
            delta = _safe_get(choices[0], "delta")
            content = _safe_get(delta, "content")
            if content:
                if first_token_perf is None:
                    first_token_perf = now
                output_parts.append(str(content))
                chunk_token_count += 1

        usage_seen = _completion_tokens(chunk)
        if usage_seen is not None:
            usage_token_count = usage_seen

    elapsed_s = time.perf_counter() - start

    ttft_ms: Optional[int] = (
        int((first_token_perf - start) * 1000) if first_token_perf is not None else None
    )

    if first_token_perf is not None:
        gen_seconds = last_perf - first_token_perf
        if gen_seconds <= 0:
            gen_seconds = elapsed_s
    else:
        gen_seconds = elapsed_s

    token_count = (
        usage_token_count if usage_token_count is not None else chunk_token_count
    )

    return RunResult(
        output="".join(output_parts),
        latency_ms=int(elapsed_s * 1000),
        ttft_ms=ttft_ms,
        tokens_per_sec=_tokens_per_sec(token_count, gen_seconds),
    )
