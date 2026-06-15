
from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from runner.metrics import tokens_per_sec_from_count


class RunResult(TypedDict):

    output: str
    latency_ms: int
    ttft_ms: Optional[int]
    tokens_per_sec: Optional[float]


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _completion_tokens(response: Any) -> Optional[int]:
    usage = _safe_get(response, "usage")
    count = _safe_get(usage, "completion_tokens")
    if count is None:
        return None
    try:
        return int(count)
    except (TypeError, ValueError):
        return None


def run_prompt(
    hf_id: str,
    prompt_text: str,
    token: str,
    max_tokens: int = 512,
    stream: bool = True,
) -> RunResult:
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=hf_id, token=token)
    messages = [{"role": "user", "content": prompt_text}]

    if stream:
        return _run_streaming(client, messages, max_tokens)
    return _run_blocking(client, messages, max_tokens)


def _run_blocking(client: Any, messages: list, max_tokens: int) -> RunResult:
    start = time.perf_counter()
    response = client.chat_completion(messages=messages, max_tokens=max_tokens)
    elapsed_s = time.perf_counter() - start

    choices = _safe_get(response, "choices") or []
    output = ""
    if choices:
        message = _safe_get(choices[0], "message")
        output = str(_safe_get(message, "content", "") or "")

    token_count = _completion_tokens(response)
    latency_ms = int(elapsed_s * 1000)
    return RunResult(
        output=output,
        latency_ms=latency_ms,
        ttft_ms=None,
        tokens_per_sec=tokens_per_sec_from_count(
            token_count, elapsed_s, output=output, latency_ms=latency_ms
        ),
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

    output = "".join(output_parts)
    latency_ms = int(elapsed_s * 1000)
    return RunResult(
        output=output,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        tokens_per_sec=tokens_per_sec_from_count(
            token_count, gen_seconds, output=output, latency_ms=latency_ms
        ),
    )
