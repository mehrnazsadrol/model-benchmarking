from __future__ import annotations

from typing import Optional

MIN_SECONDS = 1e-3

MAX_TOKENS_PER_SEC = 10_000.0

CHARS_PER_TOKEN = 4


def estimate_tokens_per_sec(
    output: Optional[str], latency_ms: Optional[int]
) -> Optional[float]:
    if not output:
        return None
    if latency_ms is None or latency_ms <= 0:
        return None
    approx_tokens = max(1, len(output) // CHARS_PER_TOKEN)
    seconds = latency_ms / 1000.0
    if seconds <= 0:
        return None
    return round(approx_tokens / seconds, 1)


def tokens_per_sec_from_count(
    token_count: Optional[int],
    gen_seconds: float,
    *,
    output: Optional[str] = None,
    latency_ms: Optional[int] = None,
) -> Optional[float]:
    computed: Optional[float] = None
    if token_count and token_count > 0:
        seconds = gen_seconds if gen_seconds and gen_seconds > 0 else MIN_SECONDS
        seconds = max(seconds, MIN_SECONDS)
        computed = token_count / seconds

    if computed is not None and 0 < computed <= MAX_TOKENS_PER_SEC:
        return round(computed, 1)

    estimated = estimate_tokens_per_sec(output, latency_ms)
    if estimated is None:
        return None
    if estimated > MAX_TOKENS_PER_SEC:
        return None
    return estimated
