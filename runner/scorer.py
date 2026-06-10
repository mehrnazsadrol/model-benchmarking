from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    cleaned = _THINK_BLOCK_RE.sub("", text)

    open_match = _THINK_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]

    cleaned = _THINK_CLOSE_RE.sub("", cleaned)

    return cleaned.strip()


def _normalize(text: str, case_sensitive: bool = False) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed if case_sensitive else collapsed.casefold()


_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")


def _extract_last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _find_json_blob(text: str) -> str | None:
    open_to_close = {"{": "}", "[": "]"}
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch in open_to_close:
            start = i
            opener = ch
            break
    if start is None:
        return None

    closer = open_to_close[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _score_exact_match(output: str, prompt: Dict[str, Any]) -> float:
    args = prompt.get("scoring_args") or {}
    case_sensitive = bool(args.get("case_sensitive", False))
    expected = str(prompt.get("expected_output", ""))
    return (
        1.0
        if _normalize(output, case_sensitive) == _normalize(expected, case_sensitive)
        else 0.0
    )


def _score_contains(output: str, prompt: Dict[str, Any]) -> float:
    args = prompt.get("scoring_args") or {}
    case_sensitive = bool(args.get("case_sensitive", False))
    expected = _normalize(str(prompt.get("expected_output", "")), case_sensitive)
    haystack = _normalize(output, case_sensitive)
    return 1.0 if expected in haystack else 0.0


def _score_regex(output: str, prompt: Dict[str, Any]) -> float:
    args = prompt.get("scoring_args") or {}
    pattern = str(prompt.get("expected_output", ""))
    flags = _resolve_regex_flags(args.get("flags"))
    try:
        return 1.0 if re.search(pattern, output, flags) is not None else 0.0
    except re.error:
        return 0.0


_FLAG_NAMES = {
    "i": re.IGNORECASE,
    "ignorecase": re.IGNORECASE,
    "s": re.DOTALL,
    "dotall": re.DOTALL,
    "m": re.MULTILINE,
    "multiline": re.MULTILINE,
    "x": re.VERBOSE,
    "verbose": re.VERBOSE,
}


def _resolve_regex_flags(spec: Any) -> int:
    if spec is None:
        return 0
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        spec = [spec]
    flags = 0
    try:
        for name in spec:
            flags |= _FLAG_NAMES.get(str(name).strip().lower(), 0)
    except TypeError:
        return 0
    return flags


def _score_numeric(output: str, prompt: Dict[str, Any]) -> float:
    args = prompt.get("scoring_args") or {}
    try:
        tol = float(args.get("tol", 1e-6))
    except (TypeError, ValueError):
        tol = 1e-6
    try:
        expected = float(
            str(prompt.get("expected_output", "")).replace(",", "").strip()
        )
    except (TypeError, ValueError):
        return 0.0
    got = _extract_last_number(output)
    if got is None:
        return 0.0
    return 1.0 if abs(got - expected) <= tol else 0.0


def _score_json_valid(output: str, prompt: Dict[str, Any]) -> float:
    args = prompt.get("scoring_args") or {}
    required_keys = args.get("required_keys") or []

    parsed: Any = _try_parse_json(output.strip())
    if parsed is _SENTINEL:
        blob = _find_json_blob(output)
        parsed = _try_parse_json(blob) if blob is not None else _SENTINEL
    if parsed is _SENTINEL:
        return 0.0

    if required_keys:
        if not isinstance(parsed, dict):
            return 0.0
        for key in required_keys:
            if key not in parsed:
                return 0.0
    return 1.0


_SENTINEL = object()


def _try_parse_json(text: str | None) -> Any:
    if not text:
        return _SENTINEL
    try:
        return json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return _SENTINEL


_SCORERS: Dict[str, Callable[[str, Dict[str, Any]], float]] = {
    "exact_match": _score_exact_match,
    "contains": _score_contains,
    "regex": _score_regex,
    "numeric": _score_numeric,
    "json_valid": _score_json_valid,
}

MANUAL_METHODS = frozenset({"manual"})


def score(output: str, prompt: Dict[str, Any]) -> float:
    method = prompt.get("scoring_method")
    if not method:
        raise ValueError("prompt is missing 'scoring_method'")
    scorer = _SCORERS.get(method)
    if scorer is None:
        raise ValueError(
            f"unknown scoring_method {method!r}; " f"known methods: {sorted(_SCORERS)}"
        )
    return scorer(str(output if output is not None else ""), prompt)
