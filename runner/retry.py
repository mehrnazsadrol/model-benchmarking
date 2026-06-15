
from __future__ import annotations

from typing import Any, Callable, Optional

_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

_PERMANENT_STATUSES = frozenset({400, 401, 402, 403, 404, 405, 410})

_TRANSIENT_MESSAGE_HINTS = (
    "currently loading",
    "model is loading",
    "is currently loading",
    "loading the model",
)


def _status_code_of(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _is_timeout_or_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    transient_names = {
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
        "ProtocolError",
        "ChunkedEncodingError",
    }
    return bool(names & transient_names)


def is_transient_error(exc: BaseException) -> bool:
    code = _status_code_of(exc)
    if code is not None:
        if code in _TRANSIENT_STATUSES:
            return True
        if code in _PERMANENT_STATUSES:
            return False
        if 500 <= code <= 599:
            return True
        return False

    if _is_timeout_or_connection_error(exc):
        return True

    message = str(exc).lower()
    if any(hint in message for hint in _TRANSIENT_MESSAGE_HINTS):
        return True

    return False


def safe_run(
    fn: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    wait_multiplier: float = 2.0,
    wait_min: float = 2.0,
    wait_max: float = 20.0,
    **kwargs: Any,
) -> Any:
    if max_attempts <= 1:
        return fn(*args, **kwargs)

    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        retry=retry_if_exception(is_transient_error),
        wait=wait_exponential(multiplier=wait_multiplier, min=wait_min, max=wait_max),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
    def _attempt() -> Any:
        return fn(*args, **kwargs)

    return _attempt()
