"""Exponential backoff retry for external calls (Groq model, MCP transport).

Daily-cap 429s are distinguished from transient 429s using two signals:
  1. Message text: "tokens per day" / "TPD" / "24h" appear only in Groq's
     TPD (daily cap) error, never in TPM (transient per-minute) errors.
  2. Retry-After header magnitude: daily cap sets it to ~86400s; transient
     sets it to 3-60s. A value > 3600 is treated as non-retryable.

413 Request Too Large is detected before the retryable-type filter because
groq.APIStatusError (the exception Groq raises for 413) is not in the
retryable set and would propagate raw without this early check.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent.errors import AgentError, RateLimitExceeded, RequestTooLarge, RetryExhausted

T = TypeVar("T")

_RETRY_MAX_ATTEMPTS: int = int(os.environ.get("RETRY_MAX_ATTEMPTS", "3"))
_RETRY_BASE_DELAY: float = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))
_RETRY_MAX_DELAY: float = 60.0
_RETRY_JITTER: float = 0.25

_DAILY_CAP_RE = re.compile(r"tokens per day|TPD|\b24h", re.IGNORECASE)

_RETRYABLE_TYPES: tuple[type[Exception], ...] = ()


def _retryable_types() -> tuple[type[Exception], ...]:
    """Return the tuple of retryable exception types, importing lazily."""
    global _RETRYABLE_TYPES
    if _RETRYABLE_TYPES:
        return _RETRYABLE_TYPES
    types: list[type[Exception]] = [OSError, ConnectionError, TimeoutError]
    try:
        import groq
        types += [groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError]
    except ImportError:
        pass
    _RETRYABLE_TYPES = tuple(types)
    return _RETRYABLE_TYPES


def _is_request_too_large(exc: Exception) -> bool:
    """Return True for a 413 Request Too Large from Groq.

    Checked before the retryable-type filter: groq.APIStatusError(413) is
    not in _retryable_types() and would propagate raw without this guard.
    Retrying a 413 is pointless; the same oversized request will fail again.
    """
    if getattr(exc, "status_code", None) == 413:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 413:
        return True
    return "request too large" in str(exc).lower()


def _is_daily_cap(exc: Exception) -> bool:
    """Return True when the 429 is a non-retryable daily token-cap error."""
    if _DAILY_CAP_RE.search(str(exc)):
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            ra = response.headers.get("retry-after", "0")
            if int(ra) > 3600:
                return True
        except (ValueError, TypeError, AttributeError):
            pass
    return False


def _model_from_exc(exc: Exception) -> str:
    """Best-effort extraction of the model name from a Groq error."""
    msg = str(exc)
    m = re.search(r"model[` ]+([^\s`'\"]+)", msg)
    return m.group(1) if m else "unknown"


def _requested_tokens_from_exc(exc: Exception) -> int | None:
    """Extract the 'Requested ~N' token count from a Groq error message."""
    m = re.search(r"[Rr]equested\s*[~]?(\d+)", str(exc))
    return int(m.group(1)) if m else None


def _retry_after_seconds(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return int(response.headers.get("retry-after", ""))
        except (ValueError, TypeError, AttributeError):
            pass
    return None


async def with_retry(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = _RETRY_MAX_ATTEMPTS,
    base_delay: float = _RETRY_BASE_DELAY,
    max_delay: float = _RETRY_MAX_DELAY,
    jitter: float = _RETRY_JITTER,
) -> T:
    """Call coro_fn(), retrying on transient errors with exponential backoff.

    Non-retryable exits (in order of check):
      1. RequestTooLarge (413): raised immediately, before retryable-type filter.
      2. RateLimitExceeded (daily-cap 429): raised immediately.
      3. AgentError subclasses: always re-raised.
      4. Any exception not in _retryable_types(): propagated immediately.

    Raises RetryExhausted after max_attempts transient failures.
    """
    last_exc: Exception | None = None
    retryable = _retryable_types()

    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except AgentError:
            raise
        except Exception as exc:
            # 413 check comes BEFORE the retryable-type filter because
            # groq.APIStatusError(413) is not in retryable and would otherwise
            # propagate raw without being converted to a typed error.
            if _is_request_too_large(exc):
                raise RequestTooLarge(
                    model=_model_from_exc(exc),
                    requested_tokens=_requested_tokens_from_exc(exc),
                ) from exc

            if not isinstance(exc, retryable):
                raise

            if _is_daily_cap(exc):
                raise RateLimitExceeded(
                    model=_model_from_exc(exc),
                    retry_after=_retry_after_seconds(exc),
                ) from exc

            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, jitter), max_delay)
                await asyncio.sleep(delay)

    raise RetryExhausted(attempts=max_attempts, last_error=last_exc)  # type: ignore[arg-type]
