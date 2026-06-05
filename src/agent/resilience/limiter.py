"""Token-bucket rate limiter for outbound model and external calls.

A single default limiter is created at import time from env vars.
Call `await apply_limiter()` at the top of any call site that should be
rate-limited. Configurable via RATE_LIMIT_RPM and RATE_LIMIT_BURST.
"""
from __future__ import annotations

import asyncio
import os
import time


class TokenBucket:
    """Async token bucket: refills at `rate` tokens/sec up to `burst`."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Block until `tokens` tokens are available, then consume them."""
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            deficit = tokens - self._tokens
            wait = deficit / self._rate
            await asyncio.sleep(wait)


def _make_default() -> TokenBucket:
    rpm = float(os.environ.get("RATE_LIMIT_RPM", "30"))
    burst = int(os.environ.get("RATE_LIMIT_BURST", "5"))
    return TokenBucket(rate=rpm / 60.0, burst=burst)


_DEFAULT_LIMITER: TokenBucket = _make_default()


async def apply_limiter(tokens: int = 1) -> None:
    """Acquire from the default limiter before an outbound call."""
    await _DEFAULT_LIMITER.acquire(tokens)
