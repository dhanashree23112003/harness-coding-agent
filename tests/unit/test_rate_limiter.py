"""Unit tests for resilience/limiter.py - no I/O required."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from agent.resilience.limiter import TokenBucket


class TestTokenBucket:
    def test_bucket_starts_full(self):
        bucket = TokenBucket(rate=1.0, burst=5)
        assert bucket._tokens == 5.0

    async def test_acquire_from_full_bucket_does_not_sleep(self):
        bucket = TokenBucket(rate=1.0, burst=5)
        sleep_calls: list[float] = []

        async def fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        with patch("agent.resilience.limiter.asyncio.sleep", fake_sleep):
            await bucket.acquire(1)

        assert sleep_calls == []

    async def test_acquire_all_burst_tokens_no_sleep(self):
        bucket = TokenBucket(rate=1.0, burst=3)
        sleep_calls: list[float] = []

        async def fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        with patch("agent.resilience.limiter.asyncio.sleep", fake_sleep):
            await bucket.acquire(1)
            await bucket.acquire(1)
            await bucket.acquire(1)

        assert sleep_calls == []
        assert bucket._tokens < 1.0

    async def test_bucket_exhaustion_causes_sleep(self):
        bucket = TokenBucket(rate=1.0, burst=2)
        sleep_calls: list[float] = []

        call_count = 0

        async def fake_sleep(d: float) -> None:
            sleep_calls.append(d)
            bucket._tokens += d * bucket._rate
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise RuntimeError("infinite loop guard")

        with patch("agent.resilience.limiter.asyncio.sleep", fake_sleep):
            await bucket.acquire(1)
            await bucket.acquire(1)
            await bucket.acquire(1)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] > 0

    async def test_refill_over_time(self):
        bucket = TokenBucket(rate=10.0, burst=10)
        bucket._tokens = 0.0
        bucket._last_refill = time.monotonic() - 1.0
        bucket._refill()
        assert bucket._tokens == pytest.approx(10.0, abs=0.5)

    async def test_refill_does_not_exceed_burst(self):
        bucket = TokenBucket(rate=100.0, burst=5)
        bucket._tokens = 0.0
        bucket._last_refill = time.monotonic() - 10.0
        bucket._refill()
        assert bucket._tokens == 5.0

    async def test_rate_zero_tokens_per_second_sleeps(self):
        bucket = TokenBucket(rate=0.5, burst=1)
        bucket._tokens = 0.0

        sleep_calls: list[float] = []
        call_count = 0

        async def fake_sleep(d: float) -> None:
            sleep_calls.append(d)
            bucket._tokens += d * bucket._rate
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise RuntimeError("infinite loop guard")

        with patch("agent.resilience.limiter.asyncio.sleep", fake_sleep):
            await bucket.acquire(1)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] > 0
