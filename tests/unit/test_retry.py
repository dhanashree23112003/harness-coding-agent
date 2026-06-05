"""Unit tests for resilience/retry.py - no network calls required."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.errors import RateLimitExceeded, RequestTooLarge, RetryExhausted
from agent.resilience.retry import _is_daily_cap, _is_request_too_large, with_retry

# Real Groq error message strings (from actual traces).
_REQUEST_TOO_LARGE_MSG = (
    "Request too large for model `llama-3.1-8b-instant` in organization `org_test` "
    "on tokens per minute (TPM): Limit 6000, Used 0, Requested ~7000. "
    "Please try again in 1m0s."
)

_DAILY_CAP_MSG = (
    "Rate limit reached for model `llama-3.1-8b-instant` in organization `org_test` "
    "on tokens per day (TPD): Limit 14400, Used 14400, Requested ~1000. "
    "Please try again in 24h0m0s."
)
_TRANSIENT_MSG = (
    "Rate limit reached for model `llama-3.1-8b-instant` on tokens per minute (TPM): "
    "Limit 6000, Used 5800, Requested ~400. Please try again in 3s. "
    "Visit https://console.groq.com/ for more info."
)


# ---------------------------------------------------------------------------
# _is_daily_cap: message-text detection
# ---------------------------------------------------------------------------

class TestIsDailyCap:
    def test_tpd_message_is_daily_cap(self):
        exc = Exception(_DAILY_CAP_MSG)
        assert _is_daily_cap(exc) is True

    def test_tpm_message_is_not_daily_cap(self):
        exc = Exception(_TRANSIENT_MSG)
        assert _is_daily_cap(exc) is False

    def test_24h_in_message_is_daily_cap(self):
        exc = Exception("Please try again in 24h0m0s.")
        assert _is_daily_cap(exc) is True

    def test_tokens_per_day_phrase(self):
        exc = Exception("tokens per day (TPD): Limit 14400")
        assert _is_daily_cap(exc) is True

    def test_empty_message_is_not_daily_cap(self):
        assert _is_daily_cap(Exception("some other error")) is False

    def test_retry_after_over_3600_is_daily_cap(self):
        exc = Exception("Rate limit")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "86400"}
        exc.response = mock_response
        assert _is_daily_cap(exc) is True

    def test_retry_after_under_3600_is_not_daily_cap(self):
        exc = Exception("Rate limit")
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "30"}
        exc.response = mock_response
        assert _is_daily_cap(exc) is False

    def test_retry_after_header_missing_falls_back_to_message(self):
        exc = Exception(_TRANSIENT_MSG)
        mock_response = MagicMock()
        mock_response.headers = {}
        exc.response = mock_response
        assert _is_daily_cap(exc) is False


# ---------------------------------------------------------------------------
# with_retry: retry behavior
# ---------------------------------------------------------------------------

class TestWithRetry:
    async def test_succeeds_on_first_try(self):
        fn = AsyncMock(return_value="ok")
        result = await with_retry(fn, max_attempts=3)
        assert result == "ok"
        fn.assert_awaited_once()

    async def test_retries_transient_error_and_succeeds(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")
            return "success"

        with patch("agent.resilience.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(flaky, max_attempts=3, base_delay=0.0)

        assert result == "success"
        assert call_count == 3

    async def test_raises_retry_exhausted_after_max_attempts(self):
        fn = AsyncMock(side_effect=OSError("always fails"))

        with patch("agent.resilience.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RetryExhausted) as exc_info:
                await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_error, OSError)

    async def test_daily_cap_raises_immediately_no_retries(self):
        call_count = 0

        async def daily_cap_fn():
            nonlocal call_count
            call_count += 1
            exc = OSError(_DAILY_CAP_MSG)
            raise exc

        with pytest.raises(RateLimitExceeded):
            await with_retry(daily_cap_fn, max_attempts=3, base_delay=0.0)

        assert call_count == 1

    async def test_transient_429_is_retried(self):
        call_count = 0

        async def tpm_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError(_TRANSIENT_MSG)
            return "ok"

        with patch("agent.resilience.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(tpm_fn, max_attempts=3, base_delay=0.0)

        assert result == "ok"
        assert call_count == 2

    async def test_daily_cap_via_retry_after_header(self):
        async def fn():
            exc = OSError("some rate limit error")
            mock_response = MagicMock()
            mock_response.headers = {"retry-after": "86400"}
            exc.response = mock_response
            raise exc

        with pytest.raises(RateLimitExceeded):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

    async def test_non_retryable_exception_propagates_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert call_count == 1

    async def test_sleep_called_with_positive_delay(self):
        fn = AsyncMock(side_effect=OSError("fail"))

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("agent.resilience.retry.asyncio.sleep", fake_sleep):
            with pytest.raises(RetryExhausted):
                await with_retry(fn, max_attempts=3, base_delay=1.0, jitter=0.0)

        assert len(sleep_calls) == 2
        assert all(d > 0 for d in sleep_calls)
        assert sleep_calls[1] >= sleep_calls[0]


# ---------------------------------------------------------------------------
# _is_request_too_large: 413 detection
# ---------------------------------------------------------------------------

class TestIsRequestTooLarge:
    def test_real_413_tpm_message_detected(self):
        exc = Exception(_REQUEST_TOO_LARGE_MSG)
        assert _is_request_too_large(exc) is True

    def test_daily_cap_message_not_detected_as_too_large(self):
        exc = Exception(_DAILY_CAP_MSG)
        assert _is_request_too_large(exc) is False

    def test_transient_429_not_detected_as_too_large(self):
        exc = Exception(_TRANSIENT_MSG)
        assert _is_request_too_large(exc) is False

    def test_status_code_413_attribute_detected(self):
        exc = Exception("some api error")
        exc.status_code = 413
        assert _is_request_too_large(exc) is True

    def test_status_code_429_not_detected(self):
        exc = Exception("rate limit")
        exc.status_code = 429
        assert _is_request_too_large(exc) is False

    def test_response_status_code_413_detected(self):
        exc = Exception("api error")
        mock_response = MagicMock()
        mock_response.status_code = 413
        exc.response = mock_response
        assert _is_request_too_large(exc) is True


# ---------------------------------------------------------------------------
# with_retry: 413 behavior
# ---------------------------------------------------------------------------

class TestWithRetry413:
    async def test_request_too_large_raises_immediately_no_retries(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise Exception(_REQUEST_TOO_LARGE_MSG)

        with pytest.raises(RequestTooLarge) as exc_info:
            await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert call_count == 1
        assert exc_info.value.model != ""

    async def test_request_too_large_via_status_code_raises_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            exc = ValueError("api error")
            exc.status_code = 413
            raise exc

        with pytest.raises(RequestTooLarge):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert call_count == 1

    async def test_413_requested_tokens_extracted(self):
        async def fn():
            raise Exception(_REQUEST_TOO_LARGE_MSG)

        with pytest.raises(RequestTooLarge) as exc_info:
            await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert exc_info.value.requested_tokens == 7000

    async def test_413_does_not_exhaust_retries(self):
        """413 exits after 1 call; RetryExhausted is never raised."""
        async def fn():
            raise Exception(_REQUEST_TOO_LARGE_MSG)

        with pytest.raises(RequestTooLarge):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

    async def test_413_error_not_in_retryable_types_still_converted(self):
        """Simulate groq.APIStatusError(413): not in retryable_types() but still
        converted to RequestTooLarge rather than propagating raw."""
        call_count = 0

        class FakeAPIStatusError(Exception):
            status_code = 413

        async def fn():
            nonlocal call_count
            call_count += 1
            raise FakeAPIStatusError("Groq 413")

        with pytest.raises(RequestTooLarge):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

        assert call_count == 1
