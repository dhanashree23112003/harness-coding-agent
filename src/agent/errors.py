"""Typed error hierarchy for the agent. All public errors inherit AgentError.

No bare exceptions should cross module boundaries; callers catch AgentError
or its specific subclasses.
"""
from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent errors."""


class RetryExhausted(AgentError):
    """All retry attempts failed on a transient error."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"retry exhausted after {attempts} attempt(s): {last_error}"
        )


class RateLimitExceeded(AgentError):
    """Non-retryable 429: daily token cap reached."""

    def __init__(self, model: str, retry_after: int | None = None) -> None:
        self.model = model
        self.retry_after = retry_after
        ra = f" retry-after={retry_after}s" if retry_after is not None else ""
        super().__init__(f"daily token cap reached for model={model}{ra}")


class RequestTooLarge(AgentError):
    """Non-retryable 413: single request exceeds the model's per-request token limit.

    Retrying sends the same oversized request and will fail identically.
    The fix is to compact the context before the next call.
    """

    def __init__(self, model: str, requested_tokens: int | None = None) -> None:
        self.model = model
        self.requested_tokens = requested_tokens
        rt = f" requested={requested_tokens} tokens" if requested_tokens is not None else ""
        super().__init__(f"request too large for model={model}{rt}")


class ToolError(AgentError):
    """A tool call failed with a typed error the model can reason about."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"tool {tool_name!r} failed: {message}")


class RetrievalMiss(AgentError):
    """Retriever returned nothing useful for the current goal."""

    def __init__(self, goal: str, k: int) -> None:
        self.goal = goal
        self.k = k
        super().__init__(f"retrieval miss at k={k} for goal: {goal[:80]!r}")


class ValidationError(AgentError):
    """Pydantic or schema validation failure at a module boundary."""
