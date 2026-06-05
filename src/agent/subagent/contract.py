"""Pydantic contracts and typed errors for the subagent system."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubagentBudget(BaseModel):
    max_steps: int = 10
    max_tokens: int = 50_000


class NamespaceScope(BaseModel):
    """Declares which tools from a namespace the subagent may use.

    tools=None means all tools in the namespace are allowed.
    tools=["read_file"] means only read_file from that namespace.
    """
    namespace: str
    tools: list[str] | None = None


class SubagentTask(BaseModel):
    brief: str
    allowed_scopes: list[NamespaceScope]
    budget: SubagentBudget = Field(default_factory=SubagentBudget)


class Finding(BaseModel):
    test_id: str
    status: Literal["failed", "error", "skipped"]
    message: str
    file_path: str | None = None
    line: int | None = None


class SubagentResult(BaseModel):
    status: Literal["completed", "budget_exceeded", "error"]
    findings: list[Finding]
    artifacts: dict[str, str]
    tokens_used: int
    steps_taken: int
    summary: str = ""
    error: str | None = None


class SubagentBudgetExceeded(Exception):
    """Raised when the subagent exhausts its step or token budget."""

    def __init__(
        self,
        steps_taken: int,
        tokens_used: int,
        budget: SubagentBudget,
    ) -> None:
        self.steps_taken = steps_taken
        self.tokens_used = tokens_used
        self.budget = budget
        super().__init__(
            f"subagent budget exceeded: {steps_taken}/{budget.max_steps} steps, "
            f"{tokens_used}/{budget.max_tokens} tokens"
        )


class ToolScopeViolation(Exception):
    """Raised when the subagent attempts to call a tool outside its allowed scope."""

    def __init__(self, tool_name: str, allowed: list[str]) -> None:
        self.tool_name = tool_name
        self.allowed = allowed
        super().__init__(
            f"tool {tool_name!r} is outside the subagent's allowed scope "
            f"(allowed: {sorted(allowed)})"
        )
