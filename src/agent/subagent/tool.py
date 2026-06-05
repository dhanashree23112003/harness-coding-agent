"""Factory for the spawn_subagent LangChain tool exposed to the parent graph."""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from agent.subagent.contract import (
    SubagentBudgetExceeded,
    SubagentResult,
    SubagentTask,
)
from agent.subagent.runner import SubagentRunner


def make_spawn_subagent_tool(runner: SubagentRunner) -> StructuredTool:
    """Return a StructuredTool the parent graph can call to spawn a subagent.

    The tool catches SubagentBudgetExceeded and all other exceptions so the
    parent always receives a typed SubagentResult, never a bare exception.
    """

    async def _spawn(
        brief: str,
        allowed_scopes: list[dict],
        budget: dict | None = None,
    ) -> dict:
        task = SubagentTask.model_validate({
            "brief": brief,
            "allowed_scopes": allowed_scopes,
            **({"budget": budget} if budget else {}),
        })
        try:
            result = await runner.run(task)
        except SubagentBudgetExceeded as exc:
            result = SubagentResult(
                status="budget_exceeded",
                findings=[],
                artifacts={},
                tokens_used=exc.tokens_used,
                steps_taken=exc.steps_taken,
                error=str(exc),
            )
        except Exception as exc:
            result = SubagentResult(
                status="error",
                findings=[],
                artifacts={},
                tokens_used=0,
                steps_taken=0,
                error=str(exc),
            )
        return result.model_dump()

    return StructuredTool.from_function(
        coroutine=_spawn,
        name="spawn_subagent",
        description=(
            "Launch an isolated subagent to perform a bounded sub-task with a scoped toolset "
            "and its own step/token budget. Returns a structured SubagentResult with status, "
            "findings (test failures), artifacts, and token usage. "
            "Use this to triage test failures: set allowed_scopes to "
            '[{"namespace": "test"}, {"namespace": "fs", "tools": ["read_file"]}].'
        ),
        return_direct=False,
    )
