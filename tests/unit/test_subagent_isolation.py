"""Unit tests for subagent isolation: scope enforcement, context isolation,
typed result, and budget exhaustion. No external API calls required."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.subagent.contract import (
    Finding,
    NamespaceScope,
    SubagentBudget,
    SubagentBudgetExceeded,
    SubagentResult,
    SubagentTask,
    ToolScopeViolation,
)
from agent.subagent.runner import SubagentRunner


def _make_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = f"mock tool {name}"
    return t


def _make_tools_by_namespace() -> dict[str, list]:
    return {
        "fs": [_make_tool("read_file"), _make_tool("write_file"), _make_tool("list_dir")],
        "git": [_make_tool("git_status"), _make_tool("git_diff"), _make_tool("commit")],
        "test": [
            _make_tool("run_suite"),
            _make_tool("run_test_file"),
            _make_tool("last_failures"),
            _make_tool("discover_tests"),
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: scope enforcement
# ---------------------------------------------------------------------------

class TestScopeEnforcement:
    def test_namespace_only_scope_includes_all_tools_in_that_namespace(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        scopes = [NamespaceScope(namespace="test")]
        tools = runner._scope_tools(scopes)
        names = {t.name for t in tools}
        assert names == {"run_suite", "run_test_file", "last_failures", "discover_tests"}

    def test_scoped_to_test_excludes_fs_and_git_tools(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        scopes = [NamespaceScope(namespace="test")]
        tools = runner._scope_tools(scopes)
        names = {t.name for t in tools}
        assert "read_file" not in names
        assert "write_file" not in names
        assert "git_status" not in names
        assert "commit" not in names

    def test_partial_fs_scope_includes_only_named_tool(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        scopes = [
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ]
        tools = runner._scope_tools(scopes)
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "write_file" not in names
        assert "list_dir" not in names

    def test_unknown_namespace_raises_value_error(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        scopes = [NamespaceScope(namespace="nonexistent")]
        with pytest.raises(ValueError, match="nonexistent"):
            runner._scope_tools(scopes)

    def test_unknown_tool_name_in_namespace_raises_value_error(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        scopes = [NamespaceScope(namespace="fs", tools=["delete_everything"])]
        with pytest.raises(ValueError, match="delete_everything"):
            runner._scope_tools(scopes)


# ---------------------------------------------------------------------------
# Test 2: typed SubagentResult return
# ---------------------------------------------------------------------------

class TestTypedResultReturn:
    async def test_run_returns_subagent_result_instance(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        task = SubagentTask(
            brief="Identify failing tests.",
            allowed_scopes=[NamespaceScope(namespace="test")],
            budget=SubagentBudget(max_steps=5, max_tokens=10_000),
        )
        final_ai = AIMessage(content="No failures found.", tool_calls=[])
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(return_value=final_ai)

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            result = await runner.run(task)

        assert isinstance(result, SubagentResult)

    async def test_result_has_all_required_fields(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        task = SubagentTask(
            brief="Run tests.",
            allowed_scopes=[NamespaceScope(namespace="test")],
        )
        final_ai = AIMessage(content="Done.", tool_calls=[])
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(return_value=final_ai)

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            result = await runner.run(task)

        assert result.status == "completed"
        assert isinstance(result.findings, list)
        assert isinstance(result.artifacts, dict)
        assert isinstance(result.tokens_used, int)
        assert isinstance(result.steps_taken, int)


# ---------------------------------------------------------------------------
# Test 3: context isolation
# ---------------------------------------------------------------------------

class TestContextIsolation:
    async def test_subagent_messages_start_with_only_system_and_brief(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        brief = "Run affected tests and report failures."
        task = SubagentTask(
            brief=brief,
            allowed_scopes=[NamespaceScope(namespace="test")],
        )
        captured_messages: list = []

        async def capture_and_return(messages):
            captured_messages.extend(messages)
            return AIMessage(content="done", tool_calls=[])

        mock_bound = MagicMock()
        mock_bound.ainvoke = capture_and_return

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            await runner.run(task)

        assert len(captured_messages) == 2, (
            f"Expected exactly 2 initial messages (system + human), "
            f"got {len(captured_messages)}: {captured_messages}"
        )
        assert isinstance(captured_messages[0], SystemMessage)
        assert isinstance(captured_messages[1], HumanMessage)
        assert captured_messages[1].content == brief

    async def test_subagent_messages_contain_no_parent_history(self):
        """Parent messages must never leak into the subagent's context."""
        runner = SubagentRunner(_make_tools_by_namespace())
        task = SubagentTask(
            brief="Triage failures.",
            allowed_scopes=[NamespaceScope(namespace="test")],
        )
        parent_message_text = "PARENT_SECRET_CONTEXT"
        captured: list = []

        async def capture(messages):
            captured.extend(messages)
            return AIMessage(content="ok", tool_calls=[])

        mock_bound = MagicMock()
        mock_bound.ainvoke = capture

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            await runner.run(task)

        for msg in captured:
            content = msg.content if hasattr(msg, "content") else str(msg)
            assert parent_message_text not in (content or ""), (
                "Parent message text leaked into subagent context"
            )


# ---------------------------------------------------------------------------
# Test 4: budget exhaustion
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    async def test_budget_exceeded_when_max_steps_reached(self):
        runner = SubagentRunner(_make_tools_by_namespace())
        task = SubagentTask(
            brief="Run forever.",
            allowed_scopes=[NamespaceScope(namespace="test")],
            budget=SubagentBudget(max_steps=1, max_tokens=100_000),
        )
        # LLM always returns a tool call so the loop would run forever.
        tc_msg = AIMessage(
            content="",
            tool_calls=[{"name": "run_suite", "args": {"cwd": "."}, "id": "tc1"}],
        )
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(return_value=tc_msg)

        # Mock the tool itself so arun doesn't actually run pytest.
        run_suite_mock = _make_tool("run_suite")
        run_suite_mock.arun = AsyncMock(return_value='{"passed": 0, "failed": 0, "results": []}')
        tools_by_ns = {
            "test": [run_suite_mock],
        }
        runner2 = SubagentRunner(tools_by_ns)

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            with pytest.raises(SubagentBudgetExceeded) as exc_info:
                await runner2.run(task)

        exc = exc_info.value
        assert exc.steps_taken >= 1
        assert exc.budget.max_steps == 1

    async def test_spawn_tool_converts_budget_exceeded_to_result(self):
        """spawn_subagent tool must catch SubagentBudgetExceeded and return
        a SubagentResult with status='budget_exceeded', never raise."""
        from agent.subagent.tool import make_spawn_subagent_tool

        run_suite_mock = _make_tool("run_suite")
        run_suite_mock.arun = AsyncMock(return_value='{"passed": 0, "failed": 0, "results": []}')
        runner = SubagentRunner({"test": [run_suite_mock]})

        spawn = make_spawn_subagent_tool(runner)

        tc_msg = AIMessage(
            content="",
            tool_calls=[{"name": "run_suite", "args": {"cwd": "."}, "id": "tc1"}],
        )
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(return_value=tc_msg)

        with patch("agent.subagent.runner.ChatGroq") as mock_groq:
            mock_groq.return_value.bind_tools.return_value = mock_bound
            result_dict = await spawn.arun({
                "brief": "run forever",
                "allowed_scopes": [{"namespace": "test"}],
                "budget": {"max_steps": 1, "max_tokens": 100_000},
            })

        result = SubagentResult.model_validate(result_dict)
        assert result.status == "budget_exceeded"
        assert result.error is not None
