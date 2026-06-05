"""Integration test: parent spawns the test-triage subagent against a fixture
repo with a deliberately failing test and consumes the structured result.

Requires GROQ_API_KEY and running MCP servers. Skipped automatically when the
key is absent so `make test` (unit only) always passes without credentials.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from agent.subagent.contract import (
    NamespaceScope,
    SubagentBudget,
    SubagentResult,
    SubagentTask,
)
from agent.subagent.runner import SubagentRunner

pytestmark = pytest.mark.integration

_NEEDS_KEY = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)


@pytest.fixture
def failing_suite_repo(tmp_path):
    """Minimal Python package with one passing and one deliberately failing test."""
    (tmp_path / "main.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )
    (tmp_path / "test_main.py").write_text(
        "from main import add\n\n"
        "def test_add_pass():\n"
        "    assert add(1, 2) == 3\n\n"
        "def test_add_fail():\n"
        "    assert add(1, 2) == 99  # deliberate failure\n"
    )
    return tmp_path


@_NEEDS_KEY
async def test_subagent_spawn_returns_structured_result(failing_suite_repo):
    """SubagentRunner.run() returns a typed SubagentResult (no external graph)."""
    from agent.mcp_client.client import mcp_tools_session_with_namespaces

    async with mcp_tools_session_with_namespaces() as (_, tools_by_ns):
        runner = SubagentRunner(all_tools_by_namespace=tools_by_ns)
        task = SubagentTask(
            brief=(
                f"Run all tests in the directory {failing_suite_repo}. "
                "Identify all failing tests and report them. "
                "Use the cwd parameter set to the directory path when running tests."
            ),
            allowed_scopes=[
                NamespaceScope(namespace="test"),
                NamespaceScope(namespace="fs", tools=["read_file"]),
            ],
            budget=SubagentBudget(max_steps=15, max_tokens=30_000),
        )
        result = await asyncio.wait_for(runner.run(task), timeout=120)

    assert isinstance(result, SubagentResult), (
        f"Expected SubagentResult, got {type(result)}"
    )
    assert result.status == "completed", (
        f"Expected status='completed', got {result.status!r}. error={result.error}"
    )
    assert result.steps_taken >= 1
    assert isinstance(result.findings, list)
    assert isinstance(result.artifacts, dict)

    # The fixture has exactly one failing test: test_add_fail.
    failing = [f for f in result.findings if f.status == "failed"]
    assert len(failing) >= 1, (
        f"Expected at least one failed finding, got findings={result.findings}"
    )
    failing_ids = [f.test_id for f in failing]
    assert any("test_add_fail" in tid for tid in failing_ids), (
        f"Expected 'test_add_fail' in a failing finding, got: {failing_ids}"
    )


@_NEEDS_KEY
async def test_subagent_scope_prevents_write_file_calls(failing_suite_repo):
    """The subagent scoped to test+fs.read_file must not have write_file available."""
    from agent.mcp_client.client import mcp_tools_session_with_namespaces

    async with mcp_tools_session_with_namespaces() as (_, tools_by_ns):
        runner = SubagentRunner(all_tools_by_namespace=tools_by_ns)
        scoped = runner._scope_tools([
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ])

    scoped_names = {t.name for t in scoped}
    assert "write_file" not in scoped_names
    assert "delete" not in scoped_names
    assert "git_commit" not in scoped_names
    assert "read_file" in scoped_names
    assert "run_suite" in scoped_names
