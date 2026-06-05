"""Deterministic proof of the context compaction strategy (SPEC Section 7).

No LLM calls. Constructs an over-threshold message history, runs
manage_context_node with the real default threshold (800 tokens), and asserts:
  - a [compaction] line is printed to stdout
  - stale tool messages (all but the last KEEP_RECENT_PAIRS pairs) are removed
  - plan and progress_ledger survive intact in the returned state
  - token estimate decreases after compaction
"""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from agent.graph.context_manager import (
    COMPACT_THRESHOLD,
    KEEP_RECENT_PAIRS,
    manage_context_node,
)


def _ai(tool_name: str, path: str = "file.py") -> AIMessage:
    tc_id = str(uuid.uuid4())
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"path": path}, "id": tc_id}],
        id=str(uuid.uuid4()),
    )


def _tool(ai_msg: AIMessage, content: str) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=ai_msg.tool_calls[0]["id"],
        id=str(uuid.uuid4()),
    )


def _make_over_threshold_state(n_pairs: int = KEEP_RECENT_PAIRS + 2) -> dict:
    """Build a state whose token estimate exceeds COMPACT_THRESHOLD.

    Each tool response carries ~900 chars of simulated file content so that
    n_pairs pairs sum to well over 3200 chars (= 800 tokens at char/4).
    """
    tool_response_body = "x" * 900

    messages: list = [
        SystemMessage(content="You are a coding agent."),
        HumanMessage(content="Add validation to divide()."),
    ]
    for i in range(n_pairs):
        ai = _ai("read_file", path=f"src/module_{i}.py")
        tm = _tool(ai, f'{{"path": "src/module_{i}.py", "content": "{tool_response_body}"}}')
        messages.extend([ai, tm])

    return {
        "task": "add validation to divide()",
        "plan": "add input validation to divide(), update callers, make suite green",
        "messages": messages,
        "available_tool_names": [],
        "retrieval_k": 12,
        "retrieval_miss_count": 0,
        "consecutive_repeat_count": 0,
        "progress_ledger": "",
        "token_estimate": 0,
        "compaction_count": 0,
        "ledger_message_id": None,
        "correlation_id": "test-cid",
    }


class TestCompactionIntegration:
    def test_state_exceeds_threshold(self):
        """Precondition: our message history actually triggers compaction."""
        from agent.graph.context_manager import estimate_tokens
        state = _make_over_threshold_state()
        tokens = estimate_tokens(state["messages"])
        assert tokens > COMPACT_THRESHOLD, (
            f"test history only has {tokens} tokens; need > {COMPACT_THRESHOLD} "
            "to trigger compaction -- increase n_pairs or tool_response_body size"
        )

    def test_compaction_line_printed(self, capsys):
        """The [compaction] line must appear in stdout so runs are observable."""
        state = _make_over_threshold_state()
        manage_context_node(state)
        captured = capsys.readouterr()
        print(captured.out, end="")  # echo so it shows in pytest -s / -v output
        assert "[compaction]" in captured.out

    def test_stale_tool_messages_are_dropped(self):
        """All but the last KEEP_RECENT_PAIRS pairs must be removed."""
        n_pairs = KEEP_RECENT_PAIRS + 2
        state = _make_over_threshold_state(n_pairs=n_pairs)

        # Collect AI messages in message order (order matters: first n are stale).
        ai_msgs_ordered = [
            msg for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.tool_calls
        ]
        stale_count = n_pairs - KEEP_RECENT_PAIRS
        stale_ai_ids = {msg.id for msg in ai_msgs_ordered[:stale_count]}
        recent_ai_ids = {msg.id for msg in ai_msgs_ordered[stale_count:]}

        result = manage_context_node(state)

        remove_ops = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        removed_ids = {op.id for op in remove_ops}

        # Each stale pair produces at least 2 removes (1 AI + 1 tool message).
        assert len(removed_ids) >= stale_count * 2, (
            f"expected at least {stale_count * 2} RemoveMessage ops "
            f"(1 AI + 1 tool per stale pair), got {len(removed_ids)}"
        )

        # Stale AI message IDs must be in the remove set.
        for sid in stale_ai_ids:
            assert sid in removed_ids, f"stale AIMessage {sid} was not removed"

        # Recent AI message IDs must NOT be removed.
        for rid in recent_ai_ids:
            assert rid not in removed_ids, f"recent AIMessage {rid} was incorrectly removed"

    def test_plan_survives_compaction(self):
        """plan is a state field, not a message; it must not appear in result."""
        state = _make_over_threshold_state()
        result = manage_context_node(state)
        assert "plan" not in result, "compact_messages must not touch the plan field"

    def test_progress_ledger_updated_and_non_empty(self):
        """The compacted pairs must be summarised into progress_ledger."""
        state = _make_over_threshold_state()
        result = manage_context_node(state)
        ledger = result.get("progress_ledger", "")
        assert "Progress ledger" in ledger
        assert "read_file" in ledger

    def test_token_estimate_decreases_after_compaction(self):
        """Dropping verbose messages must reduce the token estimate."""
        from agent.graph.context_manager import estimate_tokens
        state = _make_over_threshold_state()
        before = estimate_tokens(state["messages"])
        result = manage_context_node(state)
        after = result["token_estimate"]
        assert after < before, (
            f"token estimate should decrease after compaction: {before} -> {after}"
        )

    def test_compaction_count_incremented(self):
        state = _make_over_threshold_state()
        result = manage_context_node(state)
        assert result.get("compaction_count", 0) == 1

    def test_ledger_message_injected(self):
        """A new SystemMessage carrying the ledger must be added to messages."""
        state = _make_over_threshold_state()
        result = manage_context_node(state)
        new_msgs = [
            m for m in result["messages"]
            if isinstance(m, SystemMessage)
        ]
        assert len(new_msgs) >= 1
        assert any("Progress ledger" in m.content for m in new_msgs)
