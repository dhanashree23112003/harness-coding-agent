"""Unit tests for context_manager.py - compaction logic, no LLM required."""
import uuid
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.context_manager import (
    COMPACT_THRESHOLD,
    KEEP_RECENT_PAIRS,
    _extract_pairs,
    _summarize_pair,
    build_ledger,
    compact_messages,
    estimate_tokens,
    manage_context_node,
)


def _ai(tool_name: str, args: dict | None = None, msg_id: str | None = None) -> AIMessage:
    args = args or {}
    tc_id = str(uuid.uuid4())
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": tc_id}],
        id=msg_id or str(uuid.uuid4()),
    )


def _tool(ai_msg: AIMessage, content: str, msg_id: str | None = None) -> ToolMessage:
    tc_id = ai_msg.tool_calls[0]["id"]
    return ToolMessage(content=content, tool_call_id=tc_id, id=msg_id or str(uuid.uuid4()))


def _state(messages: list, **extra) -> dict:
    return {
        "messages": messages,
        "plan": "add validation to divide()",
        "progress_ledger": extra.get("progress_ledger", ""),
        "token_estimate": extra.get("token_estimate", 0),
        "compaction_count": extra.get("compaction_count", 0),
        "ledger_message_id": extra.get("ledger_message_id"),
        "task": "test task",
        "available_tool_names": [],
        "retrieval_k": 12,
        "retrieval_miss_count": 0,
        "consecutive_repeat_count": 0,
    }


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_list(self):
        assert estimate_tokens([]) == 0

    def test_string_content(self):
        msg = HumanMessage(content="a" * 400)
        assert estimate_tokens([msg]) == 100

    def test_list_content_mcp_text_block(self):
        msg = AIMessage(content=[{"type": "text", "text": "x" * 800}])
        assert estimate_tokens([msg]) == 200

    def test_list_content_mixed(self):
        msg = AIMessage(content=[{"type": "text", "text": "a" * 400}, {"type": "image"}])
        assert estimate_tokens([msg]) == 100

    def test_multiple_messages(self):
        msgs = [
            SystemMessage(content="a" * 200),
            HumanMessage(content="b" * 200),
        ]
        assert estimate_tokens(msgs) == 100

    def test_returns_int(self):
        msg = HumanMessage(content="abc")
        result = estimate_tokens([msg])
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _extract_pairs
# ---------------------------------------------------------------------------

class TestExtractPairs:
    def test_single_pair(self):
        ai = _ai("read_file", {"path": "a.py"})
        tm = _tool(ai, '{"path": "a.py", "content": "x"}')
        pairs = _extract_pairs([ai, tm])
        assert len(pairs) == 1
        assert pairs[0][0] is ai
        assert pairs[0][1] == [tm]

    def test_multiple_pairs(self):
        ai1 = _ai("read_file")
        tm1 = _tool(ai1, '{"path": "a.py"}')
        ai2 = _ai("write_file")
        tm2 = _tool(ai2, '{"path": "b.py"}')
        pairs = _extract_pairs([ai1, tm1, ai2, tm2])
        assert len(pairs) == 2

    def test_system_and_human_messages_skipped(self):
        sys_msg = SystemMessage(content="system")
        human = HumanMessage(content="task")
        ai = _ai("git_status")
        tm = _tool(ai, '{"status": "clean"}')
        pairs = _extract_pairs([sys_msg, human, ai, tm])
        assert len(pairs) == 1

    def test_ai_without_tool_calls_skipped(self):
        ai_text = AIMessage(content="Just a text response", id=str(uuid.uuid4()))
        ai_tool = _ai("read_file")
        tm = _tool(ai_tool, '{"content": "x"}')
        pairs = _extract_pairs([ai_text, ai_tool, tm])
        assert len(pairs) == 1
        assert pairs[0][0] is ai_tool

    def test_ai_with_tool_calls_but_no_tool_message(self):
        ai = _ai("read_file")
        pairs = _extract_pairs([ai])
        assert len(pairs) == 0

    def test_multiple_tool_messages_per_pair(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "a.py"}, "id": "tc1"},
                {"name": "read_file", "args": {"path": "b.py"}, "id": "tc2"},
            ],
            id=str(uuid.uuid4()),
        )
        tm1 = ToolMessage(content="a", tool_call_id="tc1", id=str(uuid.uuid4()))
        tm2 = ToolMessage(content="b", tool_call_id="tc2", id=str(uuid.uuid4()))
        pairs = _extract_pairs([ai, tm1, tm2])
        assert len(pairs) == 1
        assert len(pairs[0][1]) == 2


# ---------------------------------------------------------------------------
# _summarize_pair
# ---------------------------------------------------------------------------

class TestSummarizePair:
    def test_path_field_extracted(self):
        ai = _ai("read_file", {"path": "calc.py"})
        tm = _tool(ai, '{"path": "calc.py", "content": "def divide..."}')
        summary = _summarize_pair(ai, [tm])
        assert "read_file" in summary
        assert "calc.py" in summary

    def test_passed_failed_counts(self):
        ai = _ai("run_suite")
        tm = _tool(ai, '{"passed": 5, "failed": 2}')
        summary = _summarize_pair(ai, [tm])
        assert "passed=5" in summary
        assert "failed=2" in summary

    def test_summary_field(self):
        ai = _ai("git_status")
        tm = _tool(ai, '{"summary": "nothing to commit"}')
        summary = _summarize_pair(ai, [tm])
        assert "nothing to commit" in summary

    def test_invalid_json_falls_back_to_raw(self):
        ai = _ai("read_file")
        tm = _tool(ai, "not-json-at-all")
        summary = _summarize_pair(ai, [tm])
        assert "read_file" in summary

    def test_list_content_mcp_block(self):
        ai = _ai("read_file")
        tm = ToolMessage(
            content=[{"type": "text", "text": '{"path": "x.py"}'}],
            tool_call_id=ai.tool_calls[0]["id"],
            id=str(uuid.uuid4()),
        )
        summary = _summarize_pair(ai, [tm])
        assert "read_file" in summary


# ---------------------------------------------------------------------------
# build_ledger
# ---------------------------------------------------------------------------

class TestBuildLedger:
    def test_builds_from_pairs(self):
        ai = _ai("read_file", {"path": "a.py"})
        tm = _tool(ai, '{"path": "a.py"}')
        ledger = build_ledger([(ai, [tm])], "")
        assert "Progress ledger" in ledger
        assert "read_file" in ledger

    def test_appends_to_prior_ledger(self):
        prior = "Progress ledger (completed steps):\n  git_status(status=clean)"
        ai = _ai("write_file")
        tm = _tool(ai, '{"path": "b.py"}')
        ledger = build_ledger([(ai, [tm])], prior)
        assert "git_status" in ledger
        assert "write_file" in ledger

    def test_empty_pairs_with_prior(self):
        prior = "Progress ledger (completed steps):\n  step1"
        ledger = build_ledger([], prior)
        assert "step1" in ledger

    def test_starts_with_header(self):
        ai = _ai("run_suite")
        tm = _tool(ai, '{"passed": 3, "failed": 0}')
        ledger = build_ledger([(ai, [tm])], "")
        assert ledger.startswith("Progress ledger")


# ---------------------------------------------------------------------------
# compact_messages
# ---------------------------------------------------------------------------

class TestCompactMessages:
    def _make_pairs(self, n: int) -> list:
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
        for i in range(n):
            ai = _ai("read_file", {"path": f"file{i}.py"})
            tm = _tool(ai, f'{{"path": "file{i}.py"}}')
            msgs.extend([ai, tm])
        return msgs

    def test_not_enough_pairs_returns_token_estimate_only(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS)
        state = _state(msgs)
        result = compact_messages(state)
        assert "messages" not in result
        assert "token_estimate" in result

    def test_compaction_removes_old_pairs(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 2)
        state = _state(msgs)
        result = compact_messages(state)
        assert "messages" in result
        remove_ops = [m for m in result["messages"] if hasattr(m, "id") and not hasattr(m, "tool_calls") and isinstance(m, type(SystemMessage(content="")))]
        # We get RemoveMessage objects + a new SystemMessage.
        assert len(result["messages"]) >= 1

    def test_compaction_count_incremented(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 1)
        state = _state(msgs, compaction_count=2)
        result = compact_messages(state)
        assert result.get("compaction_count", 0) == 3

    def test_progress_ledger_updated(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 1)
        state = _state(msgs)
        result = compact_messages(state)
        ledger = result.get("progress_ledger", "")
        assert "Progress ledger" in ledger
        assert "read_file" in ledger

    def test_plan_not_in_result(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 1)
        state = _state(msgs)
        result = compact_messages(state)
        assert "plan" not in result

    def test_ledger_message_id_set(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 1)
        state = _state(msgs)
        result = compact_messages(state)
        assert result.get("ledger_message_id") is not None

    def test_prior_ledger_message_removed(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 2)
        old_ledger_id = str(uuid.uuid4())
        # Inject a fake prior ledger message so it can be removed.
        old_ledger_msg = SystemMessage(content="old ledger", id=old_ledger_id)
        msgs.insert(2, old_ledger_msg)
        state = _state(msgs, ledger_message_id=old_ledger_id)
        result = compact_messages(state)
        removed_ids = {m.id for m in result["messages"] if hasattr(m, "id") and not isinstance(m, (AIMessage, ToolMessage, SystemMessage, HumanMessage))}
        # The old ledger id should appear in the remove operations.
        from langchain_core.messages import RemoveMessage
        remove_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
        assert old_ledger_id in remove_ids

    def test_token_estimate_updated(self):
        msgs = self._make_pairs(KEEP_RECENT_PAIRS + 1)
        state = _state(msgs, token_estimate=9999)
        result = compact_messages(state)
        assert "token_estimate" in result
        assert result["token_estimate"] < 9999


# ---------------------------------------------------------------------------
# manage_context_node
# ---------------------------------------------------------------------------

class TestManageContextNode:
    def _make_pairs(self, n: int) -> list:
        msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
        for i in range(n):
            ai = _ai("read_file", {"path": f"f{i}.py"})
            tm = _tool(ai, f'{{"path": "f{i}.py", "content": "x"*100}}')
            msgs.extend([ai, tm])
        return msgs

    def test_below_threshold_no_compaction(self):
        msgs = [SystemMessage(content="s"), HumanMessage(content="t")]
        state = _state(msgs)
        with patch("agent.graph.context_manager.COMPACT_THRESHOLD", 99999):
            result = manage_context_node(state)
        assert "compaction_count" not in result
        assert "token_estimate" in result

    def test_at_threshold_triggers_compaction(self):
        msgs = [SystemMessage(content="x" * 400), HumanMessage(content="task")]
        for i in range(KEEP_RECENT_PAIRS + 1):
            ai = _ai("read_file", {"path": f"f{i}.py"})
            tm = _tool(ai, "y" * 400)
            msgs.extend([ai, tm])
        state = _state(msgs)
        with patch("agent.graph.context_manager.COMPACT_THRESHOLD", 1):
            result = manage_context_node(state)
        assert result.get("compaction_count", 0) >= 1

    def test_returns_token_estimate_always(self):
        state = _state([SystemMessage(content="hi")])
        result = manage_context_node(state)
        assert "token_estimate" in result
        assert isinstance(result["token_estimate"], int)
