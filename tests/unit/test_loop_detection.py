"""Unit tests for consecutive-repeat loop detection."""
import pytest
from langchain_core.messages import AIMessage

from agent.graph.graph import _should_continue
from agent.graph.nodes import _next_repeat_count


def _tc(name: str, **args: object) -> dict:
    return {"name": name, "args": args, "id": f"id_{name}"}


# ---------------------------------------------------------------------------
# _next_repeat_count: counter update logic (no LLM required)
# ---------------------------------------------------------------------------

class TestNextRepeatCount:
    def test_no_previous_returns_zero(self):
        assert _next_repeat_count([_tc("read_file", path="a.py")], None, 0) == 0

    def test_empty_response_resets_to_zero(self):
        prev = [_tc("read_file", path="a.py")]
        assert _next_repeat_count([], prev, 5) == 0

    def test_different_calls_reset_to_zero(self):
        a = [_tc("read_file", path="a.py")]
        b = [_tc("git_status", cwd=".")]
        # Even with a non-zero incoming count, a change resets to 0.
        assert _next_repeat_count(b, a, 1) == 0

    def test_matching_calls_increment(self):
        calls = [_tc("git_status", cwd=".")]
        assert _next_repeat_count(calls, calls, 0) == 1
        assert _next_repeat_count(calls, calls, 1) == 2

    def test_non_consecutive_repeat_does_not_accumulate(self):
        """read_file -> git_status -> read_file: counter stays 0 throughout.

        The second read_file is compared against git_status (the immediately
        preceding turn), not against the first read_file, so it is NOT a
        consecutive repeat and must not push the count toward the stop threshold.
        """
        read_calls = [_tc("read_file", path="a.py")]
        other_calls = [_tc("git_status", cwd=".")]

        # Turn 1: read_file, no prior AI turn.
        crc = _next_repeat_count(read_calls, None, 0)
        assert crc == 0

        # Turn 2: git_status, prev = read_file (different) -> reset.
        crc = _next_repeat_count(other_calls, read_calls, crc)
        assert crc == 0

        # Turn 3: read_file again, prev = git_status (different) -> reset, NOT increment.
        crc = _next_repeat_count(read_calls, other_calls, crc)
        assert crc == 0  # below the stop threshold of 2

    def test_consecutive_identical_turns_reach_threshold(self):
        """same_calls x4: count reaches 3 (AGENT_LOOP_THRESHOLD default), triggering stop.

        Turn 1 (no prev): crc=0
        Turn 2 (same as 1): crc=1  <- first repeat, still allowed
        Turn 3 (same as 2): crc=2  <- second repeat, still allowed (Slice 6 raised threshold)
        Turn 4 (same as 3): crc=3  <- third consecutive identical turn, stop fires
        """
        from agent.graph.nodes import _LOOP_THRESHOLD

        same_calls = [_tc("git_status", cwd=".")]

        crc = _next_repeat_count(same_calls, None, 0)
        assert crc == 0

        crc = _next_repeat_count(same_calls, same_calls, crc)
        assert crc == 1
        assert crc < _LOOP_THRESHOLD

        crc = _next_repeat_count(same_calls, same_calls, crc)
        assert crc == 2
        assert crc < _LOOP_THRESHOLD

        crc = _next_repeat_count(same_calls, same_calls, crc)
        assert crc == 3
        assert crc >= _LOOP_THRESHOLD  # stop threshold reached


# ---------------------------------------------------------------------------
# _should_continue: routing based on consecutive_repeat_count
# ---------------------------------------------------------------------------

class TestShouldContinueLoopDetection:
    def _state(self, crc: int, tool_calls: list[dict]) -> dict:
        ai_msg = AIMessage(content="", tool_calls=tool_calls)
        return {
            "messages": [ai_msg],
            "available_tool_names": [],
            "consecutive_repeat_count": crc,
            "retrieval_miss_count": 0,
        }

    def test_stops_at_crc_three(self):
        """Threshold is now 3 (AGENT_LOOP_THRESHOLD default), raised from 2 for Slice 6."""
        state = self._state(3, [_tc("read_file", path="a.py")])
        assert _should_continue(state) == "end"

    def test_stops_at_crc_above_threshold(self):
        state = self._state(5, [_tc("read_file", path="a.py")])
        assert _should_continue(state) == "end"

    def test_continues_at_crc_two(self):
        """Second repeat still allowed: agent can retry a failing test run once more."""
        state = self._state(2, [_tc("read_file", path="a.py")])
        assert _should_continue(state) == "tools"

    def test_continues_at_crc_one(self):
        """First repeat: allow the agent one more turn."""
        state = self._state(1, [_tc("read_file", path="a.py")])
        assert _should_continue(state) == "tools"

    def test_continues_at_crc_zero(self):
        state = self._state(0, [_tc("read_file", path="a.py")])
        assert _should_continue(state) == "tools"

    def test_no_tool_calls_ends_unconditionally(self):
        """A text-only response ends regardless of the counter."""
        ai_msg = AIMessage(content="Done.", tool_calls=[])
        state = {
            "messages": [ai_msg],
            "available_tool_names": [],
            "consecutive_repeat_count": 0,
            "retrieval_miss_count": 0,
        }
        assert _should_continue(state) == "end"
