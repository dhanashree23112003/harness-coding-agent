import asyncio
import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.context_manager import manage_context_node
from agent.graph.nodes import _LOOP_THRESHOLD, make_act_node, make_retrieve_node, plan_node, widen_node
from agent.graph.state import AgentState
from agent.retrieval.retriever import ToolRetriever


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return "end"

    available = set(state.get("available_tool_names", []))
    if available:
        for call in last.tool_calls:
            if call["name"] not in available:
                print(
                    f"[graph] RETRIEVAL MISS: model called '{call['name']}' "
                    f"which was not in the retrieved subset of {len(available)} tools",
                    flush=True,
                )
                return "miss"

    # Loop detection: stop only after 2 consecutive identical tool-calling turns.
    # act_node increments consecutive_repeat_count when its output matches the
    # immediately preceding AIMessage; widen_node resets it to 0. A single
    # non-consecutive repeat (read_file, other calls, read_file) stays at 0
    # and never triggers this stop.
    if state.get("consecutive_repeat_count", 0) >= _LOOP_THRESHOLD:
        print(
            "[graph] LOOP DETECTED: 2 consecutive identical tool-calling turns. Stopping.",
            flush=True,
        )
        return "end"

    return "tools"


def build_graph(tools: list[Any], retriever: ToolRetriever):
    # MCP stdio sessions are not multiplexed: concurrent requests to the same
    # server corrupt the stream. Serialize all calls through a per-graph lock.
    # Lock is created here (inside async context) so it binds to the running loop.
    tool_lock = asyncio.Lock()

    async def _log_tool_call(request, execute):
        """awrap_tool_call: serialize + log each tool call with timing."""
        async with tool_lock:
            t0 = time.perf_counter()
            args_preview = json.dumps(request.tool_call.get("args", {}))[:120]
            print(f"[tool] CALL  {request.tool_call['name']}  {args_preview}", flush=True)
            result = await execute(request)
            print(f"[tool] DONE  ({time.perf_counter() - t0:.2f}s)", flush=True)
            return result

    tool_node = ToolNode(tools, awrap_tool_call=_log_tool_call)
    retrieve_node = make_retrieve_node(retriever)
    act_node = make_act_node(tools, retriever)

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("widen", widen_node)
    g.add_node("act", act_node)
    g.add_node("tools", tool_node)
    g.add_node("manage_context", manage_context_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "act")
    g.add_conditional_edges(
        "act",
        _should_continue,
        {"tools": "tools", "miss": "widen", "end": END},
    )
    g.add_edge("widen", "retrieve")
    g.add_edge("tools", "manage_context")
    g.add_edge("manage_context", "act")

    return g.compile()
