import asyncio
import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.graph.nodes import make_act_node, plan_node, retrieve_node
from agent.graph.state import AgentState


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def build_graph(tools: list[Any]):
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

    # awrap_tool_call keeps ToolNode as the registered graph node so LangGraph's
    # config injection path is intact.
    tool_node = ToolNode(tools, awrap_tool_call=_log_tool_call)
    act_node = make_act_node(tools)

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("act", act_node)
    g.add_node("tools", tool_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "act")
    g.add_conditional_edges("act", _should_continue, {"tools": "tools", "end": END})
    g.add_edge("tools", "act")

    return g.compile()
