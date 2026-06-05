import json
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_groq import ChatGroq

from agent.graph.state import AgentState
from agent.retrieval.retriever import _DEFAULT_K, _K_WIDEN_STEP, ToolRetriever

_MISS_CAP = 3
_LOOP_THRESHOLD: int = int(os.environ.get("AGENT_LOOP_THRESHOLD", "3"))


def _next_repeat_count(
    response_calls: list[dict],
    prev_calls: list[dict] | None,
    current_count: int,
) -> int:
    """Return the updated consecutive_repeat_count.

    Increments when response_calls exactly matches prev_calls (same tool names
    and args, order-independent). Resets to 0 on any difference or when
    response_calls is empty.
    """
    def _fps(calls: list[dict] | None) -> frozenset[str]:
        return frozenset(
            c["name"] + "|" + json.dumps(c["args"], sort_keys=True)
            for c in (calls or [])
        )

    curr = _fps(response_calls)
    if not curr:
        return 0
    return current_count + 1 if curr == _fps(prev_calls) else 0


def plan_node(state: AgentState) -> dict:
    """Slice 1: trivially record the task as the plan."""
    return {"plan": state["task"]}


def widen_node(state: AgentState) -> dict:
    """Increment retrieval_k and retrieval_miss_count before re-retrieving.

    Removes the orphaned AIMessage (tool calls with no ToolMessage responses)
    so act_node re-enters with a clean conversation history and generates
    fresh, correctly-formatted tool calls. Stubs caused Groq to produce
    malformed follow-up calls when the tool name appeared in the stub text.

    Resets consecutive_repeat_count to 0: after a widen the model operates in
    a different tool context, so prior repeat counts no longer apply.
    """
    to_remove: list[RemoveMessage] = []
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            to_remove.append(RemoveMessage(id=last.id))

    return {
        "retrieval_k": state.get("retrieval_k", _DEFAULT_K) + _K_WIDEN_STEP,
        "retrieval_miss_count": state.get("retrieval_miss_count", 0) + 1,
        "consecutive_repeat_count": 0,
        "messages": to_remove,
    }


def make_retrieve_node(retriever: ToolRetriever):
    """Return a retrieve_node closed over the ToolRetriever instance."""

    async def retrieve_node(state: AgentState) -> dict:
        k = state.get("retrieval_k", _DEFAULT_K)
        names = await retriever.retrieve(_goal(state), k)
        print(
            f"[retrieve] k={k}, retrieved {len(names)} tools: {sorted(names)}",
            flush=True,
        )
        return {"available_tool_names": names}

    return retrieve_node


def _goal(state: AgentState) -> str:
    """Build a goal string from plan + the most recent human message."""
    plan = state.get("plan", "")
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return f"{plan} {msg.content}".strip()
    return plan


def make_act_node(tools: list[Any], retriever: ToolRetriever | None = None):
    """Return an act node that binds the retrieved tool subset to the LLM.

    Falls back to all tools when the available set is empty or when
    retrieval_miss_count reaches _MISS_CAP (prevents infinite miss loops).

    Updates consecutive_repeat_count in state: increments when the model
    produces the same tool calls as its immediately preceding turn, resets
    otherwise. _should_continue reads this counter to detect stuck loops.
    """
    llm = ChatGroq(model=os.environ.get("AGENT_MODEL", "llama-3.1-8b-instant"))
    tool_by_name = {t.name: t for t in tools}

    def act_node(state: AgentState) -> dict:
        available = state.get("available_tool_names", [])
        miss_count = state.get("retrieval_miss_count", 0)

        if miss_count >= _MISS_CAP or not available:
            subset = tools
            if miss_count >= _MISS_CAP:
                print(f"[act] MISS CAP reached: binding all {len(tools)} tools", flush=True)
        else:
            subset = [tool_by_name[n] for n in available if n in tool_by_name]

        bound = llm.bind_tools(subset)
        response = bound.invoke(state["messages"])

        # Find the immediately preceding AIMessage with tool_calls to compare.
        prev_calls: list[dict] | None = None
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                prev_calls = msg.tool_calls
                break

        new_crc = _next_repeat_count(
            response.tool_calls or [],
            prev_calls,
            state.get("consecutive_repeat_count", 0),
        )

        return {"messages": [response], "consecutive_repeat_count": new_crc}

    return act_node
