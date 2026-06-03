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
    tool_node = ToolNode(tools)
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
