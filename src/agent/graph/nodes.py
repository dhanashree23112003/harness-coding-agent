from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from agent.graph.state import AgentState


def plan_node(state: AgentState) -> dict:
    """Slice 1: trivially record the task as the plan."""
    return {"plan": state["task"]}


def retrieve_node(state: AgentState) -> dict:
    """Slice 1: stub. Real pgvector retrieval arrives in Slice 3."""
    return {}


def make_act_node(tools: list[Any]):
    """Return an act node that binds the given tools to the LLM."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
    llm_with_tools = llm.bind_tools(tools)

    def act_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not messages:
            messages = [HumanMessage(content=state["task"])]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return act_node
