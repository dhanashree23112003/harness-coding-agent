from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    task: str
    plan: str
    messages: Annotated[list[BaseMessage], add_messages]
    available_tool_names: list[str]   # populated by retrieve_node each step
    retrieval_k: int                  # current top-k; widens by 6 on each miss
    retrieval_miss_count: int         # number of miss-guard triggers this session
    consecutive_repeat_count: int     # acts increments on back-to-back identical calls; widen resets to 0
