from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    task: str
    plan: str
    messages: Annotated[list[BaseMessage], add_messages]
