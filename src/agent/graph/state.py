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
    progress_ledger: str              # accumulated compaction summaries; survives message drops
    token_estimate: int               # updated by manage_context_node each step
    compaction_count: int             # how many times compaction has fired
    ledger_message_id: str | None     # id of the injected ledger SystemMessage; None before first compaction
