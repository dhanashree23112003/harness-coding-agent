"""Slice 1 entry point: proves the spine end to end with fs.read_file."""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.graph import build_graph
from agent.mcp_client.client import get_mcp_tools

load_dotenv()

_DEMO_FILE = Path(__file__).resolve().parents[2] / "SPEC.md"


def _print_trace(messages: list) -> None:
    print("\n" + "=" * 60)
    print("TRACE")
    print("=" * 60)
    for i, msg in enumerate(messages):
        tag = f"[{i}]"
        if isinstance(msg, SystemMessage):
            print(f"\n{tag} SystemMessage")
            print(f"  content: {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"\n{tag} HumanMessage")
            print(f"  content: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"\n{tag} AIMessage  (tool call)")
                for tc in msg.tool_calls:
                    print(f"  tool:  {tc['name']}")
                    print(f"  args:  {json.dumps(tc['args'], indent=4)}")
                if msg.content:
                    print(f"  text:  {msg.content}")
            else:
                print(f"\n{tag} AIMessage  (final answer)")
                print(f"  content: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"\n{tag} ToolMessage  (raw result for tool_call_id={msg.tool_call_id})")
            try:
                parsed = json.loads(msg.content)
                # truncate content field so the trace stays readable
                if "content" in parsed:
                    parsed["content"] = parsed["content"][:120] + " [truncated]"
                print(f"  result: {json.dumps(parsed, indent=4)}")
            except (json.JSONDecodeError, TypeError):
                print(f"  result: {str(msg.content)[:300]}")
        else:
            print(f"\n{tag} {type(msg).__name__}: {str(msg)[:200]}")
    print("\n" + "=" * 60)


_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a precise coding agent. "
    "Execute the user's task exactly as stated, no more and no less. "
    "Do not summarize, explain, or expand beyond what is explicitly asked. "
    "When the task is complete, stop immediately."
))


async def run(task: str) -> str:
    tools = await get_mcp_tools()
    print(f"[agent] discovered tools: {[t.name for t in tools]}")
    graph = build_graph(tools)
    init_messages = [_SYSTEM_PROMPT, HumanMessage(content=task)]
    result = await graph.ainvoke({"task": task, "plan": "", "messages": init_messages})
    _print_trace(result["messages"])
    return result["messages"][-1].content


def main() -> None:
    task = f"Read the file {_DEMO_FILE} and tell me only its very first non-empty line."
    print(f"[agent] task: {task}\n")
    answer = asyncio.run(run(task))
    print(f"\n[agent] answer:\n{answer}")


if __name__ == "__main__":
    main()
