"""Slice 1 entry point: proves the spine end to end with fs.read_file."""
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent.graph.graph import build_graph
from agent.mcp_client.client import get_mcp_tools

load_dotenv()

_DEMO_FILE = Path(__file__).resolve().parents[2] / "SPEC.md"


async def run(task: str) -> str:
    tools = await get_mcp_tools()
    print(f"[agent] discovered tools: {[t.name for t in tools]}")
    graph = build_graph(tools)
    result = await graph.ainvoke({"task": task, "plan": "", "messages": []})
    return result["messages"][-1].content


def main() -> None:
    task = f"Read the file {_DEMO_FILE} and tell me only its very first non-empty line."
    print(f"[agent] task: {task}\n")
    answer = asyncio.run(run(task))
    print(f"\n[agent] answer:\n{answer}")


if __name__ == "__main__":
    main()
