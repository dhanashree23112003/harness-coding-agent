"""Entry point: agent spine with tool-retrieval layer (Slice 3)."""
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.graph import build_graph
from agent.mcp_client.client import mcp_tools_session_with_namespaces
from agent.retrieval import (
    Embedder,
    PgVectorStore,
    ToolRegistryEntry,
    ToolRetriever,
    build_registry,
    entry_text,
)
from agent.subagent import SubagentRunner, make_spawn_subagent_tool

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
            raw = msg.content
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("type") == "text":
                raw = raw[0]["text"]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "content" in parsed:
                    parsed["content"] = parsed["content"][:120] + " [truncated]"
                print(f"  result: {json.dumps(parsed, indent=4)}")
            except (json.JSONDecodeError, TypeError):
                print(f"  result: {str(raw)[:300]}")
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
    t0 = time.perf_counter()
    print("[agent] starting MCP sessions and building registry...", flush=True)

    async with mcp_tools_session_with_namespaces() as (tools, tools_by_ns):
        print(f"[agent] {len(tools)} tools discovered  ({time.perf_counter() - t0:.2f}s)", flush=True)

        # Build registry and embed all tools (once per run).
        entries = build_registry(tools_by_ns)
        texts = [entry_text(e) for e in entries]

        print(f"[agent] loading sentence-transformer and embedding {len(entries)} tools...", flush=True)
        t_embed = time.perf_counter()
        embedder = Embedder()
        vecs = embedder.embed_batch(texts)
        print(f"[agent] embeddings ready  ({time.perf_counter() - t_embed:.2f}s)", flush=True)

        db_url = os.environ["DATABASE_URL"]
        store = PgVectorStore(db_url)
        await store.init_schema()
        await store.upsert(entries, vecs)
        print(f"[agent] {len(entries)} tool embeddings stored in pgvector", flush=True)

        # Build the subagent tool and register it so the retriever can surface it.
        runner = SubagentRunner(all_tools_by_namespace=tools_by_ns)
        spawn_tool = make_spawn_subagent_tool(runner)
        tools = tools + [spawn_tool]

        spawn_entry = ToolRegistryEntry(
            namespace="subagent",
            name="spawn_subagent",
            description=(
                "Launch an isolated subagent to triage test failures. "
                "Scoped to test + fs.read_file. Returns structured findings."
            ),
            input_schema={},
        )
        spawn_vec = embedder.embed(entry_text(spawn_entry))
        await store.upsert([spawn_entry], [spawn_vec])
        print("[agent] spawn_subagent tool registered", flush=True)

        retriever = ToolRetriever(store, embedder, total=len(entries) + 1)
        graph = build_graph(tools, retriever)

        init_state = {
            "task": task,
            "plan": "",
            "messages": [_SYSTEM_PROMPT, HumanMessage(content=task)],
            "available_tool_names": [],
            "retrieval_k": 12,
            "retrieval_miss_count": 0,
            "consecutive_repeat_count": 0,
        }

        t1 = time.perf_counter()
        print("[agent] starting graph run...", flush=True)
        result = await graph.ainvoke(init_state)
        print(f"[agent] graph run done in {time.perf_counter() - t1:.2f}s", flush=True)

        _print_trace(result["messages"])
        # Find the last message with non-empty text content (loop-stopped state
        # leaves the last AIMessage as a tool-call with no text).
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return msg.content
        return "(no text answer: see tool results in trace above)"


def main() -> None:
    task = f"Read the file {_DEMO_FILE} and tell me what the document is about in one sentence."
    print(f"[agent] task: {task}\n")
    answer = asyncio.run(run(task))
    print(f"\n[agent] answer:\n{answer}")


if __name__ == "__main__":
    main()
