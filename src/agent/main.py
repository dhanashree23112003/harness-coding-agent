"""Entry point: agent spine with tool-retrieval layer + context manager (Slice 6)."""
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
_FIXTURE_REPO = Path(__file__).resolve().parents[2] / "fixture_repo"

_LONG_HORIZON_TASK_TEMPLATE = """\
Working in the repository at {repo}:

1. Check git status and list all Python files in the repo.
2. Read calculator.py and app.py to understand the existing code.
3. Use ast.find_references to locate all callers of the `divide` function.
4. Read each caller file in full.
5. Add input validation to `divide()` in calculator.py: raise ValueError when \
the divisor is zero or when either argument is not a number. \
Update app.py to catch ValueError from divide and return None instead of crashing.
6. Run the full test suite. Read test_calculator.py.
7. Add tests for the new validation behaviour: divide by zero raises ValueError, \
non-numeric input raises ValueError.
8. Run the suite again. If tests still fail, read the failure output carefully and fix.
9. If any tests fail after two runs, spawn a test-triage subagent \
(scopes: test + fs.read_file) to identify the exact failures and return findings.
10. Apply any fixes identified by the subagent findings, then run the suite one \
final time.
11. Commit the final working state with the message \
"feat: add input validation to divide".
Report which tests were added and confirm all tests pass.\
"""


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


def _walk_exc(e: BaseException, depth: int = 0) -> None:
    import traceback as _tb
    pad = "  " * depth
    if hasattr(e, "exceptions"):
        print(f"{pad}[exc-group] {len(e.exceptions)} inner:", flush=True)
        for inner in e.exceptions:
            _walk_exc(inner, depth + 1)
    else:
        print(f"{pad}[exc] {type(e).__name__}: {e}", flush=True)
        lines = _tb.format_exception(type(e), e, e.__traceback__)
        for line in "".join(lines).splitlines()[-10:]:
            print(f"{pad}  {line}", flush=True)


_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a precise coding agent. "
    "Execute the user's task exactly as stated, no more and no less. "
    "Do not summarize, explain, or expand beyond what is explicitly asked. "
    "When the task is complete, stop immediately. "
    "GROUNDING RULE: Your final answer must be based only on tool results "
    "visible in this session. Do not claim a file was written or edited, "
    "a test was added or passed, or a commit was made unless a successful "
    "write_file, run_suite, git_commit, or equivalent tool result appears "
    "in your tool call history. If a step failed or you lacked a required "
    "tool, state that explicitly instead of claiming success."
))


async def run(task: str, repo_root: str | Path | None = None) -> str:
    t0 = time.perf_counter()
    print("[agent] starting MCP sessions and building registry...", flush=True)

    try:
        async with mcp_tools_session_with_namespaces(working_dir=repo_root) as (tools, tools_by_ns):
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
            runner = SubagentRunner(all_tools_by_namespace=tools_by_ns, repo_root=repo_root)
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
                "progress_ledger": "",
                "token_estimate": 0,
                "compaction_count": 0,
                "ledger_message_id": None,
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
    except BaseException as e:
        print("\n[agent] CRASH: walking exception chain to find root cause:", flush=True)
        _walk_exc(e)
        raise


def main() -> None:
    task = f"Read the file {_DEMO_FILE} and tell me what the document is about in one sentence."
    print(f"[agent] task: {task}\n")
    answer = asyncio.run(run(task))
    print(f"\n[agent] answer:\n{answer}")


async def main_long_horizon() -> None:
    """Slice 6 demo: 20+ tool-call task against fixture_repo/.

    Set AGENT_MODEL=llama-3.1-8b-instant (default) to conserve Groq daily cap.
    Set CONTEXT_COMPACT_THRESHOLD to adjust when compaction fires.
    """
    if not _FIXTURE_REPO.exists():
        raise FileNotFoundError(
            f"fixture_repo not found at {_FIXTURE_REPO}. "
            "It should be committed at the repo root."
        )
    task = _LONG_HORIZON_TASK_TEMPLATE.format(repo=_FIXTURE_REPO)
    print(f"[agent] long-horizon task:\n{task}\n")
    answer = await run(task, repo_root=_FIXTURE_REPO)
    print(f"\n[agent] answer:\n{answer}")


if __name__ == "__main__":
    main()
