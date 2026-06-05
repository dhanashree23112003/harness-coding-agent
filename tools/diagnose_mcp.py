"""Diagnostic: open each MCP server session in isolation and report which fails.

Run from the repo root:
    python tools/diagnose_mcp.py
"""
import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from agent.mcp_client.client import _make_connections


def _walk_exc(e: BaseException, depth: int = 0) -> None:
    pad = "  " * depth
    if hasattr(e, "exceptions"):
        print(f"{pad}ExceptionGroup ({len(e.exceptions)} inner):", flush=True)
        for inner in e.exceptions:
            _walk_exc(inner, depth + 1)
    else:
        print(f"{pad}{type(e).__name__}: {e}", flush=True)
        lines = traceback.format_exception(type(e), e, e.__traceback__)
        for line in "".join(lines).splitlines()[-14:]:
            print(f"{pad}  {line}", flush=True)


async def main() -> None:
    fixture = Path("fixture_repo").resolve()
    print(f"fixture_repo: {fixture}  exists={fixture.exists()}", flush=True)
    connections = _make_connections(working_dir=fixture)
    env_sample = list(connections["fs"].get("env", {}).keys())[:6]
    print(f"fs connection env keys (first 6): {env_sample}", flush=True)
    print(f"AGENT_REPO_ROOT in fs env: {connections['fs'].get('env', {}).get('AGENT_REPO_ROOT')}", flush=True)

    client = MultiServerMCPClient(connections)
    for name in ("fs", "git", "ast", "test", "deps", "ci"):
        print(f"\n--- {name} ---", flush=True)
        try:
            async with client.session(name) as s:
                tools = await load_mcp_tools(s, server_name=name)
                print(f"  OK: {len(tools)} tools: {[t.name for t in tools]}", flush=True)
        except BaseException as e:
            print(f"  FAILED:", flush=True)
            _walk_exc(e, depth=2)

    print("\n--- all 6 together (AsyncExitStack) ---", flush=True)
    from contextlib import AsyncExitStack
    from typing import Any
    try:
        async with AsyncExitStack() as stack:
            sessions: dict[str, Any] = {}
            for name in ("fs", "git", "ast", "test", "deps", "ci"):
                sessions[name] = await stack.enter_async_context(client.session(name))
                print(f"  opened: {name}", flush=True)
            tools_by_ns: dict[str, list] = {}
            for name, sess in sessions.items():
                tools_by_ns[name] = await load_mcp_tools(sess, server_name=name)
            total = sum(len(v) for v in tools_by_ns.values())
            print(f"  all 6 open, {total} tools total", flush=True)
        print("  teardown clean", flush=True)
    except BaseException as e:
        print(f"  FAILED during all-6 test:", flush=True)
        _walk_exc(e, depth=2)


asyncio.run(main())
