"""Async helpers for connecting to MCP servers and discovering tools."""
import copy
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from agent.resilience import with_retry

_SERVERS_DIR = Path(__file__).resolve().parents[2] / "agent" / "servers"
_FS_SERVER   = _SERVERS_DIR / "fs_server.py"
_GIT_SERVER  = _SERVERS_DIR / "git_server.py"
_AST_SERVER  = _SERVERS_DIR / "ast_server.py"
_TEST_SERVER = _SERVERS_DIR / "test_server.py"
_DEPS_SERVER = _SERVERS_DIR / "deps_server.py"
_CI_SERVER   = _SERVERS_DIR / "ci_server.py"

_CONNECTIONS = {
    "fs": {
        "command": sys.executable,
        "args": [str(_FS_SERVER)],
        "transport": "stdio",
    },
    "git": {
        "command": sys.executable,
        "args": [str(_GIT_SERVER)],
        "transport": "stdio",
    },
    "ast": {
        "command": sys.executable,
        "args": [str(_AST_SERVER)],
        "transport": "stdio",
    },
    "test": {
        "command": sys.executable,
        "args": [str(_TEST_SERVER)],
        "transport": "stdio",
    },
    "deps": {
        "command": sys.executable,
        "args": [str(_DEPS_SERVER)],
        "transport": "stdio",
    },
    "ci": {
        "command": sys.executable,
        "args": [str(_CI_SERVER)],
        "transport": "stdio",
    },
}

_NS_ORDER = ("fs", "git", "ast", "test", "deps", "ci")


def _make_connections(working_dir: str | Path | None = None) -> dict:
    connections = copy.deepcopy(_CONNECTIONS)
    if working_dir is not None:
        env = {**os.environ, "AGENT_REPO_ROOT": str(Path(working_dir).resolve())}
        for name in ("fs", "ast"):
            connections[name]["env"] = env
    return connections


async def get_mcp_tools(working_dir: str | Path | None = None) -> list:
    """Discover tools via per-invocation connections. Use for smoke checks only.

    Each tool returned here spawns a fresh subprocess per call. Use
    mcp_tools_session() for actual agent runs to avoid per-call spawn cost
    and the Windows BrokenResourceError on rapid reconnects.
    """
    client = MultiServerMCPClient(_make_connections(working_dir))
    return await client.get_tools()


@asynccontextmanager
async def mcp_tools_session(working_dir: str | Path | None = None) -> AsyncIterator[list]:
    """Open persistent stdio sessions for all 6 servers and yield bound tools.

    Uses AsyncExitStack so every server's teardown runs independently.
    A broken pipe on one server does not prevent the others from closing.
    """
    client = MultiServerMCPClient(_make_connections(working_dir))
    async with AsyncExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in _NS_ORDER:
            sessions[name] = await stack.enter_async_context(client.session(name))
        tools_by_ns: dict[str, list] = {}
        for name, sess in sessions.items():
            tools_by_ns[name] = await with_retry(
                lambda s=sess, n=name: load_mcp_tools(s, server_name=n)
            )
        yield [t for name in _NS_ORDER for t in tools_by_ns[name]]


@asynccontextmanager
async def mcp_tools_session_with_namespaces(
    working_dir: str | Path | None = None,
) -> AsyncIterator[tuple[list, dict[str, list]]]:
    """Like mcp_tools_session but also yields a namespace-keyed dict.

    Yields (all_tools, tools_by_namespace) where tools_by_namespace maps
    each server name to its tool list. The retrieval layer uses this to tag
    each tool with its namespace when building the registry.

    Uses AsyncExitStack so every server's teardown runs independently.
    A broken pipe on one server does not prevent the others from closing.
    """
    client = MultiServerMCPClient(_make_connections(working_dir))
    async with AsyncExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in _NS_ORDER:
            sessions[name] = await stack.enter_async_context(client.session(name))
        tools_by_ns: dict[str, list] = {}
        for name, sess in sessions.items():
            tools_by_ns[name] = await with_retry(
                lambda s=sess, n=name: load_mcp_tools(s, server_name=n)
            )
        all_tools = [t for name in _NS_ORDER for t in tools_by_ns[name]]
        yield all_tools, tools_by_ns
