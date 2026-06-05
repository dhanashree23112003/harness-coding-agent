"""Async helpers for connecting to MCP servers and discovering tools."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

_SERVERS_DIR = Path(__file__).resolve().parents[2] / "agent" / "servers"
_FS_SERVER = _SERVERS_DIR / "fs_server.py"
_GIT_SERVER = _SERVERS_DIR / "git_server.py"

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
}


async def get_mcp_tools() -> list:
    """Discover tools via per-invocation connections. Use for smoke checks only.

    Each tool returned here spawns a fresh subprocess per call. Use
    mcp_tools_session() for actual agent runs to avoid per-call spawn cost
    and the Windows BrokenResourceError on rapid reconnects.
    """
    client = MultiServerMCPClient(_CONNECTIONS)
    return await client.get_tools()


@asynccontextmanager
async def mcp_tools_session() -> AsyncIterator[list]:
    """Open persistent stdio sessions for all servers and yield bound tools.

    Tools returned here share the open subprocess connections for the lifetime
    of this context manager, so no subprocess is spawned per tool call.
    """
    client = MultiServerMCPClient(_CONNECTIONS)
    async with client.session("fs") as fs_session:
        async with client.session("git") as git_session:
            fs_tools = await load_mcp_tools(fs_session, server_name="fs")
            git_tools = await load_mcp_tools(git_session, server_name="git")
            yield fs_tools + git_tools


@asynccontextmanager
async def mcp_tools_session_with_namespaces() -> AsyncIterator[tuple[list, dict[str, list]]]:
    """Like mcp_tools_session but also yields a namespace-keyed dict.

    Yields (all_tools, tools_by_namespace) where tools_by_namespace maps
    each server name to its tool list. The retrieval layer uses this to tag
    each tool with its namespace when building the registry.
    """
    client = MultiServerMCPClient(_CONNECTIONS)
    async with client.session("fs") as fs_session:
        async with client.session("git") as git_session:
            fs_tools = await load_mcp_tools(fs_session, server_name="fs")
            git_tools = await load_mcp_tools(git_session, server_name="git")
            tools_by_ns: dict[str, list] = {"fs": fs_tools, "git": git_tools}
            yield fs_tools + git_tools, tools_by_ns
