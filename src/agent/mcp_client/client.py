"""Async helpers for connecting to MCP servers and discovering tools."""
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_FS_SERVER = Path(__file__).resolve().parents[2] / "agent" / "servers" / "fs_server.py"


async def get_mcp_tools() -> list:
    """Discover and return LangChain-compatible tools from all MCP servers.

    In langchain-mcp-adapters >= 0.1.0, each returned tool creates its own
    short-lived subprocess connection per invocation (no persistent session).
    """
    client = MultiServerMCPClient(
        {
            "fs": {
                "command": sys.executable,
                "args": [str(_FS_SERVER)],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()
