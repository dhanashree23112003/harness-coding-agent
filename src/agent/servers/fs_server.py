"""MCP server for the fs namespace. Slice 1: read_file only."""
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure the src/ tree is importable when the server runs as a subprocess.
_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.tool_io import ReadFileInput, ReadFileOutput  # noqa: E402

mcp = FastMCP("fs")


@mcp.tool()
def read_file(path: str, encoding: str = "utf-8") -> dict:
    """Read the full contents of a file and return its content and byte size."""
    req = ReadFileInput(path=path, encoding=encoding)
    raw = Path(req.path).read_bytes()
    result = ReadFileOutput(
        path=req.path,
        content=raw.decode(req.encoding),
        size_bytes=len(raw),
    )
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
