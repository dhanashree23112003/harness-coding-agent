"""MCP server for the fs namespace. All 11 filesystem tools."""
import re
import shutil
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.fs import (  # noqa: E402
    CopyInput,
    CopyOutput,
    DeleteInput,
    DeleteOutput,
    DirEntry,
    FileStatInput,
    FileStatOutput,
    GrepInput,
    GrepMatch,
    GrepOutput,
    ListDirInput,
    ListDirOutput,
    MakeDirInput,
    MakeDirOutput,
    MoveInput,
    MoveOutput,
    ReadFileInput,
    ReadFileOutput,
    ReadFileRangeInput,
    ReadFileRangeOutput,
    SearchFilesInput,
    SearchFilesOutput,
    WriteFileInput,
    WriteFileOutput,
)

mcp = FastMCP("fs")

import os as _os  # noqa: E402

_FS_ROOT: str | None = _os.environ.get("AGENT_REPO_ROOT")


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if _FS_ROOT:
        return (Path(_FS_ROOT) / p).resolve()
    return p


@mcp.tool()
def read_file(path: str, encoding: str = "utf-8") -> dict:
    """Read the full contents of a file and return its content and byte size."""
    req = ReadFileInput(path=path, encoding=encoding)
    try:
        raw = _resolve(req.path).read_bytes()
        return ReadFileOutput(
            path=req.path,
            content=raw.decode(req.encoding),
            size_bytes=len(raw),
        ).model_dump()
    except Exception as exc:
        return ReadFileOutput(
            path=req.path, content="", size_bytes=0, success=False, error=str(exc)
        ).model_dump()


@mcp.tool()
def read_file_range(path: str, start_line: int, end_line: int, encoding: str = "utf-8") -> dict:
    """Read a specific line range from a file (1-indexed, inclusive)."""
    req = ReadFileRangeInput(path=path, start_line=start_line, end_line=end_line, encoding=encoding)
    try:
        all_lines = _resolve(req.path).read_text(encoding=req.encoding).splitlines()
        sliced = all_lines[req.start_line - 1 : req.end_line]
        return ReadFileRangeOutput(
            path=req.path,
            lines=sliced,
            start_line=req.start_line,
            end_line=req.end_line,
        ).model_dump()
    except Exception as exc:
        return ReadFileRangeOutput(
            path=req.path, lines=[], start_line=req.start_line,
            end_line=req.end_line, success=False, error=str(exc),
        ).model_dump()


@mcp.tool()
def write_file(path: str, content: str, encoding: str = "utf-8", create_dirs: bool = False) -> dict:
    """Write text content to a file, optionally creating parent directories."""
    req = WriteFileInput(path=path, content=content, encoding=encoding, create_dirs=create_dirs)
    try:
        p = _resolve(req.path)
        if req.create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content, encoding=req.encoding)
        return WriteFileOutput(path=req.path, bytes_written=p.stat().st_size).model_dump()
    except Exception as exc:
        return WriteFileOutput(path=req.path, bytes_written=0, success=False, error=str(exc)).model_dump()


@mcp.tool()
def list_dir(path: str, recursive: bool = False) -> dict:
    """List the contents of a directory, returning typed entries."""
    req = ListDirInput(path=path, recursive=recursive)
    try:
        p = _resolve(req.path)
        iterator = p.rglob("*") if req.recursive else p.iterdir()
        entries = []
        for item in sorted(iterator, key=lambda x: (x.is_file(), x.name)):
            try:
                st = item.stat()
                entries.append(DirEntry(
                    name=str(item.relative_to(p)) if req.recursive else item.name,
                    is_dir=item.is_dir(),
                    size_bytes=st.st_size if item.is_file() else 0,
                    modified_at=datetime.fromtimestamp(st.st_mtime).isoformat(),
                ))
            except OSError:
                continue
        return ListDirOutput(path=req.path, entries=entries, count=len(entries)).model_dump()
    except Exception as exc:
        return ListDirOutput(path=req.path, entries=[], count=0, success=False, error=str(exc)).model_dump()


@mcp.tool()
def search_files(root: str, pattern: str, max_results: int = 200) -> dict:
    """Search for files matching a glob pattern under a root directory."""
    req = SearchFilesInput(root=root, pattern=pattern, max_results=max_results)
    try:
        matches = [str(p) for p in islice(_resolve(req.root).rglob(req.pattern), req.max_results)]
        return SearchFilesOutput(
            root=req.root, pattern=req.pattern, matches=matches, count=len(matches)
        ).model_dump()
    except Exception as exc:
        return SearchFilesOutput(
            root=req.root, pattern=req.pattern, matches=[], count=0, success=False, error=str(exc)
        ).model_dump()


@mcp.tool()
def grep(root: str, pattern: str, file_glob: str = "*", max_results: int = 200, case_sensitive: bool = True) -> dict:
    """Search for a regex pattern inside files under a root directory."""
    req = GrepInput(root=root, pattern=pattern, file_glob=file_glob,
                    max_results=max_results, case_sensitive=case_sensitive)
    try:
        flags = 0 if req.case_sensitive else re.IGNORECASE
        compiled = re.compile(req.pattern, flags)
        matches: list[GrepMatch] = []
        for fpath in _resolve(req.root).rglob(req.file_glob):
            if not fpath.is_file():
                continue
            try:
                for i, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        matches.append(GrepMatch(file=str(fpath), line_number=i, text=line))
                        if len(matches) >= req.max_results:
                            break
            except OSError:
                continue
            if len(matches) >= req.max_results:
                break
        return GrepOutput(root=req.root, pattern=req.pattern, matches=matches, count=len(matches)).model_dump()
    except Exception as exc:
        return GrepOutput(
            root=req.root, pattern=req.pattern, matches=[], count=0, success=False, error=str(exc)
        ).model_dump()


@mcp.tool()
def file_stat(path: str) -> dict:
    """Return metadata (size, type, modification time) for a file or directory."""
    req = FileStatInput(path=path)
    try:
        p = _resolve(req.path)
        st = p.stat()
        return FileStatOutput(
            path=req.path,
            size_bytes=st.st_size,
            is_file=p.is_file(),
            is_dir=p.is_dir(),
            modified_at=datetime.fromtimestamp(st.st_mtime).isoformat(),
        ).model_dump()
    except Exception as exc:
        return FileStatOutput(
            path=req.path, size_bytes=0, is_file=False, is_dir=False,
            modified_at="", success=False, error=str(exc),
        ).model_dump()


@mcp.tool()
def make_dir(path: str, parents: bool = True, exist_ok: bool = True) -> dict:
    """Create a directory, optionally creating all intermediate parents."""
    req = MakeDirInput(path=path, parents=parents, exist_ok=exist_ok)
    try:
        _resolve(req.path).mkdir(parents=req.parents, exist_ok=req.exist_ok)
        return MakeDirOutput(path=req.path).model_dump()
    except Exception as exc:
        return MakeDirOutput(path=req.path, success=False, error=str(exc)).model_dump()


@mcp.tool()
def move(src: str, dst: str, overwrite: bool = False) -> dict:
    """Move or rename a file or directory."""
    req = MoveInput(src=src, dst=dst, overwrite=overwrite)
    try:
        dst_path = _resolve(req.dst)
        if dst_path.exists() and not req.overwrite:
            return MoveOutput(src=req.src, dst=req.dst, success=False,
                              error=f"destination exists: {req.dst}").model_dump()
        shutil.move(str(_resolve(req.src)), str(dst_path))
        return MoveOutput(src=req.src, dst=req.dst).model_dump()
    except Exception as exc:
        return MoveOutput(src=req.src, dst=req.dst, success=False, error=str(exc)).model_dump()


@mcp.tool()
def delete(path: str, recursive: bool = False) -> dict:
    """Delete a file, or a directory tree when recursive=True."""
    req = DeleteInput(path=path, recursive=recursive)
    try:
        p = _resolve(req.path)
        if p.is_dir():
            if req.recursive:
                shutil.rmtree(str(p))
            else:
                p.rmdir()
        else:
            p.unlink()
        return DeleteOutput(path=req.path).model_dump()
    except Exception as exc:
        return DeleteOutput(path=req.path, success=False, error=str(exc)).model_dump()


@mcp.tool()
def copy(src: str, dst: str, overwrite: bool = False) -> dict:
    """Copy a file or directory tree to a new location."""
    req = CopyInput(src=src, dst=dst, overwrite=overwrite)
    try:
        src_path = _resolve(req.src)
        dst_path = _resolve(req.dst)
        if dst_path.exists() and not req.overwrite:
            return CopyOutput(src=req.src, dst=req.dst, success=False,
                              error=f"destination exists: {req.dst}").model_dump()
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=req.overwrite)
        else:
            shutil.copy2(str(src_path), str(dst_path))
        return CopyOutput(src=req.src, dst=req.dst).model_dump()
    except Exception as exc:
        return CopyOutput(src=req.src, dst=req.dst, success=False, error=str(exc)).model_dump()


if __name__ == "__main__":
    mcp.run()
