"""MCP server for the git namespace. All 12 version-control tools."""
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.git import (  # noqa: E402
    BlameEntry,
    BranchCreateInput,
    BranchCreateOutput,
    BranchListInput,
    BranchListOutput,
    ChangedFile,
    GitBranch,
    GitBlameInput,
    GitBlameOutput,
    GitCheckoutInput,
    GitCheckoutOutput,
    GitCommitInput,
    GitCommitOutput,
    GitDiffInput,
    GitDiffOutput,
    GitLogInput,
    GitLogOutput,
    GitStashInput,
    GitStashOutput,
    GitStatusInput,
    GitStatusOutput,
    GitTagInput,
    GitTagOutput,
    ListChangedFilesInput,
    ListChangedFilesOutput,
    LogEntry,
    ShowCommitInput,
    ShowCommitOutput,
)

mcp = FastMCP("git")

_SEP = "\x1f"  # ASCII unit separator, safe in git log format strings
_REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _git(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a git command in cwd. Returns (returncode, stdout, stderr).

    Falls back to _REPO_ROOT when cwd does not exist (model may hallucinate
    Linux paths like /home/user/project on Windows environments).
    """
    if not Path(cwd).exists():
        cwd = _REPO_ROOT
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd,
            # DEVNULL prevents git child processes from inheriting the MCP
            # server's stdin pipe. On Windows, inherited pipe handles can block
            # the parent's stdio transport from draining, causing the MCP
            # client to hang indefinitely waiting for a response.
            stdin=subprocess.DEVNULL,
        )
        return result.returncode, result.stdout, result.stderr
    except OSError as exc:
        return 1, "", str(exc)


@mcp.tool()
def git_status(cwd: str) -> dict:
    """Return the working tree status of the git repository at cwd."""
    req = GitStatusInput(cwd=cwd)
    rc, out, err = _git(["status", "--porcelain", "-b"], req.cwd)
    if rc != 0:
        return GitStatusOutput(
            cwd=req.cwd, branch="", staged=[], unstaged=[],
            untracked=[], is_clean=False, success=False, error=err.strip(),
        ).model_dump()

    lines = out.splitlines()
    branch = ""
    staged, unstaged, untracked = [], [], []
    for line in lines:
        if line.startswith("## "):
            branch = line[3:].split("...")[0].strip()
            continue
        if len(line) < 2:
            continue
        x, y, path = line[0], line[1], line[3:]
        if x == "?" and y == "?":
            untracked.append(path)
        else:
            if x != " " and x != "?":
                staged.append(path)
            if y != " " and y != "?":
                unstaged.append(path)

    return GitStatusOutput(
        cwd=req.cwd, branch=branch,
        staged=staged, unstaged=unstaged, untracked=untracked,
        is_clean=not (staged or unstaged or untracked),
    ).model_dump()


@mcp.tool()
def git_diff(cwd: str, ref1: Optional[str] = None, ref2: Optional[str] = None, path: Optional[str] = None) -> dict:
    """Return the unified diff between two refs, or the working-tree diff."""
    req = GitDiffInput(cwd=cwd, ref1=ref1, ref2=ref2, path=path)
    args = ["diff"]
    if req.ref1:
        args.append(req.ref1)
    if req.ref2:
        args.append(req.ref2)
    if req.path:
        args += ["--", req.path]

    rc, diff_text, err = _git(args, req.cwd)
    if rc != 0:
        return GitDiffOutput(cwd=req.cwd, diff_text="", files_changed=[], success=False, error=err.strip()).model_dump()

    name_args = ["diff", "--name-only"]
    if req.ref1:
        name_args.append(req.ref1)
    if req.ref2:
        name_args.append(req.ref2)
    if req.path:
        name_args += ["--", req.path]
    _, names_out, _ = _git(name_args, req.cwd)
    files_changed = [f for f in names_out.splitlines() if f]

    return GitDiffOutput(cwd=req.cwd, diff_text=diff_text, files_changed=files_changed).model_dump()


@mcp.tool()
def git_log(cwd: str, max_count: int = 20, path: Optional[str] = None) -> dict:
    """Return the commit log as structured entries."""
    req = GitLogInput(cwd=cwd, max_count=max_count, path=path)
    fmt = f"%H{_SEP}%s{_SEP}%an{_SEP}%ae{_SEP}%ai"
    args = ["log", f"--pretty=format:{fmt}", f"-n{req.max_count}"]
    if req.path:
        args += ["--", req.path]

    rc, out, err = _git(args, req.cwd)
    if rc != 0:
        return GitLogOutput(cwd=req.cwd, entries=[], success=False, error=err.strip()).model_dump()

    entries = []
    for line in out.splitlines():
        parts = line.split(_SEP)
        if len(parts) == 5:
            entries.append(LogEntry(
                commit_hash=parts[0], message=parts[1],
                author=parts[2], author_email=parts[3], date=parts[4],
            ))
    return GitLogOutput(cwd=req.cwd, entries=entries).model_dump()


@mcp.tool()
def git_blame(cwd: str, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> dict:
    """Return line-by-line blame information for a file."""
    req = GitBlameInput(cwd=cwd, path=path, start_line=start_line, end_line=end_line)
    args = ["blame", "--line-porcelain"]
    if req.start_line and req.end_line:
        args += [f"-L{req.start_line},{req.end_line}"]
    args.append(req.path)

    rc, out, err = _git(args, req.cwd)
    if rc != 0:
        return GitBlameOutput(cwd=req.cwd, path=req.path, entries=[], success=False, error=err.strip()).model_dump()

    entries = []
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if not header:
            i += 1
            continue
        commit_hash = header[0]
        line_num = int(header[2]) if len(header) >= 3 else 0
        meta: dict[str, str] = {}
        i += 1
        while i < len(lines) and not lines[i].startswith("\t"):
            parts = lines[i].split(" ", 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
            i += 1
        content = lines[i][1:] if i < len(lines) and lines[i].startswith("\t") else ""
        entries.append(BlameEntry(
            line_number=line_num,
            commit_hash=commit_hash,
            author=meta.get("author", ""),
            date=meta.get("author-time", ""),
            content=content,
        ))
        i += 1

    return GitBlameOutput(cwd=req.cwd, path=req.path, entries=entries).model_dump()


@mcp.tool()
def branch_create(cwd: str, name: str, checkout: bool = True) -> dict:
    """Create a new branch, optionally checking it out immediately."""
    req = BranchCreateInput(cwd=cwd, name=name, checkout=checkout)
    if req.checkout:
        rc, _, err = _git(["checkout", "-b", req.name], req.cwd)
    else:
        rc, _, err = _git(["branch", req.name], req.cwd)
    if rc != 0:
        return BranchCreateOutput(cwd=req.cwd, name=req.name, checked_out=False,
                                   success=False, error=err.strip()).model_dump()
    return BranchCreateOutput(cwd=req.cwd, name=req.name, checked_out=req.checkout).model_dump()


@mcp.tool()
def branch_list(cwd: str) -> dict:
    """List all local branches, marking the currently active one."""
    req = BranchListInput(cwd=cwd)
    rc, out, err = _git(["branch"], req.cwd)
    if rc != 0:
        return BranchListOutput(cwd=req.cwd, branches=[], success=False, error=err.strip()).model_dump()
    branches = []
    for line in out.splitlines():
        is_current = line.startswith("*")
        name = line.lstrip("* ").strip()
        if name:
            branches.append(GitBranch(name=name, is_current=is_current))
    return BranchListOutput(cwd=req.cwd, branches=branches).model_dump()


@mcp.tool()
def git_checkout(cwd: str, ref: str) -> dict:
    """Check out a branch, tag, or commit in the repository."""
    req = GitCheckoutInput(cwd=cwd, ref=ref)
    rc, _, err = _git(["checkout", req.ref], req.cwd)
    if rc != 0:
        return GitCheckoutOutput(cwd=req.cwd, ref=req.ref, success=False, error=err.strip()).model_dump()
    return GitCheckoutOutput(cwd=req.cwd, ref=req.ref).model_dump()


@mcp.tool()
def git_commit(cwd: str, message: str, add_all: bool = False) -> dict:
    """Create a commit with the given message, optionally staging all tracked changes first."""
    req = GitCommitInput(cwd=cwd, message=message, add_all=add_all)
    if req.add_all:
        rc, _, err = _git(["add", "-u"], req.cwd)
        if rc != 0:
            return GitCommitOutput(cwd=req.cwd, commit_hash="", message=req.message,
                                    success=False, error=err.strip()).model_dump()
    rc, out, err = _git(["commit", "-m", req.message], req.cwd)
    if rc != 0:
        return GitCommitOutput(cwd=req.cwd, commit_hash="", message=req.message,
                                success=False, error=err.strip()).model_dump()
    # Extract the short hash from "master abc1234] message" output
    rc2, hash_out, _ = _git(["rev-parse", "HEAD"], req.cwd)
    commit_hash = hash_out.strip() if rc2 == 0 else ""
    return GitCommitOutput(cwd=req.cwd, commit_hash=commit_hash, message=req.message).model_dump()


@mcp.tool()
def git_stash(cwd: str, action: str, message: Optional[str] = None) -> dict:
    """Manage the git stash: push, pop, list, or drop."""
    req = GitStashInput(cwd=cwd, action=action, message=message)
    if req.action == "push":
        args = ["stash", "push"]
        if req.message:
            args += ["-m", req.message]
    elif req.action == "pop":
        args = ["stash", "pop"]
    elif req.action == "list":
        args = ["stash", "list"]
    elif req.action == "drop":
        args = ["stash", "drop"]
        if req.message:
            args.append(req.message)
    else:
        return GitStashOutput(cwd=req.cwd, action=req.action, entries=[],
                               success=False, error=f"unknown action: {req.action}").model_dump()

    rc, out, err = _git(args, req.cwd)
    if rc != 0:
        return GitStashOutput(cwd=req.cwd, action=req.action, entries=[],
                               success=False, error=err.strip()).model_dump()
    entries = [line for line in out.splitlines() if line]
    return GitStashOutput(cwd=req.cwd, action=req.action, entries=entries).model_dump()


@mcp.tool()
def show_commit(cwd: str, ref: str = "HEAD") -> dict:
    """Show the diff and metadata for a specific commit."""
    req = ShowCommitInput(cwd=cwd, ref=ref)
    fmt = f"%H{_SEP}%an{_SEP}%ai{_SEP}%s"
    rc, meta_out, err = _git(["log", "-1", f"--pretty=format:{fmt}", req.ref], req.cwd)
    if rc != 0:
        return ShowCommitOutput(
            cwd=req.cwd, ref=req.ref, commit_hash="", author="",
            date="", message="", diff_text="", success=False, error=err.strip(),
        ).model_dump()

    parts = meta_out.split(_SEP)
    commit_hash = parts[0] if len(parts) > 0 else ""
    author = parts[1] if len(parts) > 1 else ""
    date = parts[2] if len(parts) > 2 else ""
    message = parts[3] if len(parts) > 3 else ""

    _, diff_text, _ = _git(["show", "--unified=3", req.ref], req.cwd)
    return ShowCommitOutput(
        cwd=req.cwd, ref=req.ref, commit_hash=commit_hash,
        author=author, date=date, message=message, diff_text=diff_text,
    ).model_dump()


@mcp.tool()
def list_changed_files(cwd: str, ref1: str = "HEAD~1", ref2: str = "HEAD") -> dict:
    """List files changed between two commits, with their change status."""
    req = ListChangedFilesInput(cwd=cwd, ref1=ref1, ref2=ref2)
    rc, out, err = _git(["diff", "--name-status", req.ref1, req.ref2], req.cwd)
    if rc != 0:
        return ListChangedFilesOutput(cwd=req.cwd, files=[], success=False, error=err.strip()).model_dump()
    files = []
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            files.append(ChangedFile(status=parts[0].strip(), path=parts[1].strip()))
    return ListChangedFilesOutput(cwd=req.cwd, files=files).model_dump()


@mcp.tool()
def git_tag(cwd: str, action: str, name: Optional[str] = None, ref: Optional[str] = None, message: Optional[str] = None) -> dict:
    """Manage git tags: create, list, or delete."""
    req = GitTagInput(cwd=cwd, action=action, name=name, ref=ref, message=message)
    if req.action == "list":
        rc, out, err = _git(["tag", "--list"], req.cwd)
        tags = [t for t in out.splitlines() if t]
    elif req.action == "create":
        if not req.name:
            return GitTagOutput(cwd=req.cwd, action=req.action, tags=[],
                                 success=False, error="name is required for create").model_dump()
        args = ["tag"]
        if req.message:
            args += ["-a", req.name, "-m", req.message]
        else:
            args.append(req.name)
        if req.ref:
            args.append(req.ref)
        rc, _, err = _git(args, req.cwd)
        tags = [req.name] if rc == 0 else []
    elif req.action == "delete":
        if not req.name:
            return GitTagOutput(cwd=req.cwd, action=req.action, tags=[],
                                 success=False, error="name is required for delete").model_dump()
        rc, _, err = _git(["tag", "-d", req.name], req.cwd)
        tags = [req.name] if rc == 0 else []
    else:
        return GitTagOutput(cwd=req.cwd, action=req.action, tags=[],
                             success=False, error=f"unknown action: {req.action}").model_dump()

    if rc != 0:
        return GitTagOutput(cwd=req.cwd, action=req.action, tags=[],
                             success=False, error=err.strip()).model_dump()
    return GitTagOutput(cwd=req.cwd, action=req.action, tags=tags).model_dump()


if __name__ == "__main__":
    mcp.run()
