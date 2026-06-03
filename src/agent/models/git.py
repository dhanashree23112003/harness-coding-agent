from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    commit_hash: str
    message: str
    author: str
    author_email: str
    date: str  # ISO 8601


class BlameEntry(BaseModel):
    line_number: int
    commit_hash: str
    author: str
    date: str
    content: str


class ChangedFile(BaseModel):
    path: str
    status: str  # "M", "A", "D", "R", "?" per git short status


class GitBranch(BaseModel):
    name: str
    is_current: bool


# ── git_status ────────────────────────────────────────────────────────────────

class GitStatusInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")


class GitStatusOutput(BaseModel):
    cwd: str
    branch: str
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    is_clean: bool
    success: bool = True
    error: Optional[str] = None


# ── git_diff ──────────────────────────────────────────────────────────────────

class GitDiffInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    ref1: Optional[str] = Field(None, description="First ref (e.g. HEAD~1). Defaults to working tree diff.")
    ref2: Optional[str] = Field(None, description="Second ref (e.g. HEAD). Used with ref1.")
    path: Optional[str] = Field(None, description="Limit diff to this file path")


class GitDiffOutput(BaseModel):
    cwd: str
    diff_text: str
    files_changed: list[str]
    success: bool = True
    error: Optional[str] = None


# ── git_log ───────────────────────────────────────────────────────────────────

class GitLogInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    max_count: int = Field(20, description="Maximum number of log entries to return")
    path: Optional[str] = Field(None, description="Limit log to this file path")


class GitLogOutput(BaseModel):
    cwd: str
    entries: list[LogEntry]
    success: bool = True
    error: Optional[str] = None


# ── git_blame ─────────────────────────────────────────────────────────────────

class GitBlameInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    path: str = Field(..., description="File to blame (relative to repo root)")
    start_line: Optional[int] = Field(None, description="First line (1-indexed)")
    end_line: Optional[int] = Field(None, description="Last line (1-indexed)")


class GitBlameOutput(BaseModel):
    cwd: str
    path: str
    entries: list[BlameEntry]
    success: bool = True
    error: Optional[str] = None


# ── branch_create ─────────────────────────────────────────────────────────────

class BranchCreateInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    name: str = Field(..., description="Branch name to create")
    checkout: bool = Field(True, description="Switch to the new branch after creation")


class BranchCreateOutput(BaseModel):
    cwd: str
    name: str
    checked_out: bool
    success: bool = True
    error: Optional[str] = None


# ── branch_list ───────────────────────────────────────────────────────────────

class BranchListInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")


class BranchListOutput(BaseModel):
    cwd: str
    branches: list[GitBranch]
    success: bool = True
    error: Optional[str] = None


# ── git_checkout ──────────────────────────────────────────────────────────────

class GitCheckoutInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    ref: str = Field(..., description="Branch name, tag, or commit hash to check out")


class GitCheckoutOutput(BaseModel):
    cwd: str
    ref: str
    success: bool = True
    error: Optional[str] = None


# ── git_commit ────────────────────────────────────────────────────────────────

class GitCommitInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    message: str = Field(..., description="Commit message")
    add_all: bool = Field(False, description="Stage all tracked modified files before committing")


class GitCommitOutput(BaseModel):
    cwd: str
    commit_hash: str
    message: str
    success: bool = True
    error: Optional[str] = None


# ── git_stash ─────────────────────────────────────────────────────────────────

class GitStashInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    action: Literal["push", "pop", "list", "drop"] = Field(..., description="Stash sub-command")
    message: Optional[str] = Field(None, description="Message for 'push'; stash ref for 'drop'")


class GitStashOutput(BaseModel):
    cwd: str
    action: str
    entries: list[str]
    success: bool = True
    error: Optional[str] = None


# ── show_commit ───────────────────────────────────────────────────────────────

class ShowCommitInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    ref: str = Field("HEAD", description="Commit ref to show")


class ShowCommitOutput(BaseModel):
    cwd: str
    ref: str
    commit_hash: str
    author: str
    date: str
    message: str
    diff_text: str
    success: bool = True
    error: Optional[str] = None


# ── list_changed_files ────────────────────────────────────────────────────────

class ListChangedFilesInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    ref1: str = Field("HEAD~1", description="Base ref for comparison")
    ref2: str = Field("HEAD", description="Target ref for comparison")


class ListChangedFilesOutput(BaseModel):
    cwd: str
    files: list[ChangedFile]
    success: bool = True
    error: Optional[str] = None


# ── git_tag ───────────────────────────────────────────────────────────────────

class GitTagInput(BaseModel):
    cwd: str = Field(..., description="Absolute path to the git repository")
    action: Literal["create", "list", "delete"] = Field(..., description="Tag sub-command")
    name: Optional[str] = Field(None, description="Tag name (required for create/delete)")
    ref: Optional[str] = Field(None, description="Ref to tag (create only, defaults to HEAD)")
    message: Optional[str] = Field(None, description="Annotated tag message (create only)")


class GitTagOutput(BaseModel):
    cwd: str
    action: str
    tags: list[str]
    success: bool = True
    error: Optional[str] = None
