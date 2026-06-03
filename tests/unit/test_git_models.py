import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
    BlameEntry,
    BranchCreateInput,
    BranchCreateOutput,
    BranchListInput,
    BranchListOutput,
    ChangedFile,
    GitBlameInput,
    GitBlameOutput,
    GitBranch,
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


# ── nested types ───────────────────────────────────────────────────────────────

def test_log_entry_roundtrip():
    e = LogEntry(commit_hash="abc123", message="fix: bug", author="Alice",
                 author_email="a@a.com", date="2024-01-01T00:00:00+00:00")
    assert LogEntry(**e.model_dump()) == e


def test_blame_entry_roundtrip():
    e = BlameEntry(line_number=5, commit_hash="abc", author="Bob", date="1700000000", content="x = 1")
    assert BlameEntry(**e.model_dump()) == e


def test_changed_file_roundtrip():
    f = ChangedFile(path="src/main.py", status="M")
    assert ChangedFile(**f.model_dump()) == f


def test_git_branch_roundtrip():
    b = GitBranch(name="main", is_current=True)
    assert GitBranch(**b.model_dump()) == b


# ── git_status ─────────────────────────────────────────────────────────────────

def test_git_status_input_requires_cwd():
    with pytest.raises(ValidationError):
        GitStatusInput()  # type: ignore[call-arg]


def test_git_status_output_clean():
    out = GitStatusOutput(cwd="/repo", branch="main", staged=[], unstaged=[], untracked=[], is_clean=True)
    assert out.is_clean is True
    assert out.success is True


def test_git_status_output_error():
    out = GitStatusOutput(cwd="/repo", branch="", staged=[], unstaged=[],
                          untracked=[], is_clean=False, success=False, error="not a git repo")
    assert out.error == "not a git repo"


def test_git_status_output_roundtrip():
    out = GitStatusOutput(cwd="/repo", branch="main", staged=["a.py"],
                          unstaged=[], untracked=[], is_clean=False)
    assert GitStatusOutput(**out.model_dump()) == out


# ── git_diff ───────────────────────────────────────────────────────────────────

def test_git_diff_input_defaults():
    inp = GitDiffInput(cwd="/repo")
    assert inp.ref1 is None
    assert inp.ref2 is None
    assert inp.path is None


def test_git_diff_output_roundtrip():
    out = GitDiffOutput(cwd="/repo", diff_text="--- a\n+++ b\n", files_changed=["a.py"])
    assert GitDiffOutput(**out.model_dump()) == out


# ── git_log ────────────────────────────────────────────────────────────────────

def test_git_log_input_defaults():
    inp = GitLogInput(cwd="/repo")
    assert inp.max_count == 20
    assert inp.path is None


def test_git_log_output_roundtrip():
    entry = LogEntry(commit_hash="abc", message="init", author="A",
                     author_email="a@a.com", date="2024-01-01T00:00:00+00:00")
    out = GitLogOutput(cwd="/repo", entries=[entry])
    assert GitLogOutput(**out.model_dump()) == out


# ── git_blame ──────────────────────────────────────────────────────────────────

def test_git_blame_input_requires_path():
    with pytest.raises(ValidationError):
        GitBlameInput(cwd="/repo")  # missing path


def test_git_blame_output_roundtrip():
    out = GitBlameOutput(cwd="/repo", path="src/main.py", entries=[])
    assert GitBlameOutput(**out.model_dump()) == out


# ── branch_create ──────────────────────────────────────────────────────────────

def test_branch_create_input_defaults():
    inp = BranchCreateInput(cwd="/repo", name="feat/new")
    assert inp.checkout is True


def test_branch_create_output_roundtrip():
    out = BranchCreateOutput(cwd="/repo", name="feat/new", checked_out=True)
    assert BranchCreateOutput(**out.model_dump()) == out


# ── branch_list ────────────────────────────────────────────────────────────────

def test_branch_list_output_roundtrip():
    out = BranchListOutput(cwd="/repo", branches=[GitBranch(name="main", is_current=True)])
    assert BranchListOutput(**out.model_dump()) == out


# ── git_checkout ───────────────────────────────────────────────────────────────

def test_git_checkout_input_requires_ref():
    with pytest.raises(ValidationError):
        GitCheckoutInput(cwd="/repo")  # missing ref


def test_git_checkout_output_roundtrip():
    out = GitCheckoutOutput(cwd="/repo", ref="main")
    assert GitCheckoutOutput(**out.model_dump()) == out


# ── git_commit ─────────────────────────────────────────────────────────────────

def test_git_commit_input_defaults():
    inp = GitCommitInput(cwd="/repo", message="fix: typo")
    assert inp.add_all is False


def test_git_commit_output_roundtrip():
    out = GitCommitOutput(cwd="/repo", commit_hash="abc123", message="fix: typo")
    assert GitCommitOutput(**out.model_dump()) == out


# ── git_stash ──────────────────────────────────────────────────────────────────

def test_git_stash_input_requires_action():
    with pytest.raises(ValidationError):
        GitStashInput(cwd="/repo")  # missing action


def test_git_stash_input_invalid_action():
    with pytest.raises(ValidationError):
        GitStashInput(cwd="/repo", action="unknown")  # type: ignore[arg-type]


def test_git_stash_output_roundtrip():
    out = GitStashOutput(cwd="/repo", action="list", entries=["stash@{0}: message"])
    assert GitStashOutput(**out.model_dump()) == out


# ── show_commit ────────────────────────────────────────────────────────────────

def test_show_commit_input_defaults():
    inp = ShowCommitInput(cwd="/repo")
    assert inp.ref == "HEAD"


def test_show_commit_output_roundtrip():
    out = ShowCommitOutput(
        cwd="/repo", ref="HEAD", commit_hash="abc", author="A",
        date="2024-01-01T00:00:00+00:00", message="init", diff_text="",
    )
    assert ShowCommitOutput(**out.model_dump()) == out


# ── list_changed_files ─────────────────────────────────────────────────────────

def test_list_changed_files_input_defaults():
    inp = ListChangedFilesInput(cwd="/repo")
    assert inp.ref1 == "HEAD~1"
    assert inp.ref2 == "HEAD"


def test_list_changed_files_output_roundtrip():
    out = ListChangedFilesOutput(cwd="/repo", files=[ChangedFile(path="a.py", status="M")])
    assert ListChangedFilesOutput(**out.model_dump()) == out


# ── git_tag ────────────────────────────────────────────────────────────────────

def test_git_tag_input_requires_action():
    with pytest.raises(ValidationError):
        GitTagInput(cwd="/repo")  # missing action


def test_git_tag_input_invalid_action():
    with pytest.raises(ValidationError):
        GitTagInput(cwd="/repo", action="push")  # type: ignore[arg-type]


def test_git_tag_output_roundtrip():
    out = GitTagOutput(cwd="/repo", action="list", tags=["v1.0.0"])
    assert GitTagOutput(**out.model_dump()) == out
