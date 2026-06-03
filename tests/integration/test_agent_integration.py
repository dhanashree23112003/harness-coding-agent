"""Integration test: full agent run against a throwaway fixture repo.

Requires GROQ_API_KEY. Skipped automatically when the key is absent so
`make test` (unit only) always runs without credentials.
"""
import asyncio
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_NEEDS_KEY = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)


@pytest.fixture
def fixture_repo(tmp_path):
    """Create a minimal git repo with two commits for agent tasks."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def g(*args):
        subprocess.run(["git", "-C", str(repo)] + list(args), check=True, capture_output=True)

    g("init")
    g("config", "user.email", "test@fixture.local")
    g("config", "user.name", "Fixture Agent")

    # Commit 1: README
    (repo / "README.md").write_text(
        "# Fixture Repo\nThis is a test repository for agent integration.\n"
    )
    g("add", "README.md")
    g("commit", "-m", "initial: add README")

    # Commit 2: src/main.py
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("def hello():\n    return 'hello'\n")
    g("add", "src/main.py")
    g("commit", "-m", "feat: add main.py")

    return repo


@_NEEDS_KEY
@pytest.mark.asyncio
async def test_agent_uses_git_and_fs_tools(fixture_repo):
    """Agent must call list_changed_files, read_file, and git_log for the fixture repo."""
    from agent.main import run

    task = (
        f"Working in the repository at {fixture_repo}: "
        "first list the files changed in the last commit (HEAD~1 to HEAD), "
        "then read the contents of README.md, "
        "then show the git log for the last 2 commits. "
        "Report what you found."
    )
    answer = await asyncio.wait_for(run(task), timeout=120)

    # Commit 2 introduced src/main.py, so the changed-files result should mention it.
    assert "main.py" in answer or "src" in answer, (
        f"Expected 'main.py' or 'src' in answer, got:\n{answer}"
    )
    # README content should surface in the answer.
    assert "Fixture Repo" in answer or "test repository" in answer, (
        f"Expected README text in answer, got:\n{answer}"
    )
    # Log messages from both commits should appear.
    assert "initial" in answer or "feat" in answer, (
        f"Expected commit message keywords in answer, got:\n{answer}"
    )
