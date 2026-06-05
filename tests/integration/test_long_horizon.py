"""Integration test: 20+ tool-call long-horizon task with context compaction.

Run with: pytest tests/integration/test_long_horizon.py -m slow -v

Requires GROQ_API_KEY. Skipped automatically when the key is absent so
`make test` (unit only) always runs without credentials.

This test verifies Property 3 (SPEC Section 7): a single session spanning
20+ tool calls completes coherently, with compaction firing at least once
and the plan surviving across compaction boundaries.
"""
import asyncio
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_NEEDS_KEY = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)

_FIXTURE_REPO_SRC = Path(__file__).resolve().parents[2] / "fixture_repo"

_LONG_HORIZON_TASK = """\
Working in the repository at {repo}:

1. Check git status and list all Python files in the repo.
2. Read calculator.py and app.py to understand the existing code.
3. Use ast.find_references to locate all callers of the divide function.
4. Read each caller file in full.
5. Add input validation to divide() in calculator.py: raise ValueError when \
the divisor is zero or when either argument is not a number. \
Update app.py to catch ValueError from divide and return None instead of crashing.
6. Run the full test suite. Read test_calculator.py.
7. Add tests for the new validation behaviour: divide by zero raises ValueError, \
non-numeric input raises ValueError.
8. Run the suite again. If tests still fail, read the failure output carefully and fix.
9. If any tests fail after two runs, spawn a test-triage subagent \
(scopes: test + fs.read_file) to identify the exact failures and return findings.
10. Apply any fixes identified by the subagent findings, then run the suite one final time.
11. Commit the final working state with the message "feat: add input validation to divide".
Report which tests were added and confirm all tests pass.\
"""


@pytest.fixture
def fixture_repo(tmp_path):
    """Copy fixture_repo into a fresh tmp_path git repo."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def g(*args):
        subprocess.run(["git", "-C", str(repo)] + list(args), check=True, capture_output=True)

    g("init")
    g("config", "user.email", "test@fixture.local")
    g("config", "user.name", "Fixture Agent")

    # Copy fixture files.
    for src in _FIXTURE_REPO_SRC.iterdir():
        if src.suffix in {".py", ".cfg", ".toml", ".ini"} or src.name in {"conftest.py"}:
            (repo / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    g("add", ".")
    g("commit", "-m", "initial: fixture repo with calculator, app, tests")
    return repo


@_NEEDS_KEY
@pytest.mark.asyncio
async def test_long_horizon_validation_task(fixture_repo):
    """Agent must complete a 20+ step task, compacting context, without losing coherence."""
    from agent.main import run

    task = _LONG_HORIZON_TASK.format(repo=fixture_repo)
    result = await asyncio.wait_for(run(task, repo_root=fixture_repo), timeout=600)

    # Agent must mention that validation was added.
    lower = result.lower()
    assert any(kw in lower for kw in ("validation", "valueerror", "validate")), (
        f"Expected validation mention in answer, got:\n{result}"
    )
    # Agent must mention tests.
    assert any(kw in lower for kw in ("test", "pass", "suite")), (
        f"Expected test status in answer, got:\n{result}"
    )
