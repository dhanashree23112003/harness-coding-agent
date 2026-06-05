"""Pydantic I/O models for the test namespace (test runner, 8 tools)."""
from typing import Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class TestItem(BaseModel):
    """A collected pytest test node."""
    node_id: str   # e.g. "tests/unit/test_fs_models.py::test_read_file_output_fields"
    file: str
    name: str
    markers: list[str]


class TestResult(BaseModel):
    node_id: str
    outcome: str   # "passed" | "failed" | "error" | "skipped"
    duration_s: float
    longrepr: Optional[str] = None  # failure/error text


class CoverageModule(BaseModel):
    path: str
    stmts: int
    miss: int
    cover_pct: float


class TestFailure(BaseModel):
    """A single test failure. error_text feeds fs.grep to locate failing assertions.

    Map: failure.error_text -> GrepInput.pattern, failure.file -> GrepInput.root.
    """
    test_id: str
    file: str
    line: int
    error_text: str  # first AssertionError / E-prefixed line from longrepr


# ── discover_tests ────────────────────────────────────────────────────────────

class DiscoverTestsInput(BaseModel):
    root: str = Field(..., description="Repository root to run pytest from")
    paths: list[str] = Field(
        [],
        description=(
            "Specific paths to discover tests in. "
            "Accepts file paths from git.list_changed_files output."
        ),
    )
    markers: list[str] = Field([], description="pytest marker expressions to filter tests")


class DiscoverTestsOutput(BaseModel):
    root: str
    tests: list[TestItem]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── run_test_file ─────────────────────────────────────────────────────────────

class RunTestFileInput(BaseModel):
    path: str = Field(..., description="Path to the test file to run")
    cwd: str = Field(..., description="Working directory for the pytest invocation")
    timeout_s: int = Field(120, description="Maximum seconds to wait for the run")


class RunTestFileOutput(BaseModel):
    path: str
    passed: int
    failed: int
    errors: int
    skipped: int
    results: list[TestResult]
    duration_s: float
    success: bool = True
    error: Optional[str] = None


# ── run_test_node ─────────────────────────────────────────────────────────────

class RunTestNodeInput(BaseModel):
    node_id: str = Field(..., description="Pytest node ID, e.g. 'tests/unit/test_fs.py::test_read'")
    cwd: str = Field(..., description="Working directory for the pytest invocation")
    timeout_s: int = Field(60, description="Maximum seconds to wait")


class RunTestNodeOutput(BaseModel):
    node_id: str
    outcome: str
    longrepr: Optional[str] = None
    duration_s: float
    success: bool = True
    error: Optional[str] = None


# ── run_suite ─────────────────────────────────────────────────────────────────

class RunSuiteInput(BaseModel):
    cwd: str = Field(..., description="Working directory and implicit test root")
    paths: list[str] = Field([], description="Sub-paths to restrict the run (empty = all)")
    markers: list[str] = Field([], description="Marker filter expressions")
    timeout_s: int = Field(300, description="Maximum seconds to wait for the suite")


class RunSuiteOutput(BaseModel):
    cwd: str
    passed: int
    failed: int
    errors: int
    skipped: int
    results: list[TestResult]
    duration_s: float
    success: bool = True
    error: Optional[str] = None


# ── coverage_report ───────────────────────────────────────────────────────────

class CoverageReportInput(BaseModel):
    cwd: str = Field(..., description="Working directory for the pytest invocation")
    paths: list[str] = Field([], description="Source paths to measure coverage for (empty = all)")
    min_cover: float = Field(0.0, description="Flag modules below this coverage percentage")


class CoverageReportOutput(BaseModel):
    cwd: str
    total_cover_pct: float
    modules: list[CoverageModule]
    below_threshold: list[str]
    success: bool = True
    error: Optional[str] = None


# ── coverage_diff ─────────────────────────────────────────────────────────────

class CoverageDiffInput(BaseModel):
    cwd: str = Field(..., description="Working directory (git repo root)")
    base_ref: str = Field("HEAD~1", description="Git ref to compare coverage against")


class CoverageDiffOutput(BaseModel):
    cwd: str
    base_ref: str
    added_cover_pct: float   # positive = coverage improved
    dropped_cover_pct: float  # positive = coverage dropped
    changed_modules: list[CoverageModule]
    success: bool = True
    error: Optional[str] = None


# ── last_failures ─────────────────────────────────────────────────────────────

class LastFailuresInput(BaseModel):
    cwd: str = Field(..., description="Working directory (must contain .pytest_cache)")
    max_failures: int = Field(20, description="Maximum number of failures to return")


class LastFailuresOutput(BaseModel):
    """Output failures feed fs.grep to locate failing assertions.

    Map: failure.error_text -> GrepInput.pattern, failure.file -> search scope.
    """
    cwd: str
    failures: list[TestFailure]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── rerun_failed ──────────────────────────────────────────────────────────────

class RerunFailedInput(BaseModel):
    cwd: str = Field(..., description="Working directory with .pytest_cache from a prior run")
    timeout_s: int = Field(300, description="Maximum seconds to wait")


class RerunFailedOutput(BaseModel):
    cwd: str
    passed: int
    failed: int
    results: list[TestResult]
    success: bool = True
    error: Optional[str] = None
