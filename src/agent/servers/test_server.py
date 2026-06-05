"""MCP server for the test namespace. All 8 pytest-runner tools."""
import json
import re
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.test import (  # noqa: E402
    CoverageDiffInput,
    CoverageDiffOutput,
    CoverageModule,
    CoverageReportInput,
    CoverageReportOutput,
    DiscoverTestsInput,
    DiscoverTestsOutput,
    LastFailuresInput,
    LastFailuresOutput,
    RerunFailedInput,
    RerunFailedOutput,
    RunSuiteInput,
    RunSuiteOutput,
    RunTestFileInput,
    RunTestFileOutput,
    RunTestNodeInput,
    RunTestNodeOutput,
    TestFailure,
    TestItem,
    TestResult,
)

mcp = FastMCP("test")

import subprocess  # noqa: E402


def _pytest(args: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest"] + args,
        cwd=cwd, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=timeout,
    )


# Matches verbose pytest output: "tests/unit/test_foo.py::test_bar PASSED   [ 50%]"
_RESULT_RE = re.compile(r"^([\w/\\:.]+::[\w\[\]]+)\s+(PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)
# Matches collect-only output lines for test nodes
_COLLECT_RE = re.compile(r"^([\w/\\:.]+::[\w\[\]]+)", re.MULTILINE)


def _parse_results(output: str) -> list[TestResult]:
    results = []
    for m in _RESULT_RE.finditer(output):
        node_id, outcome = m.group(1), m.group(2).lower()
        results.append(TestResult(node_id=node_id, outcome=outcome, duration_s=0.0))
    return results


def _count_summary(output: str) -> tuple[int, int, int, int]:
    """Return (passed, failed, errors, skipped) from the pytest summary line."""
    m = re.search(
        r"(\d+) passed|(\d+) failed|(\d+) error|(\d+) skipped",
        output,
    )
    passed = int(re.search(r"(\d+) passed", output).group(1)) if re.search(r"\d+ passed", output) else 0
    failed = int(re.search(r"(\d+) failed", output).group(1)) if re.search(r"\d+ failed", output) else 0
    errors = int(re.search(r"(\d+) error", output).group(1)) if re.search(r"\d+ error", output) else 0
    skipped = int(re.search(r"(\d+) skipped", output).group(1)) if re.search(r"\d+ skipped", output) else 0
    return passed, failed, errors, skipped


# ── discover_tests ─────────────────────────────────────────────────────────────

@mcp.tool()
def discover_tests(root: str, paths: list[str] = [], markers: list[str] = []) -> dict:  # noqa: B006
    """Discover all pytest tests under root (or restricted to paths).

    The paths parameter accepts file paths from git.list_changed_files output,
    so the agent can test only files that changed.
    """
    req = DiscoverTestsInput(root=root, paths=paths, markers=markers)
    try:
        args = ["--collect-only", "-q", "--no-header"]
        if req.paths:
            args += req.paths
        if req.markers:
            args += ["-m", " or ".join(req.markers)]
        result = _pytest(args, cwd=req.root, timeout=60)
        tests: list[TestItem] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            m = re.match(r"([\w/\\:.]+::[^\s]+)", line)
            if m:
                node_id = m.group(1)
                parts = node_id.split("::")
                file_part = parts[0]
                name_part = "::".join(parts[1:]) if len(parts) > 1 else node_id
                tests.append(TestItem(node_id=node_id, file=file_part, name=name_part, markers=[]))
        return DiscoverTestsOutput(root=req.root, tests=tests, count=len(tests)).model_dump()
    except Exception as exc:
        return DiscoverTestsOutput(root=req.root, tests=[], count=0, success=False, error=str(exc)).model_dump()


# ── run_test_file ──────────────────────────────────────────────────────────────

@mcp.tool()
def run_test_file(path: str, cwd: str, timeout_s: int = 120) -> dict:
    """Run all tests in a single test file and return structured pass/fail results."""
    req = RunTestFileInput(path=path, cwd=cwd, timeout_s=timeout_s)
    try:
        t0 = time.monotonic()
        result = _pytest([req.path, "-v", "--tb=short", "--no-header"], cwd=req.cwd, timeout=req.timeout_s)
        duration = time.monotonic() - t0
        passed, failed, errors, skipped = _count_summary(result.stdout)
        results = _parse_results(result.stdout)
        return RunTestFileOutput(
            path=req.path, passed=passed, failed=failed, errors=errors,
            skipped=skipped, results=results, duration_s=round(duration, 3),
        ).model_dump()
    except subprocess.TimeoutExpired:
        return RunTestFileOutput(
            path=req.path, passed=0, failed=0, errors=0, skipped=0,
            results=[], duration_s=float(req.timeout_s),
            success=False, error=f"timeout after {req.timeout_s}s",
        ).model_dump()
    except Exception as exc:
        return RunTestFileOutput(
            path=req.path, passed=0, failed=0, errors=0, skipped=0,
            results=[], duration_s=0.0, success=False, error=str(exc),
        ).model_dump()


# ── run_test_node ──────────────────────────────────────────────────────────────

@mcp.tool()
def run_test_node(node_id: str, cwd: str, timeout_s: int = 60) -> dict:
    """Run a single pytest test node by its node ID and return the outcome."""
    req = RunTestNodeInput(node_id=node_id, cwd=cwd, timeout_s=timeout_s)
    try:
        t0 = time.monotonic()
        result = _pytest([req.node_id, "-v", "--tb=short", "--no-header"], cwd=req.cwd, timeout=req.timeout_s)
        duration = time.monotonic() - t0
        passed, failed, errors, _ = _count_summary(result.stdout)
        if failed or errors:
            outcome = "failed"
            longrepr = result.stdout[-3000:]
        else:
            outcome = "passed"
            longrepr = None
        return RunTestNodeOutput(
            node_id=req.node_id, outcome=outcome,
            longrepr=longrepr, duration_s=round(duration, 3),
        ).model_dump()
    except subprocess.TimeoutExpired:
        return RunTestNodeOutput(
            node_id=req.node_id, outcome="error",
            longrepr=f"timeout after {req.timeout_s}s",
            duration_s=float(req.timeout_s),
            success=False, error=f"timeout after {req.timeout_s}s",
        ).model_dump()
    except Exception as exc:
        return RunTestNodeOutput(
            node_id=req.node_id, outcome="error",
            longrepr=str(exc), duration_s=0.0,
            success=False, error=str(exc),
        ).model_dump()


# ── run_suite ──────────────────────────────────────────────────────────────────

@mcp.tool()
def run_suite(cwd: str, paths: list[str] = [], markers: list[str] = [], timeout_s: int = 300) -> dict:  # noqa: B006
    """Run the full test suite (or a restricted subset) and return aggregate results."""
    req = RunSuiteInput(cwd=cwd, paths=paths, markers=markers, timeout_s=timeout_s)
    try:
        args = ["-v", "--tb=short", "--no-header"]
        if req.paths:
            args += req.paths
        if req.markers:
            args += ["-m", " or ".join(req.markers)]
        t0 = time.monotonic()
        result = _pytest(args, cwd=req.cwd, timeout=req.timeout_s)
        duration = time.monotonic() - t0
        passed, failed, errors, skipped = _count_summary(result.stdout)
        results = _parse_results(result.stdout)
        return RunSuiteOutput(
            cwd=req.cwd, passed=passed, failed=failed, errors=errors,
            skipped=skipped, results=results, duration_s=round(duration, 3),
        ).model_dump()
    except subprocess.TimeoutExpired:
        return RunSuiteOutput(
            cwd=req.cwd, passed=0, failed=0, errors=0, skipped=0,
            results=[], duration_s=float(req.timeout_s),
            success=False, error=f"timeout after {req.timeout_s}s",
        ).model_dump()
    except Exception as exc:
        return RunSuiteOutput(
            cwd=req.cwd, passed=0, failed=0, errors=0, skipped=0,
            results=[], duration_s=0.0, success=False, error=str(exc),
        ).model_dump()


# ── coverage_report ────────────────────────────────────────────────────────────

@mcp.tool()
def coverage_report(cwd: str, paths: list[str] = [], min_cover: float = 0.0) -> dict:  # noqa: B006
    """Run pytest with coverage and return per-module coverage stats."""
    req = CoverageReportInput(cwd=cwd, paths=paths, min_cover=min_cover)
    try:
        cov_args = []
        if req.paths:
            for p in req.paths:
                cov_args += [f"--cov={p}"]
        else:
            cov_args = ["--cov=."]
        args = cov_args + ["--cov-report=json", "-q", "--no-header"]
        result = _pytest(args, cwd=req.cwd, timeout=300)
        cov_json = Path(req.cwd) / "coverage.json"
        if cov_json.exists():
            data = json.loads(cov_json.read_text())
            total = data.get("totals", {}).get("percent_covered", 0.0)
            modules: list[CoverageModule] = []
            for fpath, fdata in data.get("files", {}).items():
                summary = fdata.get("summary", {})
                pct = summary.get("percent_covered", 0.0)
                modules.append(CoverageModule(
                    path=fpath,
                    stmts=summary.get("num_statements", 0),
                    miss=summary.get("missing_lines", 0),
                    cover_pct=pct,
                ))
            below = [m.path for m in modules if m.cover_pct < req.min_cover]
            return CoverageReportOutput(
                cwd=req.cwd, total_cover_pct=total, modules=modules, below_threshold=below,
            ).model_dump()
        # No JSON file: try to parse terminal output
        m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
        total = float(m.group(1)) if m else 0.0
        return CoverageReportOutput(
            cwd=req.cwd, total_cover_pct=total, modules=[], below_threshold=[],
        ).model_dump()
    except Exception as exc:
        return CoverageReportOutput(
            cwd=req.cwd, total_cover_pct=0.0, modules=[], below_threshold=[],
            success=False, error=str(exc),
        ).model_dump()


# ── coverage_diff ──────────────────────────────────────────────────────────────

@mcp.tool()
def coverage_diff(cwd: str, base_ref: str = "HEAD~1") -> dict:
    """Compare test coverage between the current HEAD and a base git ref."""
    req = CoverageDiffInput(cwd=cwd, base_ref=base_ref)
    try:
        import subprocess as sp

        def _run_coverage(ref: str | None) -> float:
            if ref:
                sp.run(
                    ["git", "stash"], cwd=req.cwd, capture_output=True, stdin=subprocess.DEVNULL,
                )
                sp.run(
                    ["git", "checkout", ref], cwd=req.cwd, capture_output=True, stdin=subprocess.DEVNULL,
                )
            result = _pytest(["--cov=.", "--cov-report=json", "-q", "--no-header"], cwd=req.cwd, timeout=300)
            cov_json = Path(req.cwd) / "coverage.json"
            if cov_json.exists():
                data = json.loads(cov_json.read_text())
                return data.get("totals", {}).get("percent_covered", 0.0)
            m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
            return float(m.group(1)) if m else 0.0

        current = _run_coverage(None)
        base = _run_coverage(req.base_ref)
        sp.run(["git", "stash", "pop"], cwd=req.cwd, capture_output=True, stdin=subprocess.DEVNULL)
        added = max(0.0, current - base)
        dropped = max(0.0, base - current)
        return CoverageDiffOutput(
            cwd=req.cwd, base_ref=req.base_ref,
            added_cover_pct=added, dropped_cover_pct=dropped, changed_modules=[],
        ).model_dump()
    except Exception as exc:
        return CoverageDiffOutput(
            cwd=req.cwd, base_ref=req.base_ref,
            added_cover_pct=0.0, dropped_cover_pct=0.0, changed_modules=[],
            success=False, error=str(exc),
        ).model_dump()


# ── last_failures ──────────────────────────────────────────────────────────────

@mcp.tool()
def last_failures(cwd: str, max_failures: int = 20) -> dict:
    """Return structured info about tests that failed in the most recent pytest run.

    Output failures feed fs.grep to locate failing assertions:
    map failure.error_text -> GrepInput.pattern, failure.file -> search scope.
    """
    req = LastFailuresInput(cwd=cwd, max_failures=max_failures)
    try:
        cache_file = Path(req.cwd) / ".pytest_cache" / "v" / "cache" / "lastfailed"
        if not cache_file.exists():
            return LastFailuresOutput(cwd=req.cwd, failures=[], count=0).model_dump()

        last_failed_map: dict = json.loads(cache_file.read_text())
        node_ids = list(last_failed_map.keys())[: req.max_failures]
        if not node_ids:
            return LastFailuresOutput(cwd=req.cwd, failures=[], count=0).model_dump()

        result = _pytest(node_ids + ["--tb=long", "--no-header"], cwd=req.cwd, timeout=120)
        failures: list[TestFailure] = []
        current_node: str | None = None
        longrepr_lines: list[str] = []

        for line in result.stdout.splitlines():
            node_match = re.match(r"^FAILED (.+) -", line)
            if node_match:
                current_node = node_match.group(1).strip()
            e_match = re.match(r"^E\s+(.+)", line)
            if e_match and current_node:
                error_text = e_match.group(1).strip()
                parts = current_node.split("::")
                file_part = parts[0]
                line_m = re.search(r":(\d+):", current_node)
                line_num = int(line_m.group(1)) if line_m else 0
                failures.append(TestFailure(
                    test_id=current_node, file=file_part,
                    line=line_num, error_text=error_text,
                ))
                current_node = None

        # Fallback: produce failures from node_ids without error_text
        if not failures:
            for node_id in node_ids:
                parts = node_id.split("::")
                failures.append(TestFailure(
                    test_id=node_id, file=parts[0], line=0, error_text="",
                ))

        return LastFailuresOutput(cwd=req.cwd, failures=failures, count=len(failures)).model_dump()
    except Exception as exc:
        return LastFailuresOutput(
            cwd=req.cwd, failures=[], count=0, success=False, error=str(exc),
        ).model_dump()


# ── rerun_failed ───────────────────────────────────────────────────────────────

@mcp.tool()
def rerun_failed(cwd: str, timeout_s: int = 300) -> dict:
    """Rerun the tests that failed in the most recent pytest run."""
    req = RerunFailedInput(cwd=cwd, timeout_s=timeout_s)
    try:
        t0 = time.monotonic()
        result = _pytest(["--lf", "-v", "--tb=short", "--no-header"], cwd=req.cwd, timeout=req.timeout_s)
        duration = time.monotonic() - t0
        passed, failed, _, _ = _count_summary(result.stdout)
        results = _parse_results(result.stdout)
        return RerunFailedOutput(
            cwd=req.cwd, passed=passed, failed=failed, results=results,
        ).model_dump()
    except subprocess.TimeoutExpired:
        return RerunFailedOutput(
            cwd=req.cwd, passed=0, failed=0, results=[],
            success=False, error=f"timeout after {req.timeout_s}s",
        ).model_dump()
    except Exception as exc:
        return RerunFailedOutput(
            cwd=req.cwd, passed=0, failed=0, results=[],
            success=False, error=str(exc),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
