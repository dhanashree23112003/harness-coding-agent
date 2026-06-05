import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
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


# ── nested types ───────────────────────────────────────────────────────────────

def test_test_item_roundtrip():
    item = TestItem(
        node_id="tests/unit/test_fs.py::test_read", file="tests/unit/test_fs.py",
        name="test_read", markers=["unit"],
    )
    assert TestItem(**item.model_dump()) == item


def test_test_result_roundtrip():
    r = TestResult(node_id="tests/unit/test_fs.py::test_read", outcome="passed", duration_s=0.1)
    assert r.longrepr is None
    assert TestResult(**r.model_dump()) == r


def test_test_result_failure_longrepr():
    r = TestResult(
        node_id="tests/unit/test_fs.py::test_read",
        outcome="failed", duration_s=0.2,
        longrepr="AssertionError: expected True",
    )
    assert r.outcome == "failed"


def test_coverage_module_roundtrip():
    m = CoverageModule(path="src/agent/graph/nodes.py", stmts=100, miss=10, cover_pct=90.0)
    assert CoverageModule(**m.model_dump()) == m


def test_test_failure_roundtrip():
    f = TestFailure(
        test_id="tests/unit/test_fs.py::test_read",
        file="tests/unit/test_fs.py", line=52,
        error_text="AssertionError: expected 42 but got 0",
    )
    assert TestFailure(**f.model_dump()) == f


def test_test_failure_composition_fields():
    f = TestFailure(
        test_id="tests/unit/test_x.py::test_y",
        file="tests/unit/test_x.py", line=10,
        error_text="AssertionError: values differ",
    )
    # error_text and file are the fields that feed fs.grep
    assert f.error_text  # non-empty grep pattern
    assert f.file        # grep scope


# ── discover_tests ─────────────────────────────────────────────────────────────

def test_discover_tests_input_requires_root():
    with pytest.raises(ValidationError):
        DiscoverTestsInput()  # type: ignore[call-arg]


def test_discover_tests_input_defaults():
    inp = DiscoverTestsInput(root="/repo")
    assert inp.paths == []
    assert inp.markers == []


def test_discover_tests_input_accepts_changed_file_paths():
    paths = ["/repo/src/agent/graph/nodes.py", "/repo/src/agent/models/fs.py"]
    inp = DiscoverTestsInput(root="/repo", paths=paths)
    assert inp.paths == paths


def test_discover_tests_output_roundtrip():
    out = DiscoverTestsOutput(root="/repo", tests=[], count=0)
    assert DiscoverTestsOutput(**out.model_dump()) == out


# ── run_test_file ──────────────────────────────────────────────────────────────

def test_run_test_file_input_requires_fields():
    with pytest.raises(ValidationError):
        RunTestFileInput(path="tests/unit/test_fs.py")  # missing cwd


def test_run_test_file_input_defaults():
    inp = RunTestFileInput(path="tests/unit/test_fs.py", cwd="/repo")
    assert inp.timeout_s == 120


def test_run_test_file_output_roundtrip():
    out = RunTestFileOutput(
        path="tests/unit/test_fs.py", passed=5, failed=0, errors=0,
        skipped=0, results=[], duration_s=1.0,
    )
    assert RunTestFileOutput(**out.model_dump()) == out


def test_run_test_file_output_error_field():
    out = RunTestFileOutput(
        path="tests/unit/test_fs.py", passed=0, failed=0, errors=0,
        skipped=0, results=[], duration_s=0.0, success=False, error="timeout",
    )
    assert out.error == "timeout"


# ── run_test_node ──────────────────────────────────────────────────────────────

def test_run_test_node_input_defaults():
    inp = RunTestNodeInput(node_id="tests/unit/test_fs.py::test_read", cwd="/repo")
    assert inp.timeout_s == 60


def test_run_test_node_output_roundtrip():
    out = RunTestNodeOutput(
        node_id="tests/unit/test_fs.py::test_read", outcome="passed", duration_s=0.05,
    )
    assert RunTestNodeOutput(**out.model_dump()) == out


# ── run_suite ──────────────────────────────────────────────────────────────────

def test_run_suite_input_requires_cwd():
    with pytest.raises(ValidationError):
        RunSuiteInput()  # type: ignore[call-arg]


def test_run_suite_input_defaults():
    inp = RunSuiteInput(cwd="/repo")
    assert inp.paths == []
    assert inp.markers == []
    assert inp.timeout_s == 300


def test_run_suite_output_roundtrip():
    out = RunSuiteOutput(
        cwd="/repo", passed=10, failed=2, errors=0, skipped=1, results=[], duration_s=5.0,
    )
    assert RunSuiteOutput(**out.model_dump()) == out


# ── coverage_report ────────────────────────────────────────────────────────────

def test_coverage_report_input_defaults():
    inp = CoverageReportInput(cwd="/repo")
    assert inp.paths == []
    assert inp.min_cover == 0.0


def test_coverage_report_output_roundtrip():
    out = CoverageReportOutput(
        cwd="/repo", total_cover_pct=85.0, modules=[], below_threshold=[],
    )
    assert CoverageReportOutput(**out.model_dump()) == out


# ── coverage_diff ──────────────────────────────────────────────────────────────

def test_coverage_diff_input_defaults():
    inp = CoverageDiffInput(cwd="/repo")
    assert inp.base_ref == "HEAD~1"


def test_coverage_diff_output_roundtrip():
    out = CoverageDiffOutput(
        cwd="/repo", base_ref="HEAD~1", added_cover_pct=2.0,
        dropped_cover_pct=0.0, changed_modules=[],
    )
    assert CoverageDiffOutput(**out.model_dump()) == out


# ── last_failures ──────────────────────────────────────────────────────────────

def test_last_failures_input_defaults():
    inp = LastFailuresInput(cwd="/repo")
    assert inp.max_failures == 20


def test_last_failures_output_roundtrip():
    out = LastFailuresOutput(cwd="/repo", failures=[], count=0)
    assert LastFailuresOutput(**out.model_dump()) == out


def test_last_failures_output_error_field():
    out = LastFailuresOutput(
        cwd="/repo", failures=[], count=0, success=False, error="no cache found",
    )
    assert out.error == "no cache found"


# ── rerun_failed ───────────────────────────────────────────────────────────────

def test_rerun_failed_input_defaults():
    inp = RerunFailedInput(cwd="/repo")
    assert inp.timeout_s == 300


def test_rerun_failed_output_roundtrip():
    out = RerunFailedOutput(cwd="/repo", passed=3, failed=1, results=[])
    assert RerunFailedOutput(**out.model_dump()) == out
