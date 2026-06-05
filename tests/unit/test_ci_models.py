import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
    BuildCheckInput,
    BuildCheckOutput,
    LintViolation,
    PreCommitRunInput,
    PreCommitRunOutput,
    QualitySummary,
    RunFormatterInput,
    RunFormatterOutput,
    RunLinterInput,
    RunLinterOutput,
    RunSecurityScanInput,
    RunSecurityScanOutput,
    RunTypeCheckInput,
    RunTypeCheckOutput,
    SecurityFinding,
    SummarizeQualityInput,
    SummarizeQualityOutput,
    TypeCheckError,
)


# ── nested types ───────────────────────────────────────────────────────────────

def test_lint_violation_roundtrip():
    v = LintViolation(file="src/a.py", line=10, col=4, code="E501", message="line too long", severity="error")
    assert LintViolation(**v.model_dump()) == v


def test_type_check_error_roundtrip():
    e = TypeCheckError(file="src/a.py", line=5, col=0, message='Incompatible types', error_code="arg-type")
    assert TypeCheckError(**e.model_dump()) == e


def test_security_finding_roundtrip():
    f = SecurityFinding(
        file="src/a.py", line=20, test_id="B301",
        severity="HIGH", confidence="MEDIUM",
        message="Use of pickle detected", cwe="CWE-502",
    )
    assert SecurityFinding(**f.model_dump()) == f


def test_security_finding_no_cwe():
    f = SecurityFinding(
        file="src/a.py", line=20, test_id="B301",
        severity="MEDIUM", confidence="LOW",
        message="some finding",
    )
    assert f.cwe is None


def test_quality_summary_roundtrip():
    s = QualitySummary(
        lint_errors=2, lint_warnings=5, type_errors=1,
        security_high=0, security_medium=1, security_low=3,
        overall_ok=False,
    )
    assert QualitySummary(**s.model_dump()) == s


def test_quality_summary_overall_ok():
    s = QualitySummary(
        lint_errors=0, lint_warnings=0, type_errors=0,
        security_high=0, security_medium=0, security_low=0,
        overall_ok=True,
    )
    assert s.overall_ok is True


# ── run_linter ─────────────────────────────────────────────────────────────────

def test_run_linter_input_requires_path():
    with pytest.raises(ValidationError):
        RunLinterInput()  # type: ignore[call-arg]


def test_run_linter_input_defaults():
    inp = RunLinterInput(path="src/")
    assert inp.fix is False
    assert inp.select == []
    assert inp.ignore == []


def test_run_linter_output_roundtrip():
    out = RunLinterOutput(path="src/", violations=[], error_count=0, warning_count=0, fixed=0)
    assert RunLinterOutput(**out.model_dump()) == out


def test_run_linter_output_error_field():
    out = RunLinterOutput(
        path="src/", violations=[], error_count=0, warning_count=0, fixed=0,
        success=False, error="ruff not found",
    )
    assert out.error == "ruff not found"


# ── run_formatter ──────────────────────────────────────────────────────────────

def test_run_formatter_input_defaults():
    inp = RunFormatterInput(path="src/")
    assert inp.check_only is True


def test_run_formatter_output_roundtrip():
    out = RunFormatterOutput(path="src/", changed_files=[], already_formatted=True)
    assert RunFormatterOutput(**out.model_dump()) == out


def test_run_formatter_output_with_changes():
    out = RunFormatterOutput(path="src/", changed_files=["src/a.py"], already_formatted=False)
    assert out.already_formatted is False
    assert len(out.changed_files) == 1


# ── run_type_check ─────────────────────────────────────────────────────────────

def test_run_type_check_input_defaults():
    inp = RunTypeCheckInput(path="src/")
    assert inp.strict is False


def test_run_type_check_output_roundtrip():
    out = RunTypeCheckOutput(path="src/", errors=[], error_count=0)
    assert RunTypeCheckOutput(**out.model_dump()) == out


def test_run_type_check_output_error_field():
    out = RunTypeCheckOutput(
        path="src/", errors=[], error_count=0,
        success=False, error="mypy not found",
    )
    assert out.error == "mypy not found"


# ── build_check ────────────────────────────────────────────────────────────────

def test_build_check_input_requires_root():
    with pytest.raises(ValidationError):
        BuildCheckInput()  # type: ignore[call-arg]


def test_build_check_output_roundtrip():
    out = BuildCheckOutput(project_root="/repo", build_ok=True, artifacts=["dist/pkg-1.0-py3-none-any.whl"])
    assert BuildCheckOutput(**out.model_dump()) == out


def test_build_check_output_failure():
    out = BuildCheckOutput(
        project_root="/repo", build_ok=False, artifacts=[],
        success=False, error="build failed",
    )
    assert out.build_ok is False


# ── pre_commit_run ─────────────────────────────────────────────────────────────

def test_pre_commit_run_input_defaults():
    inp = PreCommitRunInput(project_root="/repo")
    assert inp.all_files is False
    assert inp.hook_ids == []


def test_pre_commit_run_output_roundtrip():
    out = PreCommitRunOutput(
        project_root="/repo",
        passed=["ruff", "mypy"], failed=[], skipped=["bandit"],
    )
    assert PreCommitRunOutput(**out.model_dump()) == out


# ── run_security_scan ──────────────────────────────────────────────────────────

def test_run_security_scan_input_defaults():
    inp = RunSecurityScanInput(path="src/")
    assert inp.severity == "MEDIUM"
    assert inp.confidence == "MEDIUM"


def test_run_security_scan_output_roundtrip():
    out = RunSecurityScanOutput(
        path="src/", findings=[], high_count=0, medium_count=0, low_count=0,
    )
    assert RunSecurityScanOutput(**out.model_dump()) == out


def test_run_security_scan_output_error_field():
    out = RunSecurityScanOutput(
        path="src/", findings=[], high_count=0, medium_count=0, low_count=0,
        success=False, error="bandit not found",
    )
    assert out.error == "bandit not found"


# ── summarize_quality ──────────────────────────────────────────────────────────

def test_summarize_quality_input_requires_fields():
    with pytest.raises(ValidationError):
        SummarizeQualityInput(project_root="/repo")  # missing src_path


def test_summarize_quality_output_roundtrip():
    summary = QualitySummary(
        lint_errors=0, lint_warnings=2, type_errors=0,
        security_high=0, security_medium=0, security_low=1,
        overall_ok=True,
    )
    out = SummarizeQualityOutput(project_root="/repo", summary=summary)
    assert SummarizeQualityOutput(**out.model_dump()) == out
