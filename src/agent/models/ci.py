"""Pydantic I/O models for the ci namespace (quality / CI tools, 7 tools)."""
from typing import Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class LintViolation(BaseModel):
    file: str
    line: int
    col: int
    code: str
    message: str
    severity: str  # "error" | "warning"


class TypeCheckError(BaseModel):
    file: str
    line: int
    col: int
    message: str
    error_code: str


class SecurityFinding(BaseModel):
    file: str
    line: int
    test_id: str
    severity: str    # "HIGH" | "MEDIUM" | "LOW"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    message: str
    cwe: Optional[str] = None


class QualitySummary(BaseModel):
    lint_errors: int
    lint_warnings: int
    type_errors: int
    security_high: int
    security_medium: int
    security_low: int
    overall_ok: bool


# ── run_linter ────────────────────────────────────────────────────────────────

class RunLinterInput(BaseModel):
    path: str = Field(..., description="File or directory to lint")
    fix: bool = Field(False, description="Apply auto-fixes where possible")
    select: list[str] = Field([], description="Rule codes to enable (e.g. ['E', 'W'])")
    ignore: list[str] = Field([], description="Rule codes to suppress")


class RunLinterOutput(BaseModel):
    path: str
    violations: list[LintViolation]
    error_count: int
    warning_count: int
    fixed: int
    success: bool = True
    error: Optional[str] = None


# ── run_formatter ─────────────────────────────────────────────────────────────

class RunFormatterInput(BaseModel):
    path: str = Field(..., description="File or directory to format")
    check_only: bool = Field(True, description="When True, report without writing changes")


class RunFormatterOutput(BaseModel):
    path: str
    changed_files: list[str]
    already_formatted: bool
    success: bool = True
    error: Optional[str] = None


# ── run_type_check ────────────────────────────────────────────────────────────

class RunTypeCheckInput(BaseModel):
    path: str = Field(..., description="File or package to type-check")
    strict: bool = Field(False, description="Enable mypy strict mode")


class RunTypeCheckOutput(BaseModel):
    path: str
    errors: list[TypeCheckError]
    error_count: int
    success: bool = True
    error: Optional[str] = None


# ── build_check ───────────────────────────────────────────────────────────────

class BuildCheckInput(BaseModel):
    project_root: str = Field(..., description="Root directory containing pyproject.toml")


class BuildCheckOutput(BaseModel):
    project_root: str
    build_ok: bool
    artifacts: list[str]
    success: bool = True
    error: Optional[str] = None


# ── pre_commit_run ────────────────────────────────────────────────────────────

class PreCommitRunInput(BaseModel):
    project_root: str = Field(..., description="Repository root containing .pre-commit-config.yaml")
    all_files: bool = Field(False, description="Run hooks against all files, not just staged")
    hook_ids: list[str] = Field([], description="Run only these specific hook IDs (empty = all)")


class PreCommitRunOutput(BaseModel):
    project_root: str
    passed: list[str]
    failed: list[str]
    skipped: list[str]
    success: bool = True
    error: Optional[str] = None


# ── run_security_scan ─────────────────────────────────────────────────────────

class RunSecurityScanInput(BaseModel):
    path: str = Field(..., description="File or directory to scan")
    severity: str = Field("MEDIUM", description="Minimum severity to report: LOW, MEDIUM, HIGH")
    confidence: str = Field("MEDIUM", description="Minimum confidence to report: LOW, MEDIUM, HIGH")


class RunSecurityScanOutput(BaseModel):
    path: str
    findings: list[SecurityFinding]
    high_count: int
    medium_count: int
    low_count: int
    success: bool = True
    error: Optional[str] = None


# ── summarize_quality ─────────────────────────────────────────────────────────

class SummarizeQualityInput(BaseModel):
    project_root: str = Field(..., description="Root directory of the project")
    src_path: str = Field(..., description="Source directory to analyse (relative to project_root)")


class SummarizeQualityOutput(BaseModel):
    project_root: str
    summary: QualitySummary
    success: bool = True
    error: Optional[str] = None
