"""MCP server for the ci namespace. All 7 CI / quality tools."""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.ci import (  # noqa: E402
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

mcp = FastMCP("ci")


def _run(cmd: list[str], cwd: str | None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=timeout,
    )


def _require(tool: str) -> str | None:
    """Return the full path to tool, or None if not found."""
    return shutil.which(tool)


# ── run_linter ─────────────────────────────────────────────────────────────────

@mcp.tool()
def run_linter(path: str, fix: bool = False, select: list[str] = [], ignore: list[str] = []) -> dict:  # noqa: B006
    """Run ruff linter on a file or directory and return structured violations."""
    req = RunLinterInput(path=path, fix=fix, select=select, ignore=ignore)
    ruff = _require("ruff")
    if ruff is None:
        return RunLinterOutput(
            path=req.path, violations=[], error_count=0, warning_count=0, fixed=0,
            success=False, error="ruff not found; install with: pip install ruff",
        ).model_dump()
    try:
        args = [ruff, "check", req.path, "--output-format=json"]
        if req.fix:
            args.append("--fix")
        if req.select:
            args += ["--select", ",".join(req.select)]
        if req.ignore:
            args += ["--ignore", ",".join(req.ignore)]
        result = _run(args, cwd=None)
        try:
            raw = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        violations = []
        for item in raw:
            violations.append(LintViolation(
                file=item.get("filename", ""),
                line=item.get("location", {}).get("row", 0),
                col=item.get("location", {}).get("column", 0),
                code=item.get("code", ""),
                message=item.get("message", ""),
                severity="error" if item.get("code", "").startswith("E") else "warning",
            ))
        errors = sum(1 for v in violations if v.severity == "error")
        warnings = len(violations) - errors
        # Ruff reports fixed count in stderr: "Fixed X errors"
        fixed = 0
        m = re.search(r"Fixed (\d+)", result.stderr)
        if m:
            fixed = int(m.group(1))
        return RunLinterOutput(
            path=req.path, violations=violations,
            error_count=errors, warning_count=warnings, fixed=fixed,
        ).model_dump()
    except Exception as exc:
        return RunLinterOutput(
            path=req.path, violations=[], error_count=0, warning_count=0, fixed=0,
            success=False, error=str(exc),
        ).model_dump()


# ── run_formatter ──────────────────────────────────────────────────────────────

@mcp.tool()
def run_formatter(path: str, check_only: bool = True) -> dict:
    """Run ruff formatter on a file or directory; report or apply formatting changes."""
    req = RunFormatterInput(path=path, check_only=check_only)
    ruff = _require("ruff")
    if ruff is None:
        return RunFormatterOutput(
            path=req.path, changed_files=[], already_formatted=False,
            success=False, error="ruff not found; install with: pip install ruff",
        ).model_dump()
    try:
        args = [ruff, "format", req.path]
        if req.check_only:
            args.append("--check")
        result = _run(args, cwd=None)
        # Exit 0 = nothing to change (or changes applied); exit 1 with --check = would reformat
        changed_files: list[str] = []
        if req.check_only and result.returncode != 0:
            # Parse "Would reformat: <file>" lines
            for line in result.stderr.splitlines():
                m = re.match(r"Would reformat: (.+)", line)
                if m:
                    changed_files.append(m.group(1).strip())
        elif not req.check_only:
            # Parse "Reformatted <file>" lines
            for line in result.stderr.splitlines():
                m = re.match(r"Reformatted (.+)", line)
                if m:
                    changed_files.append(m.group(1).strip())
        already_formatted = len(changed_files) == 0
        return RunFormatterOutput(
            path=req.path, changed_files=changed_files, already_formatted=already_formatted,
        ).model_dump()
    except Exception as exc:
        return RunFormatterOutput(
            path=req.path, changed_files=[], already_formatted=False,
            success=False, error=str(exc),
        ).model_dump()


# ── run_type_check ─────────────────────────────────────────────────────────────

@mcp.tool()
def run_type_check(path: str, strict: bool = False) -> dict:
    """Run mypy type checker on a file or package and return structured errors."""
    req = RunTypeCheckInput(path=path, strict=strict)
    mypy = _require("mypy")
    if mypy is None:
        return RunTypeCheckOutput(
            path=req.path, errors=[], error_count=0,
            success=False, error="mypy not found; install with: pip install mypy",
        ).model_dump()
    try:
        args = [mypy, req.path, "--output=json"]
        if req.strict:
            args.append("--strict")
        result = _run(args, cwd=None)
        errors: list[TypeCheckError] = []
        for line in result.stdout.splitlines():
            try:
                item = json.loads(line)
                if item.get("severity") in ("error", "note") and item.get("severity") == "error":
                    errors.append(TypeCheckError(
                        file=item.get("file", ""),
                        line=item.get("line", 0),
                        col=item.get("column", 0),
                        message=item.get("message", ""),
                        error_code=item.get("error_code", ""),
                    ))
            except (json.JSONDecodeError, KeyError):
                # Fallback: parse plain text "file:line:col: error: message [code]"
                m = re.match(r"(.+):(\d+):(\d+): error: (.+?) +\[(.+)\]", line)
                if m:
                    errors.append(TypeCheckError(
                        file=m.group(1), line=int(m.group(2)), col=int(m.group(3)),
                        message=m.group(4), error_code=m.group(5),
                    ))
        return RunTypeCheckOutput(path=req.path, errors=errors, error_count=len(errors)).model_dump()
    except Exception as exc:
        return RunTypeCheckOutput(
            path=req.path, errors=[], error_count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── build_check ────────────────────────────────────────────────────────────────

@mcp.tool()
def build_check(project_root: str) -> dict:
    """Verify that the project builds cleanly (sdist + wheel via python -m build)."""
    req = BuildCheckInput(project_root=project_root)
    try:
        result = _run(
            [sys.executable, "-m", "build", "--outdir", "dist"],
            cwd=req.project_root, timeout=300,
        )
        build_ok = result.returncode == 0
        artifacts = []
        dist = Path(req.project_root) / "dist"
        if dist.exists():
            artifacts = [str(p.name) for p in dist.iterdir() if p.is_file()]
        return BuildCheckOutput(
            project_root=req.project_root, build_ok=build_ok, artifacts=artifacts,
        ).model_dump()
    except Exception as exc:
        return BuildCheckOutput(
            project_root=req.project_root, build_ok=False, artifacts=[],
            success=False, error=str(exc),
        ).model_dump()


# ── pre_commit_run ─────────────────────────────────────────────────────────────

@mcp.tool()
def pre_commit_run(project_root: str, all_files: bool = False, hook_ids: list[str] = []) -> dict:  # noqa: B006
    """Run pre-commit hooks and return which hooks passed, failed, or were skipped."""
    req = PreCommitRunInput(project_root=project_root, all_files=all_files, hook_ids=hook_ids)
    pc = _require("pre-commit")
    if pc is None:
        return PreCommitRunOutput(
            project_root=req.project_root, passed=[], failed=[], skipped=[],
            success=False, error="pre-commit not found; install with: pip install pre-commit",
        ).model_dump()
    try:
        args = [pc, "run"]
        if req.all_files:
            args.append("--all-files")
        args += req.hook_ids
        result = _run(args, cwd=req.project_root, timeout=300)
        passed, failed, skipped = [], [], []
        for line in result.stdout.splitlines():
            m = re.match(r"^(.+?)\s+\.+\s+(Passed|Failed|Skipped)", line)
            if m:
                hook, status = m.group(1).strip(), m.group(2)
                if status == "Passed":
                    passed.append(hook)
                elif status == "Failed":
                    failed.append(hook)
                else:
                    skipped.append(hook)
        return PreCommitRunOutput(
            project_root=req.project_root, passed=passed, failed=failed, skipped=skipped,
        ).model_dump()
    except Exception as exc:
        return PreCommitRunOutput(
            project_root=req.project_root, passed=[], failed=[], skipped=[],
            success=False, error=str(exc),
        ).model_dump()


# ── run_security_scan ──────────────────────────────────────────────────────────

@mcp.tool()
def run_security_scan(path: str, severity: str = "MEDIUM", confidence: str = "MEDIUM") -> dict:
    """Run bandit security scanner on a file or directory and return findings."""
    req = RunSecurityScanInput(path=path, severity=severity, confidence=confidence)
    bandit = _require("bandit")
    if bandit is None:
        return RunSecurityScanOutput(
            path=req.path, findings=[], high_count=0, medium_count=0, low_count=0,
            success=False, error="bandit not found; install with: pip install bandit",
        ).model_dump()
    try:
        result = _run(
            [bandit, "-r", req.path, "-f", "json",
             "-l", req.severity, "-i", req.confidence],
            cwd=None,
        )
        try:
            raw = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            raw = {}
        findings: list[SecurityFinding] = []
        for item in raw.get("results", []):
            findings.append(SecurityFinding(
                file=item.get("filename", ""),
                line=item.get("line_number", 0),
                test_id=item.get("test_id", ""),
                severity=item.get("issue_severity", ""),
                confidence=item.get("issue_confidence", ""),
                message=item.get("issue_text", ""),
                cwe=item.get("issue_cwe", {}).get("id") if item.get("issue_cwe") else None,
            ))
        high = sum(1 for f in findings if f.severity.upper() == "HIGH")
        medium = sum(1 for f in findings if f.severity.upper() == "MEDIUM")
        low = sum(1 for f in findings if f.severity.upper() == "LOW")
        return RunSecurityScanOutput(
            path=req.path, findings=findings,
            high_count=high, medium_count=medium, low_count=low,
        ).model_dump()
    except Exception as exc:
        return RunSecurityScanOutput(
            path=req.path, findings=[], high_count=0, medium_count=0, low_count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── summarize_quality ──────────────────────────────────────────────────────────

@mcp.tool()
def summarize_quality(project_root: str, src_path: str) -> dict:
    """Run lint, type check, and security scan; return an aggregate quality summary."""
    req = SummarizeQualityInput(project_root=project_root, src_path=src_path)
    try:
        target = str(Path(req.project_root) / req.src_path)

        # Lint
        lint_errors, lint_warnings = 0, 0
        ruff = _require("ruff")
        if ruff:
            r = _run([ruff, "check", target, "--output-format=json"], cwd=None)
            try:
                raw = json.loads(r.stdout or "[]")
                for item in raw:
                    if item.get("code", "").startswith("E"):
                        lint_errors += 1
                    else:
                        lint_warnings += 1
            except json.JSONDecodeError:
                pass

        # Type check
        type_errors = 0
        mypy = _require("mypy")
        if mypy:
            r = _run([mypy, target, "--output=json"], cwd=None)
            for line in r.stdout.splitlines():
                try:
                    item = json.loads(line)
                    if item.get("severity") == "error":
                        type_errors += 1
                except (json.JSONDecodeError, KeyError):
                    if re.match(r".+:\d+:\d+: error:", line):
                        type_errors += 1

        # Security
        high, medium, low = 0, 0, 0
        bandit = _require("bandit")
        if bandit:
            r = _run([bandit, "-r", target, "-f", "json"], cwd=None)
            try:
                raw = json.loads(r.stdout or "{}")
                for item in raw.get("results", []):
                    sev = item.get("issue_severity", "").upper()
                    if sev == "HIGH":
                        high += 1
                    elif sev == "MEDIUM":
                        medium += 1
                    else:
                        low += 1
            except json.JSONDecodeError:
                pass

        summary = QualitySummary(
            lint_errors=lint_errors, lint_warnings=lint_warnings,
            type_errors=type_errors,
            security_high=high, security_medium=medium, security_low=low,
            overall_ok=(lint_errors == 0 and type_errors == 0 and high == 0),
        )
        return SummarizeQualityOutput(project_root=req.project_root, summary=summary).model_dump()
    except Exception as exc:
        summary = QualitySummary(
            lint_errors=0, lint_warnings=0, type_errors=0,
            security_high=0, security_medium=0, security_low=0, overall_ok=False,
        )
        return SummarizeQualityOutput(
            project_root=req.project_root, summary=summary,
            success=False, error=str(exc),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
