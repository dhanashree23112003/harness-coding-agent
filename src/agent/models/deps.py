"""Pydantic I/O models for the deps namespace (dependencies, 7 tools)."""
from typing import Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class Dependency(BaseModel):
    name: str
    version: str
    source: str  # "dependencies" | "optional-dependencies" | "dev"


class OutdatedPackage(BaseModel):
    name: str
    current_version: str
    latest_version: str


class ImportResolution(BaseModel):
    import_name: str
    package_name: Optional[str] = None
    installed: bool
    stdlib: bool


class DependencyEdge(BaseModel):
    source: str
    target: str
    is_optional: bool


class VulnerabilityItem(BaseModel):
    package: str
    version: str
    id: str         # CVE or GHSA id
    severity: str   # "critical" | "high" | "medium" | "low"
    description: str
    fix_version: Optional[str] = None


# ── list_dependencies ─────────────────────────────────────────────────────────

class ListDependenciesInput(BaseModel):
    project_root: str = Field(..., description="Root directory containing pyproject.toml")
    groups: list[str] = Field(
        ["dependencies"],
        description="Dependency groups to include: 'dependencies', 'optional-dependencies', 'dev'",
    )


class ListDependenciesOutput(BaseModel):
    project_root: str
    dependencies: list[Dependency]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── check_outdated ────────────────────────────────────────────────────────────

class CheckOutdatedInput(BaseModel):
    project_root: str = Field(..., description="Root directory (used as cwd for pip)")
    include_dev: bool = Field(False, description="Include dev/optional dependencies")


class CheckOutdatedOutput(BaseModel):
    project_root: str
    outdated: list[OutdatedPackage]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── resolve_import ────────────────────────────────────────────────────────────

class ResolveImportInput(BaseModel):
    import_name: str = Field(..., description="Top-level module name as used in an import statement")
    project_root: str = Field(..., description="Project root (used as cwd)")


class ResolveImportOutput(BaseModel):
    import_name: str
    resolution: ImportResolution
    success: bool = True
    error: Optional[str] = None


# ── find_unused_deps ──────────────────────────────────────────────────────────

class FindUnusedDepsInput(BaseModel):
    project_root: str = Field(..., description="Root directory containing pyproject.toml")
    src_root: str = Field(..., description="Directory tree to scan for import usage")


class FindUnusedDepsOutput(BaseModel):
    project_root: str
    unused: list[str]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── dependency_graph ──────────────────────────────────────────────────────────

class DependencyGraphInput(BaseModel):
    project_root: str = Field(..., description="Root directory containing pyproject.toml")
    depth: int = Field(1, description="How many levels of transitive deps to traverse")


class DependencyGraphOutput(BaseModel):
    project_root: str
    nodes: list[str]
    edges: list[DependencyEdge]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── vulnerability_scan ────────────────────────────────────────────────────────

class VulnerabilityScanInput(BaseModel):
    project_root: str = Field(..., description="Root directory (used as cwd for pip-audit)")


class VulnerabilityScanOutput(BaseModel):
    project_root: str
    vulnerabilities: list[VulnerabilityItem]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── add_dependency ────────────────────────────────────────────────────────────

class AddDependencyInput(BaseModel):
    project_root: str = Field(..., description="Root directory containing pyproject.toml")
    package: str = Field(..., description="Package name to add")
    version_spec: str = Field("", description="Version specifier, e.g. '>=1.0' or '==2.3.1'")
    group: str = Field("dependencies", description="Dependency group to add to")
    install: bool = Field(True, description="Run pip install after updating pyproject.toml")


class AddDependencyOutput(BaseModel):
    project_root: str
    package: str
    installed: bool
    pyproject_updated: bool
    success: bool = True
    error: Optional[str] = None
