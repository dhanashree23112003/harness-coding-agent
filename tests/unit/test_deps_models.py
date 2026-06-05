import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
    AddDependencyInput,
    AddDependencyOutput,
    CheckOutdatedInput,
    CheckOutdatedOutput,
    Dependency,
    DependencyEdge,
    DependencyGraphInput,
    DependencyGraphOutput,
    FindUnusedDepsInput,
    FindUnusedDepsOutput,
    ImportResolution,
    ListDependenciesInput,
    ListDependenciesOutput,
    OutdatedPackage,
    ResolveImportInput,
    ResolveImportOutput,
    VulnerabilityItem,
    VulnerabilityScanInput,
    VulnerabilityScanOutput,
)


# ── nested types ───────────────────────────────────────────────────────────────

def test_dependency_roundtrip():
    d = Dependency(name="pydantic", version=">=2.0", source="dependencies")
    assert Dependency(**d.model_dump()) == d


def test_outdated_package_roundtrip():
    p = OutdatedPackage(name="pydantic", current_version="2.0.0", latest_version="2.5.0")
    assert OutdatedPackage(**p.model_dump()) == p


def test_import_resolution_stdlib():
    r = ImportResolution(import_name="os", package_name=None, installed=True, stdlib=True)
    assert r.stdlib is True
    assert ImportResolution(**r.model_dump()) == r


def test_import_resolution_third_party():
    r = ImportResolution(import_name="pydantic", package_name="pydantic", installed=True, stdlib=False)
    assert r.stdlib is False


def test_dependency_edge_roundtrip():
    e = DependencyEdge(source="langchain-groq", target="langchain-core", is_optional=False)
    assert DependencyEdge(**e.model_dump()) == e


def test_vulnerability_item_roundtrip():
    v = VulnerabilityItem(
        package="requests", version="2.25.0",
        id="CVE-2023-1234", severity="high",
        description="Remote code execution", fix_version="2.31.0",
    )
    assert VulnerabilityItem(**v.model_dump()) == v


def test_vulnerability_item_no_fix():
    v = VulnerabilityItem(
        package="requests", version="2.25.0",
        id="CVE-2023-1234", severity="medium",
        description="Some issue",
    )
    assert v.fix_version is None


# ── list_dependencies ──────────────────────────────────────────────────────────

def test_list_dependencies_input_requires_root():
    with pytest.raises(ValidationError):
        ListDependenciesInput()  # type: ignore[call-arg]


def test_list_dependencies_input_defaults():
    inp = ListDependenciesInput(project_root="/repo")
    assert inp.groups == ["dependencies"]


def test_list_dependencies_output_roundtrip():
    out = ListDependenciesOutput(project_root="/repo", dependencies=[], count=0)
    assert ListDependenciesOutput(**out.model_dump()) == out


def test_list_dependencies_output_error_field():
    out = ListDependenciesOutput(
        project_root="/repo", dependencies=[], count=0,
        success=False, error="pyproject.toml not found",
    )
    assert out.error == "pyproject.toml not found"


# ── check_outdated ─────────────────────────────────────────────────────────────

def test_check_outdated_input_defaults():
    inp = CheckOutdatedInput(project_root="/repo")
    assert inp.include_dev is False


def test_check_outdated_output_roundtrip():
    out = CheckOutdatedOutput(project_root="/repo", outdated=[], count=0)
    assert CheckOutdatedOutput(**out.model_dump()) == out


# ── resolve_import ─────────────────────────────────────────────────────────────

def test_resolve_import_input_requires_fields():
    with pytest.raises(ValidationError):
        ResolveImportInput(import_name="os")  # missing project_root


def test_resolve_import_output_roundtrip():
    r = ImportResolution(import_name="os", package_name=None, installed=True, stdlib=True)
    out = ResolveImportOutput(import_name="os", resolution=r)
    assert ResolveImportOutput(**out.model_dump()) == out


# ── find_unused_deps ───────────────────────────────────────────────────────────

def test_find_unused_deps_input_requires_fields():
    with pytest.raises(ValidationError):
        FindUnusedDepsInput(project_root="/repo")  # missing src_root


def test_find_unused_deps_output_roundtrip():
    out = FindUnusedDepsOutput(project_root="/repo", unused=["unused-lib"], count=1)
    assert FindUnusedDepsOutput(**out.model_dump()) == out


# ── dependency_graph ───────────────────────────────────────────────────────────

def test_dependency_graph_input_defaults():
    inp = DependencyGraphInput(project_root="/repo")
    assert inp.depth == 1


def test_dependency_graph_output_roundtrip():
    edge = DependencyEdge(source="langchain-groq", target="groq", is_optional=False)
    out = DependencyGraphOutput(
        project_root="/repo", nodes=["langchain-groq", "groq"],
        edges=[edge], count=1,
    )
    assert DependencyGraphOutput(**out.model_dump()) == out


# ── vulnerability_scan ─────────────────────────────────────────────────────────

def test_vulnerability_scan_output_roundtrip():
    out = VulnerabilityScanOutput(project_root="/repo", vulnerabilities=[], count=0)
    assert VulnerabilityScanOutput(**out.model_dump()) == out


def test_vulnerability_scan_output_error_field():
    out = VulnerabilityScanOutput(
        project_root="/repo", vulnerabilities=[], count=0,
        success=False, error="pip-audit not found",
    )
    assert out.error == "pip-audit not found"


# ── add_dependency ─────────────────────────────────────────────────────────────

def test_add_dependency_input_requires_fields():
    with pytest.raises(ValidationError):
        AddDependencyInput(project_root="/repo")  # missing package


def test_add_dependency_input_defaults():
    inp = AddDependencyInput(project_root="/repo", package="httpx")
    assert inp.version_spec == ""
    assert inp.group == "dependencies"
    assert inp.install is True


def test_add_dependency_output_roundtrip():
    out = AddDependencyOutput(
        project_root="/repo", package="httpx", installed=True, pyproject_updated=True,
    )
    assert AddDependencyOutput(**out.model_dump()) == out
