"""MCP server for the deps namespace. All 7 dependency-management tools."""
import json
import re
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.deps import (  # noqa: E402
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

mcp = FastMCP("deps")

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _load_toml(project_root: str) -> dict:
    p = Path(project_root) / "pyproject.toml"
    if not p.exists():
        raise FileNotFoundError(f"pyproject.toml not found in {project_root}")
    if tomllib is None:
        raise ImportError("tomllib/tomli not available; install tomli on Python < 3.11")
    raw = p.read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def _pip(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip"] + args,
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=cwd,
    )


# ── list_dependencies ──────────────────────────────────────────────────────────

@mcp.tool()
def list_dependencies(project_root: str, groups: list[str] = ["dependencies"]) -> dict:  # noqa: B006
    """List all declared project dependencies from pyproject.toml."""
    req = ListDependenciesInput(project_root=project_root, groups=groups)
    try:
        data = _load_toml(req.project_root)
        project = data.get("project", {})
        deps: list[Dependency] = []

        if "dependencies" in req.groups:
            for spec in project.get("dependencies", []):
                name = re.split(r"[>=<!~\[]", spec)[0].strip()
                version = spec[len(name):].strip()
                deps.append(Dependency(name=name, version=version, source="dependencies"))

        if "optional-dependencies" in req.groups or "dev" in req.groups:
            for group_name, specs in project.get("optional-dependencies", {}).items():
                source = "dev" if group_name in ("dev", "test", "testing") else "optional-dependencies"
                for spec in specs:
                    name = re.split(r"[>=<!~\[]", spec)[0].strip()
                    version = spec[len(name):].strip()
                    deps.append(Dependency(name=name, version=version, source=source))

        return ListDependenciesOutput(
            project_root=req.project_root, dependencies=deps, count=len(deps),
        ).model_dump()
    except Exception as exc:
        return ListDependenciesOutput(
            project_root=req.project_root, dependencies=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── check_outdated ─────────────────────────────────────────────────────────────

@mcp.tool()
def check_outdated(project_root: str, include_dev: bool = False) -> dict:
    """Check which installed packages have newer versions available."""
    req = CheckOutdatedInput(project_root=project_root, include_dev=include_dev)
    try:
        result = _pip(["list", "--outdated", "--format=json"], cwd=req.project_root)
        if result.returncode != 0:
            return CheckOutdatedOutput(
                project_root=req.project_root, outdated=[], count=0,
                success=False, error=result.stderr[:500],
            ).model_dump()
        raw = json.loads(result.stdout or "[]")
        outdated = [
            OutdatedPackage(
                name=p["name"],
                current_version=p.get("version", ""),
                latest_version=p.get("latest_version", ""),
            )
            for p in raw
        ]
        return CheckOutdatedOutput(
            project_root=req.project_root, outdated=outdated, count=len(outdated),
        ).model_dump()
    except Exception as exc:
        return CheckOutdatedOutput(
            project_root=req.project_root, outdated=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── resolve_import ─────────────────────────────────────────────────────────────

@mcp.tool()
def resolve_import(import_name: str, project_root: str) -> dict:
    """Determine whether an import name is stdlib, installed, or unknown."""
    req = ResolveImportInput(import_name=import_name, project_root=project_root)
    try:
        # Check stdlib
        stdlib_names = getattr(sys, "stdlib_module_names", set())
        if req.import_name in stdlib_names:
            resolution = ImportResolution(
                import_name=req.import_name, package_name=None, installed=True, stdlib=True,
            )
            return ResolveImportOutput(import_name=req.import_name, resolution=resolution).model_dump()

        # Check via pip show
        result = _pip(["show", req.import_name], cwd=req.project_root)
        if result.returncode == 0:
            pkg_name = req.import_name
            for line in result.stdout.splitlines():
                if line.startswith("Name:"):
                    pkg_name = line.split(":", 1)[1].strip()
                    break
            resolution = ImportResolution(
                import_name=req.import_name, package_name=pkg_name, installed=True, stdlib=False,
            )
        else:
            resolution = ImportResolution(
                import_name=req.import_name, package_name=None, installed=False, stdlib=False,
            )
        return ResolveImportOutput(import_name=req.import_name, resolution=resolution).model_dump()
    except Exception as exc:
        resolution = ImportResolution(
            import_name=req.import_name, package_name=None, installed=False, stdlib=False,
        )
        return ResolveImportOutput(
            import_name=req.import_name, resolution=resolution,
            success=False, error=str(exc),
        ).model_dump()


# ── find_unused_deps ───────────────────────────────────────────────────────────

@mcp.tool()
def find_unused_deps(project_root: str, src_root: str) -> dict:
    """Find declared dependencies that are not imported anywhere in the source tree."""
    req = FindUnusedDepsInput(project_root=project_root, src_root=src_root)
    try:
        data = _load_toml(req.project_root)
        project = data.get("project", {})
        all_specs = list(project.get("dependencies", []))
        for specs in project.get("optional-dependencies", {}).values():
            all_specs += specs

        declared_names = [re.split(r"[>=<!~\[]", s)[0].strip() for s in all_specs]
        src_path = Path(req.src_root)
        py_sources = "\n".join(
            f.read_text(encoding="utf-8", errors="replace")
            for f in src_path.rglob("*.py")
        )

        unused = []
        for name in declared_names:
            canonical = name.replace("-", "_").lower()
            if canonical not in py_sources.lower():
                unused.append(name)

        return FindUnusedDepsOutput(
            project_root=req.project_root, unused=unused, count=len(unused),
        ).model_dump()
    except Exception as exc:
        return FindUnusedDepsOutput(
            project_root=req.project_root, unused=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── dependency_graph ───────────────────────────────────────────────────────────

@mcp.tool()
def dependency_graph(project_root: str, depth: int = 1) -> dict:
    """Build a graph of direct (and optionally transitive) package dependencies."""
    req = DependencyGraphInput(project_root=project_root, depth=depth)
    try:
        data = _load_toml(req.project_root)
        project = data.get("project", {})
        roots = [
            re.split(r"[>=<!~\[]", s)[0].strip()
            for s in project.get("dependencies", [])
        ]

        nodes: set[str] = set(roots)
        edges: list[DependencyEdge] = []
        queue = [(pkg, 0) for pkg in roots]
        visited: set[str] = set()

        while queue:
            pkg, level = queue.pop(0)
            if pkg in visited or level > req.depth:
                continue
            visited.add(pkg)
            result = _pip(["show", pkg])
            if result.returncode != 0:
                continue
            for line in result.stdout.splitlines():
                if line.startswith("Requires:"):
                    deps_str = line.split(":", 1)[1].strip()
                    if deps_str:
                        for dep in deps_str.split(","):
                            dep = dep.strip()
                            if dep:
                                nodes.add(dep)
                                edges.append(DependencyEdge(source=pkg, target=dep, is_optional=False))
                                if level + 1 <= req.depth:
                                    queue.append((dep, level + 1))

        return DependencyGraphOutput(
            project_root=req.project_root, nodes=sorted(nodes),
            edges=edges, count=len(edges),
        ).model_dump()
    except Exception as exc:
        return DependencyGraphOutput(
            project_root=req.project_root, nodes=[], edges=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── vulnerability_scan ─────────────────────────────────────────────────────────

@mcp.tool()
def vulnerability_scan(project_root: str) -> dict:
    """Scan installed packages for known security vulnerabilities using pip-audit."""
    req = VulnerabilityScanInput(project_root=project_root)
    try:
        import shutil
        if not shutil.which("pip-audit"):
            return VulnerabilityScanOutput(
                project_root=req.project_root, vulnerabilities=[], count=0,
                success=False, error="pip-audit not found; install with: pip install pip-audit",
            ).model_dump()

        result = subprocess.run(
            ["pip-audit", "--format=json"],
            cwd=req.project_root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        try:
            raw = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            raw = []

        vulns: list[VulnerabilityItem] = []
        for item in raw:
            pkg = item.get("name", "")
            version = item.get("version", "")
            for v in item.get("vulns", []):
                vulns.append(VulnerabilityItem(
                    package=pkg, version=version,
                    id=v.get("id", ""), severity=v.get("fix_versions", ["unknown"])[0] if v.get("fix_versions") else "unknown",
                    description=v.get("description", ""),
                    fix_version=v.get("fix_versions", [None])[0] if v.get("fix_versions") else None,
                ))
        return VulnerabilityScanOutput(
            project_root=req.project_root, vulnerabilities=vulns, count=len(vulns),
        ).model_dump()
    except Exception as exc:
        return VulnerabilityScanOutput(
            project_root=req.project_root, vulnerabilities=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── add_dependency ─────────────────────────────────────────────────────────────

@mcp.tool()
def add_dependency(project_root: str, package: str, version_spec: str = "", group: str = "dependencies", install: bool = True) -> dict:
    """Add a package to pyproject.toml and optionally install it."""
    req = AddDependencyInput(
        project_root=project_root, package=package,
        version_spec=version_spec, group=group, install=install,
    )
    try:
        pyproject = Path(req.project_root) / "pyproject.toml"
        if not pyproject.exists():
            return AddDependencyOutput(
                project_root=req.project_root, package=req.package,
                installed=False, pyproject_updated=False,
                success=False, error="pyproject.toml not found",
            ).model_dump()

        spec = req.package + req.version_spec
        content = pyproject.read_text(encoding="utf-8")

        # Simple injection: find the [project.dependencies] or [project] section and add the line
        updated = False
        if req.group == "dependencies":
            # Look for existing dependencies array and append before the closing ]
            pattern = re.compile(r'(\[project\][^\[]*?dependencies\s*=\s*\[)(.*?)(\])', re.DOTALL)
            match = pattern.search(content)
            if match:
                existing = match.group(2)
                new_line = f'    "{spec}",'
                if spec not in existing:
                    new_content = content[:match.start(2)] + existing.rstrip() + f"\n{new_line}\n" + content[match.start(3):]
                    pyproject.write_text(new_content, encoding="utf-8")
                    updated = True
        else:
            # Optional/dev group
            section = f'[project.optional-dependencies.{req.group}]'
            if section not in content:
                content += f'\n\n{section}\n{req.group} = [\n    "{spec}",\n]\n'
            else:
                insert_after = content.index(section) + len(section)
                before = content[:insert_after]
                after = content[insert_after:]
                bracket = after.index("[")
                close = after.index("]")
                existing = after[bracket:close]
                if spec not in existing:
                    new_after = after[:close] + f'    "{spec}",\n' + after[close:]
                    content = before + new_after
            pyproject.write_text(content, encoding="utf-8")
            updated = True

        # Install
        installed = False
        if req.install:
            install_spec = req.package + req.version_spec
            result = _pip(["install", install_spec], cwd=req.project_root)
            installed = result.returncode == 0

        return AddDependencyOutput(
            project_root=req.project_root, package=req.package,
            installed=installed, pyproject_updated=updated,
        ).model_dump()
    except Exception as exc:
        return AddDependencyOutput(
            project_root=req.project_root, package=req.package,
            installed=False, pyproject_updated=False,
            success=False, error=str(exc),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
