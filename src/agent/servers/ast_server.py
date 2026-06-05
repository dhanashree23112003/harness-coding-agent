"""MCP server for the ast namespace. All 9 static-analysis tools. Uses stdlib ast only."""
import ast as _ast
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from agent.models.ast import (  # noqa: E402
    ComplexityResult,
    ComputeComplexityInput,
    ComputeComplexityOutput,
    DeadCodeItem,
    DetectDeadCodeInput,
    DetectDeadCodeOutput,
    ExtractFunctionSignatureInput,
    ExtractFunctionSignatureOutput,
    FindDefinitionInput,
    FindDefinitionOutput,
    FindReferencesInput,
    FindReferencesOutput,
    FindUnusedImportsInput,
    FindUnusedImportsOutput,
    FunctionSignature,
    ImportEntry,
    ListImportsInput,
    ListImportsOutput,
    ListSymbolsInput,
    ListSymbolsOutput,
    ParseModuleInput,
    ParseModuleOutput,
    ReferenceLocation,
    SymbolInfo,
)

mcp = FastMCP("ast")


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _parse(path: str) -> _ast.Module:
    return _ast.parse(_read(path), filename=path)


# ── parse_module ───────────────────────────────────────────────────────────────

@mcp.tool()
def parse_module(path: str) -> dict:
    """Parse a Python file and return its top-level structure and any syntax errors."""
    req = ParseModuleInput(path=path)
    try:
        source = _read(req.path)
        try:
            tree = _ast.parse(source, filename=req.path)
            return ParseModuleOutput(
                path=req.path,
                module_name=Path(req.path).stem,
                top_level_nodes=len(tree.body),
                has_syntax_error=False,
            ).model_dump()
        except SyntaxError as exc:
            return ParseModuleOutput(
                path=req.path,
                module_name=Path(req.path).stem,
                top_level_nodes=0,
                has_syntax_error=True,
                error_offset=exc.offset,
            ).model_dump()
    except Exception as exc:
        return ParseModuleOutput(
            path=req.path, module_name="", top_level_nodes=0,
            has_syntax_error=False, success=False, error=str(exc),
        ).model_dump()


# ── list_symbols ───────────────────────────────────────────────────────────────

@mcp.tool()
def list_symbols(path: str, kinds: list[str] = ["function", "class"]) -> dict:  # noqa: B006
    """List all top-level symbols (functions, classes) defined in a Python file."""
    req = ListSymbolsInput(path=path, kinds=kinds)
    try:
        tree = _parse(req.path)
        kind_set = set(req.kinds)
        symbols: list[SymbolInfo] = []
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and "function" in kind_set:
                symbols.append(SymbolInfo(
                    name=node.name, kind="function",
                    line=node.lineno, col=node.col_offset,
                    end_line=getattr(node, "end_lineno", node.lineno),
                ))
            elif isinstance(node, _ast.ClassDef) and "class" in kind_set:
                symbols.append(SymbolInfo(
                    name=node.name, kind="class",
                    line=node.lineno, col=node.col_offset,
                    end_line=getattr(node, "end_lineno", node.lineno),
                ))
        symbols.sort(key=lambda s: s.line)
        return ListSymbolsOutput(path=req.path, symbols=symbols, count=len(symbols)).model_dump()
    except Exception as exc:
        return ListSymbolsOutput(path=req.path, symbols=[], count=0, success=False, error=str(exc)).model_dump()


# ── find_definition ────────────────────────────────────────────────────────────

@mcp.tool()
def find_definition(path: str, symbol_name: str) -> dict:
    """Find where a named symbol (function, class, or variable) is defined in a file."""
    req = FindDefinitionInput(path=path, symbol_name=symbol_name)
    try:
        tree = _parse(req.path)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == req.symbol_name:
                return FindDefinitionOutput(
                    path=req.path, symbol_name=req.symbol_name,
                    found=True, definition_line=node.lineno, kind="function",
                ).model_dump()
            if isinstance(node, _ast.ClassDef) and node.name == req.symbol_name:
                return FindDefinitionOutput(
                    path=req.path, symbol_name=req.symbol_name,
                    found=True, definition_line=node.lineno, kind="class",
                ).model_dump()
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    if isinstance(target, _ast.Name) and target.id == req.symbol_name:
                        return FindDefinitionOutput(
                            path=req.path, symbol_name=req.symbol_name,
                            found=True, definition_line=node.lineno, kind="variable",
                        ).model_dump()
        return FindDefinitionOutput(path=req.path, symbol_name=req.symbol_name, found=False).model_dump()
    except Exception as exc:
        return FindDefinitionOutput(
            path=req.path, symbol_name=req.symbol_name, found=False,
            success=False, error=str(exc),
        ).model_dump()


# ── find_references ────────────────────────────────────────────────────────────

@mcp.tool()
def find_references(path: str, symbol_name: str, search_root: str) -> dict:
    """Find all references to a symbol across the search_root directory.

    Output references feed fs.read_file_range to pull each call site:
    map ref.file -> ReadFileRangeInput.path, ref.line -> start_line.
    """
    req = FindReferencesInput(path=path, symbol_name=symbol_name, search_root=search_root)
    try:
        refs: list[ReferenceLocation] = []
        for py_file in Path(req.search_root).rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                lines = source.splitlines()
                tree = _ast.parse(source, filename=str(py_file))
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.Name) and node.id == req.symbol_name:
                        if isinstance(node.ctx, _ast.Load):
                            lineno = node.lineno
                            context = lines[lineno - 1] if lineno <= len(lines) else ""
                            refs.append(ReferenceLocation(
                                file=str(py_file), line=lineno,
                                col=node.col_offset, context=context,
                            ))
            except (SyntaxError, OSError):
                continue
        return FindReferencesOutput(
            path=req.path, symbol_name=req.symbol_name, references=refs, count=len(refs),
        ).model_dump()
    except Exception as exc:
        return FindReferencesOutput(
            path=req.path, symbol_name=req.symbol_name, references=[], count=0,
            success=False, error=str(exc),
        ).model_dump()


# ── list_imports ───────────────────────────────────────────────────────────────

@mcp.tool()
def list_imports(path: str) -> dict:
    """List all import statements in a Python file."""
    req = ListImportsInput(path=path)
    try:
        tree = _parse(req.path)
        imports: list[ImportEntry] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    imports.append(ImportEntry(
                        module=alias.name, names=[],
                        alias=alias.asname, line=node.lineno, is_from=False,
                    ))
            elif isinstance(node, _ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportEntry(
                    module=module, names=names,
                    alias=None, line=node.lineno, is_from=True,
                ))
        imports.sort(key=lambda i: i.line)
        return ListImportsOutput(path=req.path, imports=imports, count=len(imports)).model_dump()
    except Exception as exc:
        return ListImportsOutput(path=req.path, imports=[], count=0, success=False, error=str(exc)).model_dump()


# ── compute_complexity ─────────────────────────────────────────────────────────

class _ComplexityVisitor(_ast.NodeVisitor):
    """McCabe-style cyclomatic complexity approximation per function."""

    _BRANCH_TYPES = (
        _ast.If, _ast.For, _ast.AsyncFor, _ast.While,
        _ast.ExceptHandler, _ast.With, _ast.AsyncWith,
        _ast.Assert, _ast.comprehension,
    )

    def __init__(self) -> None:
        self.results: list[ComplexityResult] = []

    def _count(self, node: _ast.AST) -> int:
        count = 1
        for child in _ast.walk(node):
            if isinstance(child, self._BRANCH_TYPES):
                count += 1
            elif isinstance(child, _ast.BoolOp):
                count += len(child.values) - 1
        return count

    def visit_FunctionDef(self, node: _ast.FunctionDef) -> None:
        self.results.append(ComplexityResult(
            name=node.name, kind="function",
            complexity=self._count(node), line=node.lineno,
        ))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


@mcp.tool()
def compute_complexity(path: str, threshold: int = 0) -> dict:
    """Compute McCabe-style cyclomatic complexity for each function in a file."""
    req = ComputeComplexityInput(path=path, threshold=threshold)
    try:
        tree = _parse(req.path)
        visitor = _ComplexityVisitor()
        visitor.visit(tree)
        results = visitor.results
        if req.threshold > 0:
            above = [r for r in results if r.complexity > req.threshold]
        else:
            above = results
        max_c = max((r.complexity for r in results), default=0)
        return ComputeComplexityOutput(
            path=req.path, results=results,
            max_complexity=max_c, above_threshold=len(above),
        ).model_dump()
    except Exception as exc:
        return ComputeComplexityOutput(
            path=req.path, results=[], max_complexity=0, above_threshold=0,
            success=False, error=str(exc),
        ).model_dump()


# ── detect_dead_code ───────────────────────────────────────────────────────────

@mcp.tool()
def detect_dead_code(path: str) -> dict:
    """Detect functions and classes that are defined but never called within the same file."""
    req = DetectDeadCodeInput(path=path)
    try:
        tree = _parse(req.path)
        defined: dict[str, tuple[str, int]] = {}  # name -> (kind, line)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    defined[node.name] = ("function", node.lineno)
            elif isinstance(node, _ast.ClassDef):
                if not node.name.startswith("_"):
                    defined[node.name] = ("class", node.lineno)

        called: set[str] = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, _ast.Attribute):
                    called.add(node.func.attr)

        items = [
            DeadCodeItem(name=name, kind=kind, line=line, reason="never_called")
            for name, (kind, line) in defined.items()
            if name not in called
        ]
        items.sort(key=lambda i: i.line)
        return DetectDeadCodeOutput(path=req.path, items=items, count=len(items)).model_dump()
    except Exception as exc:
        return DetectDeadCodeOutput(
            path=req.path, items=[], count=0, success=False, error=str(exc),
        ).model_dump()


# ── extract_function_signature ─────────────────────────────────────────────────

@mcp.tool()
def extract_function_signature(path: str, function_name: str) -> dict:
    """Extract the full signature of a named function including parameters and return type."""
    req = ExtractFunctionSignatureInput(path=path, function_name=function_name)
    try:
        tree = _parse(req.path)
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == req.function_name:
                args = []
                for arg in node.args.args:
                    a = arg.arg
                    if arg.annotation:
                        a += f": {_ast.unparse(arg.annotation)}"
                    args.append(a)
                ret = _ast.unparse(node.returns) if node.returns else ""
                decorators = [_ast.unparse(d) for d in node.decorator_list]
                sig = FunctionSignature(
                    name=node.name, args=args, return_annotation=ret,
                    decorators=decorators,
                    is_async=isinstance(node, _ast.AsyncFunctionDef),
                    line=node.lineno,
                )
                return ExtractFunctionSignatureOutput(
                    path=req.path, function_name=req.function_name,
                    signature=sig, found=True,
                ).model_dump()
        return ExtractFunctionSignatureOutput(
            path=req.path, function_name=req.function_name, signature=None, found=False,
        ).model_dump()
    except Exception as exc:
        return ExtractFunctionSignatureOutput(
            path=req.path, function_name=req.function_name, signature=None, found=False,
            success=False, error=str(exc),
        ).model_dump()


# ── find_unused_imports ────────────────────────────────────────────────────────

@mcp.tool()
def find_unused_imports(path: str) -> dict:
    """Find import statements whose imported names are never referenced in the file."""
    req = FindUnusedImportsInput(path=path)
    try:
        tree = _parse(req.path)

        # Collect all imported names
        imported: list[tuple[str, ImportEntry]] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    effective = alias.asname or alias.name.split(".")[0]
                    imported.append((effective, ImportEntry(
                        module=alias.name, names=[],
                        alias=alias.asname, line=node.lineno, is_from=False,
                    )))
            elif isinstance(node, _ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    effective = alias.asname or alias.name
                    names = [alias.name]
                    imported.append((effective, ImportEntry(
                        module=module, names=names,
                        alias=alias.asname, line=node.lineno, is_from=True,
                    )))

        # Collect all names used in load context (excluding import nodes themselves)
        used: set[str] = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Name) and isinstance(node.ctx, _ast.Load):
                used.add(node.id)
            elif isinstance(node, _ast.Attribute):
                if isinstance(node.value, _ast.Name):
                    used.add(node.value.id)

        unused = [entry for name, entry in imported if name not in used]
        return FindUnusedImportsOutput(path=req.path, unused=unused, count=len(unused)).model_dump()
    except Exception as exc:
        return FindUnusedImportsOutput(
            path=req.path, unused=[], count=0, success=False, error=str(exc),
        ).model_dump()


if __name__ == "__main__":
    mcp.run()
