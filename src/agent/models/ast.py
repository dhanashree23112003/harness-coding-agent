"""Pydantic I/O models for the ast namespace (static analysis, 9 tools)."""
from typing import Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class SymbolInfo(BaseModel):
    name: str
    kind: str  # "function" | "class" | "method" | "variable"
    line: int
    col: int
    end_line: int


class ReferenceLocation(BaseModel):
    """A single reference to a symbol. Fields map directly to fs.read_file_range inputs."""
    file: str        # feeds ReadFileRangeInput.path
    line: int        # feeds ReadFileRangeInput.start_line (use line-2 for context)
    col: int
    context: str     # the source line text at this location


class ImportEntry(BaseModel):
    module: str
    names: list[str]  # empty list for bare `import module`
    alias: Optional[str] = None
    line: int
    is_from: bool  # True for `from module import x`


class ComplexityResult(BaseModel):
    name: str
    kind: str  # "function" | "method"
    complexity: int
    line: int


class DeadCodeItem(BaseModel):
    name: str
    kind: str   # "function" | "class"
    line: int
    reason: str  # "never_called" | "unreachable"


class FunctionSignature(BaseModel):
    name: str
    args: list[str]
    return_annotation: str
    decorators: list[str]
    is_async: bool
    line: int


# ── parse_module ──────────────────────────────────────────────────────────────

class ParseModuleInput(BaseModel):
    path: str = Field(..., description="Path to the Python file to parse")


class ParseModuleOutput(BaseModel):
    path: str
    module_name: str
    top_level_nodes: int
    has_syntax_error: bool
    error_offset: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


# ── list_symbols ──────────────────────────────────────────────────────────────

class ListSymbolsInput(BaseModel):
    path: str = Field(..., description="Path to the Python file")
    kinds: list[str] = Field(
        ["function", "class"],
        description="Symbol kinds to include: 'function', 'class', 'method', 'variable'",
    )


class ListSymbolsOutput(BaseModel):
    path: str
    symbols: list[SymbolInfo]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── find_definition ───────────────────────────────────────────────────────────

class FindDefinitionInput(BaseModel):
    path: str = Field(..., description="Path to the Python file to search")
    symbol_name: str = Field(..., description="Name of the symbol to locate")


class FindDefinitionOutput(BaseModel):
    path: str
    symbol_name: str
    found: bool
    definition_line: Optional[int] = None
    kind: Optional[str] = None  # "function" | "class" | "variable"
    success: bool = True
    error: Optional[str] = None


# ── find_references ───────────────────────────────────────────────────────────

class FindReferencesInput(BaseModel):
    path: str = Field(..., description="Canonical definition file (used to anchor the search)")
    symbol_name: str = Field(..., description="Name of the symbol to find references for")
    search_root: str = Field(..., description="Root directory to search for references")


class FindReferencesOutput(BaseModel):
    """Output references feed fs.read_file_range to pull each call site.

    Map: ref.file -> ReadFileRangeInput.path, ref.line -> start_line.
    """
    path: str
    symbol_name: str
    references: list[ReferenceLocation]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── list_imports ──────────────────────────────────────────────────────────────

class ListImportsInput(BaseModel):
    path: str = Field(..., description="Path to the Python file")


class ListImportsOutput(BaseModel):
    path: str
    imports: list[ImportEntry]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── compute_complexity ────────────────────────────────────────────────────────

class ComputeComplexityInput(BaseModel):
    path: str = Field(..., description="Path to the Python file")
    threshold: int = Field(0, description="Only report functions with complexity above this value (0 = all)")


class ComputeComplexityOutput(BaseModel):
    path: str
    results: list[ComplexityResult]
    max_complexity: int
    above_threshold: int
    success: bool = True
    error: Optional[str] = None


# ── detect_dead_code ──────────────────────────────────────────────────────────

class DetectDeadCodeInput(BaseModel):
    path: str = Field(..., description="Path to the Python file to analyse")


class DetectDeadCodeOutput(BaseModel):
    path: str
    items: list[DeadCodeItem]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── extract_function_signature ────────────────────────────────────────────────

class ExtractFunctionSignatureInput(BaseModel):
    path: str = Field(..., description="Path to the Python file")
    function_name: str = Field(..., description="Name of the function or method")


class ExtractFunctionSignatureOutput(BaseModel):
    path: str
    function_name: str
    signature: Optional[FunctionSignature] = None
    found: bool
    success: bool = True
    error: Optional[str] = None


# ── find_unused_imports ───────────────────────────────────────────────────────

class FindUnusedImportsInput(BaseModel):
    path: str = Field(..., description="Path to the Python file")


class FindUnusedImportsOutput(BaseModel):
    path: str
    unused: list[ImportEntry]
    count: int
    success: bool = True
    error: Optional[str] = None
