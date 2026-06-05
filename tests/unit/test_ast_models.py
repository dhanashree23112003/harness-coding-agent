import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
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


# ── parse_module ───────────────────────────────────────────────────────────────

def test_parse_module_input_requires_path():
    with pytest.raises(ValidationError):
        ParseModuleInput()  # type: ignore[call-arg]


def test_parse_module_output_defaults():
    out = ParseModuleOutput(path="/tmp/a.py", module_name="a", top_level_nodes=3, has_syntax_error=False)
    assert out.success is True
    assert out.error is None
    assert out.error_offset is None


def test_parse_module_output_syntax_error():
    out = ParseModuleOutput(
        path="/tmp/bad.py", module_name="bad", top_level_nodes=0,
        has_syntax_error=True, error_offset=42, success=False, error="invalid syntax",
    )
    assert out.has_syntax_error is True
    assert out.error_offset == 42


def test_parse_module_output_roundtrip():
    out = ParseModuleOutput(path="/tmp/a.py", module_name="a", top_level_nodes=5, has_syntax_error=False)
    assert ParseModuleOutput(**out.model_dump()) == out


# ── list_symbols ───────────────────────────────────────────────────────────────

def test_list_symbols_input_defaults():
    inp = ListSymbolsInput(path="/tmp/a.py")
    assert inp.kinds == ["function", "class"]


def test_symbol_info_roundtrip():
    s = SymbolInfo(name="foo", kind="function", line=10, col=0, end_line=20)
    assert SymbolInfo(**s.model_dump()) == s


def test_list_symbols_output_roundtrip():
    out = ListSymbolsOutput(path="/tmp/a.py", symbols=[], count=0)
    assert ListSymbolsOutput(**out.model_dump()) == out


# ── find_definition ────────────────────────────────────────────────────────────

def test_find_definition_input_requires_fields():
    with pytest.raises(ValidationError):
        FindDefinitionInput(path="/tmp/a.py")  # missing symbol_name


def test_find_definition_output_not_found():
    out = FindDefinitionOutput(path="/tmp/a.py", symbol_name="foo", found=False)
    assert out.definition_line is None
    assert out.kind is None
    assert out.success is True


def test_find_definition_output_roundtrip():
    out = FindDefinitionOutput(
        path="/tmp/a.py", symbol_name="bar", found=True,
        definition_line=42, kind="function",
    )
    assert FindDefinitionOutput(**out.model_dump()) == out


# ── find_references ────────────────────────────────────────────────────────────

def test_find_references_input_requires_fields():
    with pytest.raises(ValidationError):
        FindReferencesInput(path="/tmp/a.py", symbol_name="foo")  # missing search_root


def test_reference_location_roundtrip():
    ref = ReferenceLocation(file="/tmp/b.py", line=15, col=4, context="    foo(x)")
    assert ReferenceLocation(**ref.model_dump()) == ref


def test_find_references_output_composition_fields():
    ref = ReferenceLocation(file="/tmp/b.py", line=15, col=4, context="    foo(x)")
    out = FindReferencesOutput(
        path="/tmp/a.py", symbol_name="foo", references=[ref], count=1,
    )
    assert out.references[0].file == "/tmp/b.py"
    assert out.references[0].line == 15


def test_find_references_output_roundtrip():
    out = FindReferencesOutput(path="/tmp/a.py", symbol_name="foo", references=[], count=0)
    assert FindReferencesOutput(**out.model_dump()) == out


# ── list_imports ───────────────────────────────────────────────────────────────

def test_import_entry_from_import():
    e = ImportEntry(module="os", names=["path"], alias=None, line=1, is_from=True)
    assert e.is_from is True
    assert ImportEntry(**e.model_dump()) == e


def test_import_entry_bare_import():
    e = ImportEntry(module="sys", names=[], alias=None, line=2, is_from=False)
    assert e.names == []


def test_list_imports_output_roundtrip():
    out = ListImportsOutput(path="/tmp/a.py", imports=[], count=0)
    assert ListImportsOutput(**out.model_dump()) == out


# ── compute_complexity ─────────────────────────────────────────────────────────

def test_compute_complexity_input_defaults():
    inp = ComputeComplexityInput(path="/tmp/a.py")
    assert inp.threshold == 0


def test_complexity_result_roundtrip():
    r = ComplexityResult(name="foo", kind="function", complexity=5, line=10)
    assert ComplexityResult(**r.model_dump()) == r


def test_compute_complexity_output_roundtrip():
    out = ComputeComplexityOutput(
        path="/tmp/a.py", results=[], max_complexity=0, above_threshold=0,
    )
    assert ComputeComplexityOutput(**out.model_dump()) == out


# ── detect_dead_code ───────────────────────────────────────────────────────────

def test_dead_code_item_roundtrip():
    item = DeadCodeItem(name="_helper", kind="function", line=30, reason="never_called")
    assert DeadCodeItem(**item.model_dump()) == item


def test_detect_dead_code_output_roundtrip():
    out = DetectDeadCodeOutput(path="/tmp/a.py", items=[], count=0)
    assert DetectDeadCodeOutput(**out.model_dump()) == out


# ── extract_function_signature ─────────────────────────────────────────────────

def test_extract_function_signature_input_requires_fields():
    with pytest.raises(ValidationError):
        ExtractFunctionSignatureInput(path="/tmp/a.py")  # missing function_name


def test_function_signature_roundtrip():
    sig = FunctionSignature(
        name="foo", args=["x", "y"], return_annotation="int",
        decorators=[], is_async=False, line=10,
    )
    assert FunctionSignature(**sig.model_dump()) == sig


def test_extract_function_signature_output_not_found():
    out = ExtractFunctionSignatureOutput(
        path="/tmp/a.py", function_name="missing", signature=None, found=False,
    )
    assert out.signature is None
    assert out.found is False


def test_extract_function_signature_output_roundtrip():
    out = ExtractFunctionSignatureOutput(
        path="/tmp/a.py", function_name="missing", signature=None, found=False,
    )
    assert ExtractFunctionSignatureOutput(**out.model_dump()) == out


# ── find_unused_imports ────────────────────────────────────────────────────────

def test_find_unused_imports_output_roundtrip():
    out = FindUnusedImportsOutput(path="/tmp/a.py", unused=[], count=0)
    assert FindUnusedImportsOutput(**out.model_dump()) == out


def test_find_unused_imports_output_error_field():
    out = FindUnusedImportsOutput(
        path="/tmp/bad.py", unused=[], count=0, success=False, error="file not found",
    )
    assert out.error == "file not found"
