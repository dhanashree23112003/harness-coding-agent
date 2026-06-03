import pytest
from pydantic import ValidationError

from agent.models.tool_io import (
    CopyInput,
    CopyOutput,
    DeleteInput,
    DeleteOutput,
    DirEntry,
    FileStatInput,
    FileStatOutput,
    GrepInput,
    GrepMatch,
    GrepOutput,
    ListDirInput,
    ListDirOutput,
    MakeDirInput,
    MakeDirOutput,
    MoveInput,
    MoveOutput,
    ReadFileInput,
    ReadFileOutput,
    ReadFileRangeInput,
    ReadFileRangeOutput,
    SearchFilesInput,
    SearchFilesOutput,
    WriteFileInput,
    WriteFileOutput,
)


# ── read_file ──────────────────────────────────────────────────────────────────

def test_read_file_input_defaults():
    inp = ReadFileInput(path="/tmp/test.txt")
    assert inp.path == "/tmp/test.txt"
    assert inp.encoding == "utf-8"


def test_read_file_input_custom_encoding():
    inp = ReadFileInput(path="/tmp/test.txt", encoding="latin-1")
    assert inp.encoding == "latin-1"


def test_read_file_input_requires_path():
    with pytest.raises(ValidationError):
        ReadFileInput()  # type: ignore[call-arg]


def test_read_file_output_fields():
    out = ReadFileOutput(path="/tmp/test.txt", content="hello", size_bytes=5)
    assert out.content == "hello"
    assert out.size_bytes == 5
    assert out.success is True
    assert out.error is None


def test_read_file_output_serialization_roundtrip():
    out = ReadFileOutput(path="/tmp/test.txt", content="hello\nworld", size_bytes=11)
    d = out.model_dump()
    assert d == {
        "path": "/tmp/test.txt",
        "content": "hello\nworld",
        "size_bytes": 11,
        "success": True,
        "error": None,
    }
    assert ReadFileOutput(**d) == out


# ── read_file_range ────────────────────────────────────────────────────────────

def test_read_file_range_input_requires_fields():
    with pytest.raises(ValidationError):
        ReadFileRangeInput(path="/tmp/x.txt")  # missing start_line, end_line

def test_read_file_range_output_roundtrip():
    out = ReadFileRangeOutput(path="/tmp/x.txt", lines=["a", "b"], start_line=1, end_line=2)
    assert ReadFileRangeOutput(**out.model_dump()) == out


# ── write_file ─────────────────────────────────────────────────────────────────

def test_write_file_input_defaults():
    inp = WriteFileInput(path="/tmp/x.txt", content="hi")
    assert inp.encoding == "utf-8"
    assert inp.create_dirs is False


def test_write_file_output_success_flag():
    out = WriteFileOutput(path="/tmp/x.txt", bytes_written=2)
    assert out.success is True
    err_out = WriteFileOutput(path="/tmp/x.txt", bytes_written=0, success=False, error="oops")
    assert err_out.error == "oops"


# ── list_dir ───────────────────────────────────────────────────────────────────

def test_list_dir_input_defaults():
    inp = ListDirInput(path="/tmp")
    assert inp.recursive is False


def test_dir_entry_roundtrip():
    e = DirEntry(name="foo.py", is_dir=False, size_bytes=42, modified_at="2024-01-01T00:00:00")
    assert DirEntry(**e.model_dump()) == e


def test_list_dir_output_roundtrip():
    out = ListDirOutput(path="/tmp", entries=[], count=0)
    assert ListDirOutput(**out.model_dump()) == out


# ── search_files ───────────────────────────────────────────────────────────────

def test_search_files_input_defaults():
    inp = SearchFilesInput(root="/tmp", pattern="*.py")
    assert inp.max_results == 200


def test_search_files_output_count():
    out = SearchFilesOutput(root="/tmp", pattern="*.py", matches=["/tmp/a.py"], count=1)
    assert out.count == 1


# ── grep ───────────────────────────────────────────────────────────────────────

def test_grep_input_defaults():
    inp = GrepInput(root="/tmp", pattern="TODO")
    assert inp.file_glob == "*"
    assert inp.case_sensitive is True


def test_grep_match_roundtrip():
    m = GrepMatch(file="/tmp/a.py", line_number=10, text="TODO: fix this")
    assert GrepMatch(**m.model_dump()) == m


def test_grep_output_roundtrip():
    out = GrepOutput(root="/tmp", pattern="TODO", matches=[], count=0)
    assert GrepOutput(**out.model_dump()) == out


# ── file_stat ──────────────────────────────────────────────────────────────────

def test_file_stat_input_requires_path():
    with pytest.raises(ValidationError):
        FileStatInput()  # type: ignore[call-arg]


def test_file_stat_output_roundtrip():
    out = FileStatOutput(
        path="/tmp/x.txt", size_bytes=10,
        is_file=True, is_dir=False, modified_at="2024-01-01T00:00:00",
    )
    assert FileStatOutput(**out.model_dump()) == out


# ── make_dir ───────────────────────────────────────────────────────────────────

def test_make_dir_input_defaults():
    inp = MakeDirInput(path="/tmp/new")
    assert inp.parents is True
    assert inp.exist_ok is True


def test_make_dir_output_roundtrip():
    out = MakeDirOutput(path="/tmp/new")
    assert MakeDirOutput(**out.model_dump()) == out


# ── move ───────────────────────────────────────────────────────────────────────

def test_move_input_requires_src_dst():
    with pytest.raises(ValidationError):
        MoveInput(src="/tmp/a")  # missing dst

def test_move_output_roundtrip():
    out = MoveOutput(src="/tmp/a", dst="/tmp/b")
    assert MoveOutput(**out.model_dump()) == out


# ── delete ─────────────────────────────────────────────────────────────────────

def test_delete_input_defaults():
    inp = DeleteInput(path="/tmp/x")
    assert inp.recursive is False


def test_delete_output_error_field():
    out = DeleteOutput(path="/tmp/x", success=False, error="not found")
    assert out.error == "not found"


# ── copy ───────────────────────────────────────────────────────────────────────

def test_copy_input_defaults():
    inp = CopyInput(src="/tmp/a", dst="/tmp/b")
    assert inp.overwrite is False


def test_copy_output_roundtrip():
    out = CopyOutput(src="/tmp/a", dst="/tmp/b")
    assert CopyOutput(**out.model_dump()) == out
