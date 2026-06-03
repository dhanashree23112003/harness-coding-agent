from typing import Optional

from pydantic import BaseModel, Field


# ── nested types ──────────────────────────────────────────────────────────────

class DirEntry(BaseModel):
    name: str
    is_dir: bool
    size_bytes: int
    modified_at: str  # ISO 8601


class GrepMatch(BaseModel):
    file: str
    line_number: int
    text: str


# ── read_file ─────────────────────────────────────────────────────────────────

class ReadFileInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to read")
    encoding: str = Field("utf-8", description="File encoding")


class ReadFileOutput(BaseModel):
    path: str
    content: str
    size_bytes: int
    success: bool = True
    error: Optional[str] = None


# ── read_file_range ───────────────────────────────────────────────────────────

class ReadFileRangeInput(BaseModel):
    path: str = Field(..., description="Path to the file")
    start_line: int = Field(..., description="First line to read (1-indexed, inclusive)")
    end_line: int = Field(..., description="Last line to read (1-indexed, inclusive)")
    encoding: str = Field("utf-8", description="File encoding")


class ReadFileRangeOutput(BaseModel):
    path: str
    lines: list[str]
    start_line: int
    end_line: int
    success: bool = True
    error: Optional[str] = None


# ── write_file ────────────────────────────────────────────────────────────────

class WriteFileInput(BaseModel):
    path: str = Field(..., description="Path to write")
    content: str = Field(..., description="Text content to write")
    encoding: str = Field("utf-8", description="File encoding")
    create_dirs: bool = Field(False, description="Create parent directories if missing")


class WriteFileOutput(BaseModel):
    path: str
    bytes_written: int
    success: bool = True
    error: Optional[str] = None


# ── list_dir ──────────────────────────────────────────────────────────────────

class ListDirInput(BaseModel):
    path: str = Field(..., description="Directory to list")
    recursive: bool = Field(False, description="Recurse into subdirectories")


class ListDirOutput(BaseModel):
    path: str
    entries: list[DirEntry]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── search_files ──────────────────────────────────────────────────────────────

class SearchFilesInput(BaseModel):
    root: str = Field(..., description="Root directory to search from")
    pattern: str = Field(..., description="Glob pattern (e.g. '**/*.py')")
    max_results: int = Field(200, description="Maximum number of matches to return")


class SearchFilesOutput(BaseModel):
    root: str
    pattern: str
    matches: list[str]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── grep ──────────────────────────────────────────────────────────────────────

class GrepInput(BaseModel):
    root: str = Field(..., description="Root directory to search from")
    pattern: str = Field(..., description="Regex pattern to search for")
    file_glob: str = Field("*", description="Glob pattern to filter files (e.g. '*.py')")
    max_results: int = Field(200, description="Maximum number of matches to return")
    case_sensitive: bool = Field(True, description="Whether the search is case-sensitive")


class GrepOutput(BaseModel):
    root: str
    pattern: str
    matches: list[GrepMatch]
    count: int
    success: bool = True
    error: Optional[str] = None


# ── file_stat ─────────────────────────────────────────────────────────────────

class FileStatInput(BaseModel):
    path: str = Field(..., description="Path to stat")


class FileStatOutput(BaseModel):
    path: str
    size_bytes: int
    is_file: bool
    is_dir: bool
    modified_at: str  # ISO 8601
    success: bool = True
    error: Optional[str] = None


# ── make_dir ──────────────────────────────────────────────────────────────────

class MakeDirInput(BaseModel):
    path: str = Field(..., description="Directory path to create")
    parents: bool = Field(True, description="Create parent directories as needed")
    exist_ok: bool = Field(True, description="Do not raise an error if directory exists")


class MakeDirOutput(BaseModel):
    path: str
    success: bool = True
    error: Optional[str] = None


# ── move ──────────────────────────────────────────────────────────────────────

class MoveInput(BaseModel):
    src: str = Field(..., description="Source path")
    dst: str = Field(..., description="Destination path")
    overwrite: bool = Field(False, description="Overwrite destination if it exists")


class MoveOutput(BaseModel):
    src: str
    dst: str
    success: bool = True
    error: Optional[str] = None


# ── delete ────────────────────────────────────────────────────────────────────

class DeleteInput(BaseModel):
    path: str = Field(..., description="Path to delete")
    recursive: bool = Field(False, description="Delete directory trees recursively")


class DeleteOutput(BaseModel):
    path: str
    success: bool = True
    error: Optional[str] = None


# ── copy ──────────────────────────────────────────────────────────────────────

class CopyInput(BaseModel):
    src: str = Field(..., description="Source path")
    dst: str = Field(..., description="Destination path")
    overwrite: bool = Field(False, description="Overwrite destination if it exists")


class CopyOutput(BaseModel):
    src: str
    dst: str
    success: bool = True
    error: Optional[str] = None
