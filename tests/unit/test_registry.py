"""Unit tests for registry.py: ToolRegistryEntry and build_registry."""
from unittest.mock import MagicMock

import pytest

from agent.retrieval.registry import ToolRegistryEntry, build_registry, entry_text


def _mock_tool(name: str, description: str, properties: dict) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    schema = {"type": "object", "properties": properties}
    tool.args_schema.model_json_schema.return_value = schema
    return tool


def test_build_registry_single_tool():
    tool = _mock_tool("read_file", "Read file contents.", {"path": {}, "encoding": {}})
    entries = build_registry({"fs": [tool]})

    assert len(entries) == 1
    e = entries[0]
    assert e.namespace == "fs"
    assert e.name == "read_file"
    assert e.description == "Read file contents."
    assert "path" in e.input_schema["properties"]


def test_build_registry_multiple_namespaces():
    fs_tool = _mock_tool("read_file", "Read a file.", {"path": {}})
    git_tool = _mock_tool("git_status", "Show git status.", {"repo_path": {}})
    entries = build_registry({"fs": [fs_tool], "git": [git_tool]})

    namespaces = {e.namespace for e in entries}
    names = {e.name for e in entries}
    assert namespaces == {"fs", "git"}
    assert names == {"read_file", "git_status"}


def test_build_registry_pydantic_model():
    tool = _mock_tool("list_dir", "List directory.", {"path": {}})
    entries = build_registry({"fs": [tool]})
    assert isinstance(entries[0], ToolRegistryEntry)


def test_entry_text_includes_key_fields():
    tool = _mock_tool("grep", "Search file contents.", {"pattern": {}, "path": {}})
    entries = build_registry({"fs": [tool]})
    text = entry_text(entries[0])

    assert "fs.grep" in text
    assert "Search file contents." in text
    assert "pattern" in text or "path" in text


def test_build_registry_missing_args_schema():
    tool = MagicMock()
    tool.name = "broken_tool"
    tool.description = "A tool with no schema."
    del tool.args_schema  # simulate missing attribute

    entries = build_registry({"test": [tool]})
    assert len(entries) == 1
    assert entries[0].input_schema == {}
