import pytest
from pydantic import ValidationError

from agent.models.tool_io import ReadFileInput, ReadFileOutput


def test_input_defaults():
    inp = ReadFileInput(path="/tmp/test.txt")
    assert inp.path == "/tmp/test.txt"
    assert inp.encoding == "utf-8"


def test_input_custom_encoding():
    inp = ReadFileInput(path="/tmp/test.txt", encoding="latin-1")
    assert inp.encoding == "latin-1"


def test_input_requires_path():
    with pytest.raises(ValidationError):
        ReadFileInput()  # type: ignore[call-arg]


def test_output_fields():
    out = ReadFileOutput(path="/tmp/test.txt", content="hello", size_bytes=5)
    assert out.content == "hello"
    assert out.size_bytes == 5
    assert out.path == "/tmp/test.txt"


def test_output_serialization_roundtrip():
    out = ReadFileOutput(path="/tmp/test.txt", content="hello\nworld", size_bytes=11)
    d = out.model_dump()
    assert d == {"path": "/tmp/test.txt", "content": "hello\nworld", "size_bytes": 11}
    restored = ReadFileOutput(**d)
    assert restored == out
