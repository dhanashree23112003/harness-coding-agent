"""Unit tests for _resolve() path resolution helpers in fs_server and ast_server."""
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fs_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REPO_ROOT", str(tmp_path))
    sys.modules.pop("agent.servers.fs_server", None)
    import agent.servers.fs_server as mod
    yield mod, tmp_path
    sys.modules.pop("agent.servers.fs_server", None)


@pytest.fixture
def ast_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_REPO_ROOT", str(tmp_path))
    sys.modules.pop("agent.servers.ast_server", None)
    import agent.servers.ast_server as mod
    yield mod, tmp_path
    sys.modules.pop("agent.servers.ast_server", None)


# ---------------------------------------------------------------------------
# fs_server._resolve
# ---------------------------------------------------------------------------

class TestFsResolve:
    def test_absolute_passes_through(self, fs_mod):
        mod, tmp_path = fs_mod
        abs_path = str(tmp_path / "some" / "file.py")
        assert mod._resolve(abs_path) == Path(abs_path)

    def test_bare_filename_resolves_under_root(self, fs_mod):
        mod, tmp_path = fs_mod
        result = mod._resolve("calculator.py")
        assert result == (tmp_path / "calculator.py").resolve()

    def test_subdir_relative_path(self, fs_mod):
        mod, tmp_path = fs_mod
        result = mod._resolve("sub/utils.py")
        assert result == (tmp_path / "sub" / "utils.py").resolve()

    def test_dotdot_collapses(self, fs_mod):
        mod, tmp_path = fs_mod
        result = mod._resolve("sub/../calculator.py")
        assert result == (tmp_path / "calculator.py").resolve()

    def test_no_env_var_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_REPO_ROOT", raising=False)
        sys.modules.pop("agent.servers.fs_server", None)
        import agent.servers.fs_server as mod
        result = mod._resolve("calculator.py")
        assert result == Path("calculator.py")
        sys.modules.pop("agent.servers.fs_server", None)

    def test_returns_path_object(self, fs_mod):
        mod, tmp_path = fs_mod
        result = mod._resolve("file.py")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# ast_server._resolve
# ---------------------------------------------------------------------------

class TestAstResolve:
    def test_absolute_passes_through(self, ast_mod):
        mod, tmp_path = ast_mod
        abs_path = str(tmp_path / "foo.py")
        assert mod._resolve(abs_path) == abs_path

    def test_bare_filename_resolves_under_root(self, ast_mod):
        mod, tmp_path = ast_mod
        result = mod._resolve("calculator.py")
        assert result == str((tmp_path / "calculator.py").resolve())

    def test_subdir_relative_path(self, ast_mod):
        mod, tmp_path = ast_mod
        result = mod._resolve("sub/utils.py")
        assert result == str((tmp_path / "sub" / "utils.py").resolve())

    def test_dotdot_collapses(self, ast_mod):
        mod, tmp_path = ast_mod
        result = mod._resolve("sub/../calculator.py")
        assert result == str((tmp_path / "calculator.py").resolve())

    def test_no_env_var_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_REPO_ROOT", raising=False)
        sys.modules.pop("agent.servers.ast_server", None)
        import agent.servers.ast_server as mod
        result = mod._resolve("calculator.py")
        assert result == "calculator.py"
        sys.modules.pop("agent.servers.ast_server", None)

    def test_returns_str(self, ast_mod):
        mod, tmp_path = ast_mod
        result = mod._resolve("file.py")
        assert isinstance(result, str)
