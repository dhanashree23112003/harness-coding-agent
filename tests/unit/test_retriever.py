"""Unit tests for retriever.py: always-include, mock store, miss widening."""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agent.retrieval.retriever import ALWAYS_INCLUDE, ToolRetriever, _DEFAULT_K, _K_WIDEN_STEP


def _make_retriever(hits: list[tuple[str, str]], total: int = 23) -> ToolRetriever:
    store = MagicMock()
    store.top_k = AsyncMock(return_value=hits)

    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))

    return ToolRetriever(store=store, embedder=embedder, total=total)


@pytest.mark.asyncio
async def test_retrieve_returns_hits():
    retriever = _make_retriever([("fs", "read_file"), ("fs", "list_dir")])
    names = await retriever.retrieve("read a file", k=5)
    assert "read_file" in names
    assert "list_dir" in names


@pytest.mark.asyncio
async def test_always_include_present_regardless_of_hits():
    retriever = _make_retriever([("git", "git_log"), ("fs", "search_files")])
    names = await retriever.retrieve("show git log", k=5)
    for always in ALWAYS_INCLUDE:
        assert always in names, f"Always-include tool '{always}' missing from results"


@pytest.mark.asyncio
async def test_retrieve_wider_increments_k():
    retriever = _make_retriever([("fs", "read_file")], total=23)
    names, new_k = await retriever.retrieve_wider("read a file", current_k=_DEFAULT_K)
    assert new_k == _DEFAULT_K + _K_WIDEN_STEP


@pytest.mark.asyncio
async def test_retrieve_wider_caps_at_total():
    retriever = _make_retriever([("fs", "read_file")], total=23)
    names, new_k = await retriever.retrieve_wider("anything", current_k=21)
    assert new_k == 23  # capped at total


@pytest.mark.asyncio
async def test_retrieve_calls_store_top_k_with_min_k():
    store = MagicMock()
    store.top_k = AsyncMock(return_value=[])
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=np.zeros(384, dtype=np.float32))

    retriever = ToolRetriever(store=store, embedder=embedder, total=5)
    await retriever.retrieve("goal", k=100)

    # k should be capped at total (5), not 100
    store.top_k.assert_called_once()
    _, kwargs = store.top_k.call_args
    called_k = store.top_k.call_args[0][1]
    assert called_k == 5


@pytest.mark.asyncio
async def test_retrieve_returns_list():
    retriever = _make_retriever([("git", "git_status")])
    names = await retriever.retrieve("check git status")
    assert isinstance(names, list)


# --- widen_node and miss cap tests ---

from agent.graph.nodes import _MISS_CAP, widen_node  # noqa: E402


def test_widen_node_increments_k_and_miss_count():
    state = {"retrieval_k": _DEFAULT_K, "retrieval_miss_count": 0}
    result = widen_node(state)
    assert result["retrieval_k"] == _DEFAULT_K + _K_WIDEN_STEP
    assert result["retrieval_miss_count"] == 1


def test_widen_node_accumulates_across_calls():
    state = {"retrieval_k": _DEFAULT_K + _K_WIDEN_STEP, "retrieval_miss_count": 1}
    result = widen_node(state)
    assert result["retrieval_k"] == _DEFAULT_K + 2 * _K_WIDEN_STEP
    assert result["retrieval_miss_count"] == 2


def test_miss_cap_is_finite_and_reasonable():
    assert 1 <= _MISS_CAP <= 10
