"""Unit tests for embedder.py: shape, dtype, normalization."""
import numpy as np
import pytest

from agent.retrieval.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_returns_correct_shape(embedder: Embedder):
    vec = embedder.embed("read the contents of a file")
    assert vec.shape == (384,)


def test_embed_returns_float32(embedder: Embedder):
    vec = embedder.embed("list all python files")
    assert vec.dtype == np.float32


def test_embed_is_normalized(embedder: Embedder):
    vec = embedder.embed("show git status")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_embed_is_not_zero_vector(embedder: Embedder):
    vec = embedder.embed("commit staged changes")
    assert not np.allclose(vec, 0.0)


def test_embed_batch_shape(embedder: Embedder):
    texts = ["read file", "git status", "list directory"]
    vecs = embedder.embed_batch(texts)
    assert vecs.shape == (3, 384)


def test_embed_batch_normalized(embedder: Embedder):
    texts = ["search files", "show diff"]
    vecs = embedder.embed_batch(texts)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_different_texts_produce_different_vectors(embedder: Embedder):
    v1 = embedder.embed("read a file from the filesystem")
    v2 = embedder.embed("commit changes to the git repository")
    assert not np.allclose(v1, v2)


def test_similar_texts_produce_similar_vectors(embedder: Embedder):
    v1 = embedder.embed("read the contents of a file")
    v2 = embedder.embed("open and display a file")
    v3 = embedder.embed("create a new git branch")
    sim_same = float(np.dot(v1, v2))
    sim_diff = float(np.dot(v1, v3))
    assert sim_same > sim_diff, (
        f"Expected 'read file' phrasings to be more similar to each other "
        f"than to 'create branch'. sim_same={sim_same:.3f}, sim_diff={sim_diff:.3f}"
    )
