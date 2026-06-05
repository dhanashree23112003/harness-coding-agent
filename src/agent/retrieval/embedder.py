import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around SentenceTransformer. Instantiate once at startup."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized (384,) float32 vector."""
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return normalized (N, 384) float32 array."""
        return self._model.encode(texts, normalize_embeddings=True)
