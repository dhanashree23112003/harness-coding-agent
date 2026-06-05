from agent.retrieval.embedder import Embedder
from agent.retrieval.store import PgVectorStore

ALWAYS_INCLUDE: frozenset[str] = frozenset({
    "read_file", "write_file", "git_status", "git_commit",
})
_DEFAULT_K = 12
_K_WIDEN_STEP = 6


class ToolRetriever:
    """Retrieves top-k relevant tools by semantic similarity.

    Always includes a core set regardless of similarity score.
    Exposes retrieve_wider() for the miss guard to call with an enlarged k.
    """

    def __init__(self, store: PgVectorStore, embedder: Embedder, total: int) -> None:
        self._store = store
        self._embedder = embedder
        self._total = total  # total number of registered tools (caps k widening)

    async def retrieve(self, goal: str, k: int = _DEFAULT_K) -> list[str]:
        vec = self._embedder.embed(goal)
        hits = await self._store.top_k(vec, min(k, self._total))
        names = {name for _, name in hits}
        names |= ALWAYS_INCLUDE
        return list(names)

    async def retrieve_wider(
        self, goal: str, current_k: int
    ) -> tuple[list[str], int]:
        """Return (tool_names, new_k) with k widened by one step."""
        new_k = min(current_k + _K_WIDEN_STEP, self._total)
        names = await self.retrieve(goal, new_k)
        return names, new_k
