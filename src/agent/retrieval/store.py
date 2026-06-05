import asyncio
import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from agent.retrieval.registry import ToolRegistryEntry

# psycopg3 async requires SelectorEventLoop, which Windows does not support
# alongside subprocess-based MCP stdio. All DB work runs via psycopg3 sync
# API dispatched through run_in_executor so the main ProactorEventLoop is
# never touched by psycopg.

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tool_registry (
    id          SERIAL PRIMARY KEY,
    namespace   TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema JSONB NOT NULL,
    embedding   vector(384) NOT NULL,
    UNIQUE(namespace, name)
);

CREATE INDEX IF NOT EXISTS tool_registry_hnsw
    ON tool_registry USING hnsw (embedding vector_cosine_ops);
"""

_UPSERT_SQL = """
INSERT INTO tool_registry (namespace, name, description, input_schema, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (namespace, name) DO UPDATE SET
    description  = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    embedding    = EXCLUDED.embedding;
"""

_TOP_K_SQL = """
SELECT namespace, name
FROM tool_registry
ORDER BY embedding <=> %s
LIMIT %s;
"""


class PgVectorStore:
    """pgvector store for tool embeddings.

    Public API is async (awaitable from LangGraph nodes), but all I/O is
    done via psycopg3 sync connections dispatched through run_in_executor.
    This keeps ProactorEventLoop intact for MCP subprocess transport.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    # --- sync helpers (run in executor thread) ---

    def _sync_init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()

    def _sync_upsert(self, rows: list[tuple]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(_UPSERT_SQL, rows)
            conn.commit()

    def _sync_top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[str, str]]:
        with self._connect() as conn:
            cur = conn.execute(_TOP_K_SQL, (query_vec, k))
            return [(row[0], row[1]) for row in cur.fetchall()]

    def _sync_count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM tool_registry;")
            row = cur.fetchone()
            return row[0] if row else 0

    # --- async public API ---

    async def init_schema(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_init_schema)

    async def upsert(
        self,
        entries: list[ToolRegistryEntry],
        embeddings: np.ndarray,
    ) -> None:
        rows = [
            (
                e.namespace,
                e.name,
                e.description,
                json.dumps(e.input_schema),
                embeddings[i],
            )
            for i, e in enumerate(entries)
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_upsert, rows)

    async def top_k(
        self,
        query_vec: np.ndarray,
        k: int,
    ) -> list[tuple[str, str]]:
        """Return [(namespace, name), ...] in ascending cosine-distance order."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_top_k, query_vec, k)

    async def count(self) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_count)
