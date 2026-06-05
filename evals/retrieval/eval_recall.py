"""Recall@k eval for the tool-retrieval layer.

Usage:
    python -m evals.retrieval.eval_recall

Requires DATABASE_URL in the environment (pgvector must be running and
populated: run the agent at least once before evaluating, or call
PgVectorStore.upsert() directly from a setup script).

Prints recall@5, recall@10, recall@12 and a per-tool breakdown of misses.
"""
import asyncio
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on the path when run as a script.
_SRC = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent.retrieval.embedder import Embedder  # noqa: E402
from agent.retrieval.store import PgVectorStore  # noqa: E402
from evals.retrieval.labeled_pairs import PAIRS  # noqa: E402

_K_VALUES = [5, 10, 12]


async def _run() -> None:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to pgvector at: {db_url}", flush=True)
    store = PgVectorStore(db_url)
    total = await store.count()
    if total == 0:
        print(
            "ERROR: tool_registry table is empty. "
            "Run the agent once to populate it, then re-run this eval.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Registry contains {total} tool embeddings.", flush=True)
    embedder = Embedder()

    hits_at: dict[int, int] = {k: 0 for k in _K_VALUES}
    misses_by_tool: dict[str, list[str]] = defaultdict(list)  # tool → [goal texts missed]

    max_k = max(_K_VALUES)
    for goal, correct_tool in PAIRS:
        vec = embedder.embed(goal)
        results = await store.top_k(vec, max_k)
        retrieved_names = [name for _, name in results]

        for k in _K_VALUES:
            if correct_tool in retrieved_names[:k]:
                hits_at[k] += 1
            elif k == max_k:
                misses_by_tool[correct_tool].append(goal)

    n = len(PAIRS)
    print("\n--- Retrieval Recall@k ---")
    for k in _K_VALUES:
        recall = hits_at[k] / n
        print(f"  recall@{k:<2} = {recall:.3f}  ({hits_at[k]}/{n})")

    if misses_by_tool:
        print(f"\n--- Misses at k={max_k} ({sum(len(v) for v in misses_by_tool.values())} total) ---")
        for tool, goals in sorted(misses_by_tool.items()):
            print(f"  {tool}:")
            for g in goals:
                print(f"    - {g}")
    else:
        print(f"\nPerfect recall at k={max_k}. No misses.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
