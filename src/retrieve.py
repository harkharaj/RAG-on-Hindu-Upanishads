"""Semantic search over the indexed verses.

Library use:
    from src.retrieve import retrieve
    passages = retrieve("what happens to the self after death?", k=5)

CLI (prints the top-k hits so you can eyeball retrieval quality):
    python -m src.retrieve "what happens after death"
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION = "upanishads"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION)


def retrieve(query: str, k: int = 5, upanishad: str | None = None) -> list[dict]:
    """Return top-k verse chunks: {upanishad, ref, text, translator, score}."""
    embedding = _model().encode([query], normalize_embeddings=True).tolist()
    where = {"upanishad": upanishad} if upanishad else None
    res = _collection().query(
        query_embeddings=embedding, n_results=k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "upanishad": meta["upanishad"],
                "ref": meta["ref"],
                "translator": meta.get("translator", ""),
                "text": doc,
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )
    return hits


def main() -> None:
    query = " ".join(sys.argv[1:]) or "what happens to the self after death?"
    print(f"Query: {query}\n")
    for h in retrieve(query, k=5):
        print(f"[{h['upanishad']} {h['ref']}]  (score {h['score']})")
        print(f"  {h['text'][:300]}\n")


if __name__ == "__main__":
    main()
