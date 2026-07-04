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
# BGE models retrieve better when the *query* (not the passages) is prefixed
# with their instruction string.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBEDDING_MODEL.lower()
    else ""
)
# Two-stage retrieval: cast a wide net with the bi-encoder, then let a
# cross-encoder rerank down to k. On by default (local + free, measured
# hit@6 0.67 -> 0.87 on the eval set); set RERANKER=none for single-stage.
RERANKER = os.environ.get("RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
if RERANKER.lower() in ("", "none", "off"):
    RERANKER = ""
FETCH_K = int(os.environ.get("FETCH_K", "50"))


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER)


def retrieve(query: str, k: int = 5, upanishad: str | None = None) -> list[dict]:
    """Return top-k verse chunks: {upanishad, ref, text, translator, score}."""
    embedding = _model().encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    where = {"upanishad": upanishad} if upanishad else None
    res = _collection().query(
        query_embeddings=embedding,
        n_results=max(FETCH_K, k) if RERANKER else k,
        where=where,
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
    if RERANKER and len(hits) > k:
        scores = _reranker().predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = round(float(s), 4)  # cross-encoder relevance score
        hits = sorted(hits, key=lambda h: -h["score"])[:k]
    return hits


def main() -> None:
    query = " ".join(sys.argv[1:]) or "what happens to the self after death?"
    print(f"Query: {query}\n")
    for h in retrieve(query, k=5):
        print(f"[{h['upanishad']} {h['ref']}]  (score {h['score']})")
        print(f"  {h['text'][:300]}\n")


if __name__ == "__main__":
    main()
