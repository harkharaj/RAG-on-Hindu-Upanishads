"""Embed the processed verses and store them in a persistent ChromaDB index.

Indexes every corpus found in data/processed/ into one collection, tagged
with a "source" metadata field so retrieval can filter or compare:
    verses.json      -> source "upanishads"
    ashtavakra.json  -> source "ashtavakra"
    gita.json        -> source "gita"

Run once after the ingest scripts (and again whenever they change):
    python src/index.py
"""

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CORPORA = {
    "upanishads": ROOT / "data" / "processed" / "verses.json",
    "ashtavakra": ROOT / "data" / "processed" / "ashtavakra.json",
    "gita": ROOT / "data" / "processed" / "gita.json",
}
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION = "upanishads"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
BATCH = 256


def main() -> None:
    verses = []
    for source, path in CORPORA.items():
        if not path.exists():
            print(f"NOTE: {path.name} not found, skipping source '{source}'")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for r in records:
            r.setdefault("source", source)
        print(f"  {source}: {len(records)} chunks from {path.name}")
        verses.extend(records)
    # ids from ingest are unique per section; make globally unique defensively
    seen: dict[str, int] = {}
    for v in verses:
        n = seen.get(v["id"], 0)
        seen[v["id"]] = n + 1
        if n:
            v["id"] = f"{v['id']}~{n}"

    print(f"Embedding {len(verses)} chunks with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(verses), BATCH):
        batch = verses[i : i + BATCH]
        texts = [v["text"] for v in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        coll.add(
            ids=[v["id"] for v in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "source": v["source"],
                    "upanishad": v["upanishad"],
                    "ref": v["ref"],
                    "translator": v.get("translator", ""),
                }
                for v in batch
            ],
        )
        print(f"  indexed {min(i + BATCH, len(verses))}/{len(verses)}", end="\r")

    print(f"\nDone. Collection '{COLLECTION}' has {coll.count()} chunks at {CHROMA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
