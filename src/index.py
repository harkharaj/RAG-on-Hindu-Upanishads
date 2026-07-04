"""Embed the processed verses and store them in a persistent ChromaDB index.

Run once after ingest.py (and again whenever verses.json changes):
    python src/index.py
"""

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
VERSES_PATH = ROOT / "data" / "processed" / "verses.json"
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION = "upanishads"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
BATCH = 256


def main() -> None:
    verses = json.loads(VERSES_PATH.read_text(encoding="utf-8"))
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
