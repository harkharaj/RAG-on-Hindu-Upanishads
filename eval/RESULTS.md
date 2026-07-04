# Evaluation results — 2026-07-03

**System under test:** three-corpus RAG over the 108 Upanishads (5,653 chunks),
the Ashtavakra Gita (298 chunks) and the Bhagavad Gita, Shankara Bhashya ed.
(638 chunks) — 6,589 verse chunks total in one ChromaDB collection with
per-corpus `source` metadata.

**Pipeline:** all-MiniLM-L6-v2 bi-encoder (fetch 50) → ms-marco-MiniLM-L-6-v2
cross-encoder rerank (top 6) → Groq llama-3.3-70b, temperature 0.2, grounded
prompt with mandatory verse citations. In multi-corpus mode, top-k is
retrieved from each corpus independently and the model must answer
per-tradition and issue an explicit "They differ:" / "They agree:" verdict.

## Retrieval (35 questions, ground-truth verse refs, k=6)

| Corpus | hit@6 | MRR | n |
|---|---|---|---|
| Upanishads | **1.00** | 0.87 | 15 |
| Ashtavakra Gita | **1.00** | 0.95 | 10 |
| Bhagavad Gita | 0.90 | 0.85 | 10 |
| **Overall** | **0.97** | **0.89** | 35 |

The single miss (Gita 4.7–8, "When does the Lord incarnate?") traces to OCR
corruption in the source PDF ("decline one virtue and increase of vi ce").

## Comparative mode (8 cross-tradition questions)

Deterministic structure checks on answers generated with all three corpora
enabled: one section per text present, a "They differ:"/"They agree:"
verdict present, and at least one citation from *each* corpus.

**8/8 well-formed.**

## Faithfulness (LLM judge, 35 single-corpus answers)

Groq llama-3.3-70b as judge: is every claim in the generated answer
supported by the retrieved passages?

**34/35 SUPPORTED, 0 partial, 1 unsupported** (the Kena "who impels the
mind" question — the judge flags an inference the answer draws across two
passages).

## Reproduce

```
python -m eval.evaluate            # retrieval only (no API key)
python -m eval.evaluate --compare  # + comparative structure checks
python -m eval.evaluate --full     # + faithfulness (needs GROQ_API_KEY)
```
