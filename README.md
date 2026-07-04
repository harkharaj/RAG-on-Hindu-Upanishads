# Upanishad RAG — a source-cited chatbot for the 108 Upanishads

A retrieval-augmented chatbot that answers questions about life, death, the self,
and other themes **strictly from the text of the Upanishads**, and shows the
exact verse each answer draws from.

Retrieval runs locally and free. Generation runs on a free hosted open-source LLM
(Groq). No paid API is required to build, run, or demo this project.

**Status: built.** The corpus is all 108 Upanishads (5,600+ verse chunks) parsed
from a single compiled PDF, indexed in ChromaDB with local embeddings, answered
by Llama 3.3 70B on Groq with per-verse citations, wrapped in a Streamlit chat
UI, and measured with a 15-question retrieval + faithfulness eval.

---

## 1. What this project demonstrates

- A full RAG pipeline built and understood end to end (not just a framework call).
- Real-world data engineering: parsing 900 PDF pages with four different
  verse-numbering conventions, interleaved commentary, and a trailing glossary
  into clean, metadata-rich chunks.
- Semantic search over a real corpus using open-source embeddings.
- Source attribution — every answer cites Upanishad and verse (e.g. `Katha 1-III-3`).
- Grounding / anti-hallucination prompting, which matters for scripture.
- A measured evaluation of retrieval quality and answer faithfulness.

---

## 2. Architecture

**Indexing (run once, offline):**

```
108upanishads.pdf → ingest.py (split by verse + metadata) → index.py (embed locally) → ChromaDB
```

**Query (each question):**

```
Question → embed locally → top-50 similarity search → cross-encoder rerank to top-6 → Groq LLM (grounded prompt) → answer + citations
```

The same embedding model (`all-MiniLM-L6-v2`) is used at index and query time —
required for the vectors to be comparable. Retrieval is two-stage: the
bi-encoder casts a wide net, then `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranks (measured hit@6 0.67 → 1.00 on the eval set; disable with
`RERANKER=none`).

---

## 3. Tech stack

| Layer          | Choice                                          | Cost |
|----------------|--------------------------------------------------|------|
| Language       | Python 3.10+                                     | free |
| PDF parsing    | `pypdf`                                          | free |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local) | free |
| Reranker       | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)   | free |
| Vector store   | ChromaDB (persistent, on disk)                   | free |
| Generation LLM | Groq free API — `llama-3.3-70b-versatile`        | free |
| Frontend       | Streamlit                                        | free |
| Evaluation     | Custom hit@k / MRR + LLM-judge faithfulness      | free |

**Free-tier note:** Groq's free tier allows ~30 requests/minute and ~1,000
requests/day on the 70B models — plenty for development and demos. A backup key
on another free provider (Cerebras, Mistral) gives an independent quota.

---

## 4. Data

The corpus is `108upanishads.pdf` (compiled by Richard Sheppard, 2009): all 108
Upanishads of the Muktika canon in English translation, 908 pages.

What ingestion handles:

- **109 sections** detected by header scan (the compiler splits Nrisimha Tapaniya
  into Poorva/Uttara).
- **Four verse-numbering styles**, auto-detected per text: `1.` (Isa),
  `I-1:` (Prasna), `I-i-1:` (Mundaka, Chandogya, Brihadaranyaka), and
  `1-I-1.` (Katha). Texts with no verse numbers fall back to sentence-grouped
  passages.
- **Commentary removal** — the commentaries by Swami Nirmalananda Giri on the
  first 10 Upanishads are cut so answers come only from the scripture itself.
- The trailing **Sanskrit glossary** (pages 844+) is excluded.
- Invocations (shanti mantras) are kept as `Invocation` chunks.
- Verses longer than ~1,200 chars are split at sentence boundaries (same verse
  ref) so they fit the embedding model's window.

Every chunk carries metadata: `upanishad`, `ref` (verse reference as printed),
`translator`. This powers the citations in every answer.

> ⚠️ **Copyright:** these translations (Swami Gambhirananda, Swami Madhavananda,
> Swami Swahananda, V. Panoli, et al.) are **not** public domain. The PDF,
> `data/`, and everything derived from the text are gitignored — publish the
> *code*, not the corpus. Anyone cloning the repo drops their own copy of the
> PDF in the project root and rebuilds the index locally. For a publicly
> deployed demo, swap in the public-domain Max Müller translation
> (sacred-texts.com) or verify permission.

---

## 5. Setup

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Free Groq API key from https://console.groq.com  (no credit card)
copy .env.example .env            # then paste your key into .env

# 3. Build the corpus + index (one-off, ~2 min on CPU)
python src/ingest.py
python src/index.py

# 4. Sanity-check retrieval (no API key needed)
python -m src.retrieve "what happens to the self after death?"

# 5. Ask a question end-to-end (needs GROQ_API_KEY)
python -m src.rag "What is the parable of the two birds?"

# 6. Chat UI
streamlit run app.py
```

---

## 6. Project structure

```
RAG_UPANISHADS/
├── 108upanishads.pdf       # source corpus (not committed)
├── data/
│   ├── raw/pages.json      # cached PDF text extraction (not committed)
│   ├── processed/verses.json  # verse chunks + metadata (not committed)
│   └── chroma/             # persistent vector index (not committed)
├── src/
│   ├── ingest.py           # PDF → verse chunks with metadata
│   ├── index.py            # embed chunks → ChromaDB
│   ├── retrieve.py         # embed query → top-k similarity search
│   ├── generate.py         # grounded prompt → Groq
│   └── rag.py              # retrieve + generate, CLI entry point
├── eval/
│   ├── testset.json        # 15 questions with expected source verses
│   └── evaluate.py         # hit@k, MRR, LLM-judge faithfulness
├── app.py                  # Streamlit chat UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. Evaluation

```bash
python -m eval.evaluate           # retrieval metrics only (no API key)
python -m eval.evaluate --full    # + generation faithfulness (needs Groq key)
```

The test set is 15 hand-written questions targeting well-known passages (the
two birds, the chariot analogy, *tat tvam asi*, *neti neti*, the thunder's
Da-Da-Da, ...), each with the verse refs a correct retrieval should surface.

Metrics:

- **hit@6** — did any expected verse appear in the top 6 retrieved chunks?
- **MRR** — reciprocal rank of the first expected verse.
- **Faithfulness** — an LLM judge grades each generated answer
  SUPPORTED / PARTIAL / UNSUPPORTED against the retrieved passages only.

Record before/after numbers when changing chunking, k, the embedding model, or
the prompt.

| Date | Change | hit@6 | MRR |
|------|--------|-------|-----|
| 2026-07-02 | baseline: MiniLM-L6 single-stage, k=6 | 0.67 | 0.44 |
| 2026-07-02 | swap embeddings to bge-small-en-v1.5 | 0.60 ↓ | 0.47 |
| 2026-07-02 | MiniLM-L6 + ms-marco cross-encoder rerank (top-50 → 6) | 0.80 | 0.67 |
| 2026-07-02 | + testset fix: credit parallel passages (verified by hand) | **1.00** | **0.87** |

Two findings worth noting: a "better" embedding model (bge-small) actually
*hurt* on this corpus of archaic translation English — the reranker was the
real win; and on a 108-text corpus the gold labels must allow parallel
passages (e.g. Brihadaranyaka IV-iv-18 teaches the same "Mind of the mind"
doctrine as Kena I-2), or the eval undercounts genuinely correct retrieval.

---

## 8. Design notes specific to scripture

- **Always cite.** Every answer shows the source verses beneath it.
- **Faithfulness over fluency.** Temperature 0.2 + a strict grounding prompt.
  The bot says "the provided passages don't address this" rather than
  improvising spiritual claims.
- **Preserve disagreement.** Different Upanishads differ; the prompt instructs
  the model to present differing views rather than merging them.

---

## 9. Stretch goals

- ~~**Reranker (two-stage retrieval).**~~ **Done** — top-50 bi-encoder →
  ms-marco cross-encoder → top-6, on by default. hit@6 0.67 → 1.00.
- **Better embeddings.** `bge-small-en-v1.5` was tried and measured *worse*
  (see §7); `bge-base`/`bge-m3` remain untested and might behave differently.
- **Offline mode.** Point `generate.py` at a local Ollama server (also
  OpenAI-compatible) behind a config flag — zero-internet demo.
- **Indian-language questions.** Add a Sarvam open model so users can ask in
  Hindi or another Indian language.
- **Glossary boost.** The PDF's Sanskrit glossary (pages 844+) could be indexed
  as a separate collection to help with term-heavy queries (Atman, Moksha...).

---

## 10. License

Code: MIT. Source texts: copyrighted translations, **not** distributed with
this repo — see the copyright note in §4.
