# Vedanta RAG — a source-cited chatbot for the Upanishads, Ashtavakra Gita & Bhagavad Gita

A retrieval-augmented chatbot that answers questions about life, death, the
self, and other themes **strictly from the text of three Hindu scriptures** —
the 108 Upanishads, the Ashtavakra Gita, and the Bhagavad Gita — and shows the
exact verse each answer draws from.

Ask one text alone, or several at once: in comparative mode each tradition
answers in its own section, and the bot explicitly flags **where the texts
agree and where they differ** (e.g. the Upanishads' formal renunciation vs.
Ashtavakra's purely inner renunciation).

Retrieval runs locally and free. Generation runs on a free hosted open-source
LLM (Groq). No paid API is required to build, run, or demo this project.

**Status: built and measured.** 6,589 verse chunks across three corpora,
indexed in ChromaDB with local embeddings and cross-encoder reranking,
answered by Llama 3.3 70B on Groq with per-verse citations, wrapped in a
Streamlit chat UI — **hit@6 0.97 / MRR 0.89** on a 35-question ground-truth
eval, **34/35** LLM-judged faithfulness, **8/8** well-formed comparative
answers (full numbers in [`eval/RESULTS.md`](eval/RESULTS.md)).

---

## 1. What this project demonstrates

- A full RAG pipeline built and understood end to end (not just a framework call).
- Real-world data engineering: three PDFs with completely different layouts,
  parsed into clean, metadata-rich verse chunks:
  - 900 pages with four different verse-numbering conventions (Upanishads),
  - **font-aware extraction** — verse translations are italic, commentary is
    roman — validated against canonical verse counts (Ashtavakra, 298/298),
  - translation/commentary separation by structural markers (Gita).
- Multi-corpus retrieval with per-source filtering, so one text can't crowd
  the others out of the context window.
- Comparative generation across traditions with an explicit
  "They differ:" / "They agree:" verdict.
- Source attribution — every answer cites work and verse (e.g. `Katha 1-III-3`,
  `Ashtavakra 15.4`, `Gita 2.47`).
- Grounding / anti-hallucination prompting, which matters for scripture.
- A three-part evaluation: retrieval metrics per corpus, deterministic
  structure checks on comparative answers, and LLM-judged faithfulness.

---

## 2. Architecture

**Indexing (run once, offline):**

```
108upanishads.pdf ──→ ingest.py ────────────┐
ASHTAVAKRA_GITA.pdf ─→ ingest_ashtavakra.py ├─→ index.py (embed locally) → ChromaDB
Bhagavad-Gita....pdf ─→ ingest_gita.py ─────┘      (one collection, "source" metadata)
```

**Query (each question):**

```
Question → embed locally → top-50 similarity search (per selected corpus)
        → cross-encoder rerank to top-k → Groq LLM (grounded prompt)
        → answer + citations   [+ per-tradition sections & agree/differ verdict
                                 when more than one corpus is selected]
```

The same embedding model (`all-MiniLM-L6-v2`) is used at index and query time —
required for the vectors to be comparable. Retrieval is two-stage: the
bi-encoder casts a wide net, then `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranks (measured hit@6 0.67 → 1.00 on the original eval set; disable with
`RERANKER=none`). In comparative mode, retrieval runs once per selected corpus
so each tradition contributes its own best passages.

---

## 3. Tech stack

| Layer          | Choice                                           | Cost |
|----------------|--------------------------------------------------|------|
| Language       | Python 3.10+                                     | free |
| PDF parsing    | `pypdf` (incl. font-level extraction visitor)    | free |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local) | free |
| Reranker       | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)   | free |
| Vector store   | ChromaDB (persistent, on disk)                   | free |
| Generation LLM | Groq free API — `llama-3.3-70b-versatile`        | free |
| Frontend       | Streamlit                                        | free |
| Evaluation     | Custom hit@k / MRR + structure checks + LLM judge| free |

**Free-tier note:** Groq's free tier allows ~30 requests/minute and ~1,000
requests/day on the 70B models — plenty for development and demos. A backup key
on another free provider (Cerebras, Mistral) gives an independent quota.

---

## 4. Data

Three corpora, one ChromaDB collection. Every chunk carries metadata:
`source` (corpus), `upanishad` (work name), `ref` (verse reference as
printed), `translator` — this powers both citations and per-corpus retrieval.

### 4.1 The 108 Upanishads (`source: upanishads`, 5,653 chunks)

`108upanishads.pdf` (compiled by Richard Sheppard, 2009): the Muktika canon in
English translation, 908 pages. Ingestion handles:

- **109 sections** detected by header scan (Nrisimha Tapaniya split in two).
- **Four verse-numbering styles**, auto-detected per text: `1.` (Isa),
  `I-1:` (Prasna), `I-i-1:` (Mundaka), `1-I-1.` (Katha); texts with no verse
  numbers fall back to sentence-grouped passages.
- **Commentary removal** and exclusion of the trailing Sanskrit glossary.
- Verses longer than ~1,200 chars are split at sentence boundaries (same ref).

### 4.2 Ashtavakra Gita (`source: ashtavakra`, 298 chunks)

`ASHTAVAKRA_GITA.pdf` (Chinmaya Mission ed., tr. Swami Chinmayananda). Each
verse is buried between Devanagari, transliteration, a word-by-word gloss and
pages of commentary, with no paragraph markers. The parser anchors on the
transliteration line ending `(N)`, then uses **font information** — the verse
translation is set in italic, the commentary in roman — to find where the
translation ends. All **298/298 verses** extracted, validated against the
canonical per-chapter counts (one verse is mis-set in roman in the book itself
and handled by a punctuation fallback).

### 4.3 Bhagavad Gita — Shankara Bhashya ed. (`source: gita`, 638 chunks)

`Srimad-Bhagavad-Gita-Shankara-Bhashya-English.pdf` (tr. Swami Gambhirananda,
via Project Gutenberg). Verse translations are separated from Sankara's
commentary by a recurring separator line; the commentary re-opens with the
same verse ref, so a new verse starts only at a ref *beyond* the last covered
range (this matters for grouped verses like `2.42-43`). Covers chapters 2–18
from 2.10 — Sankara wrote no commentary on chapter 1 or 2.1–2.9 — and two
verses (17.20, 18.32) are missing from the source file itself.

In all three cases **only the scripture's own words are indexed** — the
commentaries (Nirmalananda, Chinmayananda, Sankara) are stripped, so each
tradition speaks in its own voice and the corpora stay comparable.

> ⚠️ **Copyright:** the Upanishad and Ashtavakra translations are **not**
> public domain (Ramakrishna Math, Chinmaya Mission, et al.). The PDFs,
> `data/`, and everything derived from the texts are gitignored — publish the
> *code*, not the corpus. Anyone cloning the repo drops their own copies of
> the PDFs in the project root and rebuilds the index locally. For a publicly
> deployed demo, swap in public-domain translations (e.g. Max Müller,
> sacred-texts.com) or verify permission.

---

## 5. Setup

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Free Groq API key from https://console.groq.com  (no credit card)
copy .env.example .env            # then paste your key into .env

# 3. Build the corpora + index (one-off, ~3 min on CPU)
python src/ingest.py              # 108 Upanishads
python src/ingest_ashtavakra.py   # Ashtavakra Gita
python src/ingest_gita.py         # Bhagavad Gita
python src/index.py

# 4. Sanity-check retrieval (no API key needed)
python -m src.retrieve "what happens to the self after death?"

# 5. Ask a question end-to-end (needs GROQ_API_KEY)
python -m src.rag "How is liberation attained?"                    # all three texts, comparative
python -m src.rag --scope gita "Do we own the fruits of action?"   # one text only

# 6. Chat UI (pick texts in the sidebar)
streamlit run app.py
```

---

## 6. Project structure

```
RAG-on-Hindu-Upanishads/
├── 108upanishads.pdf            # source corpora (not committed)
├── ASHTAVAKRA_GITA.pdf
├── Srimad-Bhagavad-Gita-....pdf
├── data/                        # caches, chunks, vector index (not committed)
├── src/
│   ├── ingest.py                # Upanishads PDF → verse chunks
│   ├── ingest_ashtavakra.py     # Ashtavakra PDF → verse chunks (font-aware)
│   ├── ingest_gita.py           # Gita PDF → verse chunks
│   ├── index.py                 # embed all corpora → ChromaDB
│   ├── retrieve.py              # top-k search, per-source filter, reranker
│   ├── generate.py              # grounded + comparative prompts → Groq
│   └── rag.py                   # retrieve + generate, scope logic, CLI
├── eval/
│   ├── testset.json             # 35 questions with expected source verses
│   ├── compare_testset.json     # 8 cross-tradition questions
│   ├── evaluate.py              # hit@k, MRR, structure checks, LLM judge
│   └── RESULTS.md               # recorded results
├── app.py                       # Streamlit chat UI
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## 7. Evaluation

```bash
python -m eval.evaluate            # retrieval metrics only (no API key)
python -m eval.evaluate --compare  # + comparative-answer structure checks
python -m eval.evaluate --full     # + generation faithfulness (needs Groq key)
```

The test set is 35 hand-written questions targeting well-known passages —
15 Upanishad (the two birds, the chariot, *tat tvam asi*, *neti neti*, ...),
10 Ashtavakra (the one witness, cloth-and-thread, "where is dream...", ...),
10 Gita (worn-out clothes, right to action alone, the field and its knower,
...) — each pinned to its corpus and carrying the verse refs a correct
retrieval should surface.

Metrics:

- **hit@6 / MRR** — per corpus and overall, retrieval restricted to the
  question's own corpus.
- **Comparative structure** — deterministic checks that a multi-corpus answer
  has one section per text, a "They differ:"/"They agree:" verdict, and at
  least one citation from *every* corpus.
- **Faithfulness** — an LLM judge grades each generated answer
  SUPPORTED / PARTIAL / UNSUPPORTED against the retrieved passages only.

| Date | Change | hit@6 | MRR |
|------|--------|-------|-----|
| 2026-07-02 | baseline: MiniLM-L6 single-stage, k=6 (15 Q) | 0.67 | 0.44 |
| 2026-07-02 | swap embeddings to bge-small-en-v1.5 | 0.60 ↓ | 0.47 |
| 2026-07-02 | MiniLM-L6 + ms-marco cross-encoder rerank (top-50 → 6) | 0.80 | 0.67 |
| 2026-07-02 | + testset fix: credit parallel passages | **1.00** | **0.87** |
| 2026-07-03 | + Ashtavakra & Bhagavad Gita corpora, testset 15 → 35 Q | **0.97** | **0.89** |

Current per-corpus numbers (see [`eval/RESULTS.md`](eval/RESULTS.md)):
Upanishads 1.00 / 0.87 · Ashtavakra 1.00 / 0.95 · Gita 0.90 / 0.85. The one
miss (Gita 4.7–8) traces to OCR corruption in the source PDF, not retrieval.
Comparative structure: **8/8** well-formed. Faithfulness: **34/35 SUPPORTED**
— the one flagged answer is the bot *refusing* to improvise when the directly
relevant Kena verses missed the top-6: an honest refusal is the designed
failure mode; a hallucinated answer would be the bad one.

Off-corpus sanity check: asked about "smartphones and social media" — the bot
replies that the passages don't address it, rather than inventing spiritual
claims.

Two findings worth noting: a "better" embedding model (bge-small) actually
*hurt* on this corpus of archaic translation English — the reranker was the
real win; and gold labels must credit parallel passages (many Upanishads teach
the same doctrine), or the eval undercounts genuinely correct retrieval.

---

## 8. Design notes specific to scripture

- **Always cite.** Every answer shows the source verses beneath it.
- **Faithfulness over fluency.** Temperature 0.2 + a strict grounding prompt.
  The bot says "the provided passages don't address this" rather than
  improvising spiritual claims.
- **Preserve disagreement.** Within one text, differing views are presented,
  not merged. Across texts, disagreement is promoted to a first-class output:
  the comparative answer must commit to "They differ:" or "They agree:".
- **Let each text speak for itself.** Commentaries are stripped at ingestion;
  in comparative mode each corpus gets its own retrieval budget and its own
  answer section, so the Gita's karma-yoga can't be averaged into Ashtavakra's
  radical non-doership.

---

## 9. Stretch goals

- ~~**Reranker (two-stage retrieval).**~~ **Done** — hit@6 0.67 → 1.00.
- ~~**More corpora.**~~ **Done** — Ashtavakra Gita + Bhagavad Gita, with
  comparative answers.
- **Offline mode.** Point `generate.py` at a local Ollama server (also
  OpenAI-compatible) behind a config flag — zero-internet demo.
- **Indian-language questions.** Add a Sarvam open model so users can ask in
  Hindi or another Indian language.
- **Commentary as a second layer.** Index the stripped commentaries
  (Sankara, Chinmayananda) as separate sources, so answers can distinguish
  "the text says" from "the commentator interprets".

---

## 10. License

Code: [MIT](LICENSE). Source texts: copyrighted translations, **not**
distributed with this repo — see the copyright note in §4.
