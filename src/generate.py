"""Grounded answer generation via Groq's free OpenAI-compatible API."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# Groq's free tier caps tokens per day *per model*. When the main model's
# quota is exhausted, retry once on a smaller model with its own separate
# quota instead of crashing. Set FALLBACK_MODEL=none to disable.
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "llama-3.1-8b-instant")

SYSTEM = """You answer questions using ONLY the scripture passages provided below.
Rules:
- Base every claim on the passages. Do not use outside knowledge.
- If the passages do not address the question, say so plainly.
- Cite each point with the bracketed reference of the passage it comes from,
  e.g. (Katha 1-II-18) or (Isavasya 5).
- Where the texts differ, present the different views rather than merging them."""

# Display metadata per corpus, in canonical order. Keys match the "source"
# metadata written by the ingest scripts.
SOURCES = {
    "upanishads": {"title": "the Upanishads", "section": "🕉️ What the Upanishads say"},
    "ashtavakra": {"title": "the Ashtavakra Gita", "section": "🪷 What Ashtavakra says"},
    "gita": {"title": "the Bhagavad Gita", "section": "🏹 What the Bhagavad Gita says"},
}

SYSTEM_COMPARE = """You answer questions using ONLY the scripture passages provided
below, drawn from these sources: {titles}.
Structure your answer in exactly these markdown sections:

{sections}

**⚖️ Comparing them** — one short paragraph. If the sources genuinely
diverge on this question, start this section with the line "**They differ:**"
and state the disagreement sharply. If they broadly agree, start with
"**They agree:**" and note any difference of emphasis or method.

Rules:
- Base every claim on the passages. Do not use outside knowledge.
- If one source's passages do not address the question, say so plainly in
  that source's section instead of forcing an answer.
- Cite each point with the bracketed reference of the passage it comes from,
  e.g. (Katha 1-II-18), (Ashtavakra 15.4) or (Gita 2.47)."""


def _client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in a .env file (see .env.example)."
        )
    return Groq(api_key=key)


def _present_sources(passages: list[dict]) -> list[str]:
    present = {p.get("source", "upanishads") for p in passages}
    return [s for s in SOURCES if s in present]


def format_context(passages: list[dict], compare: bool = False) -> str:
    if not compare:
        return "\n\n".join(
            f"[{p['upanishad']} {p['ref']}] {p['text']}" for p in passages
        )
    parts = []
    for source in _present_sources(passages):
        group = [p for p in passages if p.get("source", "upanishads") == source]
        body = "\n\n".join(f"[{p['upanishad']} {p['ref']}] {p['text']}" for p in group)
        parts.append(f"=== PASSAGES FROM {SOURCES[source]['title'].upper()} ===\n{body}")
    return "\n\n".join(parts)


def _compare_prompt(passages: list[dict]) -> str:
    present = _present_sources(passages)
    titles = ", ".join(SOURCES[s]["title"] for s in present)
    sections = "\n\n".join(
        f"**{SOURCES[s]['section']}** — answer from {SOURCES[s]['title']}'s passages only."
        for s in present
    )
    return SYSTEM_COMPARE.format(titles=titles, sections=sections)


def answer(question: str, passages: list[dict], compare: bool = False) -> str:
    from groq import RateLimitError

    messages = [
        {"role": "system", "content": _compare_prompt(passages) if compare else SYSTEM},
        {
            "role": "user",
            "content": f"Passages:\n{format_context(passages, compare)}\n\nQuestion: {question}",
        },
    ]
    client = _client()
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.2
        )  # low temp keeps it faithful to the text
    except RateLimitError:
        if FALLBACK_MODEL.lower() in ("", "none", "off"):
            raise RuntimeError(
                f"Groq's free-tier daily token limit for {GROQ_MODEL} is used up. "
                "It resets on a rolling basis — try again in ~15 minutes, or set "
                "GROQ_MODEL to another model in .env."
            )
        try:
            resp = client.chat.completions.create(
                model=FALLBACK_MODEL, messages=messages, temperature=0.2
            )
        except RateLimitError:
            raise RuntimeError(
                f"Groq's free-tier daily token limits for both {GROQ_MODEL} and "
                f"{FALLBACK_MODEL} are used up. Limits reset on a rolling basis — "
                "try again in ~15 minutes."
            )
    return resp.choices[0].message.content
