"""Grounded answer generation via Groq's free OpenAI-compatible API."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM = """You answer questions using ONLY the Upanishad passages provided below.
Rules:
- Base every claim on the passages. Do not use outside knowledge.
- If the passages do not address the question, say so plainly.
- Cite each point with the bracketed reference of the passage it comes from,
  e.g. (Katha 1-II-18) or (Isavasya 5).
- Where the texts differ, present the different views rather than merging them."""


def _client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in a .env file (see .env.example)."
        )
    return Groq(api_key=key)


def format_context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['upanishad']} {p['ref']}] {p['text']}" for p in passages
    )


def answer(question: str, passages: list[dict]) -> str:
    resp = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Passages:\n{format_context(passages)}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,  # low temp keeps it faithful to the text
    )
    return resp.choices[0].message.content
