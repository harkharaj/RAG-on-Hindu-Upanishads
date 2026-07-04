"""Evaluate the RAG system.

Retrieval metrics (no API key needed):
    hit@k  — did any expected verse appear in the top-k?
    MRR    — reciprocal rank of the first expected verse.

Faithfulness (needs GROQ_API_KEY): an LLM judge checks whether each generated
answer is fully supported by the retrieved passages.

Run:  python -m eval.evaluate            # retrieval only
      python -m eval.evaluate --full     # retrieval + generation + faithfulness
"""

import json
import sys
from pathlib import Path

from src.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "testset.json"
K = 6

JUDGE_PROMPT = """You are grading a question-answering system for faithfulness.
Given the source passages and the generated answer, reply with a single line:
SUPPORTED     — every factual claim in the answer is backed by the passages
PARTIAL       — some claims are backed, some are not
UNSUPPORTED   — the answer makes claims the passages do not back

Passages:
{context}

Answer to grade:
{answer}

Reply with exactly one word: SUPPORTED, PARTIAL, or UNSUPPORTED."""


def matches(hit: dict, expected: list[dict]) -> bool:
    return any(
        hit["upanishad"] == e["upanishad"]
        and any(hit["ref"].startswith(r) for r in e["refs"])
        for e in expected
    )


def eval_retrieval(testset: list[dict]) -> None:
    hits, rr_sum = 0, 0.0
    print(f"Retrieval @ k={K}")
    for item in testset:
        results = retrieve(item["question"], k=K)
        rank = next(
            (i + 1 for i, h in enumerate(results) if matches(h, item["expected"])),
            None,
        )
        ok = rank is not None
        hits += ok
        rr_sum += 1 / rank if ok else 0
        top = results[0]
        mark = "PASS" if ok else "MISS"
        print(f"  [{mark}] rank={rank or '-'}  {item['question'][:60]}")
        if not ok:
            print(f"         top hit was [{top['upanishad']} {top['ref']}]")
    n = len(testset)
    print(f"\nhit@{K}: {hits}/{n} = {hits / n:.2f}   MRR: {rr_sum / n:.2f}")


def eval_faithfulness(testset: list[dict]) -> None:
    from src.generate import GROQ_MODEL, _client, answer, format_context

    client = _client()
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    print("\nFaithfulness (LLM judge)")
    for item in testset:
        passages = retrieve(item["question"], k=K)
        ans = answer(item["question"], passages)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        context=format_context(passages), answer=ans
                    ),
                }
            ],
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().split()[0].upper()
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  [{verdict:<11}] {item['question'][:60]}")
    n = len(testset)
    print(f"\nsupported: {counts['SUPPORTED']}/{n}   partial: {counts['PARTIAL']}   unsupported: {counts['UNSUPPORTED']}")


def main() -> None:
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    eval_retrieval(testset)
    if "--full" in sys.argv:
        eval_faithfulness(testset)


if __name__ == "__main__":
    main()
