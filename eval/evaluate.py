"""Evaluate the RAG system across all three corpora.

Retrieval metrics (no API key needed), reported per corpus and overall:
    hit@k  — did any expected verse appear in the top-k?
    MRR    — reciprocal rank of the first expected verse.
Each testset item carries a "source" ("upanishads" default, "ashtavakra",
"gita") and retrieval is restricted to that corpus, so the three corpora
are scored independently.

Comparative mode (needs GROQ_API_KEY): cross-tradition questions are asked
with all corpora enabled; deterministic checks verify the answer has one
section per text, a verdict line ("They differ:"/"They agree:"), and at
least one citation from every corpus.

Faithfulness (needs GROQ_API_KEY): an LLM judge checks whether each
generated answer is fully supported by the retrieved passages.

Run:  python -m eval.evaluate            # retrieval only
      python -m eval.evaluate --compare  # + comparative structure checks
      python -m eval.evaluate --full     # + generation + faithfulness
"""

import json
import re
import sys
from pathlib import Path

from src.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "testset.json"
COMPARE_TESTSET = ROOT / "eval" / "compare_testset.json"
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

# Citation patterns per corpus, e.g. [Ashtavakra 15.4], (Gita 2.47),
# [Katha 1-II-18] — the model uses either bracket style; an upanishad
# citation is any other capitalized work name.
CITATION = {
    "ashtavakra": re.compile(r"[\[(]Ashtavakra \d"),
    "gita": re.compile(r"[\[(]Gita \d"),
    "upanishads": re.compile(r"[\[(](?!Ashtavakra |Gita )[A-Z][A-Za-z' \-]* [0-9IVX]"),
}
SECTION_MARKS = {"upanishads": "🕉️", "ashtavakra": "🪷", "gita": "🏹"}


def matches(hit: dict, expected: list[dict]) -> bool:
    return any(
        hit["upanishad"] == e["upanishad"]
        and any(hit["ref"].startswith(r) for r in e["refs"])
        for e in expected
    )


def eval_retrieval(testset: list[dict]) -> None:
    stats: dict[str, list] = {}  # source -> [hits, rr_sum, n]
    print(f"Retrieval @ k={K} (each question restricted to its own corpus)")
    for item in testset:
        source = item.get("source", "upanishads")
        results = retrieve(item["question"], k=K, source=source)
        rank = next(
            (i + 1 for i, h in enumerate(results) if matches(h, item["expected"])),
            None,
        )
        ok = rank is not None
        s = stats.setdefault(source, [0, 0.0, 0])
        s[0] += ok
        s[1] += 1 / rank if ok else 0
        s[2] += 1
        mark = "PASS" if ok else "MISS"
        print(f"  [{mark}] rank={rank or '-'}  ({source[:4]})  {item['question'][:56]}")
        if not ok:
            top = results[0]
            print(f"         top hit was [{top['upanishad']} {top['ref']}]")
    print()
    total = [0, 0.0, 0]
    for source, (hits, rr, n) in stats.items():
        print(f"  {source:<11} hit@{K}: {hits}/{n} = {hits / n:.2f}   MRR: {rr / n:.2f}")
        total = [total[0] + hits, total[1] + rr, total[2] + n]
    print(f"  {'OVERALL':<11} hit@{K}: {total[0]}/{total[2]} = {total[0] / total[2]:.2f}   MRR: {total[1] / total[2]:.2f}")


def eval_compare() -> None:
    """Ask cross-tradition questions with every corpus enabled and check the
    comparative answer's structure deterministically."""
    from src.rag import ask

    questions = json.loads(COMPARE_TESTSET.read_text(encoding="utf-8"))
    print("\nComparative mode (structure checks on multi-corpus answers)")
    n_ok = 0
    for q in questions:
        result = ask(q)
        ans = result["answer"]
        checks = {
            "sections": all(mark in ans for mark in SECTION_MARKS.values()),
            "verdict": ("They differ:" in ans) or ("They agree:" in ans),
            "cites-upa": bool(CITATION["upanishads"].search(ans)),
            "cites-ash": bool(CITATION["ashtavakra"].search(ans)),
            "cites-gita": bool(CITATION["gita"].search(ans)),
        }
        ok = all(checks.values())
        n_ok += ok
        failed = [name for name, passed in checks.items() if not passed]
        print(f"  [{'PASS' if ok else 'FAIL'}] {q[:56]}" + (f"  missing: {failed}" if failed else ""))
    print(f"\n  structure: {n_ok}/{len(questions)} answers well-formed")


def eval_faithfulness(testset: list[dict]) -> None:
    from src.generate import GROQ_MODEL, _client, answer, format_context

    client = _client()
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    print("\nFaithfulness (LLM judge)")
    for item in testset:
        source = item.get("source", "upanishads")
        passages = retrieve(item["question"], k=K, source=source)
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
        print(f"  [{verdict:<11}] ({source[:4]}) {item['question'][:56]}")
    n = len(testset)
    print(f"\nsupported: {counts['SUPPORTED']}/{n}   partial: {counts['PARTIAL']}   unsupported: {counts['UNSUPPORTED']}")


def main() -> None:
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    eval_retrieval(testset)
    if "--compare" in sys.argv or "--full" in sys.argv:
        eval_compare()
    if "--full" in sys.argv:
        eval_faithfulness(testset)


if __name__ == "__main__":
    main()
