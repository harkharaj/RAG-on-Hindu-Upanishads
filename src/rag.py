"""Full RAG pipeline: retrieve verses, generate a grounded, cited answer.

Corpora: the 108 Upanishads, the Ashtavakra Gita, the Bhagavad Gita.
Ask one text alone, or several — with several, each text answers in its
own section and agreements/disagreements are highlighted.

Library use:
    from src.rag import ask
    result = ask("What is the nature of Atman?")                # all texts
    result = ask("...", scope=["upanishads", "gita"])            # a pair
    result = ask("...", scope="ashtavakra")                      # one text
    result["answer"], result["passages"]

CLI:
    python -m src.rag "What did Yama teach Nachiketas about death?"
    python -m src.rag --scope gita,ashtavakra "How is liberation achieved?"
"""

import sys

from src.generate import SOURCES, answer
from src.retrieve import retrieve

TOP_K = 6


def ask(question: str, k: int = TOP_K, scope: str | list[str] | None = None) -> dict:
    """scope: source key(s) from SOURCES; None or "all"/"both" means every corpus."""
    if scope in (None, "all", "both"):
        scope = list(SOURCES)
    elif isinstance(scope, str):
        scope = [s.strip() for s in scope.split(",")]
    unknown = [s for s in scope if s not in SOURCES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; valid: {list(SOURCES)}")

    if len(scope) == 1:
        passages = retrieve(question, k=k, source=scope[0])
        return {"answer": answer(question, passages), "passages": passages}

    # Retrieve from each corpus separately so one can't crowd out the
    # others, then answer comparatively.
    k_each = max(3, round(k / len(scope)))
    passages = []
    for source in SOURCES:  # canonical order, regardless of scope order
        if source in scope:
            passages += retrieve(question, k=k_each, source=source)
    return {"answer": answer(question, passages, compare=True), "passages": passages}


def main() -> None:
    args = sys.argv[1:]
    scope = None
    if "--scope" in args:
        i = args.index("--scope")
        scope = args[i + 1]
        args = args[:i] + args[i + 2:]
    question = " ".join(args) or "What is the nature of the Self (Atman)?"
    result = ask(question, scope=scope)
    print(f"Q: {question}   (scope: {scope or 'all'})\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for p in result["passages"]:
        print(f"[{p['upanishad']} {p['ref']}] (score {p['score']}) {p['text'][:160]}")


if __name__ == "__main__":
    main()
