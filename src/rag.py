"""Full RAG pipeline: retrieve verses, generate a grounded, cited answer.

Library use:
    from src.rag import ask
    result = ask("What is the nature of Atman?")
    result["answer"], result["passages"]

CLI:
    python -m src.rag "What did Yama teach Nachiketas about death?"
"""

import sys

from src.generate import answer
from src.retrieve import retrieve

TOP_K = 6


def ask(question: str, k: int = TOP_K) -> dict:
    passages = retrieve(question, k=k)
    return {"answer": answer(question, passages), "passages": passages}


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is the nature of the Self (Atman)?"
    result = ask(question)
    print(f"Q: {question}\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for p in result["passages"]:
        print(f"[{p['upanishad']} {p['ref']}] (score {p['score']}) {p['text'][:160]}")


if __name__ == "__main__":
    main()
