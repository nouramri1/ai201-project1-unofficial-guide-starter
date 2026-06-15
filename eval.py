"""
Milestone 6: Run all evaluation-plan queries end-to-end.

Usage:
    python eval.py
"""

from __future__ import annotations

from test_retrieval import EVAL_QUERIES
from query import ask

OUT_OF_CORPUS_QUERY = "What do students say about dining halls at UCF?"


def main() -> None:
    for index, question in enumerate(EVAL_QUERIES, start=1):
        print("=" * 72)
        print(f"Eval {index}: {question}")
        print("=" * 72)
        result = ask(question)
        print(result["answer"])
        print("\nRetrieved from:")
        print(result["sources_text"])
        print()

    print("=" * 72)
    print(f"Out-of-corpus test: {OUT_OF_CORPUS_QUERY}")
    print("=" * 72)
    result = ask(OUT_OF_CORPUS_QUERY)
    print(result["answer"])
    print("\nRetrieved from:")
    print(result["sources_text"])


if __name__ == "__main__":
    main()
