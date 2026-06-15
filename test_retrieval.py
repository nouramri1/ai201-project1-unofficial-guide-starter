"""
Milestone 4 checkpoint: test retrieval on evaluation-plan queries.

Usage:
    python test_retrieval.py
"""

from __future__ import annotations

from retrieve import retrieve

EVAL_QUERIES = [
    "What do students say about Kevin Pfeil's exam grading policy in COP3223C?",
    "How is Mark Llewellyn's CNT4714 class graded?",
    "What is Paul Gazzillo's attendance policy for COP3402?",
    "How is Heath Martin's MAC2312 course graded?",
    "What teaching style does Paul Lawrence use for CHM2210, and what do students say about the workload?",
]


def main() -> None:
    for index, query in enumerate(EVAL_QUERIES, start=1):
        print("=" * 72)
        print(f"Query {index}: {query}")
        print("=" * 72)

        results = retrieve(query)
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            professor = metadata.get("professor", "Unknown")
            course = metadata.get("course", "N/A")
            source_file = metadata.get("source_file", "N/A")
            distance = result["distance"]
            print(f"\n--- Rank {rank} | distance: {distance:.4f} ---")
            print(f"Professor: {professor} | Course: {course} | File: {source_file}")
            print(result["text"])
        print()


if __name__ == "__main__":
    main()
