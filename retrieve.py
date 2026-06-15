"""
Milestone 4 — Retrieval.

Queries ChromaDB for the top-k most similar chunks. When a professor or course
code appears in the question, results are filtered to that metadata first.

Usage:
    python retrieve.py "Kevin Pfeil exam policy"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from embed import COLLECTION_NAME, get_collection

DEFAULT_TOP_K = 5
CHUNKS_PATH = Path("data/processed/chunks.json")
MAX_DISTANCE = 0.75


def load_professor_names() -> list[str]:
    if not CHUNKS_PATH.exists():
        return []

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    names = {
        chunk["metadata"]["professor"]
        for chunk in chunks
        if chunk.get("metadata", {}).get("professor")
    }
    return sorted(names, key=len, reverse=True)


def detect_professor(query: str) -> str | None:
    """Match a professor name from the query against our corpus."""
    query_lower = query.lower()
    for professor in load_professor_names():
        if professor.lower() in query_lower:
            return professor
    return None


def detect_course(query: str) -> str | None:
    match = re.search(r"\b[A-Z]{3}\d{4}[A-Z]?\b", query)
    return match.group(0) if match else None


def rerank_chunks(chunks: list[dict], query: str) -> list[dict]:
    """Boost chunks that share more keywords with the query."""
    keywords = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9']+", query)
        if len(word) > 3
    }

    def sort_key(chunk: dict) -> tuple[int, float]:
        text = chunk["text"].lower()
        keyword_hits = sum(1 for keyword in keywords if keyword in text)
        return (-keyword_hits, chunk["distance"])

    return sorted(chunks, key=sort_key)


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return top-k chunks with text, metadata, and cosine distance."""
    collection = get_collection()
    professor = detect_professor(query)
    course = detect_course(query)

    where_filter: dict | None = None
    if professor and course:
        where_filter = {"$and": [{"professor": professor}, {"course": course}]}
    elif professor:
        where_filter = {"professor": professor}

    query_kwargs: dict = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    retrieved: list[dict] = []
    for index in range(len(results["ids"][0])):
        distance = results["distances"][0][index]
        if distance > MAX_DISTANCE:
            continue
        retrieved.append(
            {
                "id": results["ids"][0][index],
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
            }
        )

    return rerank_chunks(retrieved, query)[:top_k]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = retrieve(query)

    print(f'Query: "{query}"')
    print(f"Collection: {COLLECTION_NAME} | top-{DEFAULT_TOP_K}\n")

    for rank, result in enumerate(results, start=1):
        professor = result["metadata"].get("professor", "Unknown")
        course = result["metadata"].get("course", "N/A")
        distance = result["distance"]
        preview = result["text"].replace("\n", " ")[:120]
        print(f"{rank}. {professor} ({course}) — distance: {distance:.4f}")
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
