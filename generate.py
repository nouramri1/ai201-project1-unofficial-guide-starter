"""
Milestone 5 — Grounded generation.

Retrieves relevant review chunks, sends them to Groq (llama-3.3-70b-versatile),
and returns an answer grounded in those excerpts only. Sources are attached
programmatically — the LLM does not generate them.

Usage:
    python generate.py "How is Mark Llewellyn's CNT4714 class graded?"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import DEFAULT_TOP_K, retrieve

load_dotenv()

NO_INFO_RESPONSE = "I don't have enough information on that."
MAX_TOP_DISTANCE = 0.72
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a UCF unofficial course guide assistant.

You MUST follow these rules:
1. Answer ONLY using facts explicitly stated in the retrieved review excerpts provided by the user.
2. Do NOT use general knowledge about universities, professors, or courses.
3. Do NOT guess or infer policies that are not directly supported by the excerpts.
4. If the excerpts do not contain enough information, respond with exactly:
   "I don't have enough information on that."
5. If student reviews conflict, present both perspectives briefly.
6. Do NOT include a sources section in your answer — sources are added separately by the system.
7. Never mention professors, courses, or URLs that do not appear in the excerpts."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "g":
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key from "
            "https://console.groq.com"
        )
    return Groq(api_key=api_key)


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No review excerpts were retrieved."

    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        professor = metadata.get("professor", "Unknown")
        course = metadata.get("course", "N/A")
        source_file = metadata.get("source_file", "unknown")
        url = metadata.get("url", "N/A")
        sections.append(
            f"--- Excerpt {index} ---\n"
            f"Professor: {professor}\n"
            f"Course: {course}\n"
            f"Source file: {source_file}\n"
            f"Source URL: {url}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(sections)


def format_source_strings(sources: list[dict]) -> list[str]:
    if not sources:
        return []

    source_strings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        key = (
            source.get("professor", "Unknown"),
            source.get("source_file", "unknown"),
            source.get("url", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        source_strings.append(
            f"{key[0]} ({source.get('course', 'N/A')}) — {key[1]} — {key[2]}"
        )
    return source_strings


def format_sources(sources: list[dict]) -> str:
    return "\n".join(f"• {source}" for source in format_source_strings(sources))


def should_decline(chunks: list[dict]) -> bool:
    """Skip the LLM when retrieval is empty or too weak — avoids hallucination."""
    if not chunks:
        return True
    return chunks[0]["distance"] > MAX_TOP_DISTANCE


def generate_answer(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    source_dicts = [
        {
            "professor": chunk["metadata"].get("professor", "Unknown"),
            "course": chunk["metadata"].get("course", "N/A"),
            "source_file": chunk["metadata"].get("source_file", "unknown"),
            "url": chunk["metadata"].get("url", ""),
        }
        for chunk in chunks
    ]
    source_strings = format_source_strings(source_dicts)

    if should_decline(chunks):
        return {
            "query": query,
            "answer": NO_INFO_RESPONSE,
            "sources": source_strings,
            "sources_text": format_sources(source_dicts),
            "chunks": chunks,
        }

    context = format_context(chunks)
    client = get_client()

    user_message = (
        f"Question: {query}\n\n"
        f"Retrieved review excerpts:\n\n{context}\n\n"
        "Answer the question using only the excerpts above."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        answer = NO_INFO_RESPONSE

    return {
        "query": query,
        "answer": answer,
        "sources": source_strings,
        "sources_text": format_sources(source_dicts),
        "chunks": chunks,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python generate.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = generate_answer(query)
    print(f'Question: {result["query"]}\n')
    print(result["answer"])
    print("\nRetrieved from:")
    print(result["sources_text"])


if __name__ == "__main__":
    main()
