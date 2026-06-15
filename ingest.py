"""
Milestone 3 — Ingestion & chunking.

Reads Rate My Professors .txt files from data/raw/, splits them into review-level
chunks (256 tokens, 50 overlap), and writes data/processed/chunks.json.

Usage:
    python ingest.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tiktoken

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/chunks.json")

# Chunking hyperparameters (from planning.md)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
MIN_CHUNK_TOKENS = 20

REVIEW_SPLIT = re.compile(r"(?=^Review \d+ \()", re.MULTILINE)
REVIEW_HEADER = re.compile(r"^Review (\d+) \(([^,]+), ([^)]+)\):\s*$", re.MULTILINE)


def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(text))


def chunk_by_tokens(text: str, chunk_size: int, overlap: int, encoder) -> list[str]:
    """Sliding-window split for reviews that exceed the token limit."""
    tokens = encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(encoder.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def parse_header(header_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower().replace(" ", "_")] = value.strip()
    return metadata


def format_metadata_prefix(metadata: dict[str, str]) -> str:
    professor = metadata.get("professor", "Unknown")
    department = metadata.get("department", "Unknown")
    school = metadata.get("school", "Unknown")
    url = metadata.get("url", "")
    return (
        f"Professor: {professor}\n"
        f"Department: {department}\n"
        f"School: {school}\n"
        f"Source URL: {url}\n"
    )


def split_reviews(reviews_text: str) -> list[str]:
    """Split on 'Review N (' so we never cut through the middle of a review."""
    reviews = [part.strip() for part in REVIEW_SPLIT.split(reviews_text) if part.strip()]
    return reviews


def parse_review_metadata(review_text: str) -> dict[str, str]:
    match = REVIEW_HEADER.search(review_text)
    if not match:
        return {}
    return {
        "review_number": match.group(1),
        "course": match.group(2).strip(),
        "review_date": match.group(3).strip(),
    }


def load_document(path: Path) -> tuple[dict[str, str], list[str]]:
    text = path.read_text(encoding="utf-8")
    if "--- Reviews ---" not in text:
        raise ValueError(f"Missing review section in {path.name}")

    header_text, reviews_text = text.split("--- Reviews ---", 1)
    metadata = parse_header(header_text)
    metadata["source_file"] = path.name
    reviews = split_reviews(reviews_text)
    return metadata, reviews


def chunk_document(
    metadata: dict[str, str],
    reviews: list[str],
    encoder,
) -> list[dict]:
    prefix = format_metadata_prefix(metadata)
    prefix_tokens = count_tokens(prefix, encoder)
    max_review_tokens = max(CHUNK_SIZE - prefix_tokens, 64)

    chunks: list[dict] = []
    professor_slug = metadata.get("source_file", "unknown").replace(".txt", "")

    for review_text in reviews:
        review_meta = parse_review_metadata(review_text)
        review_chunks = chunk_by_tokens(review_text, max_review_tokens, CHUNK_OVERLAP, encoder)

        for chunk_index, review_chunk in enumerate(review_chunks):
            full_text = f"{prefix}\n{review_chunk}"
            if count_tokens(full_text, encoder) < MIN_CHUNK_TOKENS:
                continue

            review_number = review_meta.get("review_number", "0")
            chunk_id = f"{professor_slug}_review{review_number}_chunk{chunk_index}"

            chunks.append(
                {
                    "id": chunk_id,
                    "text": full_text,
                    "metadata": {
                        **metadata,
                        **review_meta,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def ingest(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Load every .txt file in data/raw/ and return all chunks."""
    encoder = get_encoder()
    all_chunks: list[dict] = []

    paths = sorted(raw_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    for path in paths:
        metadata, reviews = load_document(path)
        all_chunks.extend(chunk_document(metadata, reviews, encoder))

    return all_chunks


def main() -> None:
    chunks = ingest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
