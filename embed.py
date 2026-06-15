"""
Milestone 4 — Embedding & vector store.

Embeds chunks with all-MiniLM-L6-v2 and stores them in a persistent ChromaDB
collection at data/chroma/ (cosine distance).

Usage:
    python embed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHUNKS_PATH = Path("data/processed/chunks.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "professor_reviews"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection() -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def _sanitize_metadata(metadata: dict) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def build_index(chunks_path: Path = CHUNKS_PATH) -> int:
    """Rebuild the ChromaDB collection from chunks.json."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run `python ingest.py` first."
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    client = get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[_sanitize_metadata(chunk["metadata"]) for chunk in chunks],
    )

    return len(chunks)


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks in {CHROMA_DIR} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()
