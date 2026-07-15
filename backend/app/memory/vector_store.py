"""ChromaDB-backed semantic memory.

A single persistent collection of conversation turns. Embeddings are
computed by our own Embedder (see embeddings.py) rather than Chroma's
default embedding function, so recall works with any local model and tests
can inject a fake.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.memory.embeddings import Embedder


@dataclass(frozen=True)
class MemoryHit:
    text: str
    metadata: dict[str, Any]
    distance: float


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name="turns", metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

    async def add_turn(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        turn_id = uuid.uuid4().hex
        [raw] = await self._embedder.embed([text])
        vector: Sequence[float] = raw
        self._collection.add(
            ids=[turn_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{"ts": time.time(), **(metadata or {})}],
        )
        return turn_id

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        if self._collection.count() == 0:
            return []
        [raw] = await self._embedder.embed([query])
        vector: Sequence[float] = raw
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(k, self._collection.count())
        )
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            MemoryHit(text=doc, metadata=dict(meta), distance=dist)
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]
