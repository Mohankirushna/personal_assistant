"""Local embedding generation via Ollama (nomic-embed-text).

The embedding model is ~274MB resident and loads on demand; it is small
enough not to be managed by the ModelManager. A protocol keeps the vector
store testable without Ollama.
"""

from __future__ import annotations

from typing import Protocol

from app.core.ollama_client import OllamaLike


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, client: OllamaLike, model: str = "nomic-embed-text") -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed(self._model, texts)
