"""Async Ollama client wrapper.

Every part of the backend that talks to Ollama (chat now; planner, vision and
embeddings in later phases) goes through this wrapper rather than the raw
``ollama`` package, so that:
  - connection/model errors surface as our own typed exceptions,
  - model load/unload semantics (keep_alive) live in one place for the
    ModelManager to drive,
  - tests can substitute a fake via :class:`OllamaLike`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import ollama

logger = logging.getLogger(__name__)

# Chat message; "tool" role messages and assistant tool_calls use extra keys,
# hence Any values.
Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool invocation proposed by the model (not yet executed)."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatTurn:
    """One assistant turn: text and/or proposed tool calls."""

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


class OllamaUnavailableError(RuntimeError):
    """Ollama is not reachable (not installed, or the daemon isn't running)."""

    def __init__(self, host: str) -> None:
        super().__init__(
            f"Cannot reach Ollama at {host}. Is it running? Start it with `ollama serve` "
            "or open the Ollama app."
        )


class ModelNotFoundError(RuntimeError):
    """The requested model has not been pulled."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Model '{model}' is not available in Ollama. Pull it with `ollama pull {model}` "
            "or run scripts/install_models.sh."
        )
        self.model = model


class OllamaLike(Protocol):
    """The interface the rest of the app depends on (real client or test fake)."""

    async def chat(
        self, model: str, messages: Iterable[Message], keep_alive: str | int
    ) -> str: ...

    async def chat_turn(
        self,
        model: str,
        messages: Iterable[Message],
        keep_alive: str | int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn: ...

    def chat_stream(
        self, model: str, messages: Iterable[Message], keep_alive: str | int
    ) -> AsyncIterator[str]: ...

    async def load_model(self, model: str, keep_alive: str | int) -> None: ...

    async def unload_model(self, model: str) -> None: ...

    async def loaded_models(self) -> list[str] | None: ...

    async def aclose(self) -> None: ...


class OllamaClient:
    """Thin async wrapper over :class:`ollama.AsyncClient`."""

    def __init__(self, host: str, timeout_seconds: float = 120.0) -> None:
        self._host = host
        self._client = ollama.AsyncClient(host=host, timeout=timeout_seconds)

    def _translate(self, exc: Exception, model: str | None = None) -> Exception:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return OllamaUnavailableError(self._host)
        if isinstance(exc, ollama.ResponseError) and exc.status_code == 404 and model:
            return ModelNotFoundError(model)
        return exc

    async def chat(self, model: str, messages: Iterable[Message], keep_alive: str | int) -> str:
        try:
            response = await self._client.chat(
                model=model, messages=list(messages), stream=False, keep_alive=keep_alive
            )
        except Exception as exc:  # noqa: BLE001 - translated and re-raised
            raise self._translate(exc, model) from exc
        return response.message.content or ""

    async def chat_turn(
        self,
        model: str,
        messages: Iterable[Message],
        keep_alive: str | int,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatTurn:
        """Chat with native tool-calling: the model's trained function-call
        template is used (far more reliable for qwen2.5 than hand-rolled
        JSON protocols)."""
        try:
            response = await self._client.chat(
                model=model,
                messages=list(messages),
                stream=False,
                keep_alive=keep_alive,
                tools=tools or [],
            )
        except Exception as exc:  # noqa: BLE001 - translated and re-raised
            raise self._translate(exc, model) from exc
        raw_calls = response.message.tool_calls or []
        return ChatTurn(
            content=response.message.content or "",
            tool_calls=[
                ToolCallRequest(
                    name=call.function.name,
                    arguments=dict(call.function.arguments or {}),
                )
                for call in raw_calls
            ],
        )

    async def chat_stream(
        self, model: str, messages: Iterable[Message], keep_alive: str | int
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat(
                model=model, messages=list(messages), stream=True, keep_alive=keep_alive
            )
            async for chunk in stream:
                content = chunk.message.content
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001 - translated and re-raised
            raise self._translate(exc, model) from exc

    async def load_model(self, model: str, keep_alive: str | int) -> None:
        """Load a model into memory without generating anything.

        Ollama loads a model when it receives a generate request with an
        empty prompt; keep_alive controls how long it stays resident.
        """
        logger.info("Loading model %s (keep_alive=%s)", model, keep_alive)
        try:
            await self._client.generate(model=model, keep_alive=keep_alive)
        except Exception as exc:  # noqa: BLE001 - translated and re-raised
            raise self._translate(exc, model) from exc

    async def unload_model(self, model: str) -> None:
        """Unload a model immediately (keep_alive=0)."""
        logger.info("Unloading model %s", model)
        try:
            await self._client.generate(model=model, keep_alive=0)
        except Exception as exc:  # noqa: BLE001 - translated and re-raised
            translated = self._translate(exc, model)
            # Unloading a model that isn't present is a no-op, not an error.
            if isinstance(translated, ModelNotFoundError):
                return
            raise translated from exc

    async def loaded_models(self) -> list[str] | None:
        """Names of models currently resident in memory, or None if Ollama is down."""
        try:
            response = await self._client.ps()
        except Exception:  # noqa: BLE001 - health probe never raises
            return None
        return [m.model for m in response.models if m.model]

    async def aclose(self) -> None:
        # ollama.AsyncClient has no public close API; its underlying httpx
        # client is the documented-by-convention way to shut down cleanly.
        await self._client._client.aclose()
