"""Shared fixtures: a fake Ollama client and an app wired to it, so the test
suite runs without Ollama installed or any model pulled."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.ollama_client import ChatTurn, Message
from app.main import create_app
from app.tools.registry import ToolRegistry


class FakeOllamaClient:
    """In-memory OllamaLike double. Records every call for assertions.

    `queued` replies are consumed first (one per chat call) — used to script
    planner decisions; afterwards every call returns `reply`.
    """

    def __init__(self, reply: str = "Hello from fake Jarvis.") -> None:
        self.reply = reply
        self.queued: list[str] = []
        self.queued_turns: list[ChatTurn] = []
        self.calls: list[tuple[str, str]] = []  # (operation, model)
        self.chat_messages: list[list[Message]] = []
        self.fail_with: Exception | None = None

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    async def chat(self, model: str, messages: Iterable[Message], keep_alive: str | int) -> str:
        self._maybe_fail()
        self.calls.append(("chat", model))
        self.chat_messages.append(list(messages))
        return self.queued.pop(0) if self.queued else self.reply

    async def chat_turn(
        self,
        model: str,
        messages: Iterable[Message],
        keep_alive: str | int,
        tools: list[dict] | None = None,
    ) -> ChatTurn:
        self._maybe_fail()
        self.calls.append(("chat_turn", model))
        self.chat_messages.append(list(messages))
        if self.queued_turns:
            return self.queued_turns.pop(0)
        return ChatTurn(content=self.reply)

    async def chat_stream(
        self, model: str, messages: Iterable[Message], keep_alive: str | int
    ) -> AsyncIterator[str]:
        self._maybe_fail()
        self.calls.append(("chat_stream", model))
        self.chat_messages.append(list(messages))
        for word in self.reply.split(" "):
            yield word + " "

    async def load_model(self, model: str, keep_alive: str | int) -> None:
        self._maybe_fail()
        self.calls.append(("load", model))

    async def unload_model(self, model: str) -> None:
        self.calls.append(("unload", model))

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", model))
        # Deterministic 4-dim vectors: direction picked by keyword.
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                    1.0 if "gamma" in lowered else 0.0,
                    0.1,
                ]
            )
        return vectors

    async def loaded_models(self) -> list[str] | None:
        return ["fake-model:latest"]

    async def aclose(self) -> None:
        self.calls.append(("aclose", ""))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # data_dir always points at the test's tmp dir so no test can ever
    # touch the user's real ~/Library/Application Support/Jarvis.
    return Settings(_env_file=None, data_dir=tmp_path / "jarvis-data")


@pytest.fixture
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def app(settings: Settings, fake_ollama: FakeOllamaClient) -> FastAPI:
    # Empty registry -> no planner -> plain streaming chat path; memory off.
    return create_app(
        settings=settings,
        ollama_client=fake_ollama,
        registry=ToolRegistry(),
        enable_memory=False,
    )


@pytest.fixture
def client(app: FastAPI) -> Iterable[TestClient]:
    # Context manager so the lifespan (service wiring) runs.
    with TestClient(app) as test_client:
        yield test_client
