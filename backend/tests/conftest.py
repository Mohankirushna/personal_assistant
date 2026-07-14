"""Shared fixtures: a fake Ollama client and an app wired to it, so the test
suite runs without Ollama installed or any model pulled."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.ollama_client import Message
from app.main import create_app


class FakeOllamaClient:
    """In-memory OllamaLike double. Records every call for assertions."""

    def __init__(self, reply: str = "Hello from fake Jarvis.") -> None:
        self.reply = reply
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
        return self.reply

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

    async def loaded_models(self) -> list[str] | None:
        return ["fake-model:latest"]

    async def aclose(self) -> None:
        self.calls.append(("aclose", ""))


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def app(settings: Settings, fake_ollama: FakeOllamaClient) -> FastAPI:
    return create_app(settings=settings, ollama_client=fake_ollama)


@pytest.fixture
def client(app: FastAPI) -> Iterable[TestClient]:
    # Context manager so the lifespan (service wiring) runs.
    with TestClient(app) as test_client:
        yield test_client
