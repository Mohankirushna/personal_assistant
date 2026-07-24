"""Chat endpoints: REST round-trip, session continuity, WS streaming, errors."""

from __future__ import annotations

from typing import ClassVar

from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import Settings
from app.core.ollama_client import ChatTurn
from app.main import create_app
from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool
from app.tools.registry import ToolRegistry
from tests.conftest import FakeOllamaClient


def test_chat_roundtrip(
    client: TestClient, fake_ollama: FakeOllamaClient, settings: Settings
) -> None:
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == fake_ollama.reply
    assert body["session_id"]
    # LLM was loaded through the ModelManager before chatting.
    assert fake_ollama.calls[0] == ("load", settings.llm_model)


def test_chat_keeps_session_history(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    first = client.post("/chat", json={"message": "First"}).json()
    client.post("/chat", json={"message": "Second", "session_id": first["session_id"]})
    # Second call's prompt contains: system, user1, assistant1, user2.
    prompt = fake_ollama.chat_messages[-1]
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]
    assert prompt[1]["content"] == "First"
    assert prompt[3]["content"] == "Second"


def test_chat_validates_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_reports_ollama_down(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    from app.core.ollama_client import OllamaUnavailableError

    fake_ollama.fail_with = OllamaUnavailableError("http://127.0.0.1:11434")
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_websocket_streams_tokens(client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "Hello"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert len(tokens) > 1  # actually streamed, not one blob
        assert "".join(tokens).strip() == fake_ollama.reply
        assert events[-1]["reply"].strip() == fake_ollama.reply
        assert events[-1]["session_id"]
        # This is a text surface: ordinary chat (no tools ran, empty
        # registry) must never be flagged for audio playback.
        assert events[-1]["speak"] is False


class _ReadAloudArgs(BaseModel):
    url: str | None = None


class _FakeReadUrlAloudTool(Tool):
    """Stand-in for read_url_aloud: skips the real fetch/AppleScript so the
    test only exercises the chat.py <-> planner <-> tool-registry wiring."""

    name: ClassVar[str] = "read_url_aloud"
    description: ClassVar[str] = "Read a page aloud (test double)."
    args_model: ClassVar[type[BaseModel]] = _ReadAloudArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: _ReadAloudArgs) -> ToolResult:  # type: ignore[override]
        return ToolResult(tool=self.name, ok=True, summary="Raw page text.", data={"url": args.url})


def test_websocket_flags_speak_when_read_url_aloud_ran(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """The one case this text surface should speak: an explicit 'read this
    out loud' that actually ran read_url_aloud. Needs its own app instance —
    the shared `client` fixture wires an empty (tool-less) registry."""
    registry = ToolRegistry()
    registry.register(_FakeReadUrlAloudTool())
    app = create_app(
        settings=settings, ollama_client=fake_ollama, registry=registry, enable_memory=False
    )
    fake_ollama.queued_turns = [ChatTurn(content="Here is the article summary.")]

    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "read this out loud"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
        assert events[-1]["reply"] == "Here is the article summary."
        assert events[-1]["speak"] is True


def test_websocket_error_keeps_socket_open(
    client: TestClient, fake_ollama: FakeOllamaClient
) -> None:
    from app.core.ollama_client import OllamaUnavailableError

    with client.websocket_connect("/ws/chat") as ws:
        fake_ollama.fail_with = OllamaUnavailableError("http://127.0.0.1:11434")
        ws.send_json({"message": "Hello"})
        assert ws.receive_json()["type"] == "error"
        # Socket still usable after the error.
        fake_ollama.fail_with = None
        ws.send_json({"message": "Hello again"})
        assert ws.receive_json()["type"] == "token"
