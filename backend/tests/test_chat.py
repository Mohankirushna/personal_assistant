"""Chat endpoints: REST round-trip, session continuity, WS streaming, errors."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
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
