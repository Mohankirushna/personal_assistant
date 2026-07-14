"""Bearer-token middleware: HTTP and WebSocket enforcement."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from tests.conftest import FakeOllamaClient

TOKEN = "secret-token"


@pytest.fixture
def authed_client() -> TestClient:
    settings = Settings(_env_file=None, auth_token=TOKEN)
    app = create_app(settings=settings, ollama_client=FakeOllamaClient())
    with TestClient(app) as client:
        yield client


def test_health_is_exempt(authed_client: TestClient) -> None:
    assert authed_client.get("/health").status_code == 200


def test_chat_requires_token(authed_client: TestClient) -> None:
    response = authed_client.post("/chat", json={"message": "hi"})
    assert response.status_code == 401


def test_chat_with_wrong_token(authed_client: TestClient) -> None:
    response = authed_client.post(
        "/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


def test_chat_with_token(authed_client: TestClient) -> None:
    response = authed_client.post(
        "/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200


def test_websocket_rejected_without_token(authed_client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect), authed_client.websocket_connect("/ws/chat"):
        pass


def test_websocket_accepted_with_token(authed_client: TestClient) -> None:
    with authed_client.websocket_connect(
        "/ws/chat", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as ws:
        ws.send_json({"message": "hi"})
        assert ws.receive_json()["type"] in ("token", "done")
