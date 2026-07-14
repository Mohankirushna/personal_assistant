"""Health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ollama"]["available"] is True
    assert body["ollama"]["loaded_models"] == ["fake-model:latest"]
    assert body["active_model"] is None  # nothing requested yet
