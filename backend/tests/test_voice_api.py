"""Voice endpoints with fake STT/TTS/wake — no models, no audio hardware."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.tools.registry import ToolRegistry
from tests.conftest import FakeOllamaClient
from tests.test_voice_session import FakeWake, frames


class FakeSTT:
    def __init__(self, text: str = "what time is it") -> None:
        self.text = text
        self.received: list[np.ndarray] = []

    async def transcribe(self, audio: np.ndarray) -> str:
        self.received.append(audio)
        return self.text


class FakeTTS:
    async def synthesize(self, text: str) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(np.zeros(160, dtype=np.int16).tobytes())
        return buffer.getvalue()


@pytest.fixture
def fake_stt() -> FakeSTT:
    return FakeSTT()


@pytest.fixture
def voice_app(settings: Settings, fake_ollama: FakeOllamaClient, fake_stt: FakeSTT) -> FastAPI:
    return create_app(
        settings=settings,
        ollama_client=fake_ollama,
        stt=fake_stt,
        tts=FakeTTS(),
        wake_detector=FakeWake(scores=[0.9]),
        registry=ToolRegistry(),
    )


@pytest.fixture
def voice_client(voice_app: FastAPI):  # noqa: ANN201
    with TestClient(voice_app) as client:
        yield client


def test_speak_returns_wav(voice_client: TestClient) -> None:
    response = voice_client.post("/voice/speak", json={"text": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"


def test_transcribe_accepts_wav(voice_client: TestClient, fake_stt: FakeSTT) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(frames(4, loud=True))
    response = voice_client.post(
        "/voice/transcribe",
        files={"file": ("audio.wav", buffer.getvalue(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": fake_stt.text}
    assert len(fake_stt.received) == 1


def test_voice_ws_full_loop(voice_client: TestClient, fake_ollama: FakeOllamaClient) -> None:
    """wake -> listening -> transcript -> reply -> audio -> audio_end."""
    with voice_client.websocket_connect("/ws/voice") as ws:
        ws.send_bytes(frames(1, loud=True))  # wake fires (fake score 0.9)
        assert ws.receive_json()["type"] == "wake"
        assert ws.receive_json()["type"] == "listening"

        ws.send_bytes(frames(4, loud=True))   # speech
        ws.send_bytes(frames(12, loud=False))  # trailing silence endpoint

        assert ws.receive_json() == {"type": "transcript", "text": "what time is it"}
        reply = ws.receive_json()
        assert reply["type"] == "reply"
        assert reply["text"] == fake_ollama.reply
        assert ws.receive_bytes()[:4] == b"RIFF"
        assert ws.receive_json()["type"] == "audio_end"


def test_voice_ws_push_to_talk(voice_client: TestClient) -> None:
    with voice_client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "start_listening"})
        assert ws.receive_json()["type"] == "listening"

        ws.send_bytes(frames(4, loud=True))
        ws.send_bytes(frames(12, loud=False))
        assert ws.receive_json()["type"] == "transcript"


def test_voice_ws_direct_say(voice_client: TestClient) -> None:
    with voice_client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "say", "text": "hello there"})
        assert ws.receive_bytes()[:4] == b"RIFF"
        assert ws.receive_json()["type"] == "audio_end"
