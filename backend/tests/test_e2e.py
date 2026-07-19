"""End-to-end integration across the whole stack against real models.

These exercise the full path a user hits — voice transport, planner + real
tools, memory recall — with the real Ollama LLM. Marked integration; run
with `uv run pytest -m integration`.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


def _say_wav(text: str) -> bytes:
    """Synthesize `text` to 16kHz mono PCM16 WAV via macOS `say`, for a
    real STT→planner path without a microphone."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "s.aiff"
        subprocess.run(["/usr/bin/say", "-o", str(aiff), text], check=True)
        wav = Path(tmp) / "s.wav"
        subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
             str(aiff), str(wav)],
            check=True,
        )
        return wav.read_bytes()


def test_full_chat_stack_with_real_llm(tmp_path: Path) -> None:
    """POST /chat → planner → clock tool → real LLM reply, plus memory
    persistence and the /tools and /memory endpoints, all wired by the real
    app factory (only the models are real; everything else is production)."""
    from fastapi.testclient import TestClient

    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings=settings)
    with TestClient(app) as client:
        # The full tool suite is registered.
        tools = {tool["name"] for tool in client.get("/tools").json()}
        assert "clock" in tools and "look_at_screen" in tools

        response = client.post("/chat", json={"message": "What time is it right now?"})
        assert response.status_code == 200
        body = response.json()
        assert body["reply"]

        # The turn was persisted and is semantically searchable.
        history = client.get("/memory/history").json()
        assert any("time" in item["utterance"].lower() for item in history)


async def test_voice_transcription_to_reply(tmp_path: Path) -> None:
    """Real Whisper transcription of synthesized speech, driving a real
    planner turn over the voice WebSocket."""
    pytest.importorskip("faster_whisper", reason="voice extra not installed")
    pytest.importorskip("openwakeword", reason="voice extra not installed")
    from fastapi.testclient import TestClient

    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    app = create_app(settings=settings)

    with TestClient(app) as client, client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "start_listening"})
        assert ws.receive_json()["type"] == "listening"

        # Feed synthesized speech, then silence to trigger endpointing.
        pcm = _wav_body_to_pcm(_say_wav("what time is it"))
        for offset in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[offset : offset + 3200])
        ws.send_bytes(np.zeros(16000, dtype=np.int16).tobytes())  # 1s silence

        transcript = None
        reply = None
        for _ in range(20):
            event = ws.receive_json()
            if event["type"] == "transcript":
                transcript = event["text"]
            elif event["type"] == "reply":
                reply = event["text"]
                break
            elif event["type"] == "error":
                pytest.fail(event["message"])
        assert transcript and "time" in transcript.lower()
        assert reply


def _wav_body_to_pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return wav_file.readframes(wav_file.getnframes())
