"""The /briefing/announce endpoint: audio gating and spoken output.

The audio-state osascript, the briefing composition, and macOS `say` are all
mocked; nothing here reads real volume or speaks.
"""

from __future__ import annotations

import pytest

from app.api import briefing as briefing_api
from app.planner.schemas import ToolResult
from app.tools._common import CommandOutput


def _mock_audio(monkeypatch: pytest.MonkeyPatch, muted: bool, volume: int) -> None:
    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        if "muted" in script:
            return CommandOutput(0, "true" if muted else "false", "")
        return CommandOutput(0, str(volume), "")

    monkeypatch.setattr(briefing_api, "run_osascript", fake_osascript)


async def test_audio_gate_blocks_when_muted(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_audio(monkeypatch, muted=True, volume=50)
    assert await briefing_api._audio_is_audible() is False


async def test_audio_gate_blocks_when_volume_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_audio(monkeypatch, muted=False, volume=0)
    assert await briefing_api._audio_is_audible() is False


async def test_audio_gate_allows_when_unmuted_with_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_audio(monkeypatch, muted=False, volume=30)
    assert await briefing_api._audio_is_audible() is True


async def test_announce_skips_speaking_when_muted(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_audio(monkeypatch, muted=True, volume=50)
    spoke: list[str] = []

    async def fake_say(text: str, voice: str | None) -> None:
        spoke.append(text)

    monkeypatch.setattr(briefing_api, "_say", fake_say)
    result = await _call_announce(monkeypatch)
    assert result.spoken is False
    assert result.reason == "muted"
    assert spoke == []  # never spoke


async def test_announce_speaks_the_briefing_when_audible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_audio(monkeypatch, muted=False, volume=40)
    spoke: list[str] = []

    async def fake_say(text: str, voice: str | None) -> None:
        spoke.append(text)

    async def fake_execute(self, raw_args):  # type: ignore[no-untyped-def]
        return ToolResult(tool="morning_briefing", ok=True, summary="Good morning. Sunny 30C.")

    monkeypatch.setattr(briefing_api, "_say", fake_say)
    monkeypatch.setattr(briefing_api.MorningBriefingTool, "execute", fake_execute)
    result = await _call_announce(monkeypatch)
    assert result.spoken is True
    assert result.reason == "ok"
    assert spoke == ["Good morning. Sunny 30C."]


async def _call_announce(monkeypatch: pytest.MonkeyPatch) -> briefing_api.AnnounceResponse:
    """Invoke the endpoint with a minimal fake Request carrying settings."""
    from app.core.config import Settings

    class _State:
        settings = Settings()

    class _App:
        state = _State()

    class _Request:
        app = _App()

    return await briefing_api.announce(_Request())  # type: ignore[arg-type]
