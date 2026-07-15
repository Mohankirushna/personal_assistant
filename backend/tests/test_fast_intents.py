"""The deterministic voice-command fast-path."""

from __future__ import annotations

import pytest

from app.planner.fast_intents import match_fast_intent


@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("next", "next"),
        ("Next.", "next"),
        ("next song", "next"),
        ("play next song", "next"),
        ("skip", "next"),
        ("skip this song", "next"),
        ("skip this", "next"),
        ("go to the next track", "next"),
        ("previous", "previous"),
        ("previous song", "previous"),
        ("go back", "previous"),
        ("last song", "previous"),
        ("pause", "pause"),
        ("Pause.", "pause"),
        ("pause the music", "pause"),
        ("stop", "pause"),
        ("stop the song", "pause"),
        ("resume", "play"),
        ("resume the music", "play"),
        ("continue", "play"),
        ("unpause", "play"),
        ("play", "play"),
        ("play the music", "play"),
    ],
)
def test_matches_media_commands(utterance: str, action: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "media_control"
    assert call.arguments == {"action": action}


@pytest.mark.parametrize(
    "utterance",
    [
        "play Despacito",              # specific song -> needs the real planner
        "play the latest taylor swift song",
        "what song is playing",        # a question, not a command
        "skip the meeting tomorrow",   # not about media
        "open the next folder",        # 'next' but not media
        "stop the docker container",   # 'stop' but not media
        "pause my subscription",       # 'pause' but not media
        "what time is it",
        "play a game with me",
        "",
    ],
)
def test_does_not_match_ambiguous_or_specific(utterance: str) -> None:
    assert match_fast_intent(utterance) is None, f"{utterance!r} should NOT match"
