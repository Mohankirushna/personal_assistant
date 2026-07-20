"""Morning-briefing announce endpoint.

The SwiftUI app POSTs here when the laptop wakes. The backend decides whether
audio output is appropriate (not muted, volume up), composes the briefing,
and speaks it aloud via macOS `say` — so the gating and speaking logic stays
here where it can be tested, and the app only has to notice the wake.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.tools._common import run_osascript
from app.tools.briefing import MorningBriefingTool

logger = logging.getLogger(__name__)

router = APIRouter()


class AnnounceResponse(BaseModel):
    spoken: bool
    reason: str
    text: str = ""


async def _audio_is_audible() -> bool:
    """True when speech would actually be heard: output not muted and volume
    above zero. This is the "headphones OR speakers, as long as it's not
    silenced" gate — muting is how the user says 'not now' (e.g. in class)."""
    muted = await run_osascript("output muted of (get volume settings)")
    if muted.ok and muted.stdout.strip() == "true":
        return False
    volume = await run_osascript("output volume of (get volume settings)")
    level = volume.stdout.strip()
    return volume.ok and level.isdigit() and int(level) > 0


async def _say(text: str, voice: str | None) -> None:
    argv = ["/usr/bin/say"]
    if voice:
        argv += ["-v", voice]
    argv.append(text)
    # Fire and forget: return as soon as speech starts, don't block the
    # request for the whole spoken duration.
    await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )


@router.post("/briefing/announce", response_model=AnnounceResponse)
async def announce(request: Request) -> AnnounceResponse:
    settings = request.app.state.settings
    if not await _audio_is_audible():
        return AnnounceResponse(spoken=False, reason="muted")
    result = await MorningBriefingTool(settings).execute({})
    if not result.ok or not result.summary.strip():
        return AnnounceResponse(spoken=False, reason="briefing-failed", text=result.summary)
    await _say(result.summary, settings.say_voice or None)
    logger.info("Spoke morning briefing on wake")
    return AnnounceResponse(spoken=True, reason="ok", text=result.summary)
