"""Timer tool: in-process countdown that fires a macOS notification.

Deliberately NOT the Clock app: UI-scripting Clock proved unverifiable (it
silently no-ops without Accessibility, and the tool would then claim success
for a timer that never existed). The backend outlives requests and is leashed
to the app's lifetime, so an asyncio countdown + `display notification` (which
needs no special permissions — verified live) is both simpler and honest.

Tradeoff: pending timers die with the backend. For kitchen-timer durations
(1–60 min) on an always-running assistant, that's acceptable; the summary
never claims persistence.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool

logger = logging.getLogger(__name__)


def _applescript_quote(text: str) -> str:
    """Make user text safe inside a double-quoted AppleScript string."""
    return text.replace("\\", "").replace('"', "'")


class TimerArgs(BaseModel):
    minutes: int = Field(
        ...,
        description="Duration in minutes (1–60). E.g. 10 for a 10-minute timer.",
        ge=1,
        le=60,
    )
    label: str = Field(
        default="",
        description="Optional label, e.g. 'laundry' or 'tea'.",
    )


class TimerTool(Tool):
    name: ClassVar[str] = "timer"
    description: ClassVar[str] = (
        "Set a countdown timer (1-60 minutes, optional label). When it ends, "
        "a notification with a sound appears on this Mac."
    )
    args_model: ClassVar[type[BaseModel]] = TimerArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    # Strong references so pending countdowns aren't garbage-collected.
    _active: ClassVar[set[asyncio.Task[None]]] = set()

    async def run(self, args: TimerArgs) -> ToolResult:  # type: ignore[override]
        fires_at = datetime.now() + timedelta(minutes=args.minutes)
        task = asyncio.create_task(self._countdown(args.minutes * 60, args.label))
        self._active.add(task)
        task.add_done_callback(self._active.discard)
        label_text = f" for {args.label}" if args.label else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=(
                f"Timer set{label_text}: {args.minutes} "
                f"minute{'s' if args.minutes != 1 else ''}. I'll notify you at "
                f"{fires_at.strftime('%I:%M %p').lstrip('0')}."
            ),
            data={
                "minutes": args.minutes,
                "label": args.label,
                "fires_at": fires_at.isoformat(timespec="seconds"),
            },
        )

    async def _countdown(self, seconds: float, label: str) -> None:
        await asyncio.sleep(seconds)
        await self._notify(label)

    @staticmethod
    async def _notify(label: str) -> None:
        text = f"Time's up: {label}!" if label else "Time's up!"
        script = (
            f'display notification "{_applescript_quote(text)}" '
            f'with title "Jarvis Timer" sound name "Glass"'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except OSError:
            logger.exception("Timer notification failed")
