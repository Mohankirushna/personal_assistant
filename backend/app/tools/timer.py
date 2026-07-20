"""Timer tool: set a countdown via the macOS Clock app."""

from __future__ import annotations

import asyncio
import subprocess
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class TimerArgs(BaseModel):
    minutes: int = Field(
        ...,
        description="Duration in minutes (1–60). E.g. 10 for a 10-minute timer.",
        ge=1,
        le=60,
    )
    label: str = Field(
        default="",
        description="Optional label for the timer (e.g., 'laundry', 'meeting'). "
        "If not provided, the timer has no label.",
    )


class TimerTool(Tool):
    name: ClassVar[str] = "timer"
    description: ClassVar[str] = (
        "Set a countdown timer in the macOS Clock app. "
        "Specify duration (1–60 minutes) and an optional label."
    )
    args_model: ClassVar[type[BaseModel]] = TimerArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: TimerArgs) -> ToolResult:  # type: ignore[override]
        script = f"""
tell application "Clock"
    activate
    delay 0.5
end tell

tell application "System Events"
    key code 123  -- left arrow (switch to Timer tab if needed)
    delay 0.2
    keystroke "{args.minutes}" using shift down
    delay 0.1
    key code 48  -- tab to move to label field (if visible)
    delay 0.1
    keystroke "{args.label}"
    delay 0.2
    keystroke return  -- start the timer
end tell
"""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            label_text = f" labeled '{args.label}'" if args.label else ""
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=f"Started a {args.minutes}-minute timer{label_text}.",
                data={"minutes": args.minutes, "label": args.label},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                self.name,
                "Timer setup timed out (Clock app may not be responding).",
            )
        except subprocess.CalledProcessError as e:
            return ToolResult.failure(
                self.name,
                f"Failed to set timer: {e.stderr or e.stdout or 'AppleScript error'}",
            )
