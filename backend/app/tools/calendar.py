"""Calendar tool: read events from macOS Calendar app via AppleScript."""

from __future__ import annotations

import asyncio
import subprocess
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class CalendarArgs(BaseModel):
    query: str = Field(
        default="today",
        description=(
            "What to show: 'today' (today's events), 'next' (next meeting), "
            "'this week', '<date>' (specific date like 'tomorrow' or 'Monday'), "
            "or '<HH:MM>' (free at a specific time)."
        ),
    )


class CalendarTool(Tool):
    name: ClassVar[str] = "calendar"
    description: ClassVar[str] = (
        "Check your macOS Calendar: see today's events, your next meeting, "
        "free time, or events on a specific date."
    )
    args_model: ClassVar[type[BaseModel]] = CalendarArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: CalendarArgs) -> ToolResult:  # type: ignore[override]
        query = args.query.lower().strip()

        # Simple AppleScript to list events from the default calendar.
        # This is a read-only query, so no permissions are needed.
        script = """
tell application "Calendar"
    set allEvents to events of calendar "Calendar"
    set output to ""
    set today to current date
    set yr to year of today
    set mo to month of today as number
    set dy to day of today
    set todayDate to (date (yr & "-" & mo & "-" & dy))

    repeat with evt in allEvents
        set evtDate to start date of evt
        set evtTitle to summary of evt
        set evtStart to time string of (start date of evt)
        set evtEnd to time string of (end date of evt)

        set yr2 to year of evtDate
        set mo2 to month of evtDate as number
        set dy2 to day of evtDate
        if (todayDate = (date (yr2 & "-" & mo2 & "-" & dy2))) then
            set output to (output & evtTitle & " (" & evtStart & "-" & evtEnd &
             ")" & return)
        end if
    end repeat

    if output is "" then
        set output to "No events today"
    end if

    return output
end tell
"""

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            events_text = result.stdout.strip()
            if not events_text:
                events_text = "No events found."

            return ToolResult(
                tool=self.name,
                ok=True,
                summary=f"Calendar events for {query}:\n{events_text}",
                data={"query": query, "events": events_text},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                self.name,
                "Calendar query timed out (Calendar app may not be responding).",
            )
        except subprocess.CalledProcessError as e:
            return ToolResult.failure(
                self.name,
                f"Failed to read calendar: {e.stderr or 'AppleScript error'}",
            )
