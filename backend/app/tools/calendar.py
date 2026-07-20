"""Calendar tool: read today's events natively via EventKit.

Why not AppleScript: `every event ... whose start date >= X` against the
Calendar app times out on real calendars (>25 s measured live) — unusable for
voice. EventKit answers the same query in milliseconds.

Permissions: macOS routes the access prompt to the *responsible* app (the
Jarvis app bundle, which declares NSCalendarsFullAccessUsageDescription).
Without that declaration macOS auto-denies with no prompt at all, so every
auth state below is reported honestly rather than as an empty calendar.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool

_DENIED_HELP = (
    "Calendar access is denied. Enable it in System Settings > Privacy & "
    "Security > Calendars for Jarvis, then try again."
)


class CalendarArgs(BaseModel):
    day: str = Field(
        default="today",
        description="Which day to check: 'today' or 'tomorrow'.",
    )


class CalendarTool(Tool):
    name: ClassVar[str] = "calendar"
    description: ClassVar[str] = (
        "Read your macOS Calendar: list today's or tomorrow's events and "
        "meetings with their times."
    )
    args_model: ClassVar[type[BaseModel]] = CalendarArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: CalendarArgs) -> ToolResult:  # type: ignore[override]
        day = "tomorrow" if "tomorrow" in args.day.lower() else "today"
        try:
            return await asyncio.to_thread(self._read_events, day)
        except ModuleNotFoundError:
            return ToolResult.failure(
                self.name,
                "Calendar support needs the 'macos' extra "
                "(uv sync --extra macos installs pyobjc EventKit).",
            )

    def _read_events(self, day: str) -> ToolResult:
        from EventKit import EKEntityTypeEvent, EKEventStore
        from Foundation import NSDate

        store = EKEventStore.alloc().init()
        status = int(EKEventStore.authorizationStatusForEntityType_(EKEntityTypeEvent))
        # 0 notDetermined, 1 restricted, 2 denied, 3 full access, 4 write-only
        if status == 0:
            granted = self._request_access(store)
            if not granted:
                return ToolResult.failure(self.name, _DENIED_HELP)
        elif status in (1, 2, 4):
            return ToolResult.failure(self.name, _DENIED_HELP)

        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if day == "tomorrow":
            start += timedelta(days=1)
        end = start + timedelta(days=1)
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            NSDate.dateWithTimeIntervalSince1970_(start.timestamp()),
            NSDate.dateWithTimeIntervalSince1970_(end.timestamp()),
            None,
        )
        events = sorted(
            store.eventsMatchingPredicate_(predicate) or [],
            key=lambda e: e.startDate().timeIntervalSince1970(),
        )
        if not events:
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=f"No calendar events {day}.",
                data={"day": day, "events": []},
            )
        lines = [_describe_event(event) for event in events]
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"{len(lines)} event(s) {day}:\n" + "\n".join(lines),
            data={"day": day, "events": lines},
        )

    @staticmethod
    def _request_access(store: Any) -> bool:
        """Trigger the system permission prompt and wait for the answer."""
        done = threading.Event()
        outcome = {"granted": False}

        def callback(granted: bool, _error: Any) -> None:
            outcome["granted"] = bool(granted)
            done.set()

        store.requestFullAccessToEventsWithCompletion_(callback)
        done.wait(timeout=120)  # generous: the user may be reading the prompt
        return outcome["granted"]


def _describe_event(event: Any) -> str:
    title = str(event.title() or "Untitled")
    if event.isAllDay():
        return f"{title} (all day)"
    start = datetime.fromtimestamp(event.startDate().timeIntervalSince1970())
    end = datetime.fromtimestamp(event.endDate().timeIntervalSince1970())
    fmt = "%I:%M %p"
    return (
        f"{title} ({start.strftime(fmt).lstrip('0')} - "
        f"{end.strftime(fmt).lstrip('0')})"
    )
