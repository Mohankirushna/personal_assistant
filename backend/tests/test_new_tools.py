"""Tests for the daily-driver tools: timer, focus_mode, calendar."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.planner.schemas import RiskLevel
from app.tools.calendar import CalendarArgs, CalendarTool, _describe_event
from app.tools.focus_mode import FocusModeArgs, FocusModeTool
from app.tools.timer import TimerArgs, TimerTool, _applescript_quote

# ---------------------------------------------------------------- timer


def test_timer_tool_spec() -> None:
    tool = TimerTool()
    assert tool.name == "timer"
    assert tool.risk_level == RiskLevel.SAFE


def test_timer_args_validation() -> None:
    assert TimerArgs(minutes=10).label == ""
    assert TimerArgs(minutes=5, label="laundry").label == "laundry"
    with pytest.raises(ValueError):
        TimerArgs(minutes=0)
    with pytest.raises(ValueError):
        TimerArgs(minutes=61)


async def test_timer_returns_immediately_and_schedules_countdown() -> None:
    tool = TimerTool()
    result = await tool.run(TimerArgs(minutes=1, label="tea"))
    try:
        assert result.ok
        assert "tea" in result.summary
        assert "1 minute" in result.summary
        assert result.data["fires_at"]
        # The countdown is pending in the background, not finished inline.
        assert len(TimerTool._active) == 1
    finally:
        for task in list(TimerTool._active):
            task.cancel()
        await asyncio.gather(*TimerTool._active, return_exceptions=True)


async def test_timer_countdown_fires_the_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: list[str] = []

    async def fake_notify(label: str) -> None:
        fired.append(label)

    monkeypatch.setattr(TimerTool, "_notify", staticmethod(fake_notify))
    tool = TimerTool()
    await tool._countdown(0.01, "laundry")
    assert fired == ["laundry"]


def test_timer_label_is_applescript_safe() -> None:
    # Quotes and backslashes must not escape the AppleScript string literal.
    assert _applescript_quote('say "hi" \\ there') == "say 'hi'  there"


# ---------------------------------------------------------------- focus mode


def test_focus_mode_tool_spec() -> None:
    tool = FocusModeTool()
    assert tool.name == "focus_mode"
    assert tool.risk_level == RiskLevel.SAFE


def test_focus_mode_args_validation() -> None:
    assert FocusModeArgs().action == "toggle"
    assert FocusModeArgs(action="on").action == "on"
    with pytest.raises(ValueError):
        FocusModeArgs(action="enable")


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self._out = stdout
        self._err = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, self._err


async def _run_focus_with(
    monkeypatch: pytest.MonkeyPatch, proc: _FakeProc, action: str
) -> Any:
    async def fake_exec(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return await FocusModeTool().run(FocusModeArgs(action=action))


async def test_focus_mode_reports_the_read_back_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run_focus_with(monkeypatch, _FakeProc(b"0,1\n"), "on")
    assert result.ok
    assert "now on" in result.summary


async def test_focus_mode_never_claims_success_when_the_click_did_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Click happened but the checkbox value didn't move: report failure,
    # never "Do Not Disturb is now on".
    result = await _run_focus_with(monkeypatch, _FakeProc(b"0,0\n"), "on")
    assert not result.ok
    assert "unchanged" in result.summary


async def test_focus_mode_already_in_desired_state_is_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run_focus_with(monkeypatch, _FakeProc(b"1,1\n"), "on")
    assert result.ok
    assert "already" in result.summary


async def test_focus_mode_missing_accessibility_fails_with_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(b"", b"osascript is not allowed assistive access. (-1719)", 1)
    result = await _run_focus_with(monkeypatch, proc, "toggle")
    assert not result.ok
    assert "Accessibility" in result.summary


# ---------------------------------------------------------------- calendar


def test_calendar_tool_spec() -> None:
    tool = CalendarTool()
    assert tool.name == "calendar"
    assert tool.risk_level == RiskLevel.SAFE


def test_calendar_args_default_to_today() -> None:
    assert CalendarArgs().day == "today"


async def test_calendar_day_parsing_routes_to_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_read(self: CalendarTool, day: str) -> Any:
        seen.append(day)
        from app.planner.schemas import ToolResult

        return ToolResult(tool="calendar", ok=True, summary="", data={})

    monkeypatch.setattr(CalendarTool, "_read_events", fake_read)
    tool = CalendarTool()
    await tool.run(CalendarArgs(day="What about TOMORROW?"))
    await tool.run(CalendarArgs(day="today"))
    assert seen == ["tomorrow", "today"]


async def test_calendar_without_eventkit_names_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing(self: CalendarTool, day: str) -> Any:
        raise ModuleNotFoundError("No module named 'EventKit'")

    monkeypatch.setattr(CalendarTool, "_read_events", raise_missing)
    result = await CalendarTool().run(CalendarArgs())
    assert not result.ok
    assert "macos" in result.summary


class _FakeNSDate:
    def __init__(self, ts: float):
        self._ts = ts

    def timeIntervalSince1970(self) -> float:
        return self._ts


class _FakeEvent:
    def __init__(self, title: str, start: float, end: float, all_day: bool = False):
        self._title = title
        self._start = start
        self._end = end
        self._all_day = all_day

    def title(self) -> str:
        return self._title

    def isAllDay(self) -> bool:
        return self._all_day

    def startDate(self) -> _FakeNSDate:
        return _FakeNSDate(self._start)

    def endDate(self) -> _FakeNSDate:
        return _FakeNSDate(self._end)


def test_calendar_event_formatting() -> None:
    from datetime import datetime

    start = datetime(2026, 7, 20, 9, 30).timestamp()
    end = datetime(2026, 7, 20, 10, 0).timestamp()
    assert _describe_event(_FakeEvent("Standup", start, end)) == (
        "Standup (9:30 AM - 10:00 AM)"
    )
    assert _describe_event(_FakeEvent("Trip", start, end, all_day=True)) == (
        "Trip (all day)"
    )
