"""Tests for newly added tools: timer, focus_mode, calendar."""

from __future__ import annotations

import pytest

from app.planner.schemas import RiskLevel
from app.tools.calendar import CalendarTool
from app.tools.focus_mode import FocusModeTool
from app.tools.timer import TimerTool


def test_timer_tool_spec() -> None:
    """Timer tool is properly registered with correct risk level."""
    tool = TimerTool()
    assert tool.name == "timer"
    assert tool.risk_level == RiskLevel.SAFE
    assert "countdown" in tool.description.lower() or "timer" in tool.description.lower()


def test_timer_args_validation() -> None:
    """Timer accepts duration 1–60 minutes and optional label."""
    from app.tools.timer import TimerArgs

    # Valid: minimal
    args = TimerArgs(minutes=10)
    assert args.minutes == 10
    assert args.label == ""

    # Valid: with label
    args = TimerArgs(minutes=5, label="laundry")
    assert args.minutes == 5
    assert args.label == "laundry"

    # Invalid: too short
    with pytest.raises(ValueError):
        TimerArgs(minutes=0)

    # Invalid: too long
    with pytest.raises(ValueError):
        TimerArgs(minutes=61)


def test_focus_mode_tool_spec() -> None:
    """Focus mode tool is properly registered."""
    tool = FocusModeTool()
    assert tool.name == "focus_mode"
    assert tool.risk_level == RiskLevel.SAFE
    assert "focus" in tool.description.lower() or "do not disturb" in tool.description.lower()


def test_focus_mode_args_validation() -> None:
    """Focus mode accepts 'on', 'off', 'toggle'."""
    from app.tools.focus_mode import FocusModeArgs

    # Valid: default
    args = FocusModeArgs()
    assert args.action == "toggle"

    # Valid: explicit
    args = FocusModeArgs(action="on")
    assert args.action == "on"

    args = FocusModeArgs(action="off")
    assert args.action == "off"

    # Invalid
    with pytest.raises(ValueError):
        FocusModeArgs(action="enable")


def test_calendar_tool_spec() -> None:
    """Calendar tool is properly registered."""
    tool = CalendarTool()
    assert tool.name == "calendar"
    assert tool.risk_level == RiskLevel.SAFE
    assert "calendar" in tool.description.lower() or "event" in tool.description.lower()


def test_calendar_args_validation() -> None:
    """Calendar accepts various query types."""
    from app.tools.calendar import CalendarArgs

    # Valid: defaults to 'today'
    args = CalendarArgs()
    assert args.query == "today"

    # Valid: specific queries
    for query in ["today", "next", "this week", "tomorrow", "monday"]:
        args = CalendarArgs(query=query)
        assert args.query == query
