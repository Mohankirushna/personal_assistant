"""Morning briefing: one spoken rundown of the day.

Orchestrates pieces the assistant already has — the clock, macOS Calendar,
Apple Mail, plus keyless web sources for weather (wttr.in) and headlines
(Google News RSS). Every section degrades independently: if calendar access
is denied or the network is down, that line is simply dropped rather than
failing the whole briefing.
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import ClassVar
from urllib.parse import quote

from pydantic import BaseModel

from app.core import location_state
from app.core.config import Settings, get_settings
from app.planner.schemas import RiskLevel, ToolResult
from app.tools import mail as mail_module
from app.tools._common import run_command
from app.tools.base import Tool
from app.tools.calendar import CalendarArgs, CalendarTool

_HEADLINE_ITEM = re.compile(
    r"<item>.*?<title>(?P<title>.*?)</title>.*?<pubDate>(?P<pubdate>.*?)</pubDate>",
    re.DOTALL,
)


def _greeting(now: datetime) -> str:
    hour = now.hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


class MorningBriefingArgs(BaseModel):
    pass


class MorningBriefingTool(Tool):
    name: ClassVar[str] = "morning_briefing"
    description: ClassVar[str] = (
        "Give the user's briefing for the day: greeting, date, today's calendar events, "
        "an unread-email summary, weather, and top headlines. Use for 'good morning', "
        "'morning briefing', 'brief me', or 'what does my day look like'."
    )
    args_model: ClassVar[type[BaseModel]] = MorningBriefingArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def run(self, args: MorningBriefingArgs) -> ToolResult:  # type: ignore[override]
        now = datetime.now()
        # Gather independent sections concurrently; each returns "" on failure.
        calendar, email, weather, headlines = await asyncio.gather(
            self._calendar_line(),
            self._email_line(),
            self._weather_line(),
            self._headlines_line(),
        )
        opening = f"{_greeting(now)}. It's {now.strftime('%A, %B %-d')}."
        sections = [opening, calendar, email, weather, headlines]
        spoken = " ".join(section for section in sections if section)
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=spoken,
            data={
                "date": now.date().isoformat(),
                "calendar": calendar,
                "email": email,
                "weather": weather,
                "headlines": headlines,
            },
        )

    @staticmethod
    async def _calendar_line() -> str:
        try:
            result = await CalendarTool().run(CalendarArgs(day="today"))
        except Exception:  # noqa: BLE001 - a missing extra/permission just drops the line
            return ""
        if not result.ok:
            return ""
        events = result.data.get("events", [])
        if not events:
            return "You have nothing on your calendar today."
        count = len(events)
        head = "; ".join(events[:3])
        noun = "event" if count == 1 else "events"
        return f"You have {count} calendar {noun} today: {head}."

    @staticmethod
    async def _email_line() -> str:
        try:
            scanned = await mail_module._scan_inbox(None, include_body=False, limit=3)
        except Exception:  # noqa: BLE001
            return ""
        if not isinstance(scanned, tuple):
            return ""  # permission/other error — omit rather than complain
        unread, messages = scanned
        if unread == 0:
            return "No unread email."
        senders: list[str] = []
        for message in messages:
            name = message["from"].split("<")[0].strip().strip('"')
            if name and name not in senders:  # dedupe repeated senders
                senders.append(name)
        who = ", ".join(senders[:2])
        noun = "email" if unread == 1 else "emails"
        tail = f", including from {who}" if who else ""
        return f"You have {unread} unread {noun}{tail}."

    async def _weather_line(self) -> str:
        # An explicit setting always wins. Otherwise use the current city the
        # SwiftUI app reported from Location Services (accurate WiFi-
        # positioning); the backend can't get location itself. If neither is
        # available, fall back to wttr.in's own IP guess (often off by tens
        # of km on mobile-carrier connections).
        location = (self._settings.briefing_location or "").strip()
        if not location:
            location = location_state.get_city() or ""
        url = "https://wttr.in/" + quote(location, safe="") + "?format=%C+%t&m"
        # No browser UA here: wttr.in returns its plain-text one-line format
        # only to curl-like clients; a browser UA gets the full HTML page.
        output = await run_command(["/usr/bin/curl", "-s", "--max-time", "8", url])
        text = output.stdout.strip()
        if not output.ok or not text or "Unknown location" in text or "<" in text:
            return ""
        # wttr.in prints temperatures like "+29°C"; drop the leading + for speech.
        text = re.sub(r"\+(\d)", r"\1", text)
        where = f" in {location}" if location else ""
        return f"The weather{where} is {text}."

    async def _headlines_line(self) -> str:
        url = (
            "https://news.google.com/rss?hl="
            + quote(self._settings.briefing_news_locale, safe="")
            + "&gl=" + quote(self._settings.briefing_news_country, safe="")
            + "&ceid=" + quote(f"{self._settings.briefing_news_country}:en", safe="")
        )
        output = await run_command(["/usr/bin/curl", "-s", "--max-time", "8", url])
        if not output.ok or not output.stdout.strip():
            return ""
        # Google's "top stories" feed order blends relevance with recency;
        # sort by each item's actual pubDate so "top headlines" means the
        # most recently published, not whatever Google ranked first. Only
        # <item> entries are headlines — the <channel> title is the feed name
        # ("Google News") and must not leak in.
        dated: list[tuple[datetime, str]] = []
        for match in _HEADLINE_ITEM.finditer(output.stdout):
            headline = self._clean_headline(match.group("title"))
            if not headline:
                continue
            try:
                published = parsedate_to_datetime(match.group("pubdate"))
            except (TypeError, ValueError):
                continue
            dated.append((published, headline))
        if not dated:
            return ""
        dated.sort(key=lambda pair: pair[0], reverse=True)
        headlines = [headline for _, headline in dated[:3]]
        return "Top headlines: " + "; ".join(headlines) + "."

    @staticmethod
    def _clean_headline(raw: str) -> str:
        text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw, flags=re.DOTALL)
        text = html.unescape(text).strip()
        # Google News appends " - Publisher"; keep the headline itself.
        text = text.rsplit(" - ", 1)[0].strip()
        # Some Google News titles are themselves a "; "-joined roundup of two
        # stories. Left alone, that collides with the "; " used to join
        # separate headlines together below, making one item sound like two.
        return text.replace("; ", ", ")
