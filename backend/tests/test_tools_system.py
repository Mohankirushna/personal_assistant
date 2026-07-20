"""System + clipboard tools. Mutating tests are integration-marked; the
clipboard test saves and restores the user's pasteboard."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from app.core.config import Settings
from app.tools._common import CommandOutput
from app.tools.clipboard.clipboard import ClipboardReadTool, ClipboardWriteTool
from app.tools.registry import ToolRegistry
from app.tools.reminders import CreateReminderTool
from app.tools.system import system as system_module
from app.tools.system.system import (
    BatteryStatusTool,
    BluetoothDevicesTool,
    BraveSearchOpenFirstTool,
    BrightnessTool,
    BrowserSearchTool,
    ScreenshotTool,
    SystemPowerTool,
    VolumeTool,
)
from app.tools.whatsapp import WhatsAppSendTool


def test_discovery_finds_the_whole_suite() -> None:
    registry = ToolRegistry()
    registry.discover()
    names = {tool.name for tool in registry.list()}
    expected = {
        "clock",
        "finder_search", "finder_list", "finder_create_folder", "finder_move", "open_file",
        "finder_delete", "finder_compress", "finder_extract",
        "terminal_run", "git", "vscode_open",
        "clipboard_read", "clipboard_write",
        "open_app", "open_url", "browser_search", "brave_search_open_first", "web_answer",
        "quit_app",
        "list_running_apps", "volume",
        "screenshot",
        "battery_status",
        "create_reminder",
        "system_power",
        "media_control", "youtube_play", "spotify_play", "spotify_open_playlist",
        "music_platform_prompt", "news_search",
        "whatsapp_send",
        "check_email", "send_email",
        "calendar", "timer", "focus_mode",
        "list_bluetooth_devices",
        "window_arrange", "brightness",
        "roll_dice",  # the example plugin — proves plugin discovery works
    }
    missing = expected - names
    assert not missing, f"tools not discovered: {missing}"
    # Service-dependent tools are NOT discovered; app.main injects them.
    assert "look_at_screen" not in names


def test_bluetooth_parser_lists_connected_devices_only() -> None:
    payload = {
        "SPBluetoothDataType": [
            {
                "device_connected": [
                    {"device_name": "AirPods", "device_connected": "Yes"},
                    {"device_name": "Magic Mouse", "device_connected": "Yes"},
                ],
                "device_not_connected": [
                    {"device_name": "Keyboard", "device_connected": "No"},
                ],
            }
        ]
    }
    assert BluetoothDevicesTool._connected_device_names(payload) == ["AirPods", "Magic Mouse"]


def test_battery_parser_reads_percentage_and_state() -> None:
    output = (
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0\t83%; discharging; 5:12 remaining"
    )
    assert BatteryStatusTool._parse_status(output) == (83, "discharging")


def test_system_power_always_requires_confirmation() -> None:
    assert SystemPowerTool().risk_level.value == "destructive"


def test_media_control_checks_the_expected_playback_state() -> None:
    from app.tools.system.system import MediaTool

    assert MediaTool._expected_state("pause") == "paused"
    assert MediaTool._expected_state("play") == "playing"
    assert MediaTool._expected_state("next") is None


def test_browser_search_builds_safe_search_urls() -> None:
    assert BrowserSearchTool._url("Ada Lovelace", "google").endswith("Ada+Lovelace")
    assert BrowserSearchTool._url("Ada Lovelace", "wikipedia").endswith("Ada+Lovelace")


def test_brave_search_uses_the_first_external_result() -> None:
    page = (
        '<a href="https://search.brave.com/settings">Settings</a>'
        '<a href="https://example.com/ada">Ada Lovelace</a>'
    )
    assert BraveSearchOpenFirstTool._first_result_url(page) == "https://example.com/ada"


async def test_search_open_first_prefers_ddgs_result(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        assert argv[0] == "/usr/bin/open"  # ddgs succeeded — no curl scrape
        opened.append(argv[-1])
        return CommandOutput(0, "", "")

    monkeypatch.setattr(
        BraveSearchOpenFirstTool, "_ddgs_first_url",
        staticmethod(lambda query: "https://example.com/ada"),
    )
    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    result = await BraveSearchOpenFirstTool().execute({"query": "ada lovelace"})
    assert result.ok, result.summary
    assert opened == ["https://example.com/ada"]


async def test_search_open_first_falls_back_to_brave_curl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        calls.append(argv)
        if argv[0] == "/usr/bin/curl":
            return CommandOutput(0, '<a href="https://example.com/ada">Ada Lovelace</a>', "")
        return CommandOutput(0, "", "")

    monkeypatch.setattr(
        BraveSearchOpenFirstTool, "_ddgs_first_url", staticmethod(lambda query: None)
    )
    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    result = await BraveSearchOpenFirstTool().execute({"query": "ada lovelace"})
    assert result.ok, result.summary
    assert calls[0][0] == "/usr/bin/curl"
    assert "--fail" in calls[0]  # HTTP 429 must be an error, not a block page to parse
    assert result.data["url"] == "https://example.com/ada"


async def test_rate_limited_search_opens_the_search_not_the_block_pages_first_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real bug: Brave's 429 block page came back as curl 'success', and its
    first external link — a Tor support page — was opened as 'the first
    result' for EVERY query. With --fail, curl exits non-zero on 429 and the
    tool must fall back to opening the search page itself."""
    opened: list[str] = []

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        if argv[0] == "/usr/bin/curl":
            # curl --fail on HTTP 429 → exit code 22, no body
            return CommandOutput(22, "", "curl: (22) The requested URL returned error: 429")
        opened.append(argv[-1])
        return CommandOutput(0, "", "")

    monkeypatch.setattr(
        BraveSearchOpenFirstTool, "_ddgs_first_url", staticmethod(lambda query: None)
    )
    monkeypatch.setattr(system_module, "run_command", fake_run_command)

    result = await BraveSearchOpenFirstTool().execute({"query": "price of iphone 15"})
    assert result.ok, result.summary
    assert "didn't load directly" in result.summary
    assert opened == ["https://search.brave.com/search?q=price+of+iphone+15"]


def test_web_answer_strips_html_to_readable_text() -> None:
    from app.tools.system.system import WebAnswerTool

    html_page = (
        "<html><head><title>x</title><style>.a{}</style></head>"
        "<body><script>bad()</script><h1>iPhone 15</h1>"
        "<p>The iPhone 15 starts at &pound;799.</p></body></html>"
    )
    text = WebAnswerTool._html_to_text(html_page)
    assert "iPhone 15 starts at £799." in text
    assert "bad()" not in text and ".a{}" not in text  # script/style dropped


async def test_web_answer_returns_page_text_and_snippets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tools.system.system import WebAnswerTool

    monkeypatch.setattr(
        WebAnswerTool, "_ddgs_results",
        staticmethod(lambda query: [
            {"title": "Apple", "href": "https://apple.com/iphone-15",
             "body": "iPhone 15 from $799."},
            {"title": "Wiki", "href": "https://en.wikipedia.org/iphone", "body": "A phone."},
        ]),
    )

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        assert argv[0] == "/usr/bin/curl"
        return CommandOutput(0, "<p>iPhone 15 costs $799 in the US.</p>", "")

    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    result = await WebAnswerTool().execute({"query": "iphone 15 price"})
    assert result.ok, result.summary
    assert "iPhone 15 costs $799" in result.summary  # top page text
    assert "from $799" in result.summary  # snippet
    assert result.data["url"] == "https://apple.com/iphone-15"


async def test_web_answer_falls_back_to_snippets_when_page_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tools.system.system import WebAnswerTool

    monkeypatch.setattr(
        WebAnswerTool, "_ddgs_results",
        staticmethod(lambda query: [
            {"title": "Result", "href": "https://x.com", "body": "The answer is 42."},
        ]),
    )

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        return CommandOutput(22, "", "curl: (22) 429")  # page blocked

    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    result = await WebAnswerTool().execute({"query": "meaning of life"})
    assert result.ok, result.summary
    assert "The answer is 42." in result.summary


async def test_web_answer_no_results_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tools.system.system import WebAnswerTool

    monkeypatch.setattr(WebAnswerTool, "_ddgs_results", staticmethod(lambda query: []))
    result = await WebAnswerTool().execute({"query": "asdfqwerzxcv"})
    assert not result.ok
    assert "no web results" in result.summary


async def test_spotify_play_resolves_a_track_and_plays_it_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spotify_play must use Spotify's AppleScript `play track` API (no
    Accessibility-dependent UI scripting) and ground its reply in the track
    Spotify actually reports playing."""
    from app.tools.system.system import SpotifyPlayTool

    monkeypatch.setattr(
        SpotifyPlayTool, "_ddgs_track_search",
        staticmethod(lambda query: "7qiZfU4dY1lWllzX7mPBI3"),
    )

    scripts: list[str] = []

    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        scripts.append(script)
        return CommandOutput(0, "Shape of You||Ed Sheeran||playing", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)

    result = await SpotifyPlayTool().execute({"query": "shape of you"})
    assert result.ok, result.summary
    assert result.summary == "Playing Shape of You by Ed Sheeran on Spotify."
    assert 'play track "spotify:track:7qiZfU4dY1lWllzX7mPBI3"' in scripts[0]
    assert "System Events" not in scripts[0]  # no keystroke/Accessibility path


async def test_spotify_play_falls_back_to_brave_when_ddgs_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tools.system.system import SpotifyPlayTool

    monkeypatch.setattr(
        SpotifyPlayTool, "_ddgs_track_search", staticmethod(lambda query: None)
    )

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        assert argv[0] == "/usr/bin/curl"
        return CommandOutput(
            0, '<a href="https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3">x</a>', ""
        )

    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        return CommandOutput(0, "Shape of You||Ed Sheeran||playing", "")

    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)

    result = await SpotifyPlayTool().execute({"query": "shape of you"})
    assert result.ok, result.summary
    assert result.data["track_id"] == "7qiZfU4dY1lWllzX7mPBI3"


async def test_spotify_play_falls_back_to_search_ui_when_no_track_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tools.system.system import SpotifyPlayTool

    monkeypatch.setattr(
        SpotifyPlayTool, "_ddgs_track_search", staticmethod(lambda query: None)
    )
    opened: list[list[str]] = []

    async def fake_run_command(argv: list[str], cwd=None, timeout=30.0) -> CommandOutput:
        opened.append(argv)
        if argv[0] == "/usr/bin/curl":
            return CommandOutput(0, "no track links here", "")
        return CommandOutput(0, "", "")

    monkeypatch.setattr(system_module, "run_command", fake_run_command)
    result = await SpotifyPlayTool().execute({"query": "some obscure song"})
    assert result.ok, result.summary
    assert "couldn't pin down" in result.summary
    assert opened[-1][:3] == ["/usr/bin/open", "-a", "Spotify"]


def test_reminder_tool_accepts_a_title_and_due_time() -> None:
    parsed = CreateReminderTool().parse_args(
        {"title": "submit the report", "due_at": "tomorrow at 10 AM"}
    )
    assert parsed is not None


def test_reminder_tool_parses_tomorrow_at_a_spoken_time() -> None:
    due_at = CreateReminderTool._parse_due_at(
        "tomorrow at 10am", now=datetime(2026, 7, 16, 9, 0)
    )
    assert due_at == datetime(2026, 7, 17, 10, 0)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("tomorrow 10pm", datetime(2026, 7, 17, 22, 0)),
        # Dayparts play the role of am/pm.
        ("tomorrow morning 10", datetime(2026, 7, 17, 10, 0)),
        ("tomorrow morning at 10", datetime(2026, 7, 17, 10, 0)),
        ("tomorrow night 10", datetime(2026, 7, 17, 22, 0)),
        ("today evening 7", datetime(2026, 7, 16, 19, 0)),
        # Bare daypart gets its customary hour.
        ("tomorrow morning", datetime(2026, 7, 17, 9, 0)),
        ("tomorrow evening", datetime(2026, 7, 17, 18, 0)),
        # Day only defaults to 9 AM; bare hour reads as 24-hour clock.
        ("tomorrow", datetime(2026, 7, 17, 9, 0)),
        ("tomorrow at 19", datetime(2026, 7, 17, 19, 0)),
        ("17th july", datetime(2026, 7, 17, 9, 0)),  # no time given -> 9 AM default
        ("july 17", datetime(2026, 7, 17, 9, 0)),
        ("on 17th july at 10am", datetime(2026, 7, 17, 10, 0)),
        ("17 july at 9am", datetime(2026, 7, 17, 9, 0)),
        ("december 25", datetime(2026, 12, 25, 9, 0)),
        ("25 december at 6pm", datetime(2026, 12, 25, 18, 0)),
        # No year and the date already passed this year -> roll to next year,
        # matching how people mean birthdays/anniversaries.
        ("on 5 january", datetime(2027, 1, 5, 9, 0)),
    ],
)
def test_reminder_tool_parses_absolute_month_day_dates(phrase: str, expected: datetime) -> None:
    assert CreateReminderTool._parse_due_at(phrase, now=datetime(2026, 7, 16, 9, 0)) == expected


def test_reminder_tool_rejects_nonsense_dates() -> None:
    now = datetime(2026, 7, 16, 9, 0)
    assert CreateReminderTool._parse_due_at("someday", now=now) is None
    assert CreateReminderTool._parse_due_at("next thursday", now=now) is None


@pytest.mark.integration
async def test_reminder_tool_actually_creates_a_reminder() -> None:
    """AppleScript's `date "<string>"` literal is locale-dependent and can
    fail outright ("Invalid date and time") even though the Python-side date
    parsed correctly — this exercises the real osascript call end to end."""
    result = await CreateReminderTool().execute(
        {"title": "Jarvis test reminder — safe to delete", "due_at": "17 july at 9am"}
    )
    assert result.ok, result.summary


def test_whatsapp_tool_formats_international_phone_numbers() -> None:
    tool = WhatsAppSendTool(Settings(whatsapp_default_country_code=None))
    assert tool._chat_id("+1 415-555-0123") == "14155550123@c.us"
    assert tool._chat_id("14155550123@c.us") == "14155550123@c.us"
    assert tool._chat_id("Mohan") is None




@pytest.mark.integration
async def test_volume_read() -> None:
    result = await VolumeTool().execute({})
    assert result.ok, result.summary
    assert 0 <= result.data["level"] <= 100


async def test_volume_mute_preserves_the_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mute must use macOS's real 'output muted' flag, not zero the volume —
    zeroing loses the level, so unmuting can't restore it."""
    calls: list[str] = []

    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        calls.append(script)
        return CommandOutput(0, "", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)
    result = await VolumeTool().execute({"muted": True})
    assert result.ok, result.summary
    assert result.data == {"muted": True}
    assert any("set volume output muted true" in call for call in calls)
    assert not any("output volume" in call for call in calls)  # level untouched


async def test_volume_unmute(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        calls.append(script)
        return CommandOutput(0, "", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)
    result = await VolumeTool().execute({"muted": False})
    assert result.ok, result.summary
    assert result.data == {"muted": False}
    assert any("set volume output muted false" in call for call in calls)


async def test_volume_read_reports_muted_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        if "output volume" in script:
            return CommandOutput(0, "40", "")
        return CommandOutput(0, "true", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)
    result = await VolumeTool().execute({})
    assert result.ok, result.summary
    assert result.data == {"level": 40, "muted": True}
    assert "muted" in result.summary.lower()


async def test_volume_relative_adjustment_clamps_and_applies_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        calls.append(script)
        if "get volume settings" in script:
            return CommandOutput(0, "95", "")
        return CommandOutput(0, "", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)
    result = await VolumeTool().execute({"direction": "up", "amount": 20})
    assert result.ok, result.summary
    # 95 + 20 clamps to 100, not 115.
    assert result.data["level"] == 100
    assert any("set volume output volume 100" in call for call in calls)


async def test_volume_relative_down_reads_then_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_osascript(script: str, timeout: float = 30.0) -> CommandOutput:
        if "get volume settings" in script:
            return CommandOutput(0, "50", "")
        return CommandOutput(0, "", "")

    monkeypatch.setattr(system_module, "run_osascript", fake_osascript)
    result = await VolumeTool().execute({"direction": "down"})
    assert result.ok, result.summary
    assert result.data["level"] == 40  # default amount is 10


@pytest.mark.integration
async def test_volume_relative_adjustment_actually_changes_the_level() -> None:
    tool = VolumeTool()
    before = await tool.execute({})
    assert before.ok
    try:
        up = await tool.execute({"direction": "up", "amount": 5})
        assert up.ok, up.summary
        assert up.data["level"] == min(100, before.data["level"] + 5)
    finally:
        await tool.execute({"level": before.data["level"]})


def test_brightness_relative_adjustment_clamps_and_applies_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BrightnessTool, "_get_brightness", classmethod(lambda cls: 0.95))
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        BrightnessTool, "_set_brightness",
        classmethod(lambda cls, level: captured.setdefault("level", level) or True),
    )

    async def run_it() -> None:
        result = await BrightnessTool().execute({"direction": "up", "amount": 0.2})
        assert result.ok, result.summary
        assert result.data["level"] == pytest.approx(1.0)  # 0.95 + 0.2 clamps to 1.0

    asyncio.run(run_it())
    assert captured["level"] == pytest.approx(1.0)


@pytest.mark.integration
async def test_brightness_read_and_relative_adjustment_actually_changes_the_level() -> None:
    tool = BrightnessTool()
    before = await tool.execute({})
    assert before.ok, before.summary
    try:
        up = await tool.execute({"direction": "up", "amount": 0.1})
        assert up.ok, up.summary
        assert up.data["level"] == pytest.approx(min(1.0, before.data["level"] + 0.1), abs=0.01)
    finally:
        await tool.execute({"level": before.data["level"]})


@pytest.mark.integration
async def test_screenshot(tmp_path: Path) -> None:
    target = tmp_path / "shot.png"
    result = await ScreenshotTool().execute({"path": str(target)})
    if not result.ok and "Screen Recording" in result.summary:
        pytest.skip("host process lacks Screen Recording permission")
    assert result.ok, result.summary
    assert target.exists() and target.stat().st_size > 0


@pytest.mark.integration
async def test_clipboard_roundtrip_preserves_user_data() -> None:
    # Save whatever is on the clipboard now.
    saved = await asyncio.create_subprocess_exec(
        "/usr/bin/pbpaste", stdout=asyncio.subprocess.PIPE
    )
    original, _ = await saved.communicate()
    try:
        write = await ClipboardWriteTool().execute({"text": "jarvis-clipboard-test"})
        assert write.ok
        read = await ClipboardReadTool().execute({})
        assert read.ok
        assert "jarvis-clipboard-test" in read.summary
    finally:
        restore = await asyncio.create_subprocess_exec(
            "/usr/bin/pbcopy", stdin=asyncio.subprocess.PIPE
        )
        await restore.communicate(original)
