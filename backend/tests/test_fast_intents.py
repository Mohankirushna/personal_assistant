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
        ("continue music", "play"),
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


def test_open_claude_uses_the_local_app() -> None:
    call = match_fast_intent("open claude")
    assert call is not None
    assert call.name == "open_app"
    assert call.arguments == {"name": "Claude"}


@pytest.mark.parametrize(
    ("utterance", "name"),
    [
        ("open visual studio code", "visual studio code"),
        ("launch slack", "slack"),
        ("start notes", "notes"),
    ],
)
def test_open_app_commands_use_the_local_app(utterance: str, name: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "open_app"
    assert call.arguments == {"name": name}


def test_open_website_does_not_try_to_launch_a_local_app() -> None:
    call = match_fast_intent("open youtube")
    assert call is not None
    assert call.name == "open_url"
    assert call.arguments == {"target": "youtube"}


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        (
            "search volcanoes in chrome",
            {"query": "volcanoes", "engine": "google", "browser": "Google Chrome"},
        ),
        (
            "search wikipedia for Ada Lovelace in Safari",
            {"query": "ada lovelace", "engine": "wikipedia", "browser": "Safari"},
        ),
    ],
)
def test_matches_visible_browser_searches(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "browser_search"
    assert call.arguments == arguments


@pytest.mark.parametrize("utterance", ["search for Ada Lovelace", "search Ada Lovelace in Brave"])
def test_general_search_opens_the_first_brave_result(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "brave_search_open_first"
    assert call.arguments == {"query": "ada lovelace"}


@pytest.mark.parametrize(
    "utterance",
    [
        "What is these football score which happened yesterday?",
        "What are the latest weather updates?",
        "Show me cricket results",
    ],
)
def test_live_information_questions_search_brave(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "brave_search_open_first"
    assert call.arguments == {"query": utterance.lower().rstrip("?")}


def test_ordinary_conversation_does_not_trigger_web_search() -> None:
    assert match_fast_intent("How are you today?") is None


@pytest.mark.parametrize(
    "utterance",
    [
        "What is the time now?",  # once web-searched -> a Yogi Berra book
        "what time is it",
        "What time is it right now?",
        "whats the time",
        "Tell me the time",
        "current time",
        "What day is it today?",
        "what's today's date",
        "What is the date?",
    ],
)
def test_time_and_date_questions_use_the_local_clock(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "clock"
    assert call.arguments == {}


def test_world_time_is_not_the_local_clocks_job() -> None:
    call = match_fast_intent("what time is it in tokyo")
    assert call is None or call.name != "clock"


@pytest.mark.parametrize(
    ("utterance", "path"),
    [
        # The exact phrasing that once became a web search about Downloads:
        ("check my downloads folders i want to know what all files i have", "~/Downloads"),
        ("What's in my Downloads folder?", "~/Downloads"),
        ("show me the files in my downloads", "~/Downloads"),
        ("list my documents", "~/Documents"),
        ("what files do I have on my desktop", "~/Desktop"),
        ("check my applications", "/Applications"),
    ],
)
def test_known_folder_listing_uses_finder_not_the_web(utterance: str, path: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "finder_list"
    assert call.arguments == {"path": path}


@pytest.mark.parametrize(
    "utterance",
    [
        "play some music",  # music request, not a folder listing
        "search where are my downloads on windows",  # about Windows, not this Mac
    ],
)
def test_folder_listing_does_not_swallow_other_intents(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is None or call.name != "finder_list"


@pytest.mark.parametrize(
    ("utterance", "browser"),
    [
        ("football score in google chrome", "Google Chrome"),
        ("football score in chrome", "Google Chrome"),
        ("football score in safari", "Safari"),
        ("football score in firefox", "Firefox"),
    ],
)
def test_sports_result_respects_an_explicit_non_brave_browser(
    utterance: str, browser: str
) -> None:
    """An explicit '... in <browser>' must never be silently dropped in
    favor of always opening Brave — that was a real bug: the sports/live-info
    fast-path matched on a substring search and ignored the browser entirely."""
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "browser_search"
    assert call.arguments == {"query": "football score", "engine": "google", "browser": browser}


def test_sports_result_with_brave_still_opens_the_first_result() -> None:
    call = match_fast_intent("football score in brave")
    assert call is not None
    assert call.name == "brave_search_open_first"
    assert call.arguments == {"query": "football score"}


@pytest.mark.parametrize(
    "utterance",
    ["Who is Ironman?", "What is the capital of France?", "Where is the Eiffel Tower?"],
)
def test_general_knowledge_questions_search_the_web(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "brave_search_open_first"
    assert call.arguments == {"query": utterance.lower().rstrip("?")}


@pytest.mark.parametrize(
    "utterance", ["Who are you?", "What is your name?", "Who is Jarvis?"]
)
def test_self_referential_questions_stay_conversational(utterance: str) -> None:
    assert match_fast_intent(utterance) is None


@pytest.mark.parametrize(
    ("utterance", "recipient", "message"),
    [
        ("Send WhatsApp to +1 415 555 0123 saying hello there", "1 415 555 0123", "hello there"),
        # Name and number together: the spoken number wins.
        (
            "in whatsapp send hello message to Mohan kirushna 9080209303",
            "9080209303",
            "hello",
        ),
        # Name alone: the tool resolves it via Contacts.
        ("send hello message in whatsapp to mohan kirushna", "mohan kirushna", "hello"),
        ("send a whatsapp message to Mohan saying I will be late", "mohan", "i will be late"),
        ("send good night to amma on whatsapp", "amma", "good night"),
        ("on whatsapp send call me when free to mohan", "mohan", "call me when free"),
    ],
)
def test_whatsapp_send_commands(utterance: str, recipient: str, message: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "whatsapp_send"
    assert call.arguments == {"recipient": recipient, "message": message}


def test_wikipedia_without_a_topic_needs_a_follow_up() -> None:
    assert match_fast_intent("search wikipedia in chrome") is None


@pytest.mark.parametrize(
    "utterance",
    ["what is my charge percentage in my laptop", "battery percentage", "show me my battery level"],
)
def test_matches_battery_status_commands(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "battery_status"
    assert call.arguments == {}


@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("restart laptop", "restart"),
        ("reboot my mac", "restart"),
        ("shutdown computer", "shutdown"),
        ("turn off my macbook", "shutdown"),
    ],
)
def test_matches_system_power_commands(utterance: str, action: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "system_power"
    assert call.arguments == {"action": action}


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("open youtube and play enemy song", {"query": "enemy song"}),
        (
            "open brave and open youtube and play enemy song",
            {"query": "enemy song", "browser": "Brave Browser"},
        ),
        (
            "open youtube in brave and play some tamil song",
            {"query": "some tamil song", "browser": "Brave Browser"},
        ),
        ("open enemy song in youtube", {"query": "enemy song"}),
        (
            "open enemy song in youtube in brave",
            {"query": "enemy song", "browser": "Brave Browser"},
        ),
    ],
)
def test_matches_youtube_play_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "youtube_play"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "query"),
    [
        ("i like to listen to new tamil songs", "new tamil songs"),
        ("play some tamil songs", "some tamil songs"),
    ],
)
def test_matches_general_music_requests(utterance: str, query: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "music_platform_prompt"
    assert call.arguments == {"query": query}


@pytest.mark.parametrize(
    "utterance",
    ["play enemy song in spotify", "open spotify and play enemy song"],
)
def test_matches_spotify_play_commands(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "spotify_play"
    assert call.arguments == {"query": "enemy song"}


def test_matches_spotify_playlist_commands() -> None:
    call = match_fast_intent("open feel playlist in spotify")
    assert call is not None
    assert call.name == "spotify_open_playlist"
    assert call.arguments == {"playlist": "feel"}


@pytest.mark.parametrize(
    "utterance",
    [
        "can you list all the bluetooth devices connected",
        "show connected bluetooth devices",
    ],
)
def test_matches_bluetooth_device_list_commands(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "list_bluetooth_devices"
    assert call.arguments == {}


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        (
            "give me recent news in brave about attack on titan anime",
            {"query": "attack on titan anime", "browser": "Brave Browser"},
        ),
        ("show me latest news about space", {"query": "space"}),
    ],
)
def test_matches_recent_news_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "news_search"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("mute", {"muted": True}),
        ("mute the volume", {"muted": True}),
        ("mute the sound", {"muted": True}),
        ("unmute", {"muted": False}),
        ("unmute the volume", {"muted": False}),
    ],
)
def test_matches_mute_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "volume"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("reduce the volume to 50 percent", {"level": 50}),
        ("set the volume to 30", {"level": 30}),
        ("lower the volume to 20 percent", {"level": 20}),
        ("increase the volume to 100", {"level": 100}),
    ],
)
def test_matches_volume_set_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "volume"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("set brightness to 80 percent", {"level": 0.8}),
        ("reduce the brightness to 40", {"level": 0.4}),
        ("set the display brightness to 100", {"level": 1.0}),
    ],
)
def test_matches_brightness_set_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "brightness"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("turn the volume up", {"direction": "up"}),
        ("turn volume down", {"direction": "down"}),
        ("increase the volume", {"direction": "up"}),
        ("decrease volume", {"direction": "down"}),
        ("volume up", {"direction": "up"}),
        ("volume down", {"direction": "down"}),
        ("raise the volume by 20", {"direction": "up", "amount": 20}),
        ("lower the volume by 15 percent", {"direction": "down", "amount": 15}),
    ],
)
def test_matches_volume_adjust_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "volume"
    assert call.arguments == arguments


@pytest.mark.parametrize(
    ("utterance", "arguments"),
    [
        ("turn the brightness up", {"direction": "up"}),
        ("turn brightness down", {"direction": "down"}),
        ("increase the brightness", {"direction": "up"}),
        ("decrease screen brightness", {"direction": "down"}),
        ("brightness up", {"direction": "up"}),
        ("make the screen brighter", {"direction": "up"}),
        ("make the display dimmer", {"direction": "down"}),
        ("raise the brightness by 20", {"direction": "up", "amount": 0.2}),
    ],
)
def test_matches_brightness_adjust_commands(utterance: str, arguments: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "brightness"
    assert call.arguments == arguments


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
        "continue",                    # follow-up, needs conversation context
        "continue the previous request",
        "play a game with me",
        "",
    ],
)
def test_does_not_match_ambiguous_or_specific(utterance: str) -> None:
    assert match_fast_intent(utterance) is None, f"{utterance!r} should NOT match"


@pytest.mark.parametrize(
    ("utterance", "minutes", "label"),
    [
        ("set a timer for 10 minutes", 10, ""),
        ("10 minute timer", 10, ""),
        ("5-minute timer", 5, ""),
        ("set a 15 minute timer for laundry", 15, "laundry"),
        ("timer for 3 minutes labeled cooking", 3, "cooking"),
    ],
)
def test_timer_commands_use_timer_tool(
    utterance: str, minutes: int, label: str
) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "timer"
    expected_args = {"minutes": minutes}
    if label:
        expected_args["label"] = label
    assert call.arguments == expected_args


@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("turn on do not disturb", "on"),
        ("enable focus mode", "on"),
        ("turn off focus mode", "off"),
        ("disable do not disturb", "off"),
        ("toggle focus mode", "toggle"),
    ],
)
def test_focus_mode_commands_use_focus_mode_tool(
    utterance: str, action: str
) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "focus_mode"
    assert call.arguments == {"action": action}


@pytest.mark.parametrize(
    "utterance",
    [
        "show my calendar",
        "check my calendar",
        "what's my calendar",
        "list my events",
        "view my meetings",
        "see my calendar for today",
    ],
)
def test_calendar_commands_use_calendar_tool(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "calendar"
    assert call.arguments == {"query": "today"}
