"""Deterministic fast-path for common, unambiguous voice commands.

Terse spoken commands ("next", "pause", "play") are exactly where a 3B model
is least reliable at tool selection — it often answers in text and fabricates
success. For a small, curated set of high-confidence commands we skip the LLM
entirely and map straight to a tool call. This is still routed through the
normal tool + safety layer by the planner; it only replaces the *selection*
step with a deterministic rule.

Anything not matched here (the vast majority of requests) returns None and
falls through to the LLM planner unchanged. Patterns are deliberately strict
(anchored, whole-utterance) so specific requests like "play Despacito" do NOT
match the generic "play" rule — they need the real planner.
"""

from __future__ import annotations

import re

from app.core.ollama_client import ToolCallRequest

# (compiled pattern, tool name, args) — first match wins.
_RULES: list[tuple[re.Pattern[str], str, dict[str, object]]] = [
    (re.compile(r"^(play |go to |skip to )?(the )?next( song| track| one)?$"),
     "media_control", {"action": "next"}),
    (re.compile(r"^skip( it| this| this song| song| track| this one)?$"),
     "media_control", {"action": "next"}),
    (re.compile(r"^(go (to |back to |back )?)?(the )?previous( song| track| one)?$"),
     "media_control", {"action": "previous"}),
    (re.compile(r"^(go back|last song|previous)$"),
     "media_control", {"action": "previous"}),
    (re.compile(r"^(pause|stop)( (the )?(music|song|track|playback|it|this|this song))?$"),
     "media_control", {"action": "pause"}),
    (re.compile(r"^(resume|unpause)( (the )?(music|song|track|playback|it))?$"),
     "media_control", {"action": "play"}),
    (re.compile(r"^continue( (the )?(music|song|track|playback|it))$"),
     "media_control", {"action": "play"}),
    (re.compile(r"^play( (the )?(music|song|track|it|something))?$"),
     "media_control", {"action": "play"}),
]

_YOUTUBE_PLAY = re.compile(
    r"^(?:open )?(?:(?P<browser>brave|google chrome|chrome|safari|firefox) "
    r"(?:browser )?and )?(?:open )?youtube and play (?P<query>.+)$"
)
_YOUTUBE_PLAY_AFTER_BROWSER = re.compile(
    r"^(?:open )?youtube(?: in (?P<browser>brave|google chrome|chrome|safari|firefox))?"
    r" and play (?P<query>.+)$"
)
_YOUTUBE_OPEN = re.compile(
    r"^(?:open |play )(?P<query>.+?) (?:in|on) youtube"
    r"(?: (?:in|on) (?P<browser>brave|google chrome|chrome|safari|firefox))?$"
)
_MUSIC_REQUEST = re.compile(
    r"^(?:i (?:would )?like to listen to (?P<listen_query>(?:(?:new|latest|some) )?.+ songs?)"
    r"|play (?P<play_query>(?:new|latest|some) .+ songs?))$"
)
_SPOTIFY_PLAY = re.compile(
    r"^(?:(?:open )?spotify(?: and)? play (?P<after_spotify>.+)"
    r"|play (?P<in_spotify>.+?) (?:in|on) spotify)$"
)
_SPOTIFY_OPEN_PLAYLIST = re.compile(
    r"^(?:open )?(?P<playlist>.+?) playlist (?:in|on) spotify$"
)
# WhatsApp: the recipient may be a contact name, a phone number, or both
# ("mohan kirushna 9080209303"); the tool resolves names via Contacts.
_WHATSAPP_PATTERNS = [
    # "send (a) whatsapp (message) to <recipient> saying <message>"
    re.compile(
        r"^(?:send )?(?:a )?whatsapp(?: message)? to (?P<recipient>.+?) "
        r"(?:saying|that says|with message) (?P<message>.+)$"
    ),
    # "in whatsapp send <message> (message) to <recipient>"
    re.compile(
        r"^(?:in|on|via|using) whatsapp send (?:a )?(?P<message>.+?)"
        r"(?: message)? to (?P<recipient>.+)$"
    ),
    # "send <message> (message) in whatsapp to <recipient>"
    re.compile(
        r"^send (?:a )?(?P<message>.+?)(?: message)? (?:in|on|via|using) whatsapp "
        r"to (?P<recipient>.+)$"
    ),
    # "send <message> (message) to <recipient> in whatsapp"
    re.compile(
        r"^send (?:a )?(?P<message>.+?)(?: message)? to (?P<recipient>.+?) "
        r"(?:in|on|via|using) whatsapp$"
    ),
]
# Trailing phone number in a recipient phrase such as "mohan kirushna 9080209303".
_WHATSAPP_PHONE_TAIL = re.compile(r"^(?:(?P<name>.+?) )?(?P<number>\d[\d ]{5,}\d)$")


def _whatsapp_recipient(raw: str) -> str:
    """Prefer an explicitly spoken number over the name preceding it."""
    raw = raw.strip()
    if not re.search(r"[a-zA-Z]", raw):
        return raw  # already just a number
    match = _WHATSAPP_PHONE_TAIL.fullmatch(raw)
    if match:
        return match.group("number")
    return raw
_RECENT_NEWS = re.compile(
    r"^(?:give me|show me|open) (?:the )?(?:latest|recent) news"
    r"(?: in (?P<browser>brave|google chrome|chrome|safari|firefox))?"
    r" about (?P<query>.+)$"
)
_LIST_BLUETOOTH_DEVICES = re.compile(
    r"^(?:can you )?(?:list|show)(?: me)? (?:all )?(?:the )?"
    r"(?:connected )?bluetooth devices(?: connected)?$"
)
_OPEN_CLAUDE = re.compile(r"^(?:open |go to )?claude$")
_OPEN_WEBSITE = re.compile(
    r"^(?:open|go to) (?P<site>youtube|gmail|google|maps|github|twitter|x|reddit|"
    r"chatgpt|netflix|amazon|whatsapp)$"
)
_OPEN_APP = re.compile(r"^(?:open|launch|start) (?P<name>.+)$")
_WIKIPEDIA_SEARCH = re.compile(
    r"^(?:search|look up|find) wikipedia (?:for|about) (?P<query>.+?)"
    r"(?: (?:in|on|using|with) (?P<browser>brave|google chrome|chrome|safari|firefox))?$"
)
_WIKIPEDIA_SEARCH_REVERSED = re.compile(
    r"^(?:search|look up|find) (?P<query>.+?) (?:in|on) wikipedia"
    r"(?: (?:in|on|using|with) (?P<browser>brave|google chrome|chrome|safari|firefox))?$"
)
_WIKIPEDIA_WITHOUT_QUERY = re.compile(
    r"^(?:search|look up|find) wikipedia"
    r"(?: (?:in|on|using|with) (?:brave|google chrome|chrome|safari|firefox))?$"
)
_BROWSER_SEARCH = re.compile(
    r"^(?:search|look up|find)(?: for)? (?P<query>.+?)"
    r"(?: (?:in|on|using|with) (?P<browser>brave|google chrome|chrome|safari|firefox))?$"
)
# Questions about live or changing information are poor candidates for the
# small local model: it has no current knowledge, and can otherwise respond
# with an unrelated capability prompt. Keep this deliberately narrow so
# ordinary conversation (for example, "how are you today?") still reaches the
# planner normally.
_LIVE_INFO_QUESTION = re.compile(
    r"^(?!(?:how are you|how r u)\b)(?:what|whats|who|when|where|which|how|"
    r"can|could|do|does|is|are|"
    r"tell|show|give)\b.*\b(?:today|yesterday|tomorrow|latest|recent|"
    r"current|live|now|news|weather|score|scores|result|results|price|prices|"
    r"stock|stocks|exchange rate|exchange rates)\b.*$"
)
# Factual trivia ("who is X", "what is X") is a poor fit for a 3B model's own
# knowledge — it is small, static, and prone to confidently making things up.
# Route it to a real web search instead. Self-referential questions ("who are
# you", "what is your name") are excluded so those stay ordinary conversation.
_GENERAL_KNOWLEDGE_QUESTION = re.compile(
    r"^(?:who|what|where|when|which) (?:is|are|was|were) "
    r"(?!(?:you|your|yourself|my|i|jarvis)\b).+$"
)
_SPORTS_RESULT_REQUEST = re.compile(
    r"\b(?:football|soccer|cricket|baseball|basketball|tennis|hockey|"
    r"rugby|f1|formula 1|sport|sports)\b.*\b(?:score|scores|result|results|"
    r"fixture|fixtures|standing|standings)\b"
)
_BATTERY_STATUS = re.compile(
    r"^(?:(?:what is|whats|tell me|show me) )?(?:my |the )?"
    r"(?:battery|charge)(?: percentage| percent| level| status)?"
    r"(?: (?:in|on) (?:my )?(?:laptop|macbook|mac))?$"
)
_SYSTEM_POWER = re.compile(
    r"^(?P<action>restart|reboot|shut down|shutdown|power off|turn off)"
    r"(?: (?:my )?(?:laptop|macbook|mac|computer))?$"
)
_VOLUME_MUTE = re.compile(r"^(?P<action>mute|unmute)(?: the (?:volume|sound|audio))?$")
_VOLUME_SET = re.compile(
    r"^(?:set|change|turn|put|make|reduce|lower|raise|increase) (?:the )?volume to "
    r"(?P<level>\d{1,3})(?: ?percent)?$"
)
_VOLUME_ADJUST = re.compile(
    r"^(?:turn (?:the )?volume (?P<dir1>up|down)"
    r"|(?P<dir2>increase|raise|decrease|lower|reduce) (?:the )?volume"
    r"|volume (?P<dir3>up|down))"
    r"(?: by (?P<amount>\d{1,3})(?: ?percent)?)?$"
)
_BRIGHTNESS_SET = re.compile(
    r"^(?:set|change|turn|put|make|reduce|lower|raise|increase) (?:the )?"
    r"(?:screen |display )?brightness to (?P<level>\d{1,3})(?: ?percent)?$"
)
_BRIGHTNESS_ADJUST = re.compile(
    r"^(?:turn (?:the )?(?:screen |display )?brightness (?P<dir1>up|down)"
    r"|(?P<dir2>increase|raise|decrease|lower|reduce) (?:the )?(?:screen |display )?brightness"
    r"|(?:screen |display )?brightness (?P<dir3>up|down)"
    r"|make (?:the )?(?:screen |display )?(?P<dir4>brighter|dimmer))"
    r"(?: by (?P<amount>\d{1,3})(?: ?percent)?)?$"
)

_TRAILING_BROWSER = re.compile(
    r"^(?P<query>.+?) (?:in|on|using|with) "
    r"(?P<browser>brave|google chrome|chrome|safari|firefox)$"
)


def _split_trailing_browser(normalized: str) -> tuple[str, str | None]:
    """Strip a trailing '... in <browser>' so an explicit browser choice is
    never silently dropped when routing an info/sports/trivia question."""
    match = _TRAILING_BROWSER.fullmatch(normalized)
    if match:
        return match.group("query"), match.group("browser")
    return normalized, None


_BROWSER_NAMES = {
    "brave": "Brave Browser",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "firefox": "Firefox",
}


def match_fast_intent(utterance: str) -> ToolCallRequest | None:
    """Return a tool call for a recognized terse command, else None."""
    normalized = re.sub(r"[^\w\s]", "", utterance).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    system_power = _SYSTEM_POWER.fullmatch(normalized)
    if system_power:
        raw_action = system_power.group("action")
        action = "restart" if raw_action in {"restart", "reboot"} else "shutdown"
        return ToolCallRequest(name="system_power", arguments={"action": action})
    if _BATTERY_STATUS.fullmatch(normalized):
        return ToolCallRequest(name="battery_status", arguments={})
    volume_mute = _VOLUME_MUTE.fullmatch(normalized)
    if volume_mute:
        return ToolCallRequest(
            name="volume", arguments={"muted": volume_mute.group("action") == "mute"}
        )
    volume_set = _VOLUME_SET.fullmatch(normalized)
    if volume_set:
        return ToolCallRequest(
            name="volume", arguments={"level": int(volume_set.group("level"))}
        )
    brightness_set = _BRIGHTNESS_SET.fullmatch(normalized)
    if brightness_set:
        return ToolCallRequest(
            name="brightness", arguments={"level": int(brightness_set.group("level")) / 100}
        )
    volume_adjust = _VOLUME_ADJUST.fullmatch(normalized)
    if volume_adjust:
        raw_dir = (
            volume_adjust.group("dir1")
            or volume_adjust.group("dir2")
            or volume_adjust.group("dir3")
        )
        direction = "up" if raw_dir in {"up", "increase", "raise"} else "down"
        args: dict[str, object] = {"direction": direction}
        if amount := volume_adjust.group("amount"):
            args["amount"] = int(amount)
        return ToolCallRequest(name="volume", arguments=args)
    brightness_adjust = _BRIGHTNESS_ADJUST.fullmatch(normalized)
    if brightness_adjust:
        raw_dir = (
            brightness_adjust.group("dir1")
            or brightness_adjust.group("dir2")
            or brightness_adjust.group("dir3")
            or brightness_adjust.group("dir4")
        )
        direction = "up" if raw_dir in {"up", "increase", "raise", "brighter"} else "down"
        args = {"direction": direction}
        if amount := brightness_adjust.group("amount"):
            args["amount"] = int(amount) / 100
        return ToolCallRequest(name="brightness", arguments=args)
    if _OPEN_CLAUDE.fullmatch(normalized):
        return ToolCallRequest(name="open_app", arguments={"name": "Claude"})
    open_website = _OPEN_WEBSITE.fullmatch(normalized)
    if open_website:
        return ToolCallRequest(
            name="open_url", arguments={"target": open_website.group("site")}
        )
    if _LIST_BLUETOOTH_DEVICES.fullmatch(normalized):
        return ToolCallRequest(name="list_bluetooth_devices", arguments={})
    wikipedia_search = (
        _WIKIPEDIA_SEARCH.fullmatch(normalized)
        or _WIKIPEDIA_SEARCH_REVERSED.fullmatch(normalized)
    )
    if wikipedia_search:
        args = {"query": wikipedia_search.group("query"), "engine": "wikipedia"}
        if browser := wikipedia_search.group("browser"):
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="browser_search", arguments=args)
    # Do not mistake the site name for the thing to search. The planner will
    # ask the one useful follow-up: which Wikipedia topic?
    if _WIKIPEDIA_WITHOUT_QUERY.fullmatch(normalized):
        return None
    browser_search = _BROWSER_SEARCH.fullmatch(normalized)
    if browser_search:
        query = browser_search.group("query")
        browser = browser_search.group("browser")
        if browser is None or browser == "brave":
            return ToolCallRequest(name="brave_search_open_first", arguments={"query": query})
        args = {"query": query, "engine": "google"}
        if browser:
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="browser_search", arguments=args)
    youtube_play = _YOUTUBE_PLAY.fullmatch(normalized)
    if youtube_play:
        args = {"query": youtube_play.group("query")}
        if browser := youtube_play.group("browser"):
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="youtube_play", arguments=args)
    youtube_play_after_browser = _YOUTUBE_PLAY_AFTER_BROWSER.fullmatch(normalized)
    if youtube_play_after_browser:
        args = {"query": youtube_play_after_browser.group("query")}
        if browser := youtube_play_after_browser.group("browser"):
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="youtube_play", arguments=args)
    youtube_open = _YOUTUBE_OPEN.fullmatch(normalized)
    if youtube_open:
        args = {"query": youtube_open.group("query")}
        if browser := youtube_open.group("browser"):
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="youtube_play", arguments=args)
    recent_news = _RECENT_NEWS.fullmatch(normalized)
    if recent_news:
        args = {"query": recent_news.group("query")}
        if browser := recent_news.group("browser"):
            args["browser"] = _BROWSER_NAMES[browser]
        return ToolCallRequest(name="news_search", arguments=args)
    for whatsapp_pattern in _WHATSAPP_PATTERNS:
        whatsapp_send = whatsapp_pattern.fullmatch(normalized)
        if whatsapp_send:
            return ToolCallRequest(
                name="whatsapp_send",
                arguments={
                    "recipient": _whatsapp_recipient(whatsapp_send.group("recipient")),
                    "message": whatsapp_send.group("message"),
                },
            )
    # A request such as "What football scores happened yesterday?" is an
    # implicit web search. Keep this after the explicit news command so it
    # retains its dedicated news-search behaviour.
    if (
        _LIVE_INFO_QUESTION.fullmatch(normalized)
        or _SPORTS_RESULT_REQUEST.search(normalized)
        or _GENERAL_KNOWLEDGE_QUESTION.fullmatch(normalized)
    ):
        query, browser = _split_trailing_browser(normalized)
        if browser and browser != "brave":
            return ToolCallRequest(
                name="browser_search",
                arguments={"query": query, "engine": "google", "browser": _BROWSER_NAMES[browser]},
            )
        return ToolCallRequest(
            name="brave_search_open_first", arguments={"query": query}
        )
    music_request = _MUSIC_REQUEST.fullmatch(normalized)
    if music_request:
        query = music_request.group("listen_query") or music_request.group("play_query")
        return ToolCallRequest(
            name="music_platform_prompt", arguments={"query": query}
        )
    spotify_playlist = _SPOTIFY_OPEN_PLAYLIST.fullmatch(normalized)
    if spotify_playlist:
        return ToolCallRequest(
            name="spotify_open_playlist",
            arguments={"playlist": spotify_playlist.group("playlist")},
        )
    spotify_play = _SPOTIFY_PLAY.fullmatch(normalized)
    if spotify_play:
        query = spotify_play.group("after_spotify") or spotify_play.group("in_spotify")
        return ToolCallRequest(name="spotify_play", arguments={"query": query})
    open_app = _OPEN_APP.fullmatch(normalized)
    if open_app:
        app_name = open_app.group("name")
        words = set(app_name.split())
        if (
            {"folder", "file", "document"} & words
            or app_name.startswith(("the next ", "next "))
            or "/" in app_name
            or "." in app_name
        ):
            return None
        return ToolCallRequest(
            name="open_app", arguments={"name": app_name}
        )
    for pattern, tool, args in _RULES:
        if pattern.fullmatch(normalized):
            return ToolCallRequest(name=tool, arguments=dict(args))
    return None
