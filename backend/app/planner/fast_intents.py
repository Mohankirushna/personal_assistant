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
_CHECK_EMAIL = re.compile(
    r"^(?:check|read|show me|open)(?: my)?(?: new)? (?:emails?|mail|inbox)$"
    r"|^(?:any|do i have(?: any)?) new (?:emails?|mail)$"
)
_SEND_EMAIL = re.compile(
    r"^(?:send (?:an |a )?email to|email) (?P<recipient>.+?) "
    r"(?:saying|that says|with message) (?P<body>.+)$"
)


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
# The user's own calendar/schedule is local state, never a web query — even
# with a typo ("my calender today" was once web-searched to a date-picker
# site because it started with "whats" and contained "today").
_LIVE_INFO_QUESTION = re.compile(
    r"^(?!(?:how are you|how r u)\b)"
    r"(?!.*\bmy (?:calendar|calender|calandar|schedule|meetings?|events)\b)"
    r"(?:what|whats|who|when|where|which|how|"
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
# Local-state questions MUST be checked before any web-search routing below:
# "what is the time now" otherwise matches the live-info pattern (starts with
# "what", contains "now") and gets web-searched — which once opened an
# article about the Yogi Berra book "What Time Is It? You Mean Now?".
# No "in <place>" variants: world time is not the local clock's job.
_CLOCK_QUESTION = re.compile(
    r"^(?:(?:can|could) you )?(?:please )?(?:tell me )?"
    r"(?:what time is it|what is the time|whats the time|the time|time|"
    r"current time|what is the current time|whats the current time|"
    r"what day is (?:it|today)|what is the day|whats the day|"
    r"(?:what is|whats) (?:the date|todays date)|todays date|current date|"
    r"what is today|whats today)"
    r"(?: right)?(?: now)?(?: today)?( please)?$"
)
# Listing a well-known folder is a Finder job, never a web search. Verb and
# folder are matched separately (search, not fullmatch) so conversational
# framing survives: "check my downloads folder, I want to know what files
# I have" — which once got web-searched as "contents of Downloads folder".
_FOLDER_LIST_VERB = re.compile(
    r"\b(?:check|list|show|see|view|browse|whats? in|what is in|what all files|"
    r"what files|files (?:in|inside|of)|contents of|what do i have)\b"
)
_FOLDER_LIST_NAME = re.compile(
    r"\b(?P<folder>downloads?|documents?|desktop|applications|pictures|movies)\b"
)
_KNOWN_FOLDERS = {
    "download": "~/Downloads",
    "downloads": "~/Downloads",
    "document": "~/Documents",
    "documents": "~/Documents",
    "desktop": "~/Desktop",
    "applications": "/Applications",
    "pictures": "~/Pictures",
    "movies": "~/Movies",
}
# Timer: match "set a timer for 10 minutes", "10 minute timer", etc.
# Flexible to handle various orderings and formats.
def _match_timer(normalized: str) -> tuple[int, str] | None:
    """Parse timer commands; return (minutes, label) or None."""
    # Extract the number first (most reliable anchor)
    num_match = re.search(r"(\d{1,2})", normalized)
    if not num_match:
        return None
    minutes = int(num_match.group(1))
    if minutes < 1 or minutes > 60:
        return None

    # Check that "timer" and a time unit ("minute" or "min") are present.
    # Don't use \b because normalization may have removed hyphens (5-minute -> 5minute)
    if "timer" not in normalized or "minute" not in normalized and "min" not in normalized:
        return None

    # Extract optional label.
    # First try "labeled X" or "called X" (most explicit).
    label = ""
    for prefix in ["labeled", "called"]:
        m = re.search(rf"{prefix}\s+(.+?)$", normalized)
        if m:
            label = m.group(1).strip()
            break

    # If no explicit prefix, try "for X" but only if it appears AFTER the time
    # unit, to avoid matching "timer for 10 minutes" as having a label.
    if not label:
        after_time = re.search(
            r"(?:minute|min)(?:ute)?s?[\w\s]*(?:for|labeled|called)\s+(.+?)$",
            normalized,
        )
        if after_time:
            label = after_time.group(1).strip()

    return minutes, label
# Focus mode: "turn on/off do not disturb", "enable/disable focus mode", etc.
_FOCUS_MODE_ON = re.compile(
    r"^(?:turn on|enable|activate|start) (?:do not disturb|focus mode|dnd)$"
)
_FOCUS_MODE_OFF = re.compile(
    r"^(?:turn off|disable|deactivate|stop) (?:do not disturb|focus mode|dnd)$"
)
_FOCUS_MODE_TOGGLE = re.compile(
    r"^(?:toggle|switch) (?:do not disturb|focus mode|dnd)$"
)
# Calendar: "show my calendar", "whats my calendar today", "next meeting"...
# "for" must be optional: "whats my calendar today" otherwise falls through
# to the live-info pattern (starts with "whats", contains "today") and gets
# web-searched — observed live.
_CALENDAR = re.compile(
    r"^(?:show|check|what'?s|list|view|see|tell me) (?:my |the )?"
    r"(?:calendar|calender|calandar|schedule|events|meetings)"
    r"(?: (?:for )?(?P<when>today|tomorrow|this week))?$"
    r"|^(?:what'?s|show|check) (?:my |the )?next meeting$"
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
    if _CLOCK_QUESTION.fullmatch(normalized):
        return ToolCallRequest(name="clock", arguments={})
    folder_name = _FOLDER_LIST_NAME.search(normalized)
    if folder_name and _FOLDER_LIST_VERB.search(normalized):
        return ToolCallRequest(
            name="finder_list",
            arguments={"path": _KNOWN_FOLDERS[folder_name.group("folder")]},
        )
    timer_result = _match_timer(normalized)
    if timer_result:
        minutes, label = timer_result
        args: dict[str, object] = {"minutes": minutes}
        if label:
            args["label"] = label
        return ToolCallRequest(name="timer", arguments=args)
    if _FOCUS_MODE_ON.fullmatch(normalized):
        return ToolCallRequest(name="focus_mode", arguments={"action": "on"})
    if _FOCUS_MODE_OFF.fullmatch(normalized):
        return ToolCallRequest(name="focus_mode", arguments={"action": "off"})
    if _FOCUS_MODE_TOGGLE.fullmatch(normalized):
        return ToolCallRequest(name="focus_mode", arguments={"action": "toggle"})
    calendar_match = _CALENDAR.fullmatch(normalized)
    if calendar_match:
        when = calendar_match.group("when")
        day = "tomorrow" if when == "tomorrow" else "today"
        return ToolCallRequest(name="calendar", arguments={"day": day})
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
        args: dict[str, object] = {"direction": direction}  # type: ignore[no-redef]
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
    if _CHECK_EMAIL.fullmatch(normalized):
        return ToolCallRequest(name="check_email", arguments={})
    # Email needs a gentler normalization: the standard one strips @ and .
    # which would mangle a typed address like a@b.co into "abco".
    address_friendly = re.sub(r"[^\w\s@.\-]", "", utterance).strip().lower()
    address_friendly = re.sub(r"\s+", " ", address_friendly)
    send_email = _SEND_EMAIL.fullmatch(address_friendly)
    if send_email:
        return ToolCallRequest(
            name="send_email",
            arguments={
                "recipient": send_email.group("recipient"),
                "body": send_email.group("body"),
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
        # An explicit browser ("...in chrome") means open a page there; a bare
        # question means the user wants an answer, so read the web and reply.
        if browser is None:
            return ToolCallRequest(name="web_answer", arguments={"query": query})
        if browser == "brave":
            return ToolCallRequest(name="brave_search_open_first", arguments={"query": query})
        return ToolCallRequest(
            name="browser_search",
            arguments={"query": query, "engine": "google", "browser": _BROWSER_NAMES[browser]},
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
