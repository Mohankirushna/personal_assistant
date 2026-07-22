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
_MORNING_BRIEFING = re.compile(
    r"^(?:good morning"
    r"|(?:give me |run )?(?:my )?(?:morning |daily )?brief(?:ing)?(?: me)?"
    r"|brief me"
    r"|what(?:s| is| does)? my day(?: look)?(?: like)?"
    r"|what(?:s| is)? on today)$"
)
_CHECK_EMAIL = re.compile(
    r"^(?:check|read|show me|open)(?: my)?(?: new)? (?:emails?|mail|inbox)$"
    r"|^(?:any|do i have(?: any)?) new (?:emails?|mail)$"
)
_SEND_EMAIL = re.compile(
    r"^(?:send (?:an |a )?email to|email) (?P<recipient>.+?) "
    r"(?:saying|that says|with message) (?P<body>.+)$"
)
_SUMMARIZE_INBOX = re.compile(
    r"^(?:summari[sz]e|sum up|give me a summary of|what(?:'s| is| are)? in) "
    r"(?:my )?(?:unread )?(?:emails?|inbox|mail)"
    r"(?: (?:about|regarding))?$"
    r"|^what(?:'s| are)? my (?:unread )?(?:emails?|mail) about$"
)
# General "what are the recent mails" / "show me my emails" / "what's in my
# inbox" — summarize the newest unread mail (no sender/topic filter).
# Note: normalization strips apostrophes, so "what's" arrives as "whats" —
# match the bare "s", not "'s".
_RECENT_MAILS = re.compile(
    r"^(?:what(?:s| is| are)?|show me|list|read|give me|tell me about|do i have)\s+"
    r"(?:the |my |any )?(?:recent|latest|new|unread|newest|last)?\s*"
    r"(?:e-?mails?|mails?|messages?|inbox)$"
    r"|^(?:recent|latest|new|unread|newest)\s+(?:e-?mails?|mails?)$"
    r"|^what(?:s| is)? in my (?:inbox|mail|mailbox)$"
)
_CHECK_EMAIL_FROM = re.compile(
    r"^(?:any|do i have(?: any)?|is there(?: any)?|check(?: for)?) "
    r"(?:new |unread )?(?:emails?|mail) from (?P<sender>.+)$"
)
# "what is the recent mail from X", "read the latest email from X",
# "show me mail from X" — read/summarize a sender's mail (regardless of read
# status). Must be checked before web-search routing, since "recent"/"latest"
# would otherwise trip the live-info pattern and google the question.
_READ_MAIL_FROM = re.compile(
    r"^(?:what(?:'s| is| are)?|show me|read|tell me|get)\s+"
    r"(?:the |my )?(?:recent|latest|last|newest|new)?\s*"
    r"(?:e-?mails?|messages?|mail)\s+(?:from|by)\s+(?P<sender>.+)$"
)
# Catch-all so any free-form question about email from someone reaches the
# mail tool instead of web search, however it's phrased ("what about mails
# from github, was there a recent one?"). Requires an email keyword AND a
# "from <sender>" clause; the sender is the address or the first few
# non-filler words after "from".
_MAIL_KEYWORD = re.compile(r"\b(?:e-?mails?|mails?|inbox)\b")
_MAIL_FROM = re.compile(r"\bfrom\s+(?P<rest>.+)$")
_MAIL_ABOUT = re.compile(r"\b(?:about|regarding|mentioning|on the topic of)\s+(?P<rest>.+)$")
_SENDER_STOPWORDS = frozenset({
    "was", "is", "are", "were", "there", "any", "recent", "recently", "one",
    "ones", "the", "a", "an", "new", "latest", "last", "newest", "today",
    "please", "me", "i", "yet", "so", "far", "and", "or", "about", "regarding",
    "saying", "that", "with", "in", "my", "unread", "lately", "still",
})


def _clean_target(rest: str) -> str | None:
    words = rest.split()
    # A leading article introduces the target ("from the professor") — skip it.
    while words and words[0] in {"the", "a", "an", "my"}:
        words = words[1:]
    if not words:
        return None
    if "@" in words[0]:  # a full email address stands alone
        return words[0]
    picked: list[str] = []
    for word in words:
        if word in _SENDER_STOPWORDS:
            break
        picked.append(word)
        if len(picked) >= 3:
            break
    return " ".join(picked) or None


def _extract_mail_target(text: str) -> tuple[str, str] | None:
    """For a free-form email question, return ('sender', X) for "mail from X"
    or ('query', X) for "mail about X" (a topic search), else None."""
    if not _MAIL_KEYWORD.search(text):
        return None
    from_match = _MAIL_FROM.search(text)
    if from_match and (sender := _clean_target(from_match.group("rest"))):
        return "sender", sender
    about_match = _MAIL_ABOUT.search(text)
    if about_match and (topic := _clean_target(about_match.group("rest"))):
        return "query", topic
    return None
_REPLY_EMAIL = re.compile(
    r"^reply(?: to)?(?: the)?(?: latest| last| newest)?(?: email| mail)?"
    r"(?: from (?P<sender>.+?))? (?:saying|that says|with) (?P<body>.+)$"
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

# A bare article/pronoun ("open the github" with no project named) can end up
# captured as the name itself: the optional "(?:the\s+)?" in these patterns
# backtracks and lets ".+?" swallow "the" when there's nothing after it for
# the literal suffix ("github", "repo", ...) to attach to otherwise. Reject
# these rather than resolving to a nonsense project like "the" — the caller
# should fall through to the full LLM planner, which has conversation
# history and can resolve "it"/"the" contextually.
_BARE_REFERENCE_WORDS = frozenset({"the", "my", "a", "an", "it", "this", "that"})


# "where is the X project", "path for X", "locate the X repo" — a question
# about a LOCAL project's location, never a web search. This MUST be matched
# before the live-info/general-knowledge web routing below, because a project
# named with a trigger word ("stocks", "news") would otherwise be web-searched
# ("where is the stocks project" starts with "where" and contains "stocks").
_LOCATE_PROJECT_PATTERNS = [
    re.compile(
        r"^(?:where is|wheres|where)\s+(?:the\s+)?(?P<name>.+?)\s+"
        r"(?:project|repo|repository|folder|directory)(?:\s+(?:located|at))?$"
    ),
    re.compile(
        r"^locate\s+(?:the\s+)?(?P<name>.+?)\s+"
        r"(?:project|repo|repository|folder|directory)$"
    ),
    re.compile(
        r"^(?:give me\s+|tell me\s+)?(?:the\s+)?(?:local\s+|folder\s+|file\s+)?"
        r"(?:path|location)\s+(?:for|of|to)\s+(?:the\s+)?(?P<name>.+?)"
        r"(?:\s+(?:project|repo|repository|folder))?$"
    ),
    re.compile(
        r"^what(?:s| is)?\s+(?:the\s+)?(?:local\s+|folder\s+|file\s+)?"
        r"(?:path|location)\s+(?:for|of|to)\s+(?:the\s+)?(?P<name>.+?)"
        r"(?:\s+(?:project|repo|repository|folder))?$"
    ),
]


def _match_locate_project(normalized: str) -> str | None:
    """Return the project name for a 'where is X project' question, else None."""
    for pattern in _LOCATE_PROJECT_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            name = match.group("name").strip()
            name = re.sub(r"^(?:the|my|a)\s+", "", name).strip()
            name = re.sub(
                r"\s+(?:project|repo|repository|folder|directory)$", "", name
            ).strip()
            if name and name not in _BARE_REFERENCE_WORDS:
                return name
    return None


# "delete the X repo", "remove X from github" — a destructive action that requires
# the planner to route to a tool with explicit confirmation, never a chat answer.
_DELETE_REPO_PATTERNS = [
    re.compile(
        r"^(?:delete|remove)\s+(?:the\s+)?(?:github\s+)?repo(?:sitory)?\s+for\s+"
        r"(?:the\s+)?(?P<name>.+?)(?:\s+(?:on|from)\s+github)?$"
    ),
    re.compile(
        r"^(?:delete|remove)\s+(?P<name>.+?)\s+(?:from|on)\s+github$"
    ),
    re.compile(
        r"^(?:delete|remove)\s+(?:the\s+)?(?P<name>.+?)\s+repo(?:sitory)?$"
    ),
]


def _match_delete_repo(normalized: str) -> str | None:
    """Return the project name for a 'delete X repo' command, else None."""
    for pattern in _DELETE_REPO_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            name = match.group("name").strip()
            name = re.sub(r"^(?:the|my|a|github)\s+", "", name).strip()
            name = re.sub(
                r"\s+(?:repo|repository|project|github)?$", "", name
            ).strip()
            if name and name not in _BARE_REFERENCE_WORDS:
                return name
    return None


# "open fitness project in github", "open fitness github repo", "show me
# fitness on github", "open the github repo for fitness" — with an explicit
# project named, this must go straight to github_open_repo. Without this,
# the generic "open X" catch-all (_OPEN_APP, below) would try to launch a
# macOS application literally named "fitness project in github".
_OPEN_REPO_PATTERNS = [
    re.compile(
        r"^(?:open|show me|show|view)\s+(?:the\s+)?(?P<name>.+?)"
        r"\s+(?:project\s+)?(?:repo\s+)?(?:in|on)\s+github$"
    ),
    re.compile(
        r"^(?:open|show me|show|view)\s+(?:the\s+)?(?P<name>.+?)\s+github"
        r"(?:\s+repo(?:sitory)?)?$"
    ),
    re.compile(
        r"^(?:open|show me|show|view)\s+(?:the\s+)?github\s+(?:repo|page|link)?"
        r"\s*for\s+(?:the\s+)?(?P<name>.+)$"
    ),
]


def _match_open_repo(normalized: str) -> str | None:
    """Return the project name for an 'open X on github' command, else None."""
    for pattern in _OPEN_REPO_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            name = match.group("name").strip()
            name = re.sub(r"^(?:the|my|a)\s+", "", name).strip()
            name = re.sub(
                r"\s+(?:project|repo|repository)$", "", name
            ).strip()
            if name and name not in _BARE_REFERENCE_WORDS:
                return name
    return None


# "open fitness project", "open the jarvis folder", "show skin project" —
# with "project"/"folder" suffix, this is clearly a local folder intent,
# never a macOS app-launcher attempt. Routes to locate_project which will
# report the folder path and status.
_OPEN_PROJECT_PATTERNS = [
    re.compile(
        r"^(?:open|show|view)\s+(?:the\s+)?(?P<name>.+?)"
        r"\s+(?:project|folder|directory)(?:\s+(?:in\s+finder|in\s+vscode))?$"
    ),
]


def _match_open_project(normalized: str) -> str | None:
    """Return the project name for an 'open X project/folder' command, else None."""
    for pattern in _OPEN_PROJECT_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            name = match.group("name").strip()
            name = re.sub(r"^(?:the|my|a)\s+", "", name).strip()
            # Exclude generic terms that aren't project names
            if (
                name
                and name not in _BARE_REFERENCE_WORDS
                and name not in {"next"}
            ):
                return name
    return None


# "push jarvis project to github", "push fitness to github as fitness-app" —
# the codebase also has a generic low-level `git` tool (raw arguments + repo
# path) whose description overlaps with github_push's ("push", "commit").
# The small planner has picked the wrong one and failed to fill its required
# fields (observed live: called `git` with only {"project": "jarvis_v2"},
# missing its required `arguments`/`repo`). Matching this deterministically
# removes the ambiguity entirely for the common phrasing.
_PUSH_REPO_PATTERNS = [
    re.compile(
        r"^push\s+(?:the\s+)?(?P<name>.+?)\s+(?:project\s+)?to\s+github"
        r"(?:\s+as\s+(?P<repo>.+))?$"
    ),
    # "create a repo and push fitness project", "create repo and push fitness to github"
    re.compile(
        r"^create\s+(?:a\s+)?repo(?:sitory)?\s+(?:and\s+)?push\s+(?:the\s+)?(?P<name>.+?)"
        r"(?:\s+(?:project|repo))?(?:\s+to\s+github)?(?:\s+as\s+(?P<repo>.+))?$"
    ),
]

# "push changes to github", "push my changes to github", "push it to github"
# — these name no actual project, just refer generically to whatever's
# outstanding. Must fall through (to the full planner, which can ask which
# project) rather than treating "changes" as a literal project name.
_PUSH_GENERIC_WORDS = frozenset(
    {"changes", "my changes", "the changes", "it", "everything", "all", "code", "this"}
)


def _match_push_repo(normalized: str) -> tuple[str, str | None] | None:
    """Return (project, repo_name) for a 'push X to github' command, else None."""
    for pattern in _PUSH_REPO_PATTERNS:
        match = pattern.fullmatch(normalized)
        if match:
            name = match.group("name").strip()
            name = re.sub(r"^(?:the|my|a)\s+", "", name).strip()
            if (
                name
                and name not in _BARE_REFERENCE_WORDS
                and name not in _PUSH_GENERIC_WORDS
            ):
                repo = match.group("repo")
                return name, (repo.strip() if repo else None)
    return None


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
    # A gentler normalization that keeps @ . - so email addresses survive
    # ("from a.b@x.co" stays intact instead of becoming "abxco"). Used by the
    # email fast-paths that capture a sender or recipient.
    address_friendly = re.sub(r"[^\w\s@.\-]", "", utterance).strip().lower()
    address_friendly = re.sub(r"\s+", " ", address_friendly)
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
    # Project-location questions before any web routing (a project named
    # "stocks"/"news" would otherwise trip the live-info web search).
    locate_name = _match_locate_project(normalized)
    if locate_name:
        return ToolCallRequest(name="locate_project", arguments={"project": locate_name})
    # Delete-repo commands are destructive and require tool routing (for confirmation).
    delete_name = _match_delete_repo(normalized)
    if delete_name:
        return ToolCallRequest(name="github_delete_repo", arguments={"project": delete_name})
    # "open X on github" with an explicit project — before the generic "open
    # X" app-launcher below, which would otherwise try to open a macOS app
    # literally named "fitness project in github".
    open_repo_name = _match_open_repo(normalized)
    if open_repo_name:
        return ToolCallRequest(name="github_open_repo", arguments={"project": open_repo_name})
    # "open fitness project", "open the jarvis folder" — route to open_app,
    # which is project-aware: it resolves the name against the local projects
    # and opens the folder in VS Code (falling back to launching an app if it's
    # not a project). "where is X" / "locate X" still go to locate_project
    # above when the user only wants the path, not to open it.
    open_project_name = _match_open_project(normalized)
    if open_project_name:
        return ToolCallRequest(name="open_app", arguments={"name": open_project_name})
    # "push X (project) to github" — bypasses the ambiguity between
    # github_push and the low-level `git` tool for this common phrasing.
    push_repo_match = _match_push_repo(normalized)
    if push_repo_match:
        push_project, push_repo_name = push_repo_match
        push_args: dict[str, object] = {"project": push_project}
        if push_repo_name:
            push_args["repo_name"] = push_repo_name
        return ToolCallRequest(name="github_push", arguments=push_args)
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
    if _MORNING_BRIEFING.fullmatch(normalized):
        return ToolCallRequest(name="morning_briefing", arguments={})
    if _SUMMARIZE_INBOX.fullmatch(normalized) or _RECENT_MAILS.fullmatch(normalized):
        return ToolCallRequest(name="summarize_inbox", arguments={})
    check_from = _CHECK_EMAIL_FROM.fullmatch(address_friendly)
    if check_from:
        return ToolCallRequest(
            name="check_email", arguments={"sender": check_from.group("sender").strip()}
        )
    read_from = _READ_MAIL_FROM.fullmatch(address_friendly)
    if read_from:
        return ToolCallRequest(
            name="summarize_inbox", arguments={"sender": read_from.group("sender").strip()}
        )
    if _CHECK_EMAIL.fullmatch(normalized):
        return ToolCallRequest(name="check_email", arguments={})
    send_email = _SEND_EMAIL.fullmatch(address_friendly)
    if send_email:
        return ToolCallRequest(
            name="send_email",
            arguments={
                "recipient": send_email.group("recipient"),
                "body": send_email.group("body"),
            },
        )
    reply_email = _REPLY_EMAIL.fullmatch(normalized)
    if reply_email:
        reply_args: dict[str, object] = {"body": reply_email.group("body")}
        if reply_email.group("sender"):
            reply_args["sender"] = reply_email.group("sender").strip()
        return ToolCallRequest(name="reply_email", arguments=reply_args)
    # Broad catch-all: any remaining email question ("mail from X", "mail
    # about <topic>") routes to the mail tool rather than web search, whatever
    # the phrasing. After the specific send/reply intents so those win first.
    mail_target = _extract_mail_target(address_friendly)
    if mail_target:
        kind, value = mail_target
        return ToolCallRequest(name="summarize_inbox", arguments={kind: value})
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
            {"folder", "file", "document", "github", "repo", "repository"} & words
            or app_name.startswith(("the next ", "next "))
            or "/" in app_name
            or "." in app_name
        ):
            # "github"/"repo" here means _match_open_repo above didn't find an
            # explicit project name (e.g. bare "open the github" referring
            # back to whatever was just discussed) — that needs conversation
            # history to resolve, which only the full LLM planner has. Never
            # let this become an attempt to launch a literal app named
            # "the github".
            return None
        return ToolCallRequest(
            name="open_app", arguments={"name": app_name}
        )
    for pattern, tool, args in _RULES:
        if pattern.fullmatch(normalized):
            return ToolCallRequest(name=tool, arguments=dict(args))
    return None
