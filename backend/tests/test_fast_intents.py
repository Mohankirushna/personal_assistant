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
def test_live_information_questions_read_and_answer(utterance: str) -> None:
    # A bare question should read the web and answer, not just open a page.
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "web_answer"
    assert call.arguments == {"query": utterance.lower().rstrip("?")}


@pytest.mark.parametrize(
    ("utterance", "expected_tool"),
    [
        ("what is the price of iphone 15 in chrome", "browser_search"),
        ("what is the price of iphone 15 in brave", "brave_search_open_first"),
    ],
)
def test_question_with_explicit_browser_opens_a_page(utterance: str, expected_tool: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == expected_tool


@pytest.mark.parametrize(
    ("utterance", "project"),
    [
        ("where is the fitness project", "fitness"),
        ("where is the stocks project", "stocks"),  # 'stocks' must NOT web-search
        ("where is the stocks project located", "stocks"),
        ("give me the folder path for the fitness project", "fitness"),
        ("locate the jarvis repo", "jarvis"),
        ("whats the local path for skin", "skin"),
    ],
)
def test_project_location_questions_use_the_locate_tool(utterance: str, project: str) -> None:
    # A project-location question must reach locate_project, never web search —
    # even when the project name is a live-info trigger word like "stocks".
    call = match_fast_intent(utterance)
    assert call is not None, utterance
    assert call.name == "locate_project"
    assert call.arguments == {"project": project}


@pytest.mark.parametrize(
    "utterance",
    [
        "what is the stock price today",
        "whats the latest news",
        "what is the tesla stock price",
    ],
)
def test_stock_and_news_questions_still_web_search(utterance: str) -> None:
    # The locate-project fast-path must not steal genuine live-info questions.
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "web_answer"


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
def test_general_knowledge_questions_read_and_answer(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "web_answer"
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
    "utterance",
    ["good morning", "brief me", "morning briefing", "give me my morning brief",
     "whats my day look like", "run my brief"],
)
def test_matches_morning_briefing(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "morning_briefing"
    assert call.arguments == {}


@pytest.mark.parametrize(
    "utterance",
    # "check my email" = quick count/list; "read my mail" now summarizes.
    ["check my email", "check my inbox", "any new emails", "check email"],
)
def test_matches_check_email_commands(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "check_email"
    assert call.arguments == {}


@pytest.mark.parametrize(
    ("utterance", "recipient", "body"),
    [
        ("send an email to mohan kirushna saying meeting at 5", "mohan kirushna", "meeting at 5"),
        ("email mohan saying I will be late", "mohan", "i will be late"),
        ("send email to a@b.co with message hello there", "a@b.co", "hello there"),
    ],
)
def test_matches_send_email_commands(utterance: str, recipient: str, body: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "send_email"
    assert call.arguments == {"recipient": recipient, "body": body}


@pytest.mark.parametrize(
    "utterance",
    ["summarize my emails", "summarize my unread emails", "summarize my inbox",
     "what are my unread emails about", "sum up my mail",
     # General "recent mails" phrasings — must reach mail, not web search.
     "what are the recent mails", "recent emails", "show me my emails",
     "whats in my inbox", "do i have new mail", "list my latest emails"],
)
def test_matches_summarize_inbox(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "summarize_inbox"
    assert call.arguments == {}


@pytest.mark.parametrize(
    ("utterance", "sender"),
    [
        ("any mail from alice", "alice"),
        ("do i have any new emails from mohan kirushna", "mohan kirushna"),
        ("check for mail from the professor", "the professor"),
    ],
)
def test_matches_check_email_from_sender(utterance: str, sender: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "check_email"
    assert call.arguments == {"sender": sender}


@pytest.mark.parametrize(
    ("utterance", "sender"),
    [
        # "recent"/"latest" must NOT trip the web-search route for mail queries.
        ("what is the recent mail from noreply.cdcinfo@vitstudent.ac.in",
         "noreply.cdcinfo@vitstudent.ac.in"),
        ("read the latest email from alice", "alice"),
        ("show me mail from the professor", "the professor"),
        ("what are the emails from a.b@x.co", "a.b@x.co"),
    ],
)
def test_read_mail_from_sender_beats_web_search(utterance: str, sender: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "summarize_inbox"
    assert call.arguments == {"sender": sender}  # full email address preserved


@pytest.mark.parametrize(
    ("utterance", "sender"),
    [
        # Free-form phrasings the strict patterns miss must still reach mail,
        # not web search.
        ("what about mails from github was there any recent one", "github"),
        ("did i get an email from the professor", "professor"),
        ("anything from cdc in my inbox lately", "cdc"),
        ("were there emails from noreply.cdcinfo@vitstudent.ac.in",
         "noreply.cdcinfo@vitstudent.ac.in"),
    ],
)
def test_freeform_email_questions_reach_mail_not_web(utterance: str, sender: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "summarize_inbox"
    assert call.arguments == {"sender": sender}


def test_broad_mail_catchall_does_not_hijack_send_or_reply() -> None:
    # "mail from X" appears, but these are send/reply intents and must win.
    send = match_fast_intent("reply to the mail from alice saying got it")
    assert send is not None and send.name == "reply_email"


@pytest.mark.parametrize(
    ("utterance", "query"),
    [
        ("was there any mail about supabase", "supabase"),
        ("any mail about the placement drive", "placement drive"),
        ("did i get an email regarding my results", "results"),  # leading "my" stripped
    ],
)
def test_mail_about_topic_routes_to_keyword_search(utterance: str, query: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "summarize_inbox"
    assert call.arguments == {"query": query}


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("reply to the latest email saying I will be there", {"body": "i will be there"}),
        ("reply saying thanks", {"body": "thanks"}),
        ("reply to the last email from alice saying got it",
         {"sender": "alice", "body": "got it"}),
    ],
)
def test_matches_reply_email(utterance: str, expected: dict[str, object]) -> None:
    call = match_fast_intent(utterance)
    assert call is not None
    assert call.name == "reply_email"
    assert call.arguments == expected


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
    ("utterance", "day"),
    [
        ("show my calendar", "today"),
        ("check my calendar", "today"),
        ("what's my calendar", "today"),
        ("list my events", "today"),
        ("view my meetings", "today"),
        ("see my calendar for today", "today"),
        ("show my calendar for tomorrow", "tomorrow"),
        # Leaked to web search live: "for" must be optional before the day.
        ("whats my calendar today", "today"),
        ("check my calendar tomorrow", "tomorrow"),
        ("what's my next meeting", "today"),
        # Typed with a typo in the chat window — was web-searched to a
        # date-picker site. Misspellings must still reach the local tool.
        ("what's my calender today", "today"),
        ("whats my schedule today", "today"),
        ("show my schedule", "today"),
    ],
)
def test_calendar_commands_use_calendar_tool(utterance: str, day: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match"
    assert call.name == "calendar"
    assert call.arguments == {"day": day}


@pytest.mark.parametrize(
    ("utterance", "project"),
    [
        ("delete the fitness repo", "fitness"),
        ("delete fitness repo", "fitness"),
        ("remove the fitness repo", "fitness"),
        ("delete fitness from github", "fitness"),
        ("remove fitness from github", "fitness"),
        ("delete the github repo for fitness", "fitness"),
        ("delete repository for the fitness project", "fitness"),
        # "...repo in github" / "...project repo in github" tails — these did
        # NOT match before, so they fell to the LLM, which skipped the tool on
        # a repeated delete (saw a prior "Deleted" in history) and neither
        # asked nor deleted. They must route deterministically to the tool.
        ("delete fitness repo in github", "fitness"),
        ("delete the fitness project repo in github", "fitness"),
        ("delete fitness in github", "fitness"),
        ("delete fitness app repo in github", "fitness app"),
    ],
)
def test_delete_repo_commands_route_to_tool(utterance: str, project: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match delete pattern"
    assert call.name == "github_delete_repo"
    assert call.arguments == {"project": project}


@pytest.mark.parametrize(
    ("utterance", "project"),
    [
        ("open fitness project in github", "fitness"),
        ("open fitness in github", "fitness"),
        ("open fitness github", "fitness"),
        ("open fitness github repo", "fitness"),
        ("show me fitness on github", "fitness"),
        ("open the github repo for fitness", "fitness"),
        ("view jarvis on github", "jarvis"),
    ],
)
def test_open_repo_commands_route_to_tool(utterance: str, project: str) -> None:
    # The original bug: these were caught by the generic "open X" app
    # launcher, which tried (and failed) to open a macOS app literally named
    # "fitness project in github".
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match open-repo pattern"
    assert call.name == "github_open_repo"
    assert call.arguments == {"project": project}


@pytest.mark.parametrize(
    "utterance",
    [
        "open the github",
        "open github",  # handled separately by _OPEN_WEBSITE, not open_repo
        "delete the repo",
        "delete repo",
        "where is the project",
        "where is the repo",
    ],
)
def test_bare_reference_words_never_become_a_fake_project_name(utterance: str) -> None:
    # A bare "the"/"it" left over from optional-article backtracking must
    # never be treated as a literal project name (e.g. github_open_repo
    # called with project="the"). These must fall through — either to a
    # different fast-intent rule or to the full LLM planner, which has
    # conversation history to resolve what "the" or "it" refers to.
    call = match_fast_intent(utterance)
    if call is not None:
        assert call.arguments.get("project") not in {"the", "it", "my", "a", "an", "this", "that"}


@pytest.mark.parametrize(
    ("utterance", "project", "repo_name"),
    [
        ("push jarvis project to github", "jarvis", None),
        ("push jarvis to github", "jarvis", None),
        ("push the jarvis project to github", "jarvis", None),
        ("push fitness to github as fitnessapp", "fitness", "fitnessapp"),
        ("create a repo and push fitness project", "fitness", None),
        ("create repo and push fitness", "fitness", None),
        ("create a repo and push fitness to github", "fitness", None),
        ("create a repo and push fitness project as fitness-app", "fitness", "fitnessapp"),
        # "into"/"onto" prepositions — these dropped to the LLM before, which
        # then failed to recreate a deleted remote / push. Must route here.
        ("push fitness app into github", "fitness app", None),
        ("push fitness into github", "fitness", None),
        ("push fitness onto github", "fitness", None),
    ],
)
def test_push_repo_commands_route_to_github_push(
    utterance: str, project: str, repo_name: str | None
) -> None:
    # The original bug: this phrasing was ambiguous between github_push and
    # the low-level generic `git` tool (raw arguments + repo path); the small
    # planner picked `git` and failed to fill its required fields. This must
    # route deterministically to github_push instead.
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match push-repo pattern"
    assert call.name == "github_push"
    expected = {"project": project}
    if repo_name:
        expected["repo_name"] = repo_name
    assert call.arguments == expected


# Two phrasings shipped broken and were only caught live, each time falling
# through to the LLM planner (which re-searched the topic instead of reading
# the page the user had actually opened) rather than matching here: bare
# "read out loud" first, then bare "read out" (no "loud") after that. Both
# are pinned explicitly below so a future edit to the signal list can't
# silently drop either again.
@pytest.mark.parametrize(
    "utterance",
    [
        "read this out loud",
        "read that aloud",
        "read it to me",
        "read this news content outloud",  # normalization keeps "outloud" as one token
        "read this article to me",
        "read the article out loud",
        "read the page aloud",
        "read the story to me",
        "read this",
        "read that",
        "read it",
        "read out loud",  # bare, no object — caught live
        "read aloud",
        "read to me",
        "read out",  # bare, no "loud" — caught live
        "read this out",
        "read that out",
        "read the article out",
    ],
)
def test_read_aloud_phrasings_match_read_url_aloud(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match read_url_aloud"
    assert call.name == "read_url_aloud"
    assert call.arguments == {}  # url is resolved from session context by the planner


@pytest.mark.parametrize(
    "utterance",
    [
        "read my email",
        "read the news",
        "read out my emails",
        "read out my messages",
        "ready out",  # must not fuzzy-match "read out"
    ],
)
def test_read_aloud_does_not_swallow_unrelated_read_requests(utterance: str) -> None:
    call = match_fast_intent(utterance)
    assert call is None or call.name != "read_url_aloud", (
        f"{utterance!r} should not route to read_url_aloud"
    )


@pytest.mark.parametrize(
    "utterance",
    ["do it again", "say it again", "read it again", "repeat that", "repeat", "once more"],
)
def test_repeat_phrasings_match_the_repeat_sentinel(utterance: str) -> None:
    # The planner (not fast_intents) decides whether this actually becomes
    # another read_url_aloud call — see test_planner.py's
    # test_do_it_again_reruns_read_url_aloud_when_last_turn_spoke and
    # test_do_it_again_falls_through_when_last_turn_was_unrelated.
    call = match_fast_intent(utterance)
    assert call is not None, f"{utterance!r} should match the repeat sentinel"
    assert call.name == "repeat_last_speech"


@pytest.mark.parametrize(
    "utterance",
    [
        "push changes to github",
        "push my changes to github",
        "push it to github",
        "push everything to github",
    ],
)
def test_push_generic_phrases_do_not_fake_a_project_name(utterance: str) -> None:
    # "changes"/"it"/"everything" are not project names — must fall through
    # rather than routing to github_push with a nonsense project.
    call = match_fast_intent(utterance)
    if call is not None and call.name == "github_push":
        assert call.arguments.get("project") not in {
            "changes", "my changes", "the changes", "it", "everything", "all", "code",
        }
