"""The Planner: iterative plan-act loop over Ollama's native tool-calling.

Each turn the model sees the conversation, the tool catalog (passed through
Ollama's `tools` API so the model's own trained function-call template is
used — hand-rolled JSON protocols measurably break down on 3B models), and
prior tool results. It either proposes tool calls or answers in text.

Proposals are only ever *proposals*: arguments are validated against the
tool's Pydantic schema and the call passes the SafetyGate before anything
executes (docs/ARCHITECTURE.md section 3).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any
from urllib.parse import unquote

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import Message, OllamaLike, ToolCallRequest
from app.core.safety import ConfirmationRequest, Confirmer, SafetyGate
from app.planner.fast_intents import match_fast_intent
from app.planner.schemas import PlanExecution, PlanStep, RiskLevel, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Observes tool execution for live UI feedback: called with (tool_name,
# status), status one of "running" | "ok" | "failed" | "denied". Observers
# are best-effort — a failure (e.g. the UI's socket died) never aborts the
# plan itself.
StepObserver = Callable[[str, str], Awaitable[None]]

PLANNER_PROMPT = """\
You are Jarvis, a macOS desktop assistant. You control the computer ONLY \
through the provided tools; a separate system validates and executes them.

You CANNOT do anything yourself — every real action or fact requires a tool \
call. This includes: the date/time, files and folders, clipboard, running \
apps, system state (volume, screen), CONTROLLING MUSIC (play, pause, next, \
skip, previous), OPENING APPS OR WEBSITES, running commands, and web content.

Short commands are actions, not chit-chat. Map them to a tool call:
  "next" / "skip"        -> media_control(action="next")
  "pause" / "stop"       -> media_control(action="pause")
  "play" / "resume"      -> media_control(action="play")
  "open youtube"         -> open_url(target="youtube")
  "open Claude"          -> open_app(name="Claude")
  "open/launch <app>"    -> open_app(name=<app>)
  "battery percentage"   -> battery_status
  "what time is it" / "today's date" -> clock
  "what's in my Downloads folder" / "check my downloads" -> \
finder_list(path="~/Downloads")
  "find my <name> files" -> finder_search
  "restart/reboot Mac"   -> system_power(action="restart")
  "shut down Mac"        -> system_power(action="shutdown")
  "turn the volume up/down" -> volume(direction="up"/"down")
  "set volume to <N>"    -> volume(level=<N>)
  "turn the brightness up/down" -> brightness(direction="up"/"down")
  "set brightness to <N>%" -> brightness(level=<N/100>)
  "remind me about <X> on <date>" -> create_reminder(title=<X>, due_at=<date>)
  "set a timer for <N> minutes" -> timer(minutes=<N>)
  "set a <N>-minute timer for <label>" -> timer(minutes=<N>, label=<label>)
  "turn on do not disturb" / "enable focus mode" -> focus_mode(action="on")
  "turn off focus mode" -> focus_mode(action="off")
  "what's my next meeting" / "show my calendar" -> calendar(day="today")
  "what's on my calendar tomorrow" -> calendar(day="tomorrow")
  "open <website> in <browser>" -> open_url
  "what is/who is/how ... <question>" -> web_answer(query=<question>)  # read + answer
  "search <topic>" -> brave_search_open_first(query=<topic>)  # opens a page
  "search <topic> in <browser>" -> browser_search(query=<topic>, browser=<browser>)
  "search Wikipedia for <topic>" -> browser_search(query=<topic>, engine="wikipedia")
  "open YouTube and play <song>" -> youtube_play(query=<song>)
  "play <song> in Spotify" -> spotify_play(query=<song>)
  "open <name> playlist in Spotify" -> spotify_open_playlist(playlist=<name>)
  "send <message> on WhatsApp to <name or number>" -> \
whatsapp_send(recipient=<contact name or number>, message=<message>)
  "good morning" / "brief me" / "what's my day like" -> morning_briefing
  "check my email" -> check_email
  "any mail from <name>" -> check_email(sender=<name>)
  "summarize my emails" -> summarize_inbox
  "send an email to <name> saying <text>" -> \
send_email(recipient=<contact name or address>, body=<text>)
  "reply to the latest email saying <text>" -> reply_email(body=<text>)

When the user gives only a brief ("email X asking them to join Monday, make
it professional"), YOU compose the full message body in the requested tone
(a complete, well-formatted letter with greeting and sign-off) and pass it
as send_email's `body`. Do not send a one-line fragment. The system shows
the user the drafted email and asks them to confirm before it is sent, so
always go ahead and call send_email once you have a recipient and a brief.
  "list connected Bluetooth devices" -> list_bluetooth_devices
  "play some Tamil songs" / "I want new Tamil songs" -> music_platform_prompt
  "recent news about <topic>" -> news_search(query=<topic>)

For reminders, only two details are required: what to remind the user about
and when. Do not ask for urgency, a category, or whether it is an event or a
task. If either required detail is missing, ask only for that detail. When
both are present, call create_reminder. Its due_at can be a natural phrase
such as 'tomorrow at 10 AM' or an ISO 8601 local time.

Conversation follow-ups refer to the immediately preceding exchange. "yes"
means the user accepts the last question, not a new request. If the last
question still needs a missing detail (for example, a Wikipedia topic), say
exactly which detail is needed. "continue the previous request" means resume
that request; it is never a music command unless the user explicitly mentions
music, a song, or playback.

For compound commands, keep going until every requested step has either
completed or failed. Opening YouTube is NOT the same as finding and playing
a requested song. Never say a song is playing unless youtube_play reports it.
Never use youtube_play for a news, web-search, article, or research request
unless the user explicitly asks for YouTube.

A bare topic or name with no verb ("amazon forest", "ironman", "the eiffel
tower") is a request to look it up and show it, not small talk — use
brave_search_open_first, not web_search. web_search only returns text and
opens nothing; reserve it for when the user explicitly wants a written
summary/list, not as the default for an ordinary lookup.
Any question about the user's own EMAIL — "mail from X", "did I get an email
from Y", "what's in my inbox", "any recent mail from Z" — is about their Mail
app, NEVER a web search. Use summarize_inbox (optionally sender=<name>) or
check_email(sender=<name>), never web_search or brave_search_open_first.
When the user asks to play music but does not name a platform, ask whether
they want YouTube, Spotify, or Apple Music. Do not choose a platform yourself.

File operations: "list/show files", "create folder", "delete file" → use
finder_* tools (faster, no shell dependency). Use terminal_run only for
complex shell logic or piped commands (grepping, parsing, chaining).

LOCAL vs WEB: anything about THIS Mac — the user's files, folders,
Downloads, Documents, Desktop, battery, running apps, volume, screen,
clipboard, or the current time or date — is answered with local tools
(clock, finder_list, finder_search, battery_status, list_running_apps).
NEVER use a web or browser search for it: the internet does not know what
is on this Mac or what time it is here. Web search is only for information
that lives on the internet (facts, news, weather, prices, places).

Rules:
- NEVER say you did, played, opened, or checked something unless a tool \
result in this conversation confirms it. Claiming an action you did not take \
via a tool is a lie — do not do it.
- If no tool fits or the request is unclear, say so or ask; do not pretend.
- If a tool was denied or failed, tell the user honestly.
- Only pure conversation (greetings, questions about yourself) skips tools.
- Answers are SPOKEN ALOUD: one or two short sentences of plain text. Never \
use Markdown, links, image syntax, or URL-encoding. Mention file paths \
exactly as the tool reported them."""


def _tool_spec(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


# Fast-path tools that return raw content to read rather than a finished
# reply — their output is handed to the model to synthesize a spoken answer.
_ANSWER_FROM_RESULT_TOOLS = {"web_answer", "summarize_inbox", "read_url_aloud"}

# read_url_aloud's raw fetch is a full page's worth of text — including site
# boilerplate (subscription banners, nav, bylines, "read later" widgets) that
# a real news site's markup doesn't cleanly separate from the article. A
# fixed "answer my question in 1-2 sentences" instruction (the default below)
# is wrong for it: there's no question, and 1-2 sentences discards the
# article entirely. It gets its own instruction, prompting the model to
# narrate the actual content and drop everything that isn't part of it.
_READ_ALOUD_SUMMARY_INSTRUCTION = (
    "The user asked to have this page read aloud. Speak the actual article content "
    "in your own words, as a natural spoken narration of a few sentences to a short "
    "paragraph — cover the real substance, not just a one-line gist. Skip site "
    "navigation, subscription/login prompts, ads, bylines, share buttons, and any "
    "other boilerplate that isn't part of the article itself. Do not call any more tools."
)
_DEFAULT_ANSWER_INSTRUCTION = (
    "Answer my question using the information above, in one or two spoken sentences. "
    "If it isn't there, say you couldn't find it. Do not call any more tools."
)

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# Names the small model reaches for that are not in the real tool catalog. A
# 3B model sometimes writes a tool call as plain JSON text instead of using
# the tool-calling API — usually inventing a generic name for "search the
# web" rather than picking the actual registered tool. Recover the intent
# instead of reading raw JSON aloud to the user.
_FAKE_TOOL_NAME_ALIASES = {
    "web_search": "brave_search_open_first",
    "websearch": "brave_search_open_first",
    "search_web": "brave_search_open_first",
    "internet_search": "brave_search_open_first",
    "google_search": "brave_search_open_first",
    "search": "brave_search_open_first",
}
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _parse_fake_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    """If the model wrote `{"name": ..., "arguments": {...}}` as plain text
    instead of a real tool call, extract (name, arguments). Returns None for
    ordinary text, which should be spoken as-is."""
    text = _CODE_FENCE.sub("", content).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name, arguments = payload.get("name"), payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


# Words that only ever describe a *reference* to something Jarvis already
# found/said, never new literal content someone wants sent verbatim. A
# message built entirely out of these ("this website link", "the url",
# "that page") is a reference; matching whole words this way — rather than
# a fixed phrase list — generalizes to variations without enumerating them.
_REFERENCE_WORDS = {
    "this", "that", "it", "the",
    "link", "url", "website", "web", "site", "page", "webpage",
    "result", "results", "answer", "article", "info", "information",
}
_STOP_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "and", "or", "is", "are", "was",
    "were", "for", "me", "please", "you", "that", "this", "it", "what", "send",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _refers_to_last_content(message: str, last_query: str | None) -> bool:
    """True if `message` (the text a fast-matched whatsapp_send would send)
    is a reference to something Jarvis already found/said ("this", "the
    website link", "that page") or clearly restates the topic of the last
    search/browse query ("the cricket score" after searching "cricket score
    yesterday"), rather than new literal content the user actually wants
    sent verbatim (e.g. "hello")."""
    words = re.findall(r"[a-z0-9]+", message.strip().lower())
    if words and all(w in _REFERENCE_WORDS for w in words):
        return True
    if not last_query:
        return False
    message_words = _significant_words(message)
    query_words = _significant_words(last_query)
    return bool(message_words) and bool(message_words & query_words)


def _last_turn_read_aloud(history: list[Message]) -> bool:
    """True if the most recent assistant turn actually ran read_url_aloud.

    ChatService appends a compact tool-outcome trace ("[tool: summary]") to
    every assistant history entry (see _update_shareable_content's docstring
    in chat_service.py) — the same mechanism follow-ups like "open the
    screenshot you just took" rely on for concrete detail. Reused here so
    "do it again" only re-triggers a read-aloud when that's actually what
    just happened, not after e.g. "turn up the volume".
    """
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        return "[read_url_aloud:" in str(message.get("content", ""))
    return False


def sanitize_spoken_reply(text: str) -> str:
    """Make a reply speakable: replies are read aloud by TTS, but small
    models emit Markdown anyway (notably image syntax with URL-encoded
    screenshot paths) regardless of prompt instructions. Deterministically
    rewrite link/image syntax to plain text and drop emphasis markers."""
    text = _MD_IMAGE.sub(lambda m: unquote(m.group(1)), text)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _has_unrelated_denials(steps: list[PlanStep], threshold: int = 2) -> bool:
    """Detect when the model proposes unrelated actions after denials.
    This catches hallucinations like: original request is 'create a folder',
    model tries mkdir (denied), then proposes unrelated system_power calls."""
    denied_steps = [s for s in steps if s.denied and s.result is not None]
    if len(denied_steps) < threshold:
        return False

    recent_denied = denied_steps[-threshold:]
    tools_used = {s.tool for s in recent_denied}
    return len(tools_used) == 1  # repeated same tool despite denials is ok-ish


def _is_hallucinating_unrelated_actions(
    steps: list[PlanStep], utterance: str
) -> bool:
    """Detect hallucination: model produces repeated failed denials of
    unrelated actions. E.g., request for 'create folder' gets 1 terminal_run
    attempt (denied), then 2+ system_power shutdown attempts (all denied)."""
    if len(steps) < 2:
        return False

    # If the last 2+ steps are both denied, and they're a different tool
    # from the first successful attempt, the model is hallucinating.
    recent_denied = [s for s in steps[-3:] if s.denied]
    if len(recent_denied) < 2:
        return False

    # All recent denials are the same unrelated tool = hallucination.
    recent_tools = [s.tool for s in recent_denied]
    if len(set(recent_tools)) == 1:
        logger.warning(
            f"Hallucination detected: {len(recent_denied)} repeated denials "
            f"of {recent_tools[0]} for '{utterance[:60]}'"
        )
        return True

    return False


class Planner:
    def __init__(
        self,
        client: OllamaLike,
        model_manager: ModelManager,
        registry: ToolRegistry,
        gate: SafetyGate,
        settings: Settings,
    ) -> None:
        self._client = client
        self._model_manager = model_manager
        self._registry = registry
        self._gate = gate
        self._settings = settings

    def _tool_specs(self, allowed_names: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            _tool_spec(
                spec["name"], spec["description"], spec["args_schema"]
            )
            for spec in (
                tool.llm_spec()
                for tool in self._registry.list()
                if allowed_names is None or tool.name in allowed_names
            )
        ]

    def _prune_tools(
        self, utterance: str, max_tools: int = 18
    ) -> set[str] | None:
        """Per-request tool catalog pruning via keyword matching. Returns a
        set of tool names to include, or None (include all) if pruning would
        lose too much.

        Motivation: the full 36-tool catalog is ~4.2K tokens before the
        planner prompt (~1.3K), which overflows Ollama's default 4K context.
        Pruning to ~18 tools (top matches only) cuts tool spec overhead
        in half while keeping the model's view focused on what matters.
        Fallback: if no tools match, or if matching yields < 3 tools (too
        restrictive), return None (send all tools).
        """
        # Tokenize the utterance: lowercase, split on whitespace and punctuation.
        normalized = utterance.lower()
        tokens = {
            w
            for w in re.split(r"[\s\-_.,:;!?()\"]+", normalized)
            if w and len(w) > 2  # skip common noise (a, an, to, by, etc.)
        }
        if not tokens:
            return None  # empty utterance; send all tools

        # Score each tool: match against name, description, and keyword synonyms.
        scores: dict[str, float] = {}
        for tool in self._registry.list():
            score = 0.0
            name_tokens = set(re.split(r"[\s\-_.]+", tool.name.lower()))
            desc_tokens = set(re.split(r"[\s\-_.,:;!?()\"]+", tool.description.lower()))

            # Exact name match is highest priority.
            if tool.name.lower() in tokens or tokens & name_tokens:
                score += 100

            # Keywords in the description get a boost.
            if tokens & desc_tokens:
                score += 10

            # Keyword prefix matches (e.g., "find" matches "finder_*").
            for token in tokens:
                if tool.name.lower().startswith(token):
                    score += 50

            scores[tool.name] = score

        # Filter to scored tools; fallback if too few.
        scored = sorted(
            (name for name, score in scores.items() if score > 0),
            key=lambda name: scores[name],
            reverse=True,
        )
        if len(scored) < 2:
            # Pruning was too aggressive; send all tools.
            return None

        # Return top N scored tools (or all if < max_tools qualify).
        return set(scored[:max_tools])

    @staticmethod
    async def _notify(on_step: StepObserver | None, tool: str, status: str) -> None:
        if on_step is None:
            return
        try:
            await on_step(tool, status)
        except Exception:  # noqa: BLE001 - observers must never abort the plan
            logger.debug("Step observer failed for %s/%s", tool, status, exc_info=True)

    @staticmethod
    def _step_status(step: PlanStep) -> str:
        if step.denied:
            return "denied"
        if step.result is not None and step.result.ok:
            return "ok"
        return "failed"

    async def run(
        self,
        utterance: str,
        history: list[Message],
        confirmer: Confirmer | None = None,
        max_steps: int = 5,
        memory_context: str | None = None,
        last_query: str | None = None,
        last_url: str | None = None,
        last_text: str | None = None,
        on_step: StepObserver | None = None,
    ) -> PlanExecution:
        """Execute the plan-act loop for one user turn."""
        execution = PlanExecution(utterance=utterance)

        # A reminder must never become an unrelated system action. In addition
        # to teaching the model how reminders work, narrow its available
        # capability to the one relevant tool for this kind of request.
        reminder_request = bool(re.search(r"\bremind(?:er)?\b", utterance, re.IGNORECASE))
        allowed_tools = {"create_reminder"} if reminder_request else None

        # Deterministic fast-path for terse, unambiguous commands ("next",
        # "pause", …) that the small model handles unreliably. Still routed
        # through the tool + safety layer; only tool *selection* is skipped.
        fast_call = match_fast_intent(utterance)
        if fast_call is not None and fast_call.name == "whatsapp_send":
            message = fast_call.arguments.get("message", "")
            if _refers_to_last_content(str(message), last_query):
                content = last_url or last_text
                if content is None:
                    execution.reply = (
                        "I don't have anything recent to send — ask me something first, "
                        "then say 'send that to <name>'."
                    )
                    return execution
                fast_call = ToolCallRequest(
                    name="whatsapp_send",
                    arguments={**fast_call.arguments, "message": content},
                )
        if fast_call is not None and fast_call.name == "repeat_last_speech":
            # "do it again" is only unambiguous when the last turn actually
            # spoke something; otherwise we don't know what "it" refers to
            # and the ordinary LLM planner (with the same history) is the
            # right fallback, same as any other unmatched utterance. When it
            # does apply, re-resolving through the read_url_aloud branch
            # below (rather than reusing the old reply) means "do it again"
            # gets a fresh fetch + summary, not the same cached text.
            fast_call = (
                ToolCallRequest(name="read_url_aloud", arguments={})
                if _last_turn_read_aloud(history)
                else None
            )
        if fast_call is not None and fast_call.name == "read_url_aloud":
            # The tool itself prefers Brave's actual current tab (the user
            # may have clicked through since Jarvis last opened something);
            # last_url is only the fallback for when Brave can't be queried.
            fast_call = ToolCallRequest(name="read_url_aloud", arguments={"url": last_url})
        # A tool whose result is raw content (web page text), not a finished
        # reply, is run deterministically here but its output is then handed
        # to the model to synthesize a spoken answer, rather than read aloud.
        prefetched: PlanStep | None = None
        if fast_call is not None and self._registry.get(fast_call.name) is not None:
            await self._notify(on_step, fast_call.name, "running")
            step = await self._execute_tool_call(fast_call, confirmer)
            await self._notify(on_step, fast_call.name, self._step_status(step))
            execution.steps.append(step)
            if step.result is not None:
                if fast_call.name in _ANSWER_FROM_RESULT_TOOLS and step.result.ok:
                    prefetched = step  # summarized by the model below
                else:
                    execution.reply = step.result.summary
                    return execution
            else:
                # Registry/execution hiccup — fall through to the LLM planner.
                execution.steps.pop()

        system_prompt = PLANNER_PROMPT
        if self._settings.user_full_name:
            system_prompt += (
                f"\n\nThe user's name is {self._settings.user_full_name}. When you draft "
                "an email, sign it with this name — never leave a '[Your Name]' placeholder."
            )
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": utterance},
        ]
        if prefetched is not None and prefetched.result is not None:
            # Seed the conversation with the deterministic fetch so the model
            # answers from real content instead of trying to search itself.
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": prefetched.tool, "arguments": prefetched.args}}
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": prefetched.result.summary,
                    "tool_name": prefetched.tool,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        _READ_ALOUD_SUMMARY_INSTRUCTION
                        if prefetched.tool == "read_url_aloud"
                        else _DEFAULT_ANSWER_INSTRUCTION
                    ),
                }
            )
        model = await self._model_manager.ensure_llm()
        # Prune tool catalog per request to stay within context budget. The full
        # 36-tool catalog is ~4.2K tokens, which overflows the default 4K context
        # when paired with the planner prompt (~1.3K). Pruning to top-scoring
        # tools (matched against the utterance) halves the overhead.
        pruned_tools = self._prune_tools(utterance)
        pruned_allowed = (
            allowed_tools & pruned_tools if pruned_tools and allowed_tools else
            pruned_tools or allowed_tools
        )
        tool_specs = self._tool_specs(pruned_allowed)
        # Greedy decoding: tool selection must be deterministic, not sampled.
        options = {
            "temperature": self._settings.planner_temperature,
            "num_ctx": self._settings.llm_context_size,
        }

        for _step in range(max_steps):
            turn = await self._client.chat_turn(
                model=model,
                messages=messages,
                keep_alive=self._settings.llm_keep_alive,
                tools=tool_specs,
                options=options,
            )

            if not turn.tool_calls and not turn.content.strip():
                # Small models occasionally emit an entirely empty turn on
                # complex requests; nudge once before giving up honestly.
                logger.info("Model returned an empty turn; retrying with a nudge")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was empty. Either call the first tool "
                            "needed for my request, or answer in text."
                        ),
                    }
                )
                turn = await self._client.chat_turn(
                    model=model,
                    messages=messages,
                    keep_alive=self._settings.llm_keep_alive,
                    tools=tool_specs,
                    options=options,
                )

            if not turn.tool_calls and turn.content.strip():
                parsed = _parse_fake_tool_call(turn.content)
                if parsed is not None:
                    name, arguments = parsed
                    resolved = _FAKE_TOOL_NAME_ALIASES.get(name, name)
                    if self._registry.get(resolved) is not None:
                        fake_call = ToolCallRequest(name=resolved, arguments=arguments)
                        turn = replace(turn, content="", tool_calls=[fake_call])
                    else:
                        # Never speak raw JSON, even when the intended tool
                        # can't be identified.
                        turn = replace(turn, content="")

            if not turn.tool_calls:
                reply = sanitize_spoken_reply(turn.content)
                if not reply:
                    # Never fabricate success ("Done.") without tool evidence.
                    reply = (
                        "I wasn't able to work out how to do that. "
                        "Could you rephrase, or break it into smaller steps?"
                    )
                execution.reply = reply
                return execution

            # Record the assistant turn in the native format so the model
            # sees its own calls on the next iteration.
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        }
                        for call in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                if allowed_tools is not None and call.name not in allowed_tools:
                    step = PlanStep(
                        tool=call.name,
                        args=call.arguments,
                        risk=RiskLevel.SAFE,
                        result=ToolResult.failure(
                            call.name,
                            "A reminder request may only create a reminder; "
                            "no other action was run.",
                        ),
                    )
                else:
                    await self._notify(on_step, call.name, "running")
                    step = await self._execute_tool_call(call, confirmer)
                    await self._notify(on_step, call.name, self._step_status(step))
                # A small model occasionally reissues an identical successful
                # call instead of recognizing the task is already done (e.g.
                # setting volume to the same level three times), looping
                # until the step cap. Stop the moment it repeats itself
                # rather than re-running the action or burning more steps.
                is_repeat = (
                    step.result is not None
                    and step.result.ok
                    and any(
                        prior.tool == step.tool
                        and prior.args == step.args
                        and prior.result is not None
                        and prior.result.ok
                        for prior in execution.steps
                    )
                )
                execution.steps.append(step)
                if is_repeat and step.result is not None:
                    # The model is stuck reissuing a done call instead of
                    # answering. The repeated tool is not necessarily the one
                    # that answered the question (seen live: clock answered,
                    # then the model wandered into volume twice — and the old
                    # "reply with the repeated summary" said "Volume is 50%"
                    # to a time question). Force one final TOOL-FREE turn so
                    # the reply addresses the original request.
                    messages.append(
                        {
                            "role": "tool",
                            "content": step.result.summary,
                            "tool_name": call.name,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That action already ran; its result is above. "
                                "Using the tool results in this conversation, answer "
                                "my original request now in one or two short "
                                "sentences. Do not request any more tools."
                            ),
                        }
                    )
                    final = await self._client.chat_turn(
                        model=model,
                        messages=messages,
                        keep_alive=self._settings.llm_keep_alive,
                        options=options,
                    )
                    execution.reply = (
                        sanitize_spoken_reply(final.content) or step.result.summary
                    )
                    return execution
                if _is_hallucinating_unrelated_actions(execution.steps, utterance):
                    # Clear failed steps so the reply is honest ("couldn't" not
                    # "tried and failed"). This prevents fabricated success claims.
                    execution.steps.clear()
                    execution.reply = (
                        "I wasn't able to work out how to do that. "
                        "Could you rephrase, or break it into smaller steps?"
                    )
                    return execution
                if step.result is None:
                    outcome = "no result"
                elif step.denied:
                    outcome = f"DENIED: {step.result.summary}"
                else:
                    outcome = step.result.summary
                messages.append(
                    {"role": "tool", "content": outcome, "tool_name": call.name}
                )

        execution.reply = (
            "I hit my step limit before finishing — here's where things stand: "
            + "; ".join(
                f"{step.tool}: {step.result.summary if step.result else 'no result'}"
                for step in execution.steps[-3:]
            )
        )
        return execution

    async def _execute_tool_call(
        self, call: ToolCallRequest, confirmer: Confirmer | None
    ) -> PlanStep:
        tool = self._registry.get(call.name)
        if tool is None:
            return PlanStep(
                tool=call.name,
                args=call.arguments,
                risk=RiskLevel.SAFE,
                result=ToolResult.failure(
                    call.name,
                    f"unknown tool '{call.name}'; available: "
                    + ", ".join(t.name for t in self._registry.list()),
                ),
            )

        parsed = tool.parse_args(call.arguments)
        risk = tool.assess_risk(parsed) if parsed is not None else tool.risk_level
        preview = tool.confirmation_action(parsed) if parsed is not None else None
        action_text = preview or f"{tool.name} {json.dumps(call.arguments, ensure_ascii=False)}"
        gate_decision = await self._gate.check(
            ConfirmationRequest(tool=tool.name, risk=risk, action=action_text),
            confirmer=confirmer,
        )
        if not gate_decision.allowed:
            return PlanStep(
                tool=tool.name,
                args=call.arguments,
                risk=risk,
                denied=True,
                result=ToolResult.failure(tool.name, gate_decision.reason),
            )

        result = await tool.execute(call.arguments)
        return PlanStep(tool=tool.name, args=call.arguments, risk=risk, result=result)
