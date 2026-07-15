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
    (re.compile(r"^(resume|unpause|continue)( (the )?(music|song|track|playback|it))?$"),
     "media_control", {"action": "play"}),
    (re.compile(r"^play( (the )?(music|song|track|it|something))?$"),
     "media_control", {"action": "play"}),
]


def match_fast_intent(utterance: str) -> ToolCallRequest | None:
    """Return a tool call for a recognized terse command, else None."""
    normalized = re.sub(r"[^\w\s]", "", utterance).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return None
    for pattern, tool, args in _RULES:
        if pattern.fullmatch(normalized):
            return ToolCallRequest(name=tool, arguments=dict(args))
    return None
