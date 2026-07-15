"""Safety / confirmation gate.

Policy (docs/ARCHITECTURE.md section 5):
  - SAFE runs immediately.
  - SENSITIVE asks once per session for each exact action, then remembers
    the approval for identical repeats.
  - DESTRUCTIVE always asks.

"Asking" is transport-dependent: interactive channels (WS chat, voice)
register a confirmer callback that shows the user the exact action verbatim
and awaits their answer. Channels that cannot ask (plain REST) get the
default confirmer, which denies anything requiring confirmation with an
explanatory message rather than silently proceeding.

`auto_approve=True` (JARVIS_AUTO_APPROVE) bypasses prompts for development
and tests — never the default.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.planner.schemas import RiskLevel

logger = logging.getLogger(__name__)

# Given a human-readable action description, return the user's yes/no.
Confirmer = Callable[["ConfirmationRequest"], Awaitable[bool]]


@dataclass(frozen=True)
class ConfirmationRequest:
    tool: str
    risk: RiskLevel
    # Exact action shown to the user — never a paraphrase.
    action: str


@dataclass
class GateDecision:
    allowed: bool
    reason: str = ""


async def _deny_by_default(request: ConfirmationRequest) -> bool:
    logger.warning(
        "No interactive confirmer for %s action %r — denying", request.risk.value, request.action
    )
    return False


class SafetyGate:
    def __init__(self, auto_approve: bool = False) -> None:
        self._auto_approve = auto_approve
        self._approved_sensitive: set[str] = set()

    async def check(
        self,
        request: ConfirmationRequest,
        confirmer: Confirmer | None = None,
    ) -> GateDecision:
        if request.risk is RiskLevel.SAFE:
            return GateDecision(allowed=True)
        if self._auto_approve:
            logger.info("auto_approve: allowing %s (%s)", request.action, request.risk.value)
            return GateDecision(allowed=True)

        remembered = request.risk is RiskLevel.SENSITIVE and request.action in (
            self._approved_sensitive
        )
        if remembered:
            return GateDecision(allowed=True, reason="approved earlier this session")

        ask = confirmer or _deny_by_default
        approved = await ask(request)
        if not approved:
            return GateDecision(
                allowed=False,
                reason=(
                    "The user declined this action."
                    if confirmer
                    else "This action needs confirmation, which this channel can't ask for. "
                    "Use the Jarvis app (or the WebSocket API) to approve it."
                ),
            )
        if request.risk is RiskLevel.SENSITIVE:
            self._approved_sensitive.add(request.action)
        return GateDecision(allowed=True)
