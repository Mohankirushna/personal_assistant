"""Typed contracts between the planner, tools, and the safety gate.

Tool-call proposals arrive via Ollama's native tool-calling API
(app.core.ollama_client.ToolCallRequest); everything here is validated
before execution — raw model output never reaches a subprocess or
filesystem call.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(enum.StrEnum):
    """How dangerous a tool call is; drives the confirmation policy."""

    SAFE = "safe"            # read-only or trivially reversible
    SENSITIVE = "sensitive"  # consequential but recoverable
    DESTRUCTIVE = "destructive"  # hard/impossible to reverse — always confirm


class ToolResult(BaseModel):
    """Structured outcome of one tool execution."""

    tool: str
    ok: bool
    # Human/model-readable summary of what happened; shown to the LLM.
    summary: str
    # Structured payload for programmatic consumers (UI, memory).
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def failure(cls, tool: str, message: str) -> ToolResult:
        return cls(tool=tool, ok=False, summary=message)


class PlanStep(BaseModel):
    """One executed step, kept for history/memory."""

    tool: str
    args: dict[str, Any]
    risk: RiskLevel
    result: ToolResult | None = None
    denied: bool = False


class PlanExecution(BaseModel):
    """Record of a full planner run for one user turn."""

    utterance: str
    steps: list[PlanStep] = Field(default_factory=list)
    reply: str = ""
