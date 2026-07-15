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
from typing import Any

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import Message, OllamaLike, ToolCallRequest
from app.core.safety import ConfirmationRequest, Confirmer, SafetyGate
from app.planner.fast_intents import match_fast_intent
from app.planner.schemas import PlanExecution, PlanStep, RiskLevel, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

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
  "open <website> in <browser>" -> open_url

Rules:
- NEVER say you did, played, opened, or checked something unless a tool \
result in this conversation confirms it. Claiming an action you did not take \
via a tool is a lie — do not do it.
- If no tool fits or the request is unclear, say so or ask; do not pretend.
- If a tool was denied or failed, tell the user honestly.
- Only pure conversation (greetings, questions about yourself) skips tools.
- Keep answers to one or two short sentences; they are often spoken aloud."""


def _tool_spec(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


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

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            _tool_spec(
                spec["name"], spec["description"], spec["args_schema"]
            )
            for spec in (tool.llm_spec() for tool in self._registry.list())
        ]

    async def run(
        self,
        utterance: str,
        history: list[Message],
        confirmer: Confirmer | None = None,
        max_steps: int = 5,
        memory_context: str | None = None,
    ) -> PlanExecution:
        """Execute the plan-act loop for one user turn."""
        execution = PlanExecution(utterance=utterance)

        # Deterministic fast-path for terse, unambiguous commands ("next",
        # "pause", …) that the small model handles unreliably. Still routed
        # through the tool + safety layer; only tool *selection* is skipped.
        fast_call = match_fast_intent(utterance)
        if fast_call is not None and self._registry.get(fast_call.name) is not None:
            step = await self._execute_tool_call(fast_call, confirmer)
            execution.steps.append(step)
            if step.result is not None:
                execution.reply = step.result.summary
                return execution
            # Registry/execution hiccup — fall through to the LLM planner.
            execution.steps.pop()

        system_prompt = PLANNER_PROMPT
        if memory_context:
            system_prompt = f"{PLANNER_PROMPT}\n\n{memory_context}"
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": utterance},
        ]
        model = await self._model_manager.ensure_llm()
        tool_specs = self._tool_specs()
        # Greedy decoding: tool selection must be deterministic, not sampled.
        options = {"temperature": self._settings.planner_temperature}

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

            if not turn.tool_calls:
                reply = turn.content.strip()
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
                step = await self._execute_tool_call(call, confirmer)
                execution.steps.append(step)
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
        action_text = f"{tool.name} {json.dumps(call.arguments, ensure_ascii=False)}"
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
