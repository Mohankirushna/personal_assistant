"""Planner loop, safety gate, and confirmation flow with scripted decisions."""

from __future__ import annotations

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import ChatTurn, ToolCallRequest
from app.core.safety import ConfirmationRequest, SafetyGate
from app.main import create_app
from app.planner.planner import Planner
from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool
from app.tools.registry import ToolRegistry
from tests.conftest import FakeOllamaClient


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo text back."
    args_model: ClassVar[type[BaseModel]] = EchoArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self) -> None:
        self.executions: list[str] = []

    async def run(self, args: EchoArgs) -> ToolResult:  # type: ignore[override]
        self.executions.append(args.text)
        return ToolResult(tool=self.name, ok=True, summary=f"echoed {args.text!r}")


class WipeArgs(BaseModel):
    path: str


class WipeTool(Tool):
    name: ClassVar[str] = "wipe"
    description: ClassVar[str] = "Delete things (test double)."
    args_model: ClassVar[type[BaseModel]] = WipeArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.DESTRUCTIVE

    def __init__(self) -> None:
        self.executions: list[str] = []

    async def run(self, args: WipeArgs) -> ToolResult:  # type: ignore[override]
        self.executions.append(args.path)
        return ToolResult(tool=self.name, ok=True, summary=f"wiped {args.path}")


def tool_call(tool: str, **args: object) -> ChatTurn:
    return ChatTurn(tool_calls=[ToolCallRequest(name=tool, arguments=dict(args))])


def respond(text: str) -> ChatTurn:
    return ChatTurn(content=text)


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def wipe_tool() -> WipeTool:
    return WipeTool()


@pytest.fixture
def registry(echo_tool: EchoTool, wipe_tool: WipeTool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(echo_tool)
    reg.register(wipe_tool)
    return reg


@pytest.fixture
def planner(
    settings: Settings, fake_ollama: FakeOllamaClient, registry: ToolRegistry
) -> Planner:
    manager = ModelManager(fake_ollama, settings)
    return Planner(fake_ollama, manager, registry, SafetyGate(), settings)


async def test_tool_then_respond(
    planner: Planner, fake_ollama: FakeOllamaClient, echo_tool: EchoTool
) -> None:
    fake_ollama.queued_turns = [tool_call("echo", text="hi"), respond("Echoed for you.")]
    execution = await planner.run("please echo hi", history=[])
    assert echo_tool.executions == ["hi"]
    assert execution.reply == "Echoed for you."
    assert len(execution.steps) == 1
    assert execution.steps[0].result is not None and execution.steps[0].result.ok


async def test_direct_respond_needs_no_tools(
    planner: Planner, fake_ollama: FakeOllamaClient, echo_tool: EchoTool
) -> None:
    fake_ollama.queued_turns = [respond("Hello there!")]
    execution = await planner.run("hello", history=[])
    assert execution.reply == "Hello there!"
    assert execution.steps == []
    assert echo_tool.executions == []


async def test_empty_turn_retries_with_nudge(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    """One empty turn -> nudge retry; the second (default) reply is used."""
    fake_ollama.queued_turns = [ChatTurn()]
    execution = await planner.run("hello", history=[])
    assert execution.reply == fake_ollama.reply
    # The nudge message was actually sent on the retry.
    assert any(
        "empty" in message.get("content", "")
        for message in fake_ollama.chat_messages[-1]
        if message["role"] == "user"
    )


async def test_two_empty_turns_admit_inability(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    """Persistent empty turns produce an honest reply, never a fake 'done'."""
    fake_ollama.queued_turns = [ChatTurn(), ChatTurn()]
    execution = await planner.run("hello", history=[])
    assert "wasn't able" in execution.reply


async def test_unknown_tool_feeds_back_and_recovers(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.queued_turns = [tool_call("teleport", where="mars"), respond("No such ability.")]
    execution = await planner.run("teleport me", history=[])
    assert execution.reply == "No such ability."
    assert execution.steps[0].result is not None
    assert "unknown tool" in execution.steps[0].result.summary


async def test_destructive_denied_without_confirmer(
    planner: Planner, fake_ollama: FakeOllamaClient, wipe_tool: WipeTool
) -> None:
    fake_ollama.queued_turns = [tool_call("wipe", path="/tmp/x"), respond("I couldn't do that.")]
    execution = await planner.run("wipe /tmp/x", history=[])
    assert wipe_tool.executions == []  # never executed
    assert execution.steps[0].denied is True


async def test_destructive_runs_with_approval(
    planner: Planner, fake_ollama: FakeOllamaClient, wipe_tool: WipeTool
) -> None:
    seen: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> bool:
        seen.append(request)
        return True

    fake_ollama.queued_turns = [tool_call("wipe", path="/tmp/x"), respond("Done.")]
    execution = await planner.run("wipe /tmp/x", history=[], confirmer=approve)
    assert wipe_tool.executions == ["/tmp/x"]
    assert execution.reply == "Done."
    # The user saw the exact action, not a paraphrase.
    assert seen[0].action == 'wipe {"path": "/tmp/x"}'
    assert seen[0].risk is RiskLevel.DESTRUCTIVE


async def test_step_cap(planner: Planner, fake_ollama: FakeOllamaClient) -> None:
    fake_ollama.queued_turns = [tool_call("echo", text=f"{i}") for i in range(10)]
    execution = await planner.run("echo forever", history=[], max_steps=3)
    assert len(execution.steps) == 3
    assert "step limit" in execution.reply


async def test_invalid_args_fail_soft(
    planner: Planner, fake_ollama: FakeOllamaClient, echo_tool: EchoTool
) -> None:
    fake_ollama.queued_turns = [
        ChatTurn(tool_calls=[ToolCallRequest(name="echo", arguments={"wrong": 1})]),
        respond("Bad args."),
    ]
    execution = await planner.run("echo", history=[])
    assert echo_tool.executions == []
    assert execution.steps[0].result is not None
    assert "invalid arguments" in execution.steps[0].result.summary


def test_ws_confirmation_roundtrip(
    settings: Settings, registry: ToolRegistry, wipe_tool: WipeTool
) -> None:
    """Full transport test: WS chat asks, client approves, tool runs."""
    fake = FakeOllamaClient()
    fake.queued_turns = [tool_call("wipe", path="/tmp/y"), respond("Wiped it.")]
    app = create_app(settings=settings, ollama_client=fake, registry=registry)
    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "wipe /tmp/y"})
        event = ws.receive_json()
        assert event["type"] == "confirm_request"
        assert event["risk"] == "destructive"
        assert event["action"] == 'wipe {"path": "/tmp/y"}'
        ws.send_json({"type": "confirm_response", "approved": True})
        assert ws.receive_json()["type"] == "token"
        done = ws.receive_json()
        assert done["type"] == "done"
        assert done["reply"] == "Wiped it."
    assert wipe_tool.executions == ["/tmp/y"]


def test_ws_confirmation_denied(
    settings: Settings, registry: ToolRegistry, wipe_tool: WipeTool
) -> None:
    fake = FakeOllamaClient()
    fake.queued_turns = [tool_call("wipe", path="/tmp/z"), respond("Okay, I won't.")]
    app = create_app(settings=settings, ollama_client=fake, registry=registry)
    with TestClient(app) as client, client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "wipe /tmp/z"})
        assert ws.receive_json()["type"] == "confirm_request"
        ws.send_json({"type": "confirm_response", "approved": False})
        ws.receive_json()  # token
        assert ws.receive_json()["reply"] == "Okay, I won't."
    assert wipe_tool.executions == []


def test_rest_chat_denies_destructive(
    settings: Settings, registry: ToolRegistry, wipe_tool: WipeTool
) -> None:
    """Plain REST has no way to ask, so destructive calls are denied."""
    fake = FakeOllamaClient()
    fake.queued_turns = [tool_call("wipe", path="/tmp/q"), respond("That needs the app.")]
    app = create_app(settings=settings, ollama_client=fake, registry=registry)
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "wipe /tmp/q"})
        assert response.status_code == 200
    assert wipe_tool.executions == []


def test_tools_endpoint_lists_registered(settings: Settings, registry: ToolRegistry) -> None:
    app = create_app(settings=settings, ollama_client=FakeOllamaClient(), registry=registry)
    with TestClient(app) as client:
        body = client.get("/tools").json()
    names = {tool["name"] for tool in body}
    assert names == {"echo", "wipe"}
    by_name = {tool["name"]: tool for tool in body}
    assert by_name["wipe"]["risk_level"] == "destructive"
