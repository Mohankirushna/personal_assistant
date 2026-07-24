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


class ReminderArgs(BaseModel):
    title: str


class ReminderTool(Tool):
    name: ClassVar[str] = "create_reminder"
    description: ClassVar[str] = "Create a reminder (test double)."
    args_model: ClassVar[type[BaseModel]] = ReminderArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self) -> None:
        self.executions: list[str] = []

    async def run(self, args: ReminderArgs) -> ToolResult:  # type: ignore[override]
        self.executions.append(args.title)
        return ToolResult(tool=self.name, ok=True, summary=f"Reminder set: {args.title}")


class WhatsAppArgs(BaseModel):
    recipient: str
    message: str


class WhatsAppEchoTool(Tool):
    """Stand-in for the real whatsapp_send tool, so reference-resolution
    tests don't need OpenWA/WAHA or a network call."""

    name: ClassVar[str] = "whatsapp_send"
    description: ClassVar[str] = "Send a WhatsApp message (test double)."
    args_model: ClassVar[type[BaseModel]] = WhatsAppArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def run(self, args: WhatsAppArgs) -> ToolResult:  # type: ignore[override]
        self.sent.append((args.recipient, args.message))
        return ToolResult(
            tool=self.name, ok=True, summary=f"Sent {args.message!r} to {args.recipient}."
        )


class WebAnswerArgs(BaseModel):
    query: str


class FakeWebAnswerTool(Tool):
    """Stand-in for web_answer that returns fixed page content, so the
    fetch-then-summarize flow can be tested without the network."""

    name: ClassVar[str] = "web_answer"
    description: ClassVar[str] = "Read the web (test double)."
    args_model: ClassVar[type[BaseModel]] = WebAnswerArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def run(self, args: WebAnswerArgs) -> ToolResult:  # type: ignore[override]
        self.calls += 1
        return ToolResult(tool=self.name, ok=True, summary=self.content, data={"query": args.query})


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


async def test_on_step_observer_sees_tool_lifecycle(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    """The observer gets (tool, "running") before execution and the outcome
    after — the voice overlay's live activity feed depends on this order."""
    fake_ollama.queued_turns = [tool_call("echo", text="hi"), respond("Echoed for you.")]
    events: list[tuple[str, str]] = []

    async def observer(tool: str, status: str) -> None:
        events.append((tool, status))

    await planner.run("please echo hi", history=[], on_step=observer)
    assert events == [("echo", "running"), ("echo", "ok")]


async def test_on_step_observer_sees_denial(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.queued_turns = [tool_call("wipe", path="/tmp/x"), respond("I couldn't.")]
    events: list[tuple[str, str]] = []

    async def observer(tool: str, status: str) -> None:
        events.append((tool, status))

    await planner.run("wipe /tmp/x", history=[], on_step=observer)
    assert events == [("wipe", "running"), ("wipe", "denied")]


async def test_on_step_observer_failure_never_breaks_the_plan(
    planner: Planner, fake_ollama: FakeOllamaClient, echo_tool: EchoTool
) -> None:
    """A dead UI socket must not abort the command it was watching."""
    fake_ollama.queued_turns = [tool_call("echo", text="hi"), respond("Echoed for you.")]

    async def broken_observer(tool: str, status: str) -> None:
        raise RuntimeError("socket closed")

    execution = await planner.run("please echo hi", history=[], on_step=broken_observer)
    assert echo_tool.executions == ["hi"]
    assert execution.reply == "Echoed for you."


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


async def test_fake_json_tool_call_is_recovered_not_read_aloud(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """A 3B model sometimes writes a tool call as JSON text instead of using
    the tool-calling API, and often invents 'web_search' for a real tool
    that's actually named something else. Recover the call rather than
    speaking raw JSON to the user."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, registry, SafetyGate(), settings)

    class SearchArgs(BaseModel):
        query: str

    class SearchTool(Tool):
        name: ClassVar[str] = "brave_search_open_first"
        description: ClassVar[str] = "Search the web (test double)."
        args_model: ClassVar[type[BaseModel]] = SearchArgs
        risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

        async def run(self, args: SearchArgs) -> ToolResult:  # type: ignore[override]
            return ToolResult(tool=self.name, ok=True, summary=f"found {args.query}")

    registry.register(SearchTool())
    fake_ollama.queued_turns = [
        ChatTurn(content='{"name": "web_search", "arguments": {"query": "Ironman"}}'),
        respond("Ironman is a Marvel superhero."),
    ]
    # Phrased to avoid the deterministic fast-path (tested separately in
    # test_fast_intents.py) so this exercises the LLM-turn recovery path.
    execution = await planner.run("tell me about ironman", history=[])
    assert execution.steps[0].tool == "brave_search_open_first"
    assert execution.steps[0].result is not None and execution.steps[0].result.ok
    assert execution.reply == "Ironman is a Marvel superhero."


async def test_unrecoverable_fake_json_falls_back_to_honest_text(
    planner: Planner, fake_ollama: FakeOllamaClient
) -> None:
    fake_ollama.queued_turns = [
        ChatTurn(content='{"name": "not_a_real_tool", "arguments": {"x": 1}}'),
    ]
    execution = await planner.run("do the thing", history=[])
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


async def test_identical_repeated_tool_call_stops_the_loop_instead_of_looping(
    planner: Planner, fake_ollama: FakeOllamaClient, echo_tool: EchoTool
) -> None:
    """A small model sometimes reissues the exact same successful call
    instead of recognizing the task is done (e.g. setting volume to the same
    level three times in a row) — this must stop immediately, not burn the
    whole step budget repeating the action."""
    fake_ollama.queued_turns = [
        tool_call("echo", text="same"),
        tool_call("echo", text="same"),
        tool_call("echo", text="same"),
        respond("should never be reached"),
    ]
    execution = await planner.run("echo same forever", history=[], max_steps=5)
    assert len(execution.steps) == 2
    assert echo_tool.executions == ["same", "same"]
    assert execution.reply == "echoed 'same'"


async def test_repeat_after_derailing_still_answers_the_original_question(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """Seen live: clock answered a time question, then the model wandered
    into two identical volume calls. The repeat-stop must not parrot the
    wandered tool's summary ("Volume is 50%") — it forces a final tool-free
    turn that answers from all results in context."""
    echo = EchoTool()
    reminder = ReminderTool()
    reg = ToolRegistry()
    reg.register(echo)
    reg.register(reminder)
    planner = Planner(
        fake_ollama, ModelManager(fake_ollama, settings), reg, SafetyGate(), settings
    )
    fake_ollama.queued_turns = [
        tool_call("echo", text="4 AM"),           # the actual answer
        tool_call("create_reminder", title="x"),  # derail…
        tool_call("create_reminder", title="x"),  # …and repeat -> stop
        respond("It is 4 AM."),                   # forced final tool-free turn
    ]
    execution = await planner.run("what time is it basically", history=[])
    assert execution.reply == "It is 4 AM."
    assert echo.executions == ["4 AM"]
    # The repeated call executes once more before detection (matching
    # test_identical_repeated_tool_call_stops_the_loop_instead_of_looping),
    # but the loop stops there instead of burning the step budget.
    assert reminder.executions == ["x", "x"]


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


async def test_reminder_request_cannot_trigger_a_system_power_call(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    registry = ToolRegistry()
    registry.register(ReminderTool())
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, registry, SafetyGate(), settings)
    fake_ollama.queued_turns = [
        tool_call("system_power", action="restart"),
        respond("What would you like me to remind you about on 17 July at 10 AM?"),
    ]

    execution = await planner.run("Add a reminder on 17 July at 10 AM", history=[])

    assert execution.steps[0].tool == "system_power"
    assert execution.steps[0].denied is False
    assert execution.steps[0].result is not None
    assert "no other action was run" in execution.steps[0].result.summary
    assert execution.reply.startswith("What would you like")


def test_ws_confirmation_roundtrip(
    settings: Settings, registry: ToolRegistry, wipe_tool: WipeTool
) -> None:
    """Full transport test: WS chat asks, client approves, tool runs."""
    fake = FakeOllamaClient()
    fake.queued_turns = [tool_call("wipe", path="/tmp/y"), respond("Wiped it.")]
    app = create_app(
        settings=settings, ollama_client=fake, registry=registry, enable_memory=False
    )
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
    app = create_app(
        settings=settings, ollama_client=fake, registry=registry, enable_memory=False
    )
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
    app = create_app(
        settings=settings, ollama_client=fake, registry=registry, enable_memory=False
    )
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "wipe /tmp/q"})
        assert response.status_code == 200
    assert wipe_tool.executions == []


def test_followup_sees_tool_outcomes(
    settings: Settings, registry: ToolRegistry, echo_tool: EchoTool
) -> None:
    """The session history carries a compact trace of executed tools, so a
    follow-up turn can reference concrete outcomes (paths, names) that the
    spoken reply paraphrased away."""
    fake = FakeOllamaClient()
    fake.queued_turns = [
        tool_call("echo", text="hello-trace"),
        respond("Done!"),           # reply that omits the detail
        respond("It said hello-trace."),
    ]
    app = create_app(
        settings=settings, ollama_client=fake, registry=registry, enable_memory=False
    )
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "echo hello-trace"}).json()
        client.post(
            "/chat",
            json={"message": "what did it say?", "session_id": first["session_id"]},
        )
    # The second turn's prompt history contains the tool trace line.
    history_texts = [m.get("content", "") for m in fake.chat_messages[-1]]
    assert any("[echo: echoed 'hello-trace']" in text for text in history_texts)


def test_tools_endpoint_lists_registered(settings: Settings, registry: ToolRegistry) -> None:
    app = create_app(
        settings=settings,
        ollama_client=FakeOllamaClient(),
        registry=registry,
        enable_memory=False,
    )
    with TestClient(app) as client:
        body = client.get("/tools").json()
    names = {tool["name"] for tool in body}
    assert names == {"echo", "wipe"}
    by_name = {tool["name"]: tool for tool in body}
    assert by_name["wipe"]["risk_level"] == "destructive"


def test_sanitize_spoken_reply() -> None:
    from app.planner.planner import sanitize_spoken_reply

    assert (
        sanitize_spoken_reply(
            "Here is your screenshot: ![](Screenshot%202026-07-15%20at%2020.22.19.png)."
        )
        == "Here is your screenshot: Screenshot 2026-07-15 at 20.22.19.png."
    )
    assert sanitize_spoken_reply("See [the docs](https://example.com).") == "See the docs."
    assert sanitize_spoken_reply("**Done** — saved to `~/Desktop`.") == "Done — saved to ~/Desktop."
    assert sanitize_spoken_reply("plain text stays untouched") == "plain text stays untouched"


@pytest.fixture
def whatsapp_tool() -> WhatsAppEchoTool:
    return WhatsAppEchoTool()


@pytest.fixture
def whatsapp_planner(
    settings: Settings, fake_ollama: FakeOllamaClient, whatsapp_tool: WhatsAppEchoTool
) -> Planner:
    reg = ToolRegistry()
    reg.register(whatsapp_tool)
    manager = ModelManager(fake_ollama, settings)
    return Planner(fake_ollama, manager, reg, SafetyGate(), settings)


async def _approve(request: ConfirmationRequest) -> bool:
    return True


async def test_web_answer_fetches_then_model_answers_from_the_content(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """A bare question fast-paths to web_answer, whose fetched content is fed
    to the model to synthesize a spoken reply — not read aloud verbatim."""
    web = FakeWebAnswerTool(content="The iPhone 15 starts at $799 in the US.")
    reg = ToolRegistry()
    reg.register(web)
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, reg, SafetyGate(), settings)
    fake_ollama.queued_turns = [respond("The iPhone 15 starts at $799.")]

    execution = await planner.run("what is the price of iphone 15", history=[])

    assert web.calls == 1  # deterministic fetch happened
    assert execution.steps[0].tool == "web_answer"
    # The reply is the model's synthesis, and the raw page text was given to it.
    assert execution.reply == "The iPhone 15 starts at $799."
    sent_messages = fake_ollama.chat_messages[0]
    tool_msgs = [m for m in sent_messages if m.get("role") == "tool"]
    assert any("starts at $799 in the US" in str(m.get("content")) for m in tool_msgs)


class ReadUrlAloudArgs(BaseModel):
    url: str | None = None


class FakeReadUrlAloudTool(Tool):
    """Stand-in for read_url_aloud: skips the real fetch/AppleScript."""

    name: ClassVar[str] = "read_url_aloud"
    description: ClassVar[str] = "Read a page aloud (test double)."
    args_model: ClassVar[type[BaseModel]] = ReadUrlAloudArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def run(self, args: ReadUrlAloudArgs) -> ToolResult:  # type: ignore[override]
        self.calls += 1
        return ToolResult(tool=self.name, ok=True, summary=self.content, data={"url": args.url})


async def test_read_this_out_loud_fast_paths_and_summarizes(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """'read this out loud' fast-paths to read_url_aloud using last_url, and
    the raw page text is handed to the model to narrate — not spoken as-is
    (see _READ_ALOUD_SUMMARY_INSTRUCTION)."""
    reader = FakeReadUrlAloudTool(content="Raw page: lots of nav junk. Real headline. Body text.")
    reg = ToolRegistry()
    reg.register(reader)
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, reg, SafetyGate(), settings)
    fake_ollama.queued_turns = [respond("Here's a clean narration of the article.")]

    execution = await planner.run(
        "read this out loud", history=[], last_url="https://example.com/article"
    )

    assert reader.calls == 1
    assert execution.steps[0].tool == "read_url_aloud"
    assert execution.reply == "Here's a clean narration of the article."


async def test_do_it_again_reruns_read_url_aloud_when_last_turn_spoke(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """'do it again' after a turn that actually read something aloud
    (detected via the [read_url_aloud: ...] trace ChatService appends to
    history) re-fetches and re-narrates, rather than echoing stale text."""
    reader = FakeReadUrlAloudTool(content="Raw page text.")
    reg = ToolRegistry()
    reg.register(reader)
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, reg, SafetyGate(), settings)
    fake_ollama.queued_turns = [respond("Fresh narration.")]
    history = [
        {"role": "user", "content": "read this out loud"},
        {
            "role": "assistant",
            "content": "Old narration.\n[read_url_aloud: Raw page text.]",
        },
    ]

    execution = await planner.run(
        "do it again", history=history, last_url="https://example.com/article"
    )

    assert reader.calls == 1  # actually re-fetched, not just echoed
    assert execution.steps[0].tool == "read_url_aloud"
    assert execution.reply == "Fresh narration."


async def test_do_it_again_falls_through_when_last_turn_was_unrelated(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """'do it again' after an unrelated action (no read-aloud trace) must
    NOT trigger read_url_aloud — "it" is ambiguous, so this falls through to
    the ordinary LLM planner rather than guessing."""
    reader = FakeReadUrlAloudTool(content="Raw page text.")
    reg = ToolRegistry()
    reg.register(reader)
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, reg, SafetyGate(), settings)
    history = [
        {"role": "user", "content": "turn up the volume"},
        {"role": "assistant", "content": "Volume increased.\n[volume: Volume set to 60%]"},
    ]

    execution = await planner.run(
        "do it again", history=history, last_url="https://example.com/article"
    )

    assert reader.calls == 0
    assert "read_url_aloud" not in [s.tool for s in execution.steps]


async def test_send_pronoun_reference_forwards_last_url(
    whatsapp_planner: Planner, whatsapp_tool: WhatsAppEchoTool
) -> None:
    execution = await whatsapp_planner.run(
        "send this link to mohan on whatsapp",
        history=[],
        confirmer=_approve,
        last_url="https://example.com/cricket-scores",
        last_text="Here's what I found.",
    )
    assert whatsapp_tool.sent == [("mohan", "https://example.com/cricket-scores")]
    assert execution.reply == "Sent 'https://example.com/cricket-scores' to mohan."


@pytest.mark.parametrize(
    "phrase",
    [
        "this website link",
        "that web page",
        "the url",
        "this page",
        "the website",
        "that site",
    ],
)
async def test_send_natural_reference_variants_forward_last_url(
    whatsapp_planner: Planner, whatsapp_tool: WhatsAppEchoTool, phrase: str
) -> None:
    """Real bug: only a fixed set of exact phrases ("this link", "that
    link") was recognized as a reference, so a natural variant like "this
    website link" fell through and was sent as literal text instead of the
    actual URL."""
    whatsapp_tool.sent.clear()
    execution = await whatsapp_planner.run(
        f"send {phrase} to mohan on whatsapp",
        history=[],
        confirmer=_approve,
        last_url="https://example.com/cricket-scores",
    )
    assert whatsapp_tool.sent == [("mohan", "https://example.com/cricket-scores")]
    assert "example.com" in execution.reply


async def test_send_topic_overlap_forwards_last_content(
    whatsapp_planner: Planner, whatsapp_tool: WhatsAppEchoTool
) -> None:
    """'send the cricket score' after searching 'cricket score yesterday'
    should forward the last result, not literally send the phrase."""
    execution = await whatsapp_planner.run(
        "send the cricket score to mohan on whatsapp",
        history=[],
        confirmer=_approve,
        last_query="cricket score yesterday",
        last_url="https://example.com/cricket-scores",
    )
    assert whatsapp_tool.sent == [("mohan", "https://example.com/cricket-scores")]
    assert "example.com" in execution.reply


async def test_send_literal_message_is_not_treated_as_a_reference(
    whatsapp_planner: Planner, whatsapp_tool: WhatsAppEchoTool
) -> None:
    execution = await whatsapp_planner.run(
        "send hello to mohan on whatsapp",
        history=[],
        confirmer=_approve,
        last_url="https://example.com/cricket-scores",
    )
    assert whatsapp_tool.sent == [("mohan", "hello")]
    assert execution.reply == "Sent 'hello' to mohan."


async def test_send_reference_with_no_prior_content_fails_honestly(
    whatsapp_planner: Planner, whatsapp_tool: WhatsAppEchoTool
) -> None:
    execution = await whatsapp_planner.run(
        "send this to mohan on whatsapp", history=[], confirmer=_approve,
    )
    assert whatsapp_tool.sent == []
    assert "don't have anything recent" in execution.reply


def test_tool_pruning_saves_tokens(settings: Settings, fake_ollama: FakeOllamaClient) -> None:
    """Tool pruning reduces the tool-spec overhead from ~4.2K to ~400-2K
    tokens on typical commands, keeping the planner's context under budget."""
    import json

    from app.tools.registry import ToolRegistry

    full_registry = ToolRegistry()
    full_registry.discover()
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, full_registry, SafetyGate(), settings)

    all_tools = list(full_registry.list())
    all_specs = [t.llm_spec() for t in all_tools]
    all_json = json.dumps(all_specs)
    base_tokens = len(all_json) // 4

    utterances = [
        "what time is it",     # ~3 tools (clock, ...)
        "take a screenshot",   # ~2 tools
        "play some music",     # ~4 tools (media, spotify, youtube, ...)
        "find my files",       # ~9 tools (finder_*)
    ]

    for utterance in utterances:
        pruned_names = planner._prune_tools(utterance)
        assert pruned_names is not None, f"{utterance!r} should prune to some tools"
        assert len(pruned_names) < len(
            all_tools
        ), f"{utterance!r} should reduce tool count"

        pruned_specs = [t.llm_spec() for t in all_tools if t.name in pruned_names]
        pruned_json = json.dumps(pruned_specs)
        pruned_tokens = len(pruned_json) // 4
        savings_pct = 100 * (1 - pruned_tokens / base_tokens)

        # Pruning should save at least 50% of token overhead.
        assert (
            savings_pct > 50
        ), f"{utterance!r}: only {savings_pct:.0f}% savings, need > 50%"


def test_tool_pruning_fallback_on_vague_query(
    settings: Settings, fake_ollama: FakeOllamaClient
) -> None:
    """When a query is too vague (< 2 matching tools), pruning returns None
    (include all tools) rather than risk losing what's needed."""
    from app.tools.registry import ToolRegistry

    full_registry = ToolRegistry()
    full_registry.discover()
    manager = ModelManager(fake_ollama, settings)
    planner = Planner(fake_ollama, manager, full_registry, SafetyGate(), settings)

    # A completely opaque phrase matches nothing.
    result = planner._prune_tools("xyz zyx abc")
    assert result is None, "opaque phrase should get all tools (fallback)"

    # A query that matches only one tool should fallback.
    result = planner._prune_tools("reminiscent")  # matches ~1 tool (reminder)
    assert result is None, "< 2 matches should fallback to all tools"
