"""End-to-end planner runs against real Ollama + qwen2.5:3b.

Verifies the model actually produces valid decision JSON and uses tools when
appropriate — the make-or-break question for a 3B planner.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import OllamaClient
from app.core.safety import SafetyGate
from app.planner.planner import Planner
from app.tools.clock import ClockTool
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.integration


@pytest.fixture
async def real_planner():  # noqa: ANN201
    settings = Settings(_env_file=None)
    client = OllamaClient(host=settings.ollama_host, timeout_seconds=120)
    registry = ToolRegistry()
    registry.register(ClockTool())
    manager = ModelManager(client, settings)
    yield Planner(client, manager, registry, SafetyGate(), settings)
    await manager.release_all()
    await client.aclose()


async def test_uses_clock_tool_for_time_question(real_planner: Planner) -> None:
    execution = await real_planner.run("What time is it right now?", history=[])
    tools_used = [step.tool for step in execution.steps]
    assert "clock" in tools_used, f"expected clock call, got steps={tools_used}"
    assert execution.reply, "expected a final natural-language reply"


async def test_responds_directly_to_greeting(real_planner: Planner) -> None:
    execution = await real_planner.run("Hello! How are you today?", history=[])
    assert execution.steps == [], f"greeting should need no tools, used {execution.steps}"
    assert execution.reply
