"""End-to-end planner runs against real Ollama + qwen2.5:3b.

Verifies the model actually produces valid decision JSON and uses tools when
appropriate — the make-or-break question for a 3B planner.
"""

from __future__ import annotations

from pathlib import Path

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


@pytest.fixture
async def full_registry_planner():  # noqa: ANN201
    """Planner over the complete discovered tool suite (21 tools)."""
    settings = Settings(_env_file=None)
    client = OllamaClient(host=settings.ollama_host, timeout_seconds=120)
    registry = ToolRegistry()
    registry.discover()
    manager = ModelManager(client, settings)
    yield Planner(client, manager, registry, SafetyGate(), settings)
    await manager.release_all()
    await client.aclose()


@pytest.fixture
def short_workdir():  # noqa: ANN201
    """A short, human-looking path. qwen2.5:3b reliably returns empty turns
    when prompts contain pytest's very long tmp paths, so e2e tests must use
    paths shaped like what users actually say."""
    import os
    import shutil

    path = Path(f"/tmp/jarvis-e2e-{os.getpid()}")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


async def test_picks_finder_list_among_all_tools(
    full_registry_planner: Planner, short_workdir: Path
) -> None:
    """Tool selection accuracy with the full catalog, not a toy registry."""
    (short_workdir / "alpha.txt").write_text("a")
    (short_workdir / "beta.txt").write_text("b")
    execution = await full_registry_planner.run(
        f"List the files in the folder {short_workdir}", history=[]
    )
    tools_used = [step.tool for step in execution.steps]
    assert "finder_list" in tools_used, f"steps={tools_used}, reply={execution.reply!r}"
    assert execution.reply


async def test_compound_request_never_fabricates_success(
    full_registry_planner: Planner, short_workdir: Path
) -> None:
    """Compound (multi-step) requests are at the capability edge of a 3B
    model: sometimes it executes both steps, sometimes it produces no plan
    at all. Both are acceptable. What is NEVER acceptable — and what this
    test pins down — is claiming success without tool evidence.

    (For reliable compound commands, users enable power mode / the 7B
    model; single-step commands are reliable on 3B — see the test above.)"""
    execution = await full_registry_planner.run(
        f"Create a folder named 'reports' inside {short_workdir}, then list "
        f"{short_workdir} to confirm it exists.",
        history=[],
    )
    if execution.steps:
        # The model planned: the folder must then genuinely exist.
        tools_used = [step.tool for step in execution.steps]
        assert "finder_create_folder" in tools_used, f"steps={tools_used}"
        assert (short_workdir / "reports").is_dir(), "claimed but not created"
    else:
        # The model declined: the reply must be an honest inability,
        # not a fabricated "done".
        assert not (short_workdir / "reports").exists()
        lowered = execution.reply.lower()
        assert any(
            phrase in lowered
            for phrase in ("wasn't able", "not able", "couldn't", "could not", "rephrase")
        ), f"reply fabricates success: {execution.reply!r}"
