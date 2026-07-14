"""ModelManager state machine: the single-heavy-model invariant."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.model_manager import HeavyModelKind, ModelManager
from tests.conftest import FakeOllamaClient


@pytest.fixture
def manager(settings: Settings, fake_ollama: FakeOllamaClient) -> ModelManager:
    return ModelManager(fake_ollama, settings)


async def test_starts_idle(manager: ModelManager) -> None:
    assert manager.current_kind is HeavyModelKind.NONE
    assert manager.current_model is None


async def test_ensure_llm_loads_once(
    manager: ModelManager, fake_ollama: FakeOllamaClient, settings: Settings
) -> None:
    model = await manager.ensure_llm()
    assert model == settings.llm_model
    assert manager.current_kind is HeavyModelKind.LLM
    # Second call is a no-op: no extra load, no unload.
    await manager.ensure_llm()
    assert fake_ollama.calls == [("load", settings.llm_model)]


async def test_vision_unloads_llm_first(
    manager: ModelManager, fake_ollama: FakeOllamaClient, settings: Settings
) -> None:
    await manager.ensure_llm()
    await manager.ensure_vision()
    assert fake_ollama.calls == [
        ("load", settings.llm_model),
        ("unload", settings.llm_model),
        ("load", settings.vision_model),
    ]
    assert manager.current_kind is HeavyModelKind.VISION


async def test_failed_load_resets_to_idle(
    manager: ModelManager, fake_ollama: FakeOllamaClient
) -> None:
    await manager.ensure_llm()
    fake_ollama.fail_with = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await manager.ensure_vision()
    # The old model was unloaded and the new one failed: state must be IDLE,
    # not a stale claim that something is resident.
    assert manager.current_kind is HeavyModelKind.NONE
    assert manager.current_model is None


async def test_release_all(
    manager: ModelManager, fake_ollama: FakeOllamaClient, settings: Settings
) -> None:
    await manager.ensure_llm()
    await manager.release_all()
    assert manager.current_kind is HeavyModelKind.NONE
    assert ("unload", settings.llm_model) in fake_ollama.calls
