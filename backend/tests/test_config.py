"""Settings behavior: defaults, env overrides, loopback enforcement."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.auth_token is None
    # Non-thinking instruct model by design; see config.py.
    assert settings.llm_model == "qwen2.5:3b-instruct-q4_K_M"
    assert settings.active_llm_model == settings.llm_model


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_PORT", "9001")
    monkeypatch.setenv("JARVIS_LLM_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    settings = Settings(_env_file=None)
    assert settings.port == 9001
    assert settings.llm_model == "qwen2.5:3b-instruct-q4_K_M"


def test_power_mode_switches_active_model() -> None:
    settings = Settings(_env_file=None, power_mode=True)
    assert settings.active_llm_model == settings.llm_power_model


def test_non_loopback_host_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, host="0.0.0.0")
