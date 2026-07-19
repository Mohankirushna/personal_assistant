"""Application settings.

All values can be overridden via environment variables prefixed with
``JARVIS_`` (e.g. ``JARVIS_PORT=9000``) or a ``.env`` file in the backend
directory. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful voice-controlled desktop assistant running "
    "locally on the user's Mac. Be concise: your answers are often spoken "
    "aloud, so prefer one or two short sentences unless the user asks for "
    "detail. Never pretend to have taken an action on the computer; tool "
    "execution is handled separately by the system, not by you."
)


class Settings(BaseSettings):
    """Backend configuration, loaded from env vars / .env at startup."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server — loopback only, by design (see docs/ARCHITECTURE.md section 8).
    host: str = "127.0.0.1"
    port: int = 8765
    # When set, every request except /health must carry
    # "Authorization: Bearer <auth_token>". The SwiftUI app (Phase 3) will
    # generate this per session and pass it via the backend's environment.
    auth_token: str | None = None

    # Ollama / models
    ollama_host: str = "http://127.0.0.1:11434"
    # Non-thinking instruct models only: reasoning models (e.g. qwen3) spend
    # tens of seconds "thinking" per reply on 8GB hardware, which is unusable
    # for voice interaction.
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    llm_power_model: str = "qwen2.5:7b-instruct-q4_K_M"
    power_mode: bool = False
    vision_model: str = "qwen2.5vl:3b"
    # How long Ollama keeps the LLM resident after the last request.
    llm_keep_alive: str = "30m"
    # Load the LLM into Ollama at startup (in the background) so the first
    # command doesn't pay the multi-second model-load latency.
    prewarm_llm: bool = True
    request_timeout_seconds: float = 120.0

    # Chat
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # History passed back to the model is trimmed to this many messages to
    # bound prompt size for small quantized models.
    max_history_messages: int = 20
    # Planning/tool-selection is decoded greedily: on a small model, default
    # sampling (~0.8) makes tool choice flip between runs for identical
    # requests. 0.0 = deterministic. Bump slightly if replies feel too flat.
    planner_temperature: float = 0.0

    # Voice
    whisper_model: str = "base.en"
    whisper_compute: str = "int8"
    wake_word_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    tts_engine: Literal["auto", "piper", "say"] = "auto"
    piper_voice_path: str | None = None
    say_voice: str | None = None
    # Endpointing: an utterance ends after this much trailing silence.
    vad_silence_ms: int = 800
    # RMS (on float32 [-1,1]) below which a frame counts as silence.
    vad_energy_threshold: float = 0.015
    max_utterance_seconds: float = 15.0

    # Memory
    data_dir: Path = Path("~/Library/Application Support/Jarvis")
    # How many recalled memory snippets to give the planner per turn (kept
    # tiny to protect the 3B model's context budget).
    memory_context_hits: int = 2

    # Development escape hatch: skip confirmation prompts entirely.
    auto_approve: bool = False

    # WAHA (optional self-hosted WhatsApp gateway, https://waha.devlike.pro).
    # The API key is kept in the local environment, never in a chat session
    # or tool argument.
    waha_base_url: str | None = None
    waha_api_key: str | None = None
    waha_session: str = "default"
    # Prepended to bare 10-digit numbers (spoken commands and locally saved
    # contacts rarely include a country code, but WhatsApp chat IDs need one).
    # Digits only, e.g. "91" for India, "1" for the US.
    whatsapp_default_country_code: str | None = None

    log_level: str = "INFO"

    @field_validator("host")
    @classmethod
    def _host_must_be_loopback(cls, value: str) -> str:
        if value not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"host must be loopback ({', '.join(sorted(_LOOPBACK_HOSTS))}); "
                "the backend is never exposed beyond localhost by design"
            )
        return value

    @property
    def active_llm_model(self) -> str:
        return self.llm_power_model if self.power_mode else self.llm_model

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def sqlite_path(self) -> Path:
        return self.resolved_data_dir / "jarvis.db"

    @property
    def chroma_path(self) -> Path:
        return self.resolved_data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
