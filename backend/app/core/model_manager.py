"""ModelManager: enforces the single-heavy-model RAM budget.

See docs/adr/0001-model-manager-ram-budget.md. On 8GB hardware only one heavy
model (text LLM *or* vision model) may be resident in Ollama at a time; this
class owns that invariant. STT/TTS/wake-word are small and are not managed
here.

All transitions are serialized behind an asyncio lock so concurrent requests
cannot race Ollama into loading two heavy models at once.
"""

from __future__ import annotations

import asyncio
import enum
import logging

from app.core.config import Settings
from app.core.ollama_client import OllamaLike

logger = logging.getLogger(__name__)


class HeavyModelKind(enum.Enum):
    NONE = "none"
    LLM = "llm"
    VISION = "vision"


class ModelManager:
    def __init__(self, client: OllamaLike, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._lock = asyncio.Lock()
        self._kind = HeavyModelKind.NONE
        self._model: str | None = None

    @property
    def current_kind(self) -> HeavyModelKind:
        return self._kind

    @property
    def current_model(self) -> str | None:
        return self._model

    async def ensure_llm(self) -> str:
        """Make the text LLM the resident heavy model; returns its name.

        Honors the power_mode setting, so toggling power mode mid-session
        swaps 4B out for 8B on the next request.
        """
        return await self._ensure(HeavyModelKind.LLM, self._settings.active_llm_model)

    async def ensure_vision(self) -> str:
        """Make the vision model the resident heavy model; returns its name."""
        return await self._ensure(HeavyModelKind.VISION, self._settings.vision_model)

    async def _ensure(self, kind: HeavyModelKind, model: str) -> str:
        async with self._lock:
            if self._kind is kind and self._model == model:
                return model
            if self._model is not None:
                await self._client.unload_model(self._model)
                self._kind, self._model = HeavyModelKind.NONE, None
            await self._client.load_model(model, keep_alive=self._settings.llm_keep_alive)
            self._kind, self._model = kind, model
            logger.info("Active heavy model is now %s (%s)", model, kind.value)
            return model

    async def release_all(self) -> None:
        """Unload whatever heavy model is resident (used at shutdown)."""
        async with self._lock:
            if self._model is not None:
                await self._client.unload_model(self._model)
                self._kind, self._model = HeavyModelKind.NONE, None
