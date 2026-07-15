"""Screen understanding via Qwen2.5-VL.

Only ever invoked on explicit user request ("look at my screen"). The
ModelManager unloads the text LLM before the vision model loads — both do
not fit in 8GB (see docs/adr/0001-model-manager-ram-budget.md) — and the
next chat turn transparently swaps the LLM back in.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import OllamaLike

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = (
    "Describe what is on this screen: which applications are visible, what "
    "the user appears to be doing, and any errors or dialogs. If there is an "
    "error message, quote it and suggest a likely fix."
)


class VisionService:
    def __init__(
        self, client: OllamaLike, model_manager: ModelManager, settings: Settings
    ) -> None:
        self._client = client
        self._model_manager = model_manager
        self._settings = settings

    async def describe_image(self, image_path: Path, question: str | None = None) -> str:
        """Load the vision model (swapping out the LLM) and describe an image.

        Tries Metal/GPU first; on the Metal out-of-memory failures this
        hardware hits under pressure (or a silently empty reply, another
        symptom of the same), retries fully on CPU — slower, but vision is
        an occasional, explicitly requested operation.
        """
        if not image_path.is_file():  # noqa: ASYNC240 - one tiny stat call
            raise FileNotFoundError(image_path)
        model = await self._model_manager.ensure_vision()
        logger.info("Vision inference on %s with %s", image_path, model)
        prompt = question or DEFAULT_QUESTION
        try:
            reply = await self._client.chat_with_image(
                model=model,
                prompt=prompt,
                image_path=str(image_path),
                keep_alive=0,  # release VL RAM immediately after answering
            )
        except Exception as exc:  # noqa: BLE001 - inspect and maybe fall back
            if "memory" not in str(exc).lower():
                raise
            logger.warning("Vision GPU inference hit OOM; retrying on CPU")
            reply = ""
        if not reply.strip():
            reply = await self._client.chat_with_image(
                model=model,
                prompt=prompt,
                image_path=str(image_path),
                keep_alive=0,
                options={"num_gpu": 0},  # CPU-only retry
            )
        return reply
