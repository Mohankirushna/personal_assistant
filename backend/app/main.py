"""FastAPI application entrypoint.

Binds to loopback only (enforced by Settings), optionally requires a bearer
token (see app.core.auth), and wires the core services together in the
lifespan. Run locally with:

    uv run jarvis-backend
    # or: uv run uvicorn app.main:create_app --factory

Voice endpoints are mounted only when the `voice` extra is installed
(`uv sync --extra voice`); the chat API works without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI

from app.api import briefing, chat, health, tools
from app.core.auth import TokenAuthMiddleware
from app.core.chat_service import ChatService
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.core.model_manager import ModelManager
from app.core.ollama_client import OllamaClient, OllamaLike
from app.core.safety import SafetyGate
from app.core.sessions import SessionStore
from app.planner.planner import Planner
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _voice_imports_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import numpy  # noqa: F401
        import openwakeword  # noqa: F401
    except ImportError:
        return False
    return True


def _memory_imports_available() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def _browser_imports_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


def create_app(
    settings: Settings | None = None,
    ollama_client: OllamaLike | None = None,
    stt: Any | None = None,
    tts: Any | None = None,
    wake_detector: Any | None = None,
    registry: ToolRegistry | None = None,
    memory: Any | None = None,
    enable_memory: bool | None = None,
) -> FastAPI:
    """Application factory.

    All service parameters are injectable for tests; production callers pass
    nothing and get env-derived settings plus real clients/models.
    """
    app_settings = settings or get_settings()
    overrides_given = stt is not None and tts is not None and wake_detector is not None
    voice_enabled = overrides_given or _voice_imports_available()
    memory_enabled = (
        enable_memory
        if enable_memory is not None
        else (memory is not None or _memory_imports_available())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_client = ollama_client is None
        client: OllamaLike = ollama_client or OllamaClient(
            host=app_settings.ollama_host,
            timeout_seconds=app_settings.request_timeout_seconds,
        )
        model_manager = ModelManager(client, app_settings)
        sessions = SessionStore(max_messages=app_settings.max_history_messages)

        tool_registry = registry if registry is not None else ToolRegistry()
        if registry is None:
            tool_registry.discover()
            # Service-dependent tools are injected here, not discovered.
            from app.tools.vision.vision import LookAtScreenTool
            from app.vision.qwen_vl import VisionService

            vision_service = VisionService(client, model_manager, app_settings)
            tool_registry.register(LookAtScreenTool(vision_service))

            if _browser_imports_available():
                from app.tools.browser.browser import (
                    BrowserDownloadTool,
                    BrowserFillTool,
                    BrowserOpenTool,
                    BrowserSearchTool,
                    BrowserSession,
                )

                browser_session = BrowserSession()
                app.state.browser_session = browser_session
                for browser_tool in (
                    BrowserSearchTool(browser_session),
                    BrowserOpenTool(browser_session),
                    BrowserFillTool(browser_session),
                    BrowserDownloadTool(browser_session),
                ):
                    tool_registry.register(browser_tool)
        gate = SafetyGate(auto_approve=app_settings.auto_approve)
        planner = (
            Planner(client, model_manager, tool_registry, gate, app_settings)
            if len(tool_registry)
            else None
        )

        memory_service = memory
        if memory_service is None and memory_enabled:
            from app.memory.embeddings import OllamaEmbedder
            from app.memory.service import MemoryService
            from app.memory.store import MemoryStore
            from app.memory.vector_store import VectorStore

            memory_service = MemoryService(
                MemoryStore(app_settings.sqlite_path),
                VectorStore(app_settings.chroma_path, OllamaEmbedder(client)),
                context_hits=app_settings.memory_context_hits,
            )

        app.state.settings = app_settings
        app.state.ollama = client
        app.state.model_manager = model_manager
        app.state.sessions = sessions
        app.state.registry = tool_registry
        app.state.safety_gate = gate
        app.state.memory = memory_service
        app.state.chat_service = ChatService(
            client, model_manager, sessions, app_settings,
            planner=planner, memory=memory_service,
        )

        if voice_enabled:
            from app.speech.stt import WhisperSTT
            from app.speech.wake_word import OpenWakeWord
            from app.tts.engine import build_tts_engine

            app.state.stt = stt or WhisperSTT(
                model_name=app_settings.whisper_model,
                compute_type=app_settings.whisper_compute,
            )
            app.state.tts = tts or build_tts_engine(app_settings)
            app.state.wake_detector = wake_detector or OpenWakeWord(
                model_name=app_settings.wake_word_model,
                threshold=app_settings.wake_threshold,
            )

        logger.info(
            "Backend ready on %s:%s (llm=%s, auth=%s, voice=%s, memory=%s, tools=%d)",
            app_settings.host,
            app_settings.port,
            app_settings.active_llm_model,
            "on" if app_settings.auth_token else "off",
            "on" if voice_enabled else "off (install the 'voice' extra)",
            "on" if memory_service else "off (install the 'memory' extra)",
            len(tool_registry),
        )

        # Pre-warm the LLM so the first command doesn't pay Ollama's
        # multi-second cold load. Background task: startup must not block on
        # it, and a failure (Ollama not running yet) is non-fatal — the first
        # real request will load the model as before. Only when we own a real
        # client: injected test doubles shouldn't see phantom load calls.
        prewarm_task: asyncio.Task[Any] | None = None
        if owns_client and app_settings.prewarm_llm:

            async def _prewarm() -> None:
                try:
                    await model_manager.ensure_llm()
                    logger.info("LLM pre-warmed")
                except Exception:  # noqa: BLE001 - best-effort warm-up
                    logger.warning("LLM pre-warm failed; will load on first request")

            prewarm_task = asyncio.create_task(_prewarm())

        try:
            yield
        finally:
            if prewarm_task is not None and not prewarm_task.done():
                prewarm_task.cancel()
            active_browser = getattr(app.state, "browser_session", None)
            if active_browser is not None:
                try:
                    await active_browser.close()
                except Exception:  # noqa: BLE001 - shutdown is best-effort
                    logger.warning("Could not close browser at shutdown", exc_info=True)
            try:
                await model_manager.release_all()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                logger.warning("Could not unload models at shutdown", exc_info=True)
            if owns_client:
                await client.aclose()

    app = FastAPI(title="Jarvis Backend", lifespan=lifespan)
    app.add_middleware(TokenAuthMiddleware, token=app_settings.auth_token)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(tools.router)
    app.include_router(briefing.router)
    if voice_enabled:
        from app.api import voice

        app.include_router(voice.router)
    if memory_enabled:
        from app.api import memory as memory_api

        app.include_router(memory_api.router)
    return app


def watch_stdin_and_terminate(
    stream: Any | None = None,
    on_eof: Callable[[], None] | None = None,
) -> threading.Thread:
    """Exit when stdin reaches EOF — i.e. when the process that spawned us
    (the menu-bar app, holding the write end of our stdin pipe) dies.

    A force-killed app never runs its clean-quit path, which used to leak an
    authenticated backend on the port; the next app session would then attach
    to a backend whose token it doesn't know and every authenticated call
    (including /ws/voice) failed. SIGTERM lets uvicorn shut down gracefully,
    unloading models via the lifespan.
    """

    def _watch() -> None:
        source = stream if stream is not None else sys.stdin.buffer
        with contextlib.suppress(Exception):
            while source.read(4096):
                pass  # discard until EOF
        logger.info("stdin closed (parent app exited) — shutting down")
        if on_eof is not None:
            on_eof()
        else:
            os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=_watch, name="stdin-watchdog", daemon=True)
    thread.start()
    return thread


def run() -> None:
    """Console entrypoint (`uv run jarvis-backend`)."""
    settings = get_settings()
    setup_logging(settings.log_level)
    if os.environ.get("JARVIS_EXIT_ON_STDIN_CLOSE") == "1":
        watch_stdin_and_terminate()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # logging is configured by setup_logging
    )
