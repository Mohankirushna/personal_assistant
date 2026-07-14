"""FastAPI application entrypoint.

Binds to loopback only (enforced by Settings), optionally requires a bearer
token (see app.core.auth), and wires the core services together in the
lifespan. Run locally with:

    uv run jarvis-backend
    # or: uv run uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api import chat, health
from app.core.auth import TokenAuthMiddleware
from app.core.chat_service import ChatService
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.core.model_manager import ModelManager
from app.core.ollama_client import OllamaClient, OllamaLike
from app.core.sessions import SessionStore

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    ollama_client: OllamaLike | None = None,
) -> FastAPI:
    """Application factory.

    `settings` and `ollama_client` are injectable for tests; production
    callers pass nothing and get env-derived settings plus a real client.
    """
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_client = ollama_client is None
        client: OllamaLike = ollama_client or OllamaClient(
            host=app_settings.ollama_host,
            timeout_seconds=app_settings.request_timeout_seconds,
        )
        model_manager = ModelManager(client, app_settings)
        sessions = SessionStore(max_messages=app_settings.max_history_messages)

        app.state.settings = app_settings
        app.state.ollama = client
        app.state.model_manager = model_manager
        app.state.sessions = sessions
        app.state.chat_service = ChatService(client, model_manager, sessions, app_settings)

        logger.info(
            "Backend ready on %s:%s (llm=%s, auth=%s)",
            app_settings.host,
            app_settings.port,
            app_settings.active_llm_model,
            "on" if app_settings.auth_token else "off",
        )
        try:
            yield
        finally:
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
    return app


def run() -> None:
    """Console entrypoint (`uv run jarvis-backend`)."""
    settings = get_settings()
    setup_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # logging is configured by setup_logging
    )
