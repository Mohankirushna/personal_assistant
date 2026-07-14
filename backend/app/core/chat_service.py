"""Chat orchestration shared by the REST and WebSocket chat endpoints.

Phase 2: user message -> LLM -> reply, with per-session history.
Phase 5 (Planner) reroutes this through the planner/tool/safety-gate flow;
the transport-facing interface (respond / respond_stream) stays the same.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import Message, OllamaLike
from app.core.sessions import ChatSession, SessionStore

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        client: OllamaLike,
        model_manager: ModelManager,
        sessions: SessionStore,
        settings: Settings,
    ) -> None:
        self._client = client
        self._model_manager = model_manager
        self._sessions = sessions
        self._settings = settings

    def open_session(self, session_id: str | None) -> ChatSession:
        return self._sessions.ensure(session_id)

    def _prompt_messages(self, session: ChatSession) -> list[Message]:
        # The system prompt is injected per call rather than stored, so
        # changing it in settings affects existing sessions too.
        return [{"role": "system", "content": self._settings.system_prompt}, *session.messages]

    async def respond(self, session: ChatSession, user_message: str) -> str:
        self._sessions.append(session, "user", user_message)
        model = await self._model_manager.ensure_llm()
        reply = await self._client.chat(
            model=model,
            messages=self._prompt_messages(session),
            keep_alive=self._settings.llm_keep_alive,
        )
        self._sessions.append(session, "assistant", reply)
        return reply

    async def respond_stream(
        self, session: ChatSession, user_message: str
    ) -> AsyncIterator[str]:
        """Yield reply tokens; history is updated with the full reply at the
        end (or with whatever was produced, if the stream is interrupted)."""
        self._sessions.append(session, "user", user_message)
        model = await self._model_manager.ensure_llm()
        parts: list[str] = []
        try:
            async for token in self._client.chat_stream(
                model=model,
                messages=self._prompt_messages(session),
                keep_alive=self._settings.llm_keep_alive,
            ):
                parts.append(token)
                yield token
        finally:
            if parts:
                self._sessions.append(session, "assistant", "".join(parts))
