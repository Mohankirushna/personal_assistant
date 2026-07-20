"""Chat orchestration shared by the REST, WebSocket, and voice endpoints.

With tools registered, every turn goes through the Planner
(User → Planner → Tools → Safety Gate → Result → reply); `confirmer` is the
transport's way of asking the user to approve sensitive/destructive actions.
With no tools (minimal installs, some tests), it falls back to plain LLM
chat. `respond_stream` streams tokens only on the plain path — planner
replies arrive as one chunk, which is what TTS needs anyway.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import Message, OllamaLike
from app.core.safety import Confirmer
from app.core.sessions import ChatSession, SessionStore
from app.memory.service import MemoryService
from app.planner.planner import Planner, StepObserver
from app.planner.schemas import PlanExecution

logger = logging.getLogger(__name__)

# Tools whose output is worth forwarding when the user later says
# "send that/this to <name>" — deliberately just the info/browse tools,
# not actions like opening apps or controlling media.
_SHAREABLE_CONTENT_TOOLS = {
    "brave_search_open_first", "browser_search", "news_search", "youtube_play", "web_answer",
}


class ChatService:
    def __init__(
        self,
        client: OllamaLike,
        model_manager: ModelManager,
        sessions: SessionStore,
        settings: Settings,
        planner: Planner | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self._client = client
        self._model_manager = model_manager
        self._sessions = sessions
        self._settings = settings
        self._planner = planner
        self._memory = memory

    def open_session(self, session_id: str | None) -> ChatSession:
        return self._sessions.ensure(session_id)

    @staticmethod
    def _update_shareable_content(session: ChatSession, execution: PlanExecution) -> None:
        for step in execution.steps:
            if step.tool not in _SHAREABLE_CONTENT_TOOLS:
                continue
            if step.result is None or not step.result.ok:
                continue
            data = step.result.data
            if data.get("query"):
                session.last_query = data["query"]
            if data.get("url"):
                session.last_url = data["url"]
        if execution.reply:
            session.last_text = execution.reply

    def _prompt_messages(self, session: ChatSession) -> list[Message]:
        # The system prompt is injected per call rather than stored, so
        # changing it in settings affects existing sessions too.
        return [{"role": "system", "content": self._settings.system_prompt}, *session.messages]

    async def respond(
        self,
        session: ChatSession,
        user_message: str,
        confirmer: Confirmer | None = None,
        on_step: StepObserver | None = None,
    ) -> str:
        if self._planner is not None:
            history = list(session.messages)
            self._sessions.append(session, "user", user_message)
            memory_context = (
                await self._memory.context_for(user_message) if self._memory else None
            )
            execution = await self._planner.run(
                user_message,
                history=history,
                confirmer=confirmer,
                memory_context=memory_context,
                last_query=session.last_query,
                last_url=session.last_url,
                last_text=session.last_text,
                on_step=on_step,
            )
            # Store the reply PLUS a compact trace of what actually ran, so
            # follow-ups can reference concrete outcomes (file paths, track
            # names) even when the spoken reply paraphrased them away —
            # e.g. "open the screenshot you just took" needs the saved path.
            trace = "".join(
                f"\n[{step.tool}: {step.result.summary[:160]}]"
                for step in execution.steps
                if step.result is not None and step.result.ok
            )
            self._sessions.append(session, "assistant", execution.reply + trace)
            self._update_shareable_content(session, execution)
            if self._memory:
                await self._memory.record_turn(session.id, execution)
            return execution.reply

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
        self,
        session: ChatSession,
        user_message: str,
        confirmer: Confirmer | None = None,
        on_step: StepObserver | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply chunks. Planner replies arrive as a single chunk;
        plain chat streams token-by-token."""
        if self._planner is not None:
            reply = await self.respond(
                session, user_message, confirmer=confirmer, on_step=on_step
            )
            yield reply
            return

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
