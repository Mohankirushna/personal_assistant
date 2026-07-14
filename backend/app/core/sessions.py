"""In-memory chat session store.

Phase 2 keeps conversation history in process memory only; Phase 7 (Memory)
adds SQLite persistence and semantic recall on top. Voice sessions (Phase 4)
reuse this same store so spoken and typed turns share one history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.core.ollama_client import Message


@dataclass
class ChatSession:
    id: str
    messages: list[Message] = field(default_factory=list)


class SessionStore:
    def __init__(self, max_messages: int = 20) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._max_messages = max_messages

    def ensure(self, session_id: str | None) -> ChatSession:
        """Return the existing session, or create one (also for unknown ids,
        so a client resuming after a backend restart degrades gracefully)."""
        if session_id is not None and session_id in self._sessions:
            return self._sessions[session_id]
        session = ChatSession(id=session_id or uuid.uuid4().hex)
        self._sessions[session.id] = session
        return session

    def append(self, session: ChatSession, role: str, content: str) -> None:
        session.messages.append({"role": role, "content": content})
        if len(session.messages) > self._max_messages:
            del session.messages[: len(session.messages) - self._max_messages]
