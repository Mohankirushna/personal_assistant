"""MemoryService: the facade the rest of the app talks to.

Records every completed turn into both stores and produces the compact
"relevant past context" snippet the planner sees. Recording is fire-and-
forget from the caller's perspective — a memory failure must never break a
user's request.
"""

from __future__ import annotations

import logging

from app.memory.store import HistoryEntry, MemoryStore
from app.memory.vector_store import MemoryHit, VectorStore
from app.planner.schemas import PlanExecution

logger = logging.getLogger(__name__)

# Only recall reasonably close matches (cosine distance).
_MAX_DISTANCE = 0.55


class MemoryService:
    def __init__(self, store: MemoryStore, vectors: VectorStore, context_hits: int = 2) -> None:
        self._store = store
        self._vectors = vectors
        self._context_hits = context_hits

    async def record_turn(self, session_id: str, execution: PlanExecution) -> None:
        try:
            steps = [
                {
                    "tool": step.tool,
                    "args": step.args,
                    "risk": step.risk.value,
                    "ok": bool(step.result and step.result.ok),
                    "denied": step.denied,
                }
                for step in execution.steps
            ]
            await self._store.record_turn(
                session_id, execution.utterance, execution.reply, steps
            )
            tools_used = ", ".join(step.tool for step in execution.steps) or "no tools"
            await self._vectors.add_turn(
                f"User asked: {execution.utterance}\nOutcome ({tools_used}): {execution.reply}",
                metadata={"session_id": session_id},
            )
        except Exception:  # noqa: BLE001 - memory must never break a turn
            logger.warning("Failed to record turn in memory", exc_info=True)

    async def context_for(self, utterance: str) -> str | None:
        """A short recall snippet for the planner, or None."""
        try:
            hits = await self._vectors.search(utterance, k=self._context_hits)
        except Exception:  # noqa: BLE001 - recall is best-effort
            logger.warning("Memory recall failed", exc_info=True)
            return None
        relevant = [hit for hit in hits if hit.distance <= _MAX_DISTANCE]
        if not relevant:
            return None
        return "Possibly relevant past interactions:\n" + "\n---\n".join(
            hit.text for hit in relevant
        )

    async def history(self, limit: int = 20) -> list[HistoryEntry]:
        return await self._store.history(limit)

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        return await self._vectors.search(query, k)
