"""Memory endpoints: command history and semantic search.

Mounted only when memory is enabled (chromadb installed).
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


class HistoryItem(BaseModel):
    ts: float
    session_id: str
    utterance: str
    reply: str
    steps: list[dict]


class SearchHit(BaseModel):
    text: str
    distance: float


@router.get("/memory/history", response_model=list[HistoryItem])
async def history(
    request: Request, limit: int = Query(default=20, ge=1, le=200)
) -> list[HistoryItem]:
    entries = await request.app.state.memory.history(limit)
    return [
        HistoryItem(
            ts=entry.ts,
            session_id=entry.session_id,
            utterance=entry.utterance,
            reply=entry.reply,
            steps=entry.steps,
        )
        for entry in entries
    ]


@router.get("/memory/search", response_model=list[SearchHit])
async def search(
    request: Request,
    q: str = Query(min_length=1),
    k: int = Query(default=5, ge=1, le=20),
) -> list[SearchHit]:
    hits = await request.app.state.memory.search(q, k)
    return [SearchHit(text=hit.text, distance=hit.distance) for hit in hits]
