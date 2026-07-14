"""Chat endpoints.

POST /chat            — request/response, full reply in one payload.
WS   /ws/chat         — streaming: send {"message", "session_id"?}, receive
                        {"type": "token"}* then {"type": "done"}; errors come
                        back as {"type": "error"} without closing the socket.

Phase 5 (Planner) reroutes ChatService through the planner; these transports
stay unchanged.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.core.chat_service import ChatService
from app.core.ollama_client import ModelNotFoundError, OllamaUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def _service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = _service(request)
    session = service.open_session(payload.session_id)
    try:
        reply = await service.respond(session, payload.message)
    except (OllamaUnavailableError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(session_id=session.id, reply=reply)


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    service: ChatService = websocket.app.state.chat_service
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                parsed = ChatRequest.model_validate(payload)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            session = service.open_session(parsed.session_id)
            parts: list[str] = []
            try:
                async for token in service.respond_stream(session, parsed.message):
                    parts.append(token)
                    await websocket.send_json({"type": "token", "content": token})
            except (OllamaUnavailableError, ModelNotFoundError) as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            await websocket.send_json(
                {"type": "done", "session_id": session.id, "reply": "".join(parts)}
            )
    except WebSocketDisconnect:
        logger.debug("chat websocket disconnected")
