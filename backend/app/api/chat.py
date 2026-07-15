"""Chat endpoints.

POST /chat — request/response. Cannot prompt for confirmation, so
             sensitive/destructive tool calls are denied with an explanation
             (use the app/WebSocket for interactive approval).
WS /ws/chat — streaming + interactive confirmations:
    client -> {"message", "session_id"?}
           -> {"type": "confirm_response", "approved": bool}
    server -> {"type": "token", "content"}*
           -> {"type": "confirm_request", "tool", "risk", "action"}
           -> {"type": "done", "session_id", "reply"}
           -> {"type": "error", "message"} (socket stays open)
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.core.chat_service import ChatService
from app.core.ollama_client import ModelNotFoundError, OllamaUnavailableError
from app.core.safety import ConfirmationRequest

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIRM_TIMEOUT_SECONDS = 120


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

    async def ws_confirmer(request: ConfirmationRequest) -> bool:
        """Show the exact action to the user and await their verdict."""
        await websocket.send_json(
            {
                "type": "confirm_request",
                "tool": request.tool,
                "risk": request.risk.value,
                "action": request.action,
            }
        )
        try:
            while True:
                message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=CONFIRM_TIMEOUT_SECONDS
                )
                if message.get("type") == "confirm_response":
                    return bool(message.get("approved"))
                logger.debug("ignoring message while awaiting confirmation: %s", message)
        except TimeoutError:
            logger.info("confirmation timed out; denying")
            return False

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
                async for token in service.respond_stream(
                    session, parsed.message, confirmer=ws_confirmer
                ):
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
