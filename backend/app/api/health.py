"""Health-check router.

Open (no auth) so the SwiftUI app can probe liveness before it has the
session token. Reports Ollama reachability without failing the endpoint —
the backend being up and Ollama being up are separate facts.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class OllamaStatus(BaseModel):
    available: bool
    loaded_models: list[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    ollama: OllamaStatus
    active_model: str | None


def _backend_version() -> str:
    try:
        return version("jarvis-backend")
    except PackageNotFoundError:
        return "0.0.0-dev"


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    loaded = await request.app.state.ollama.loaded_models()
    return HealthResponse(
        version=_backend_version(),
        ollama=OllamaStatus(available=loaded is not None, loaded_models=loaded or []),
        active_model=request.app.state.model_manager.current_model,
    )
