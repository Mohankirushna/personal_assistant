"""Tools router: what tools exist and how risky they are.

Used by the SwiftUI app's settings/tooling views and by anyone poking the
API. Execution never goes through here — tools only run inside a plan.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class ToolInfo(BaseModel):
    name: str
    description: str
    risk_level: str


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(request: Request) -> list[ToolInfo]:
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            risk_level=tool.risk_level.value,
        )
        for tool in request.app.state.registry.list()
    ]
