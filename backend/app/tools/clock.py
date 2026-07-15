"""Clock tool: current local date and time.

The smallest possible real tool — useful for a voice assistant, and it lets
the planner be exercised end-to-end before the Phase 6 tool suite lands.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class ClockArgs(BaseModel):
    pass


class ClockTool(Tool):
    name: ClassVar[str] = "clock"
    description: ClassVar[str] = (
        "Get the current local date and time on this Mac. Takes no arguments."
    )
    args_model: ClassVar[type[BaseModel]] = ClockArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: BaseModel) -> ToolResult:
        now = datetime.now().astimezone()
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=now.strftime("It is %A, %B %d, %Y at %I:%M %p (%Z)."),
            data={"iso": now.isoformat()},
        )
