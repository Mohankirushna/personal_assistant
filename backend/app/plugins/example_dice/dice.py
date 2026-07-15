"""A dice-rolling tool — the reference plugin.

This entire file is what writing a Jarvis plugin looks like: subclass Tool,
declare a name/description/args model/risk level, implement run(). Drop the
package in app/plugins/ and it is discovered at startup.
"""

from __future__ import annotations

import secrets
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class DiceArgs(BaseModel):
    sides: int = Field(default=6, ge=2, le=1000, description="Number of sides on the die.")
    count: int = Field(default=1, ge=1, le=20, description="How many dice to roll.")


class DiceTool(Tool):
    name: ClassVar[str] = "roll_dice"
    description: ClassVar[str] = "Roll one or more dice, e.g. for a decision or a game."
    args_model: ClassVar[type[BaseModel]] = DiceArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: DiceArgs) -> ToolResult:  # type: ignore[override]
        rolls = [secrets.randbelow(args.sides) + 1 for _ in range(args.count)]
        summary = (
            f"Rolled {rolls[0]} on a d{args.sides}."
            if args.count == 1
            else f"Rolled {rolls} on {args.count}d{args.sides} (total {sum(rolls)})."
        )
        return ToolResult(tool=self.name, ok=True, summary=summary, data={"rolls": rolls})
