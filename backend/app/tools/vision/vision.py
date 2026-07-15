"""Vision tool: screenshot the screen and describe it with Qwen2.5-VL.

Needs the VisionService injected, so it is registered explicitly in
app.main (not by discovery — the registry skips tools whose constructors
require arguments). Vision only ever runs on an explicit user request, per
the product spec and the RAM budget (ADR 0001).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import run_command
from app.tools.base import Tool
from app.vision.qwen_vl import VisionService


class LookAtScreenArgs(BaseModel):
    question: str | None = Field(
        default=None,
        description="Optional specific question about the screen; omit for a "
        "general description.",
    )


class LookAtScreenTool(Tool):
    name: ClassVar[str] = "look_at_screen"
    description: ClassVar[str] = (
        "Take a screenshot and visually analyze what is on the user's screen "
        "(apps, errors, dialogs). Use ONLY when the user explicitly asks you "
        "to look at their screen. Slow: swaps the language model out of "
        "memory temporarily."
    )
    args_model: ClassVar[type[BaseModel]] = LookAtScreenArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, vision: VisionService) -> None:
        self._vision = vision

    async def run(self, args: LookAtScreenArgs) -> ToolResult:  # type: ignore[override]
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "screen.png"
            output = await run_command(["/usr/sbin/screencapture", "-x", str(shot)])
            if not output.ok or not shot.exists():
                hint = (
                    " — grant Screen Recording permission in System Settings → "
                    "Privacy & Security → Screen Recording"
                    if "could not create image" in output.stderr
                    else ""
                )
                return ToolResult.failure(
                    self.name, f"could not capture the screen: {output.combined()}{hint}"
                )
            description = await self._vision.describe_image(shot, args.question)
        return ToolResult(tool=self.name, ok=True, summary=description)
