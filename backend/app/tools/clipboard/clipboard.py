"""Clipboard tools: read/write the system pasteboard.

Uses pbpaste/pbcopy (built-in, no dependencies) rather than pyobjc —
text-only, which covers the assistant's use cases.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class ReadClipboardArgs(BaseModel):
    pass


class ClipboardReadTool(Tool):
    name: ClassVar[str] = "clipboard_read"
    description: ClassVar[str] = "Read the current text contents of the clipboard."
    args_model: ClassVar[type[BaseModel]] = ReadClipboardArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: BaseModel) -> ToolResult:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/pbpaste", stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        text = stdout.decode(errors="replace")
        if not text:
            return ToolResult(
                tool=self.name, ok=True, summary="The clipboard is empty (or not text)."
            )
        shown = text if len(text) <= 2000 else text[:2000] + "… (truncated)"
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Clipboard contains:\n{shown}",
            data={"length": len(text)},
        )


class WriteClipboardArgs(BaseModel):
    text: str = Field(description="Text to place on the clipboard.")


class ClipboardWriteTool(Tool):
    name: ClassVar[str] = "clipboard_write"
    description: ClassVar[str] = "Copy text to the clipboard (replaces current contents)."
    args_model: ClassVar[type[BaseModel]] = WriteClipboardArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: WriteClipboardArgs) -> ToolResult:  # type: ignore[override]
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/pbcopy", stdin=asyncio.subprocess.PIPE
        )
        await process.communicate(args.text.encode())
        if process.returncode != 0:
            return ToolResult.failure(self.name, "pbcopy failed")
        return ToolResult(
            tool=self.name, ok=True, summary=f"Copied {len(args.text)} characters to the clipboard"
        )
