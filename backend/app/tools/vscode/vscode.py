"""VS Code tool: open files/folders in Visual Studio Code.

Prefers the `code` CLI (supports --goto file:line); falls back to
`open -a "Visual Studio Code"` when the CLI isn't on PATH.
"""

from __future__ import annotations

import shutil
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import expand_path, run_command
from app.tools.base import Tool


class VSCodeOpenArgs(BaseModel):
    path: str = Field(description="File or project folder to open in VS Code.")
    line: int | None = Field(default=None, description="Jump to this line (files only).")


class VSCodeOpenTool(Tool):
    name: ClassVar[str] = "vscode_open"
    description: ClassVar[str] = (
        "Open a file or project folder in Visual Studio Code, optionally at a line."
    )
    args_model: ClassVar[type[BaseModel]] = VSCodeOpenArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: VSCodeOpenArgs) -> ToolResult:  # type: ignore[override]
        target = expand_path(args.path)
        if not target.exists():
            return ToolResult.failure(self.name, f"{target} does not exist")

        code_cli = shutil.which("code")
        if code_cli:
            argv = [code_cli]
            if args.line and target.is_file():
                argv += ["--goto", f"{target}:{args.line}"]
            else:
                argv.append(str(target))
        else:
            argv = ["/usr/bin/open", "-a", "Visual Studio Code", str(target)]

        output = await run_command(argv, timeout=20)
        if not output.ok:
            return ToolResult.failure(self.name, f"could not open VS Code: {output.combined()}")
        where = f" at line {args.line}" if args.line and target.is_file() else ""
        return ToolResult(tool=self.name, ok=True, summary=f"Opened {target}{where} in VS Code")
