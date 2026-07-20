"""Focus mode / Do Not Disturb: toggle via Control Center UI scripting.

Why not `defaults write ...controlcenter DoNotDisturb`: since macOS 12 the
real Focus state lives with `donotdisturbd` (TCC-protected), and writing that
legacy plist key succeeds while changing nothing — the tool would then claim
"Do Not Disturb enabled" for a no-op. There is no public CLI for Focus, so
this clicks the Focus toggle in Control Center and — crucially — reads the
toggle's value back afterwards: the summary reports the state the UI
actually shows, never an assumption.

Requires Accessibility permission for Jarvis (System Settings > Privacy &
Security > Accessibility); without it the AppleScript errors and the tool
fails honestly with instructions.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool

_ACCESSIBILITY_HELP = (
    "I can't reach Control Center: Jarvis needs Accessibility permission "
    "(System Settings > Privacy & Security > Accessibility)."
)

# Opens Control Center, finds the Focus toggle, applies the desired state,
# and returns "before,after" checkbox values so Python can verify the click
# actually changed something.
_FOCUS_SCRIPT = """
on run argv
    set desired to item 1 of argv
    tell application "System Events"
        tell process "ControlCenter"
            click (first menu bar item of menu bar 1 whose name contains "Control Center")
            delay 0.7
            set focusToggle to (first checkbox of window 1 whose name contains "Focus")
            set beforeVal to value of focusToggle as integer
            set needsClick to (desired = "toggle")
            if desired = "on" and beforeVal = 0 then set needsClick to true
            if desired = "off" and beforeVal = 1 then set needsClick to true
            if needsClick then
                click focusToggle
                delay 0.5
                set afterVal to value of focusToggle as integer
            else
                set afterVal to beforeVal
            end if
            key code 53 -- Esc: close Control Center
            return (beforeVal as text) & "," & (afterVal as text)
        end tell
    end tell
end run
"""


class FocusModeArgs(BaseModel):
    action: Literal["on", "off", "toggle"] = Field(
        default="toggle",
        description="'on' to enable, 'off' to disable, 'toggle' to switch.",
    )


class FocusModeTool(Tool):
    name: ClassVar[str] = "focus_mode"
    description: ClassVar[str] = (
        "Enable, disable, or toggle Do Not Disturb / Focus on this Mac via "
        "Control Center."
    )
    args_model: ClassVar[type[BaseModel]] = FocusModeArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: FocusModeArgs) -> ToolResult:  # type: ignore[override]
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                _FOCUS_SCRIPT,
                args.action,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        except TimeoutError:
            return ToolResult.failure(self.name, "Control Center did not respond.")
        except OSError as exc:
            return ToolResult.failure(self.name, f"Could not run osascript: {exc}")

        if proc.returncode != 0:
            error = stderr.decode().strip()
            if "assistive access" in error or "-1719" in error:
                return ToolResult.failure(self.name, _ACCESSIBILITY_HELP)
            return ToolResult.failure(
                self.name, f"Focus toggle failed: {error or 'unknown UI error'}"
            )

        before, _, after = stdout.decode().strip().partition(",")
        if after not in ("0", "1"):
            return ToolResult.failure(
                self.name, "Couldn't read the Focus state back from Control Center."
            )
        wanted_change = args.action == "toggle" or (args.action == "on") != (
            before == "1"
        )
        if wanted_change and before == after:
            return ToolResult.failure(
                self.name,
                "Clicking the Focus toggle didn't change its state — "
                "Do Not Disturb is unchanged.",
            )
        state = "on" if after == "1" else "off"
        already = " (it already was)" if before == after else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Do Not Disturb is now {state}{already}.",
            data={"action": args.action, "state": state},
        )
