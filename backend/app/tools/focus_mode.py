"""Focus mode / Do Not Disturb control: toggle via macOS defaults."""

from __future__ import annotations

import asyncio
import subprocess
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool


class FocusModeArgs(BaseModel):
    action: Literal["on", "off", "toggle"] = Field(
        default="toggle",
        description="'on' to enable, 'off' to disable, 'toggle' to switch.",
    )


class FocusModeTool(Tool):
    name: ClassVar[str] = "focus_mode"
    description: ClassVar[str] = (
        "Enable, disable, or toggle Do Not Disturb / Focus Mode on this Mac. "
        "Use 'on' to enable, 'off' to disable, or 'toggle' to switch."
    )
    args_model: ClassVar[type[BaseModel]] = FocusModeArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: FocusModeArgs) -> ToolResult:  # type: ignore[override]
        action = args.action
        plist = "~/Library/Preferences/ByHost/com.apple.controlcenter.plist"
        # Use `defaults` command to toggle Focus Mode
        if action == "on":
            script_actual = rf"""
defaults write {plist} DoNotDisturb -bool true
killall -u $(whoami) "Control Center" 2>/dev/null || true
"""
        elif action == "off":
            script_actual = rf"""
defaults write {plist} DoNotDisturb -bool false
killall -u $(whoami) "Control Center" 2>/dev/null || true
"""
        else:  # toggle
            script_actual = rf"""
CURRENT=$(defaults read {plist} DoNotDisturb 2>/dev/null || echo "0")
if [ "$CURRENT" == "1" ]; then
  defaults write {plist} DoNotDisturb -bool false
else
  defaults write {plist} DoNotDisturb -bool true
fi
killall -u $(whoami) "Control Center" 2>/dev/null || true
"""

        try:
            await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", script_actual],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            action_label = (
                "enabled" if action == "on"
                else "disabled" if action == "off"
                else "toggled"
            )
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=f"Do Not Disturb is now {action_label}.",
                data={"action": action},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(self.name, "Focus mode toggle timed out.")
        except subprocess.CalledProcessError as e:
            return ToolResult.failure(
                self.name,
                f"Failed to toggle focus mode: {e.stderr or e.stdout or 'unknown error'}",
            )
