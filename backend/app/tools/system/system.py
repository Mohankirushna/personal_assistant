"""System tools: apps, volume, screenshots, media, windows, brightness.

Everything goes through AppleScript / built-in CLIs — no compiled helper
needed. Window management requires the host process to have Accessibility
permission (System Settings → Privacy & Security → Accessibility); the tool
reports that clearly when missing rather than failing cryptically.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import applescript_quote, expand_path, run_command, run_osascript
from app.tools.base import Tool


class OpenAppArgs(BaseModel):
    name: str = Field(description="Application name, e.g. 'Safari' or 'Notes'.")


class OpenAppTool(Tool):
    name: ClassVar[str] = "open_app"
    description: ClassVar[str] = "Open (or bring to front) a macOS application by name."
    args_model: ClassVar[type[BaseModel]] = OpenAppArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: OpenAppArgs) -> ToolResult:  # type: ignore[override]
        output = await run_command(["/usr/bin/open", "-a", args.name])
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not open {args.name!r}: {output.combined()}"
            )
        return ToolResult(tool=self.name, ok=True, summary=f"Opened {args.name}")


class QuitAppArgs(BaseModel):
    name: str = Field(description="Application to quit, e.g. 'Safari'.")


class QuitAppTool(Tool):
    name: ClassVar[str] = "quit_app"
    description: ClassVar[str] = (
        "Quit a running application (it may prompt to save unsaved work)."
    )
    args_model: ClassVar[type[BaseModel]] = QuitAppArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    async def run(self, args: QuitAppArgs) -> ToolResult:  # type: ignore[override]
        script = f"tell application {applescript_quote(args.name)} to quit"
        output = await run_osascript(script)
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not quit {args.name!r}: {output.combined()}"
            )
        return ToolResult(tool=self.name, ok=True, summary=f"Quit {args.name}")


class ListAppsArgs(BaseModel):
    pass


class ListAppsTool(Tool):
    name: ClassVar[str] = "list_running_apps"
    description: ClassVar[str] = "List the applications currently running (visible ones)."
    args_model: ClassVar[type[BaseModel]] = ListAppsArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: BaseModel) -> ToolResult:
        script = (
            'tell application "System Events" to get name of every process '
            "whose background only is false"
        )
        output = await run_osascript(script)
        if not output.ok:
            return ToolResult.failure(self.name, output.combined())
        apps = [name.strip() for name in output.stdout.split(",") if name.strip()]
        return ToolResult(
            tool=self.name,
            ok=True,
            summary="Running apps: " + ", ".join(apps),
            data={"apps": apps},
        )


class VolumeArgs(BaseModel):
    level: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Volume 0-100 to set; omit to just read the current volume.",
    )


class VolumeTool(Tool):
    name: ClassVar[str] = "volume"
    description: ClassVar[str] = "Get or set the system output volume (0-100)."
    args_model: ClassVar[type[BaseModel]] = VolumeArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: VolumeArgs) -> ToolResult:  # type: ignore[override]
        if args.level is None:
            output = await run_osascript("output volume of (get volume settings)")
            if not output.ok:
                return ToolResult.failure(self.name, output.combined())
            level = output.stdout.strip()
            return ToolResult(
                tool=self.name, ok=True, summary=f"Volume is {level}%", data={"level": int(level)}
            )
        output = await run_osascript(f"set volume output volume {args.level}")
        if not output.ok:
            return ToolResult.failure(self.name, output.combined())
        return ToolResult(tool=self.name, ok=True, summary=f"Volume set to {args.level}%")


class ScreenshotArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Where to save; default is a timestamped file on the Desktop."
    )


class ScreenshotTool(Tool):
    name: ClassVar[str] = "screenshot"
    description: ClassVar[str] = "Take a screenshot of the whole screen and save it."
    args_model: ClassVar[type[BaseModel]] = ScreenshotArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: ScreenshotArgs) -> ToolResult:  # type: ignore[override]
        if args.path:
            target = expand_path(args.path)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
            target = expand_path(f"~/Desktop/Screenshot {stamp}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        output = await run_command(["/usr/sbin/screencapture", "-x", str(target)])
        if not output.ok or not target.exists():
            hint = ""
            if "could not create image" in output.stderr:
                hint = (
                    " — grant Screen Recording permission to Jarvis/your terminal in "
                    "System Settings → Privacy & Security → Screen Recording"
                )
            return ToolResult.failure(
                self.name, f"screencapture failed: {output.combined()}{hint}"
            )
        return ToolResult(
            tool=self.name, ok=True, summary=f"Screenshot saved to {target}",
            data={"path": str(target)},
        )


class MediaArgs(BaseModel):
    action: Literal["play", "pause", "next", "previous"] = Field(
        description="Media action. Use 'pause' for stop/pause/quiet, 'play' "
        "for play/resume/continue, 'next' to skip, 'previous' to go back."
    )


class MediaTool(Tool):
    name: ClassVar[str] = "media_control"
    description: ClassVar[str] = (
        "Control music playback in Music or Spotify: play, pause (also for "
        "'stop'), skip to next, or go to the previous track."
    )
    args_model: ClassVar[type[BaseModel]] = MediaArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    # Explicit verbs, never the `playpause` toggle: "pause then stop" must not
    # flip playback back on. Each verb is idempotent for its intent.
    _ACTIONS: ClassVar[dict[str, str]] = {
        "play": "play",
        "pause": "pause",
        "next": "next track",
        "previous": "previous track",
    }

    async def _running(self, app: str) -> bool:
        output = await run_osascript(f"application {applescript_quote(app)} is running")
        return output.ok and output.stdout.strip() == "true"

    async def run(self, args: MediaArgs) -> ToolResult:  # type: ignore[override]
        verb = self._ACTIONS[args.action]
        for player in ("Spotify", "Music"):
            if not await self._running(player):
                continue
            quoted = applescript_quote(player)
            output = await run_osascript(f"tell application {quoted} to {verb}")
            if not output.ok:
                return ToolResult.failure(
                    self.name, f"could not control {player}: {output.combined()}"
                )
            # Read the real resulting state AND current track, so the reply is
            # grounded in what actually happened — "next" that didn't advance
            # can't be reported as success, and the model can't invent a title.
            state_out = await run_osascript(f"tell application {quoted} to player state")
            state = state_out.stdout.strip() or "unknown"
            track_out = await run_osascript(
                f'tell application {quoted} to name of current track'
            )
            artist_out = await run_osascript(
                f'tell application {quoted} to artist of current track'
            )
            track = track_out.stdout.strip() if track_out.ok else ""
            artist = artist_out.stdout.strip() if artist_out.ok else ""
            now = f"'{track}'" + (f" by {artist}" if artist else "") if track else "nothing"
            verb_past = {"play": "playing", "pause": "paused",
                         "next": "skipped to", "previous": "went back to"}[args.action]
            summary = (
                f"{player} is {state}."
                if args.action in ("play", "pause")
                else f"{player} {verb_past} {now} ({state})."
            )
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=summary,
                data={"player": player, "state": state, "track": track, "artist": artist},
            )
        return ToolResult.failure(self.name, "Neither Music nor Spotify is running.")


class WindowArgs(BaseModel):
    app: str = Field(description="Application whose front window to arrange, e.g. 'Safari'.")
    position: Literal["left_half", "right_half", "maximize", "center"] = Field(
        description="Where to place the window."
    )


class WindowArrangeTool(Tool):
    name: ClassVar[str] = "window_arrange"
    description: ClassVar[str] = (
        "Move/resize an app's front window: left half, right half, maximize, or center."
    )
    args_model: ClassVar[type[BaseModel]] = WindowArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: WindowArgs) -> ToolResult:  # type: ignore[override]
        bounds_output = await run_osascript(
            'tell application "Finder" to get bounds of window of desktop'
        )
        if not bounds_output.ok:
            return ToolResult.failure(self.name, bounds_output.combined())
        try:
            _x, _y, width, height = (int(v.strip()) for v in bounds_output.stdout.split(","))
        except ValueError:
            return ToolResult.failure(
                self.name, f"could not parse screen bounds: {bounds_output.stdout!r}"
            )

        frames = {
            "left_half": (0, 25, width // 2, height - 25),
            "right_half": (width // 2, 25, width // 2, height - 25),
            "maximize": (0, 25, width, height - 25),
            "center": (width // 6, height // 8, width * 2 // 3, height * 3 // 4),
        }
        x, y, w, h = frames[args.position]
        app_name = applescript_quote(args.app)
        script = (
            f'tell application "System Events" to tell process {app_name}\n'
            f"  set position of front window to {{{x}, {y}}}\n"
            f"  set size of front window to {{{w}, {h}}}\n"
            "end tell"
        )
        output = await run_osascript(script)
        if not output.ok:
            hint = ""
            if "assistive access" in output.stderr.lower() or "1002" in output.stderr:
                hint = (
                    " (grant Accessibility permission to your terminal/Jarvis in "
                    "System Settings → Privacy & Security → Accessibility)"
                )
            return ToolResult.failure(self.name, f"could not arrange window: "
                                                 f"{output.combined()}{hint}")
        return ToolResult(
            tool=self.name, ok=True,
            summary=f"Placed {args.app}'s front window at {args.position}",
        )


class BrightnessArgs(BaseModel):
    level: float = Field(ge=0.0, le=1.0, description="Screen brightness, 0.0-1.0.")


class BrightnessTool(Tool):
    name: ClassVar[str] = "brightness"
    description: ClassVar[str] = "Set the display brightness (0.0 to 1.0)."
    args_model: ClassVar[type[BaseModel]] = BrightnessArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: BrightnessArgs) -> ToolResult:  # type: ignore[override]
        binary = shutil.which("brightness")
        if binary is None:
            return ToolResult.failure(
                self.name,
                "The 'brightness' CLI is not installed. Install it with "
                "`brew install brightness` to enable this.",
            )
        output = await run_command([binary, str(args.level)])
        if not output.ok:
            return ToolResult.failure(self.name, output.combined())
        return ToolResult(
            tool=self.name, ok=True, summary=f"Brightness set to {args.level:.0%}"
        )
