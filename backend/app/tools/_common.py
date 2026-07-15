"""Shared helpers for tool implementations.

Underscore-prefixed so the registry's discovery walk skips this module.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CommandOutput:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def combined(self, limit: int = 4000) -> str:
        """stdout+stderr trimmed to a size a 3B model can digest."""
        text = self.stdout
        if self.stderr:
            text = f"{text}\n[stderr] {self.stderr}" if text else f"[stderr] {self.stderr}"
        text = text.strip()
        if len(text) > limit:
            text = text[:limit] + f"\n… (truncated, {len(text)} chars total)"
        return text


async def run_command(
    argv: list[str],
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CommandOutput:
    """Run a program (no shell) and capture output."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandOutput(-1, "", f"timed out after {timeout:.0f}s")
    return CommandOutput(
        process.returncode if process.returncode is not None else -1,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def run_shell(
    command: str,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CommandOutput:
    """Run a command through the shell (for the terminal tool, where the
    user's command genuinely is shell syntax)."""
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        executable="/bin/zsh",
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandOutput(-1, "", f"timed out after {timeout:.0f}s")
    return CommandOutput(
        process.returncode if process.returncode is not None else -1,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def run_osascript(script: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> CommandOutput:
    """Run an AppleScript snippet."""
    return await run_command(["/usr/bin/osascript", "-e", script], timeout=timeout)


def applescript_quote(value: str) -> str:
    """Quote a string for safe embedding in an AppleScript literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def expand_path(raw: str) -> Path:
    """~ and env expansion; callers get an absolute Path."""
    return Path(raw).expanduser().resolve()
