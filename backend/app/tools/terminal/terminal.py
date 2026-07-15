"""Terminal tool: run shell commands with a risk-classification policy.

Three tiers, decided by `classify_command` BEFORE execution:
  - SAFE: an allowlist of clearly read-only commands (ls, cat, git status,
    docker ps, …) — run immediately.
  - DESTRUCTIVE: pattern-matched dangerous commands (rm, sudo, mkfs,
    diskutil erase, uninstalls, force-push, …) — always require confirmation.
  - SENSITIVE: everything else — confirmed once per exact command per
    session.

The command string the user approves is the exact string executed.
"""

from __future__ import annotations

import re
import shlex
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import expand_path, run_shell
from app.tools.base import Tool

# Commands that only read state. First token (after env assignments) matched.
_SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "pwd", "echo", "which", "whoami", "date",
    "df", "du", "ps", "top", "uname", "uptime", "wc", "file", "stat",
    "printenv", "env", "hostname", "sw_vers", "system_profiler",
}
# Read-only subcommands of common multi-tools: "git status", "docker ps", …
_SAFE_SUBCOMMANDS = {
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "show"),
    ("git", "branch"), ("git", "remote"),
    ("docker", "ps"), ("docker", "images"), ("docker", "version"),
    ("npm", "ls"), ("npm", "list"), ("npm", "outdated"), ("npm", "--version"),
    ("yarn", "list"), ("brew", "list"), ("brew", "info"), ("brew", "outdated"),
    ("pip", "list"), ("pip", "show"), ("uv", "tree"),
}

_DESTRUCTIVE_PATTERNS = [
    r"\brm\b", r"\brmdir\b", r"\bshred\b", r"\bmkfs\b", r"\bdd\b",
    r"\bdiskutil\s+(erase|partition|apfs\s+delete)", r">\s*/dev/",
    r"\bsudo\b", r"\bkillall\b", r"\bkill\s+-9\b",
    r"\bbrew\s+(uninstall|remove|purge)\b", r"\bnpm\s+(uninstall|rm)\b",
    r"\byarn\s+remove\b", r"\bpip\s+uninstall\b", r"\buv\s+remove\b",
    r"\bgit\s+push\s+.*--force", r"\bgit\s+reset\s+--hard", r"\bgit\s+clean\b",
    r"\bchmod\s+-R\b", r"\bchown\s+-R\b", r"\btruncate\b",
    r"\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm)\b",
    r"\blaunchctl\s+(unload|remove)\b", r"\bdefaults\s+delete\b",
    r"\bcurl\b.*\|\s*(ba|z)?sh", r"\bwget\b.*\|\s*(ba|z)?sh",
]
_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_PATTERNS), re.IGNORECASE)


def classify_command(command: str) -> RiskLevel:
    if _DESTRUCTIVE_RE.search(command):
        return RiskLevel.DESTRUCTIVE
    try:
        tokens = shlex.split(command)
    except ValueError:
        return RiskLevel.SENSITIVE
    # Skip leading VAR=value assignments.
    while tokens and "=" in tokens[0] and not tokens[0].startswith("="):
        tokens = tokens[1:]
    if not tokens:
        return RiskLevel.SENSITIVE
    # Pipelines/compound commands never qualify as SAFE.
    if any(token in {"|", "||", "&&", ";", ">", ">>", "<"} for token in tokens):
        return RiskLevel.SENSITIVE
    if tokens[0] in _SAFE_COMMANDS:
        return RiskLevel.SAFE
    if len(tokens) >= 2 and (tokens[0], tokens[1]) in _SAFE_SUBCOMMANDS:
        return RiskLevel.SAFE
    return RiskLevel.SENSITIVE


class TerminalArgs(BaseModel):
    command: str = Field(description="The shell command to run, e.g. 'ls -la ~/Downloads'.")
    cwd: str | None = Field(default=None, description="Working directory; default is home.")


class TerminalTool(Tool):
    name: ClassVar[str] = "terminal_run"
    description: ClassVar[str] = (
        "Run a shell command (zsh) and return its output — python, git, npm, "
        "yarn, brew, docker, anything on PATH. Dangerous commands require "
        "user confirmation."
    )
    args_model: ClassVar[type[BaseModel]] = TerminalArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def assess_risk(self, args: BaseModel) -> RiskLevel:
        assert isinstance(args, TerminalArgs)
        return classify_command(args.command)

    async def run(self, args: TerminalArgs) -> ToolResult:  # type: ignore[override]
        cwd = expand_path(args.cwd) if args.cwd else expand_path("~")
        if not cwd.is_dir():
            return ToolResult.failure(self.name, f"cwd {cwd} is not a directory")
        output = await run_shell(args.command, cwd=cwd, timeout=60)
        body = output.combined() or "(no output)"
        if output.ok:
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=f"$ {args.command}\n{body}",
                data={"returncode": output.returncode},
            )
        return ToolResult(
            tool=self.name,
            ok=False,
            summary=f"$ {args.command}\nexited {output.returncode}\n{body}",
            data={"returncode": output.returncode},
        )
