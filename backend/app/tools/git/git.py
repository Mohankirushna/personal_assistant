"""Git tool: run git operations in a repository with per-subcommand risk.

Read-only subcommands (status, log, diff, …) are SAFE; state-changing ones
(commit, checkout, pull, …) are SENSITIVE; history-destroying ones
(push --force, reset --hard, clean) are DESTRUCTIVE.
"""

from __future__ import annotations

import re
import shlex
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import expand_path, run_command
from app.tools.base import Tool

_SAFE_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "tag", "describe",
    "blame", "shortlog", "rev-parse", "ls-files", "stash",  # bare stash list via args check below
}
_DESTRUCTIVE_RE = re.compile(
    r"(push\s+.*(--force|-f)\b)|(reset\s+--hard)|(\bclean\b)|(branch\s+-D)|"
    r"(rebase(?!\s+--(abort|continue)))|(filter-branch)|(reflog\s+expire)",
    re.IGNORECASE,
)


def classify_git(arguments: str) -> RiskLevel:
    if _DESTRUCTIVE_RE.search(arguments):
        return RiskLevel.DESTRUCTIVE
    try:
        tokens = shlex.split(arguments)
    except ValueError:
        return RiskLevel.SENSITIVE
    if not tokens:
        return RiskLevel.SENSITIVE
    subcommand = tokens[0]
    if subcommand == "stash" and len(tokens) > 1 and tokens[1] not in ("list", "show"):
        return RiskLevel.SENSITIVE
    if subcommand in _SAFE_SUBCOMMANDS:
        return RiskLevel.SAFE
    return RiskLevel.SENSITIVE


class GitArgs(BaseModel):
    arguments: str = Field(
        description="Arguments after 'git', e.g. 'status', 'log --oneline -5', "
        "'commit -m \"message\"'."
    )
    repo: str = Field(description="Path to the repository, e.g. '~/projects/myapp'.")


class GitTool(Tool):
    name: ClassVar[str] = "git"
    description: ClassVar[str] = (
        "Run a git command in a repository: status, log, diff, add, commit, "
        "branch, checkout, pull, push, and so on."
    )
    args_model: ClassVar[type[BaseModel]] = GitArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def assess_risk(self, args: BaseModel) -> RiskLevel:
        assert isinstance(args, GitArgs)
        return classify_git(args.arguments)

    async def run(self, args: GitArgs) -> ToolResult:  # type: ignore[override]
        repo = expand_path(args.repo)
        if not (repo / ".git").exists():
            return ToolResult.failure(self.name, f"{repo} is not a git repository")
        try:
            argv = ["git", *shlex.split(args.arguments)]
        except ValueError as exc:
            return ToolResult.failure(self.name, f"could not parse arguments: {exc}")
        output = await run_command(argv, cwd=repo, timeout=60)
        body = output.combined() or "(no output)"
        return ToolResult(
            tool=self.name,
            ok=output.ok,
            summary=f"$ git {args.arguments}\n{body}",
            data={"returncode": output.returncode},
        )
