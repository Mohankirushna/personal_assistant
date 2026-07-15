"""Git tool against a throwaway repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.planner.schemas import RiskLevel
from app.tools._common import run_command
from app.tools.git.git import GitTool, classify_git


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("status", RiskLevel.SAFE),
        ("log --oneline", RiskLevel.SAFE),
        ("diff HEAD~1", RiskLevel.SAFE),
        ("branch", RiskLevel.SAFE),
        ("stash list", RiskLevel.SAFE),
        ("commit -m 'x'", RiskLevel.SENSITIVE),
        ("checkout -b feature", RiskLevel.SENSITIVE),
        ("pull", RiskLevel.SENSITIVE),
        ("stash pop", RiskLevel.SENSITIVE),
        ("push --force", RiskLevel.DESTRUCTIVE),
        ("push -f origin main", RiskLevel.DESTRUCTIVE),
        ("reset --hard HEAD~1", RiskLevel.DESTRUCTIVE),
        ("clean -fd", RiskLevel.DESTRUCTIVE),
        ("branch -D old", RiskLevel.DESTRUCTIVE),
        ("rebase main", RiskLevel.DESTRUCTIVE),
        ("rebase --abort", RiskLevel.SENSITIVE),
    ],
)
def test_classification(arguments: str, expected: RiskLevel) -> None:
    assert classify_git(arguments) is expected


@pytest.fixture
async def repo(tmp_path: Path) -> Path:
    await run_command(["git", "init", "-q"], cwd=tmp_path)
    await run_command(["git", "config", "user.email", "test@jarvis.local"], cwd=tmp_path)
    await run_command(["git", "config", "user.name", "Jarvis Test"], cwd=tmp_path)
    (tmp_path / "README.md").write_text("# test\n")
    return tmp_path


async def test_status_and_commit_flow(repo: Path) -> None:
    tool = GitTool()
    status = await tool.execute({"arguments": "status --short", "repo": str(repo)})
    assert status.ok
    assert "README.md" in status.summary

    add = await tool.execute({"arguments": "add README.md", "repo": str(repo)})
    assert add.ok

    commit = await tool.execute(
        {"arguments": "commit -m 'initial commit'", "repo": str(repo)}
    )
    assert commit.ok, commit.summary

    log = await tool.execute({"arguments": "log --oneline", "repo": str(repo)})
    assert log.ok
    assert "initial commit" in log.summary


async def test_rejects_non_repo(tmp_path: Path) -> None:
    result = await GitTool().execute({"arguments": "status", "repo": str(tmp_path)})
    assert not result.ok
    assert "not a git repository" in result.summary
