"""GitHub tools: project resolution against real local repos, smart push workflows.

No fabricated URLs anywhere here: every repo used in these tests is a real
`git init` fixture with a real `origin` remote, matching how ProjectRegistry
actually works in production.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.model_manager import ModelManager
from app.core.project_registry import ProjectRegistry
from app.tools import github as github_module
from app.tools._common import CommandOutput
from app.tools.github import (
    DeleteRepoArgs,
    GitHubDeleteRepoTool,
    GitHubOpenRepoTool,
    GitHubPushTool,
    LocateProjectArgs,
    LocateProjectTool,
    OpenRepoArgs,
    PushChangesArgs,
    RefreshProjectsTool,
    _resolve_project,
)
from tests.conftest import FakeOllamaClient


def _make_repo(root: Path, name: str, remote: str | None) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    if remote is not None:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=repo, check=True)
    return repo


async def test_resolve_project_keyword_match(tmp_path: Path, settings: Settings) -> None:
    _make_repo(tmp_path, "skin_analyser", "git@github.com:mohan/skin-analyser.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    result = await _resolve_project("open skin", registry, fake, manager, settings)

    assert result is not None
    assert result.name == "skin_analyser"
    # Keyword match found it — LLM was never consulted.
    assert ("chat", settings.llm_model) not in fake.calls


async def test_resolve_project_llm_fallback(tmp_path: Path, settings: Settings) -> None:
    _make_repo(tmp_path, "jarvis_v2", "https://github.com/mohan/Personal_Assistant.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("jarvis_v2")
    manager = ModelManager(fake, settings)

    result = await _resolve_project(
        "the thing i use for talking to my computer", registry, fake, manager, settings
    )

    assert result is not None
    assert result.name == "jarvis_v2"
    assert ("chat", settings.llm_model) in fake.calls


async def test_resolve_project_not_found(tmp_path: Path, settings: Settings) -> None:
    _make_repo(tmp_path, "jarvis_v2", "https://github.com/mohan/Personal_Assistant.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("unknown")
    manager = ModelManager(fake, settings)

    result = await _resolve_project("completely unrelated banana", registry, fake, manager, settings)

    assert result is None


async def test_open_repo_uses_real_remote(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL that gets opened is the repo's real 'origin', not a guess."""
    _make_repo(tmp_path, "skin_analyser", "git@github.com:mohan/skin-analyser.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    opened_urls: list[str] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        if cmd[0] == "open":
            opened_urls.append(cmd[1])
            return CommandOutput(0, "", "")
        return CommandOutput(1, "", "unexpected command in this test")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubOpenRepoTool(registry, fake, manager, settings)
    result = await tool.run(OpenRepoArgs(project="skin"))

    assert result.ok, result.summary
    assert opened_urls == ["https://github.com/mohan/skin-analyser"]


async def test_open_repo_not_found_lists_known_projects(
    tmp_path: Path, settings: Settings
) -> None:
    _make_repo(tmp_path, "jarvis_v2", "https://github.com/mohan/Personal_Assistant.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("unknown")
    manager = ModelManager(fake, settings)

    tool = GitHubOpenRepoTool(registry, fake, manager, settings)
    result = await tool.run(OpenRepoArgs(project="nonexistent thing"))

    assert not result.ok
    assert "jarvis_v2" in result.summary


async def test_open_repo_no_remote_configured(tmp_path: Path, settings: Settings) -> None:
    """A local repo with no 'origin' must fail honestly, never open a fake URL."""
    _make_repo(tmp_path, "fresh_project", remote=None)
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = GitHubOpenRepoTool(registry, fake, manager, settings)
    result = await tool.run(OpenRepoArgs(project="fresh_project"))

    assert not result.ok
    assert "no GitHub remote" in result.summary


async def test_open_repo_reports_a_deleted_remote_instead_of_pretending_it_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo's local git remote survives it being deleted on GitHub — the
    tool must actually check, not just fire `open` at a stale URL and claim
    success."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    async def fake_exists(remote_url: str, token: str | None) -> bool | None:
        return False  # simulates a 404 from the GitHub API

    monkeypatch.setattr(github_module, "_repo_exists_on_github", fake_exists)
    opened: list[str] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        if cmd[0] == "open":
            opened.append(cmd[1])
        return CommandOutput(0, "", "")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubOpenRepoTool(registry, fake, manager, settings)
    result = await tool.run(OpenRepoArgs(project="fitness"))

    assert not result.ok
    assert "no longer exists on github" in result.summary.lower()
    assert opened == []  # never opened a dead link


async def test_open_repo_proceeds_when_existence_cant_be_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No token (or a network hiccup) must degrade to the old best-effort
    behaviour, not block every open with an unrelated failure."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token=None)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    opened: list[str] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        if cmd[0] == "open":
            opened.append(cmd[1])
        return CommandOutput(0, "", "")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubOpenRepoTool(registry, fake, manager, settings)
    result = await tool.run(OpenRepoArgs(project="fitness"))

    assert result.ok, result.summary
    assert opened == ["https://github.com/mohan/fitness-app"]


async def test_locate_project_reports_a_deleted_remote_instead_of_pretending_it_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    async def fake_exists(remote_url: str, token: str | None) -> bool | None:
        return False

    monkeypatch.setattr(github_module, "_repo_exists_on_github", fake_exists)

    tool = LocateProjectTool(registry, fake, manager, settings)
    result = await tool.run(LocateProjectArgs(project="fitness"))

    assert result.ok, result.summary  # the project itself is still found locally
    assert "no longer exists on github" in result.summary.lower()


class _FakeClient:
    """A minimal httpx.AsyncClient stand-in: async-with, one .get() call."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None):
        class _Response:
            status_code = self._status_code

        return _Response()


async def test_repo_exists_on_github_returns_false_for_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        github_module.httpx, "AsyncClient", lambda timeout=None: _FakeClient(404)
    )
    result = await github_module._repo_exists_on_github(
        "https://github.com/mohan/gone", "fake_token"
    )
    assert result is False


async def test_repo_exists_on_github_returns_true_for_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        github_module.httpx, "AsyncClient", lambda timeout=None: _FakeClient(200)
    )
    result = await github_module._repo_exists_on_github(
        "https://github.com/mohan/alive", "fake_token"
    )
    assert result is True


async def test_repo_exists_on_github_returns_none_without_token() -> None:
    result = await github_module._repo_exists_on_github(
        "https://github.com/mohan/whatever", None
    )
    assert result is None


async def test_push_no_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Initialize a real git repo for this test
    _make_repo(tmp_path, "repo", "https://github.com/mohan/repo.git")
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    async def mock_run(cmd, cwd=None, timeout=30.0):
        if "status" in cmd:
            return CommandOutput(0, "", "")  # clean tree
        return CommandOutput(0, "", "")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="repo"))

    assert result.ok
    assert "No changes" in result.summary


async def test_push_recreates_a_deleted_remote_even_with_a_clean_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original bug: a project with a clean working tree but a remote
    that's been deleted on GitHub must NOT short-circuit as 'no changes' —
    it needs the repo recreated and its existing local history pushed."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    async def fake_exists(remote_url: str, token: str | None) -> bool | None:
        return False  # simulates the repo having been deleted on GitHub

    monkeypatch.setattr(github_module, "_repo_exists_on_github", fake_exists)

    created_repos: list[str] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        if cmd[0] == "curl":
            created_repos.append("called")
            return CommandOutput(0, '{"id": 1, "name": "fitness-app"}', "")
        if "config" in cmd and "remote.origin.url" in cmd:
            return CommandOutput(0, "https://github.com/mohan/fitness-app.git\n", "")
        if "status" in cmd:
            return CommandOutput(0, "", "")  # clean tree
        if "rev-parse" in cmd:
            return CommandOutput(0, "main\n", "")
        if cmd[0] == "open":
            return CommandOutput(0, "", "")
        return CommandOutput(0, "", "")  # add/commit/pull/push all succeed

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="fitness"))

    assert result.ok, result.summary
    assert created_repos == ["called"]  # the repo WAS recreated
    assert "no changes" not in result.summary.lower()  # never short-circuited


async def test_push_executes_full_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_repo(tmp_path, "repo", "https://github.com/mohan/repo.git")
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("Fix bug in app.py")
    manager = ModelManager(fake, settings)

    calls: list[list[str]] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        calls.append(cmd)
        if "status" in cmd:
            return CommandOutput(0, "M app.py\n", "")
        if "add" in cmd:
            return CommandOutput(0, "", "")
        if "diff" in cmd:
            return CommandOutput(0, "diff --git a/app.py b/app.py\n+fixed the bug\n", "")
        if "commit" in cmd:
            return CommandOutput(0, "", "")
        if "push" in cmd:
            return CommandOutput(0, "", "")
        if "rev-parse" in cmd:
            return CommandOutput(0, "main\n", "")
        if "config" in cmd:
            return CommandOutput(0, "https://github.com/mohan/repo.git\n", "")
        if cmd[0] == "open":
            return CommandOutput(0, "", "")
        return CommandOutput(1, "", "unexpected: " + " ".join(cmd))

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="repo"))

    assert result.ok, result.summary
    assert "Fix bug in app.py" in result.summary
    cmd_kinds = [c[1] if len(c) > 1 else c[0] for c in calls]
    add_idx = cmd_kinds.index("add")
    commit_idx = cmd_kinds.index("commit")
    push_idx = cmd_kinds.index("push")
    assert add_idx < commit_idx < push_idx


async def test_push_unknown_project_fails_honestly(
    tmp_path: Path, settings: Settings
) -> None:
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("unknown")
    manager = ModelManager(fake, settings)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="nonexistent"))

    assert not result.ok
    assert "Could not find" in result.summary


async def test_locate_project_returns_real_path_and_remote(
    tmp_path: Path, settings: Settings
) -> None:
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = LocateProjectTool(registry, fake, manager, settings)
    result = await tool.run(LocateProjectArgs(project="give me the folder path for fitness"))

    assert result.ok, result.summary
    assert result.data["path"] == str(tmp_path / "fitness")
    assert result.data["is_git"] is True
    assert result.data["remote_url"] == "https://github.com/mohan/fitness-app"


async def test_locate_project_plain_folder_no_git(tmp_path: Path) -> None:
    (tmp_path / "newthing").mkdir()
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("unknown")  # LLM fallback finds nothing
    manager = ModelManager(fake, settings)

    tool = LocateProjectTool(registry, fake, manager, settings)
    result = await tool.run(LocateProjectArgs(project="newthing"))

    assert result.ok, result.summary
    assert result.data["is_git"] is False
    assert result.data["path"] == str(tmp_path / "newthing")


async def test_locate_project_not_found(tmp_path: Path, settings: Settings) -> None:
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("unknown")
    manager = ModelManager(fake, settings)

    tool = LocateProjectTool(registry, fake, manager, settings)
    result = await tool.run(LocateProjectArgs(project="nonexistent-xyz"))

    assert not result.ok
    assert "couldn't find" in result.summary.lower()


async def test_refresh_projects_reports_count(tmp_path: Path) -> None:
    _make_repo(tmp_path, "skin_analyser", "git@github.com:mohan/skin-analyser.git")
    _make_repo(tmp_path, "jarvis_v2", "https://github.com/mohan/Personal_Assistant.git")
    registry = ProjectRegistry(tmp_path)

    tool = RefreshProjectsTool(registry)
    result = await tool.run(github_module.RefreshProjectsArgs())

    assert result.ok
    assert "2" in result.summary
    assert "skin_analyser" in result.summary and "jarvis_v2" in result.summary


async def test_push_creates_repo_if_no_git_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a project folder exists but has no .git, git init + remote add + push."""
    folder = tmp_path / "new_project"
    folder.mkdir()
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    fake = FakeOllamaClient()
    fake.queued.append("Add new feature")
    manager = ModelManager(fake, settings)

    calls: list[list[str]] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        calls.append(cmd)
        if "init" in cmd:
            return CommandOutput(0, "", "")
        if "remote" in cmd:
            return CommandOutput(0, "", "")
        if "status" in cmd:
            return CommandOutput(0, "M file.py\n", "")
        if "add" in cmd:
            return CommandOutput(0, "", "")
        if "diff" in cmd:
            return CommandOutput(0, "diff --git a/file.py b/file.py\n+new feature\n", "")
        if "commit" in cmd:
            return CommandOutput(0, "", "")
        if "push" in cmd:
            return CommandOutput(0, "", "")
        if "rev-parse" in cmd:
            return CommandOutput(0, "main\n", "")
        if "config" in cmd:
            return CommandOutput(0, "https://github.com/mohan/new_project.git\n", "")
        if cmd[0] == "open":
            return CommandOutput(0, "", "")
        return CommandOutput(1, "", "unexpected: " + " ".join(cmd))

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="new_project", repo_name="new_project"))

    assert result.ok, result.summary
    # Verify git init and remote add were called
    cmd_kinds = [" ".join(c[:2]) for c in calls]
    assert "git init" in cmd_kinds
    assert "git remote" in cmd_kinds


async def test_push_requires_repo_name_if_no_git(tmp_path: Path) -> None:
    """If no .git and no repo_name provided, fail with helpful message."""
    folder = tmp_path / "new_project"
    folder.mkdir()
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(project="new_project"))

    assert not result.ok
    assert "repo name" in result.summary.lower()
    assert "git repo found" in result.summary.lower()


async def test_push_requires_a_project_even_with_repo_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """repo_name alone, with no project named, must fail clearly rather than
    bootstrapping a folder or touching anything on disk."""
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    calls: list[list[str]] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        calls.append(cmd)
        return CommandOutput(1, "", "should not be called")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs(repo_name="test"))

    assert not result.ok
    assert "which project" in result.summary.lower()
    assert calls == []  # no git command ever ran
    assert not (tmp_path / "test").exists()  # no folder was created


async def test_push_never_falls_back_to_the_servers_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither project nor repo_name given: must fail clearly, never touch
    Path.cwd() (the backend server's own directory, which has no .gitignore
    and would leak .env if `git add .` ran there)."""
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    calls: list[list[str]] = []

    async def mock_run(cmd, cwd=None, timeout=30.0):
        calls.append(cmd)
        return CommandOutput(1, "", "should not be called")

    monkeypatch.setattr(github_module, "run_command", mock_run)

    tool = GitHubPushTool(registry, fake, manager, settings)
    result = await tool.run(PushChangesArgs())

    assert not result.ok
    assert calls == []  # no git command ever ran


async def test_delete_repo_requires_github_token(tmp_path: Path) -> None:
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token=None)
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    result = await tool.run(DeleteRepoArgs(project="fitness"))

    assert not result.ok
    assert "token" in result.summary.lower()


async def test_delete_repo_requires_github_remote(tmp_path: Path) -> None:
    _make_repo(tmp_path, "fitness", remote=None)
    registry = ProjectRegistry(tmp_path)
    settings = Settings(
        _env_file=None, projects_dir=tmp_path, github_token="fake_token_123"
    )
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    result = await tool.run(DeleteRepoArgs(project="fitness"))

    assert not result.ok
    assert "no github remote" in result.summary.lower()


class _DeleteVerifyClient:
    """httpx.AsyncClient stand-in covering both the DELETE call and the
    follow-up existence-check GETs, sharing state across the multiple
    `AsyncClient()` instances the real code creates (one per call)."""

    def __init__(self, delete_status: int, get_sequence: list[int]) -> None:
        self._delete_status = delete_status
        self._get_sequence = get_sequence  # shared, mutated in place

    async def __aenter__(self) -> "_DeleteVerifyClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def delete(self, url: str, headers: dict[str, str] | None = None):
        class _Resp:
            status_code = self._delete_status
            text = ""

        return _Resp()

    async def get(self, url: str, headers: dict[str, str] | None = None):
        code = self._get_sequence.pop(0) if self._get_sequence else 404

        class _Resp:
            status_code = code

        return _Resp()


async def test_delete_repo_verifies_before_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original bug: GitHub's DELETE answered 2xx, but a GET moments
    later still showed the repo as existing (propagation delay) — the tool
    must poll until it actually confirms gone, never claim success on the
    2xx alone."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    real_sleep = asyncio.sleep
    monkeypatch.setattr(github_module.asyncio, "sleep", lambda *_: real_sleep(0))
    get_sequence = [200, 200, 404]  # still exists twice, then confirmed gone
    monkeypatch.setattr(
        github_module.httpx, "AsyncClient",
        lambda timeout=None: _DeleteVerifyClient(204, get_sequence),
    )

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    result = await tool.run(DeleteRepoArgs(project="fitness"))

    assert result.ok, result.summary
    assert "deleted" in result.summary.lower()
    assert get_sequence == []  # all three checks were consumed


async def test_delete_repo_reports_pending_if_never_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If GitHub never actually confirms the repo is gone within the retry
    budget, the tool must say so plainly rather than asserting 'Deleted'."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(_env_file=None, projects_dir=tmp_path, github_token="fake_token")
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    real_sleep = asyncio.sleep
    monkeypatch.setattr(github_module.asyncio, "sleep", lambda *_: real_sleep(0))
    get_sequence = [200, 200, 200, 200, 200]  # never confirms gone
    monkeypatch.setattr(
        github_module.httpx, "AsyncClient",
        lambda timeout=None: _DeleteVerifyClient(204, get_sequence),
    )

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    result = await tool.run(DeleteRepoArgs(project="fitness"))

    assert not result.ok
    assert "still shows as existing" in result.summary.lower()


async def test_delete_repo_confirmation_preview(tmp_path: Path) -> None:
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    settings = Settings(
        _env_file=None, projects_dir=tmp_path, github_token="fake_token"
    )
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    preview = tool.confirmation_action(DeleteRepoArgs(project="fitness"))

    assert preview is not None
    assert "delete" in preview.lower()
    assert "fitness" in preview.lower()
    assert "cannot be undone" in preview.lower()


async def test_delete_repo_confirmation_opens_browser_when_cache_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the registry has already been scanned (the common case — the user
    typically located/pushed the project earlier), confirmation_action must
    open the repo in the browser so the user can verify it before approving."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)
    await registry.refresh()  # warm the cache, as a prior tool call would
    settings = Settings(
        _env_file=None, projects_dir=tmp_path, github_token="fake_token"
    )
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    opened: list[list[str]] = []
    monkeypatch.setattr(
        github_module.subprocess, "Popen", lambda argv: opened.append(argv)
    )

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    preview = tool.confirmation_action(DeleteRepoArgs(project="fitness"))

    assert opened == [["open", "https://github.com/mohan/fitness-app"]]
    assert preview is not None
    assert "https://github.com/mohan/fitness-app" in preview
    assert "cannot be undone" in preview.lower()


async def test_delete_repo_confirmation_no_browser_when_cache_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before any scan has happened, find_cached can't know about the repo —
    confirmation must degrade to the plain text, never raise or hang."""
    _make_repo(tmp_path, "fitness", "https://github.com/mohan/fitness-app.git")
    registry = ProjectRegistry(tmp_path)  # never refreshed
    settings = Settings(
        _env_file=None, projects_dir=tmp_path, github_token="fake_token"
    )
    fake = FakeOllamaClient()
    manager = ModelManager(fake, settings)

    opened: list[list[str]] = []
    monkeypatch.setattr(
        github_module.subprocess, "Popen", lambda argv: opened.append(argv)
    )

    tool = GitHubDeleteRepoTool(registry, fake, manager, settings)
    preview = tool.confirmation_action(DeleteRepoArgs(project="fitness"))

    assert opened == []
    assert preview == "Delete the GitHub repository for 'fitness'? This cannot be undone."
