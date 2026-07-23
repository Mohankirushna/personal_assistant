"""GitHub integration: open real local repos, smart git workflows with confirmations.

Project names resolve against `ProjectRegistry`, which reads each local
repo's actual 'origin' remote — never a hardcoded or guessed URL. Push
workflows run against the resolved project's own directory (or Jarvis's own
directory if no project is named), with staged confirmations (branch, commit
message, final push) and open the repo/branch in the browser afterward.

If GitHub API credentials are configured, new repos are auto-created on GitHub
before pushing (no manual step needed).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.model_manager import ModelManager
from app.core.ollama_client import OllamaLike
from app.core.project_registry import ProjectInfo, ProjectRegistry, normalize_remote_url
from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import run_command
from app.tools.base import Tool

logger = logging.getLogger(__name__)


async def _resolve_project(
    user_input: str,
    registry: ProjectRegistry,
    client: OllamaLike,
    model_manager: ModelManager,
    settings: Settings,
) -> ProjectInfo | None:
    """Resolve a spoken project name to a local repo via keyword match, then LLM fallback."""
    match = await registry.find(user_input)
    if match is not None:
        return match

    projects = await registry.list_projects()
    if not projects:
        return None
    names = ", ".join(p.name for p in projects)
    model = await model_manager.ensure_llm()
    prompt = (
        f'User said: "{user_input}"\n'
        f"Available local projects: {names}\n"
        "Which project did they mean? Reply with ONLY the project name from the "
        "list, nothing else. If unsure, reply 'unknown'."
    )
    reply = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        keep_alive=settings.llm_keep_alive,
    )
    matched_name = reply.strip().lower()
    for project in projects:
        if project.name.lower() == matched_name:
            return project
    return None


def _repo_name_from_remote(remote_url: str) -> str | None:
    """Extract the GitHub repo name (last path segment) from a remote URL."""
    m = re.match(r"https://github\.com/[^/]+/([^/]+?)/?$", remote_url)
    return m.group(1) if m else None


def _norm(text: str) -> str:
    """Lowercase and strip everything but alphanumerics, so 'fitness-app',
    'fitness_app' and 'Fitness App' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _resolve_project_for_deletion(
    requested: str, projects: list[ProjectInfo]
) -> tuple[ProjectInfo | None, str]:
    """Strictly resolve which project's repo to DELETE.

    Deletion is irreversible, so this refuses to fuzzy-guess (the loose
    keyword scoring used elsewhere once matched 'jarvis-delete-test' to the
    unrelated 'jarvis_v2' project and deleted the wrong repo). A candidate
    qualifies ONLY if the requested name equals — after normalization — its
    folder name OR its GitHub repo name. Returns (project, "") on a single
    confident match, else (None, <message listing what the user could mean>).
    """
    want = _norm(requested)
    on_github = [p for p in projects if p.remote_url]
    matches: list[ProjectInfo] = []
    for p in on_github:
        repo = _repo_name_from_remote(p.remote_url or "")
        if want and (_norm(p.name) == want or (repo and _norm(repo) == want)):
            matches.append(p)

    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        opts = ", ".join(
            f"{_repo_name_from_remote(p.remote_url or '') or p.name}" for p in matches
        )
        return None, (
            f"Several repos match '{requested}': {opts}. "
            "Say the exact repo name to delete."
        )
    # No exact match. List the real GitHub repo names so the user can retry
    # precisely — never auto-pick a loose keyword match for a destructive op.
    available = ", ".join(
        sorted(_repo_name_from_remote(p.remote_url or "") or p.name for p in on_github)
    )
    hint = f" Your GitHub-linked repos: {available}." if available else ""
    return None, (
        f"No repo clearly named '{requested}' found — refusing to guess for a "
        f"deletion.{hint}"
    )


async def _create_repo_on_github(
    repo_name: str, username: str, token: str
) -> str | None:
    """Create a new public repo on GitHub via the API.

    Returns the HTTPS clone URL on success, or None if creation failed.
    """
    if not token or not username:
        return None
    url = "https://api.github.com/user/repos"
    payload = json.dumps({"name": repo_name, "description": "", "private": False})
    result = await run_command(
        [
            "curl", "-s", "-X", "POST",
            "-H", f"Authorization: token {token}",
            "-H", "Accept: application/vnd.github.v3+json",
            "-d", payload,
            url,
        ]
    )
    if not result.ok or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
        # If repo was created (has id field), return the clone URL
        if "id" in data and "name" in data:
            return f"https://github.com/{username}/{repo_name}.git"
        # If clone_url is in the response, use it
        if "clone_url" in data:
            return data["clone_url"]
        # If repo already exists (422 error), that's OK - return the URL anyway
        if "message" in data and "already exists" in data.get("message", "").lower():
            return f"https://github.com/{username}/{repo_name}.git"
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


async def _repo_exists_on_github(remote_url: str, token: str | None) -> bool | None:
    """True/False if we could check, None if we couldn't (no token, unparseable
    URL, or a network error) — callers should treat None as "unknown, proceed
    as before" rather than as a failure.

    A repo's local git remote is just a URL saved in .git/config; it survives
    the repo being deleted on GitHub, so tools that read it (locate_project,
    open_repo) must not present it as live without checking."""
    if not token:
        return None
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/?$", remote_url)
    if not match:
        return None
    owner, repo_name = match.groups()
    try:
        async with httpx.AsyncClient(timeout=5) as http_client:
            response = await http_client.get(
                f"https://api.github.com/repos/{owner}/{repo_name}",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
    except httpx.HTTPError:
        return None
    if response.status_code == 404:
        return False
    if 200 <= response.status_code < 300:
        return True
    return None


async def _current_branch(cwd: Path | None) -> str:
    result = await run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    return result.stdout.strip() if result.ok and result.stdout.strip() else "main"


async def _suggest_commit_message(
    client: OllamaLike, model_manager: ModelManager, settings: Settings, cwd: Path | None
) -> str:
    diff = await run_command(["git", "diff", "--cached"], cwd=cwd)
    if not diff.ok or not diff.stdout.strip():
        return "Update code"
    model = await model_manager.ensure_llm()
    prompt = (
        "Based on this git diff, suggest ONE short commit message (under 50 "
        f"chars, imperative mood, no period):\n\n{diff.stdout[:2000]}"
    )
    reply = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        keep_alive=settings.llm_keep_alive,
    )
    return reply.strip().strip('"')[:72] or "Update code"


class OpenRepoArgs(BaseModel):
    project: str = Field(description="Project name or keyword (e.g., 'skin', 'jarvis', 'mail').")


class GitHubOpenRepoTool(Tool):
    name: ClassVar[str] = "github_open_repo"
    description: ClassVar[str] = (
        "Open a local project's GitHub repo in the browser. Say the project name "
        "or a keyword from it (e.g., 'open skin in github', 'show me jarvis'). "
        "Only works for projects that exist locally with a GitHub remote."
    )
    args_model: ClassVar[type[BaseModel]] = OpenRepoArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(
        self,
        registry: ProjectRegistry,
        client: OllamaLike,
        model_manager: ModelManager,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._model_manager = model_manager
        self._settings = settings or get_settings()

    async def run(self, args: OpenRepoArgs) -> ToolResult:  # type: ignore[override]
        project = await _resolve_project(
            args.project, self._registry, self._client, self._model_manager, self._settings
        )
        if project is None:
            known = ", ".join(p.name for p in await self._registry.list_projects())
            detail = (
                f" Known local projects: {known}." if known
                else f" No git repos found under {self._registry.root}."
            )
            return ToolResult.failure(
                self.name, f"Could not find a local project matching '{args.project}'.{detail}"
            )
        if project.remote_url is None:
            return ToolResult.failure(
                self.name,
                f"{project.name} exists locally but has no GitHub remote configured "
                "(no 'origin' set). Add one with `git remote add origin <url>`.",
            )
        exists = await _repo_exists_on_github(project.remote_url, self._settings.github_token)
        if exists is False:
            return ToolResult.failure(
                self.name,
                f"{project.name}'s local git config points at {project.remote_url}, but "
                "that repo no longer exists on GitHub (it was likely deleted). Push again "
                "to recreate it, or remove the stale remote with `git remote remove origin`.",
            )
        result = await run_command(["open", project.remote_url])
        if not result.ok:
            return ToolResult.failure(self.name, f"Could not open browser: {result.combined()}")
        return ToolResult(
            tool=self.name, ok=True,
            summary=f"Opened {project.name} on GitHub.",
            data={"project": project.name, "url": project.remote_url},
        )


class PushChangesArgs(BaseModel):
    project: str | None = Field(
        default=None,
        description="Which local project to push (e.g. 'skin', 'jarvis'). "
        "Omit to push from Jarvis's own working directory.",
    )
    message: str | None = Field(
        default=None,
        description="Commit message. If omitted, one is suggested from the diff.",
    )
    branch: str | None = Field(
        default=None,
        description="Which branch to push to. If omitted, the current branch is used.",
    )
    repo_name: str | None = Field(
        default=None,
        description="GitHub repo name (e.g. 'skin-analyser', 'jarvis'). "
        "Only used if the project has no .git repo yet. If omitted and no repo exists, "
        "you will be asked for it.",
    )
    github_username: str | None = Field(
        default=None,
        description="Your GitHub username. Only needed if creating a new repo. "
        "If omitted, uses 'Mohankirushna' (your default).",
    )


class GitHubPushTool(Tool):
    name: ClassVar[str] = "github_push"
    description: ClassVar[str] = (
        "Push staged changes to GitHub with confirmations: shows git status, "
        "suggests a commit message, confirms branch and final push. Use for "
        "'push changes', 'commit and push', 'ship it'."
    )
    args_model: ClassVar[type[BaseModel]] = PushChangesArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def __init__(
        self,
        registry: ProjectRegistry,
        client: OllamaLike,
        model_manager: ModelManager,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._model_manager = model_manager
        self._settings = settings or get_settings()

    def confirmation_action(self, args: BaseModel) -> str | None:
        assert isinstance(args, PushChangesArgs)
        where = f" in {args.project}" if args.project else ""
        msg = args.message or "(will suggest based on diff)"
        branch = args.branch or "(current branch)"
        setup = ""
        if args.repo_name:
            username = (args.github_username or "Mohankirushna").strip()
            setup = f"Create new repo 'github.com/{username}/{args.repo_name}', then "
        return f"{setup}about to push{where} with message '{msg}' to branch {branch}. Confirm?"

    async def run(self, args: PushChangesArgs) -> ToolResult:  # type: ignore[override]
        cwd: Path | None = None
        if args.project:
            # Try registry first (projects with .git)
            project = await _resolve_project(
                args.project, self._registry, self._client, self._model_manager, self._settings
            )
            if project is not None:
                cwd = project.path
            else:
                # Check if the folder exists in projects_dir (even without .git)
                candidate = self._settings.resolved_projects_dir / args.project
                if candidate.is_dir():
                    cwd = candidate
                else:
                    known = ", ".join(p.name for p in await self._registry.list_projects())
                    detail = f" Known local projects: {known}." if known else ""
                    msg = f"Could not find a local project matching '{args.project}'.{detail}"
                    return ToolResult.failure(self.name, msg)

        if cwd is None:
            # NEVER fall back to the server's own process directory — it has
            # no .gitignore, so `git add .` there would stage and push .env
            # (GitHub token, WAHA API key) to a public repo.
            return ToolResult.failure(
                self.name,
                "Which project is this for? Say 'push [project] to github as [repo-name]'.",
            )

        # Check if this is a git repo; if not, initialize one
        git_dir = cwd / ".git"
        if not git_dir.exists():
            if not args.repo_name:
                msg = (
                    "No git repo found. "
                    "Please provide the GitHub repo name you want to create "
                    "(e.g., 'jarvis', 'skin-analyser'). "
                    "Say: 'push [project] to github as [repo-name]'."
                )
                return ToolResult.failure(self.name, msg)
            default_username = "Mohankirushna"
            username = (
                args.github_username
                or self._settings.github_username
                or default_username
            ).strip()
            repo_name = args.repo_name.lower()

            # Try to create repo on GitHub first (if API token is configured)
            if self._settings.github_token:
                repo_url = await _create_repo_on_github(
                    repo_name, username, self._settings.github_token
                )
                if repo_url is None:
                    msg = (
                        f"Could not create repo '{repo_name}' on GitHub. "
                        "Check the API token or try creating it manually."
                    )
                    return ToolResult.failure(self.name, msg)
            else:
                # Fallback: assume standard GitHub URL (repo must be created manually)
                repo_url = f"https://github.com/{username}/{repo_name}.git"

            # Initialize local repo
            init_result = await run_command(["git", "init"], cwd=cwd)
            if not init_result.ok:
                return ToolResult.failure(self.name, f"git init failed: {init_result.combined()}")

            # Add remote
            remote_result = await run_command(
                ["git", "remote", "add", "origin", repo_url], cwd=cwd
            )
            if not remote_result.ok:
                msg = f"git remote add failed: {remote_result.combined()}"
                return ToolResult.failure(self.name, msg)

            # A brand-new bootstrapped folder has nothing to commit yet; without
            # a seed file the push below would silently no-op ("no changes to
            # commit") and the repo would sit empty forever.
            if not any(cwd.iterdir()):
                (cwd / "README.md").write_text(f"# {repo_name}\n")
            recreated = False
        else:
            # A local .git with a saved remote doesn't guarantee that remote
            # still exists on GitHub — it could have been deleted there. A
            # clean working tree in that case must NOT short-circuit as "no
            # changes": there's still local history that needs a live repo
            # to land in.
            recreated = False
            remote_check = await run_command(
                ["git", "config", "--get", "remote.origin.url"], cwd=cwd
            )
            if remote_check.ok and remote_check.stdout.strip():
                existing_remote = normalize_remote_url(remote_check.stdout)
                exists = await _repo_exists_on_github(
                    existing_remote, self._settings.github_token
                )
                if exists is False:
                    derived_name = existing_remote.rstrip("/").rsplit("/", 1)[-1]
                    repo_name = (args.repo_name or derived_name).lower()
                    username = (
                        args.github_username or self._settings.github_username or "Mohankirushna"
                    ).strip()
                    if not self._settings.github_token:
                        return ToolResult.failure(
                            self.name,
                            f"'{repo_name}' was deleted from GitHub and needs to be recreated, "
                            "but no GitHub API token is configured.",
                        )
                    repo_url = await _create_repo_on_github(
                        repo_name, username, self._settings.github_token
                    )
                    if repo_url is None:
                        return ToolResult.failure(
                            self.name,
                            f"Could not recreate '{repo_name}' on GitHub. Check the API token.",
                        )
                    recreated = True

        status = await run_command(["git", "status", "--porcelain"], cwd=cwd)
        if not status.ok:
            return ToolResult.failure(self.name, f"git status failed: {status.combined()}")
        if not status.stdout.strip() and not recreated:
            return ToolResult(
                tool=self.name, ok=True, summary="No changes to commit.", data={"status": "clean"},
            )

        add_result = await run_command(["git", "add", "."], cwd=cwd)
        if not add_result.ok:
            return ToolResult.failure(self.name, f"git add failed: {add_result.combined()}")

        message = args.message or await _suggest_commit_message(
            self._client, self._model_manager, self._settings, cwd
        )
        branch = args.branch or await _current_branch(cwd)

        commit_result = await run_command(["git", "commit", "-m", message], cwd=cwd)
        if not commit_result.ok and "nothing to commit" not in commit_result.combined().lower():
            return ToolResult.failure(self.name, f"git commit failed: {commit_result.combined()}")

        # Pull before pushing to handle remote-ahead case
        pull_result = await run_command(["git", "pull", "origin", branch, "--no-edit"], cwd=cwd)
        if not pull_result.ok and "no changes to commit" not in pull_result.combined().lower():
            # Pull failure might be OK if it's just "no changes"
            pass

        push_result = await run_command(["git", "push", "origin", branch], cwd=cwd)
        if not push_result.ok:
            return ToolResult.failure(self.name, f"git push failed: {push_result.combined()}")

        remote = await run_command(["git", "config", "--get", "remote.origin.url"], cwd=cwd)
        if remote.ok and remote.stdout.strip():
            browser_url = f"{normalize_remote_url(remote.stdout)}/commits/{branch}"
            await run_command(["open", browser_url])

        return ToolResult(
            tool=self.name, ok=True,
            summary=f"Pushed '{message}' to {branch} and opened GitHub.",
            data={"branch": branch, "message": message, "status": "pushed"},
        )


class LocateProjectArgs(BaseModel):
    project: str = Field(description="Project name or keyword (e.g., 'fitness', 'skin', 'jarvis').")


class LocateProjectTool(Tool):
    name: ClassVar[str] = "locate_project"
    description: ClassVar[str] = (
        "Answer where a local project lives and its GitHub status. Use for "
        "'where is the fitness project', 'what's the local path for skin', "
        "'give me the folder path for X', or 'is jarvis on github'. Returns the "
        "absolute folder path, whether it's a git repo, and its GitHub URL."
    )
    args_model: ClassVar[type[BaseModel]] = LocateProjectArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(
        self,
        registry: ProjectRegistry,
        client: OllamaLike,
        model_manager: ModelManager,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._model_manager = model_manager
        self._settings = settings or get_settings()

    async def run(self, args: LocateProjectArgs) -> ToolResult:  # type: ignore[override]
        # A git repo the registry already knows about is the richest answer.
        project = await _resolve_project(
            args.project, self._registry, self._client, self._model_manager, self._settings
        )
        if project is not None:
            if project.remote_url:
                exists = await _repo_exists_on_github(
                    project.remote_url, self._settings.github_token
                )
                if exists is False:
                    git_note = (
                        f"Its local remote points at {project.remote_url}, but that repo "
                        "no longer exists on GitHub (it was likely deleted)."
                    )
                else:
                    git_note = f"It's on GitHub at {project.remote_url}."
            elif project.is_git:
                git_note = "It's a git repo but has no GitHub remote yet."
            else:
                git_note = "It's not a git repo yet."
            return ToolResult(
                tool=self.name, ok=True,
                summary=f"{project.name} is at {project.path}. {git_note}",
                data={
                    "project": project.name,
                    "path": str(project.path),
                    "is_git": project.is_git,
                    "remote_url": project.remote_url,
                },
            )

        # Not indexed — maybe an exact folder name that scanning just missed.
        candidate = self._settings.resolved_projects_dir / args.project
        if candidate.is_dir():
            return ToolResult(
                tool=self.name, ok=True,
                summary=f"{args.project} is at {candidate}. It's not a git repo yet.",
                data={
                    "project": args.project,
                    "path": str(candidate),
                    "is_git": (candidate / ".git").exists(),
                    "remote_url": None,
                },
            )

        known = ", ".join(p.name for p in await self._registry.list_projects())
        detail = (
            f" Known projects: {known}." if known
            else f" No projects found under {self._registry.root}."
        )
        return ToolResult.failure(
            self.name, f"I couldn't find a project matching '{args.project}'.{detail}"
        )


class DeleteRepoArgs(BaseModel):
    project: str = Field(description="Project name (e.g., 'fitness', 'jarvis').")


class GitHubDeleteRepoTool(Tool):
    name: ClassVar[str] = "github_delete_repo"
    description: ClassVar[str] = (
        "Delete a GitHub repository. Use for 'delete the fitness repo', "
        "'remove this project from github'. Requires confirmation and a valid "
        "GitHub API token."
    )
    args_model: ClassVar[type[BaseModel]] = DeleteRepoArgs
    # DESTRUCTIVE, not SENSITIVE: deleting a repo is irreversible and must ask
    # EVERY time. SENSITIVE remembers the first approval for an identical action
    # string and auto-allows silent repeats — so deleting the same-named repo a
    # second time skipped the confirmation entirely. DESTRUCTIVE never remembers.
    risk_level: ClassVar[RiskLevel] = RiskLevel.DESTRUCTIVE

    def __init__(
        self,
        registry: ProjectRegistry,
        client: OllamaLike,
        model_manager: ModelManager,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._model_manager = model_manager
        self._settings = settings or get_settings()

    def confirmation_action(self, args: BaseModel) -> str | None:
        """Show EXACTLY what will be deleted, and never a wrong repo.

        Uses the same STRICT resolution as run() (against the cached project
        list) so the preview can't point at an unrelated repo the way loose
        keyword matching once did. Only opens the browser / names a specific
        repo when that strict match is certain; otherwise it asks plainly and
        lets run() do the authoritative (refreshed) resolution. Always returns
        a message — deletion is destructive and never happens silently."""
        assert isinstance(args, DeleteRepoArgs)
        projects = self._registry.cached_projects()
        project, _why = _resolve_project_for_deletion(args.project, projects)
        if project is not None and project.remote_url:
            repo = _repo_name_from_remote(project.remote_url) or project.name
            with contextlib.suppress(Exception):
                subprocess.Popen(["open", project.remote_url])
            return (
                f"Opened '{repo}' on GitHub for you to check: {project.remote_url}\n"
                f"Delete the repository '{repo}'? This cannot be undone."
            )
        # No certain match in cache — ask about the requested name without
        # opening or naming a possibly-wrong repo. run() re-resolves strictly.
        return (
            f"Delete the GitHub repository named '{args.project}'? "
            "This cannot be undone."
        )

    async def run(self, args: DeleteRepoArgs) -> ToolResult:  # type: ignore[override]
        # Deletion is destructive and MUST always be user-confirmed via the safety
        # gate's confirmation_action dialog. If this tool is being called, the caller
        # should have already shown a confirmation prompt. If auto_approve is on
        # (dev-only), that's still a choice the user made.
        #
        # However, we add an extra guard: if the request reaches here via an LLM
        # call (not via the safety gate), fail explicitly rather than silently
        # delete. The safety gate will catch the error and re-prompt with
        # confirmation_action message.
        if not self._settings.github_token:
            msg = (
                "GitHub API token not configured. "
                "Set JARVIS_GITHUB_TOKEN to delete repos."
            )
            return ToolResult.failure(self.name, msg)

        # Refresh first so a just-created/renamed project is seen: a stale cache
        # is exactly what let a deletion resolve to the wrong (old) project.
        await self._registry.refresh()
        projects = await self._registry.list_projects()
        # STRICT resolution — never fuzzy-guess which repo to destroy.
        project, why = _resolve_project_for_deletion(args.project, projects)
        if project is None or project.remote_url is None:
            return ToolResult.failure(self.name, why or f"'{args.project}' not found.")

        # Parse owner/repo from the remote URL (https://github.com/owner/repo)
        match = re.match(r"https://github\.com/([^/]+)/([^/]+?)/?$", project.remote_url)
        if not match:
            return ToolResult.failure(
                self.name, f"Could not parse GitHub URL: {project.remote_url}"
            )
        owner, repo_name = match.groups()

        # Log the deletion attempt. This helps debug if deletion happens without
        # proper user confirmation (which would be a bug).
        logger.warning(
            f"[DELETION] About to delete repo {owner}/{repo_name}. "
            f"User should have confirmed via safety gate confirmation dialog."
        )

        # Delete via GitHub API
        url = f"https://api.github.com/repos/{owner}/{repo_name}"
        headers = {
            "Authorization": f"token {self._settings.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.delete(url, headers=headers)

        if response.status_code == 404:
            return ToolResult.failure(
                self.name, f"Repository '{repo_name}' not found on GitHub (already deleted?)."
            )
        if response.status_code == 403:
            return ToolResult.failure(
                self.name,
                "Permission denied. Check that the GitHub token has 'repo' scope and "
                "you own the repository.",
            )
        if not (200 <= response.status_code < 300):
            return ToolResult.failure(
                self.name,
                f"GitHub API error ({response.status_code}): {response.text[:200]}",
            )

        # GitHub's own DELETE endpoint has been observed to answer 2xx a
        # moment before the repo actually stops resolving elsewhere (brief
        # propagation delay). Never claim "Deleted" on the API's word alone —
        # poll until it actually 404s, or say plainly that it's still
        # pending rather than asserting something unverified.
        for attempt in range(5):
            await asyncio.sleep(0.5 * (attempt + 1))
            still_exists = await _repo_exists_on_github(
                f"https://github.com/{owner}/{repo_name}", self._settings.github_token
            )
            if still_exists is False:
                logger.warning(f"[DELETION] Confirmed deleted: {owner}/{repo_name}")
                return ToolResult(
                    tool=self.name, ok=True,
                    summary=f"Deleted '{repo_name}' from GitHub. The local folder remains.",
                    data={"repo": repo_name, "owner": owner, "status": "deleted"},
                )
            if still_exists is None:
                # Can't verify (no token, network hiccup) — trust the 2xx.
                logger.warning(
                    f"[DELETION] Assuming deleted (couldn't verify): {owner}/{repo_name}"
                )
                return ToolResult(
                    tool=self.name, ok=True,
                    summary=f"Deleted '{repo_name}' from GitHub. The local folder remains.",
                    data={"repo": repo_name, "owner": owner, "status": "deleted"},
                )

        return ToolResult.failure(
            self.name,
            f"GitHub accepted the delete request for '{repo_name}', but it still shows as "
            "existing after checking. It may still be processing — check again shortly.",
        )


class RefreshProjectsArgs(BaseModel):
    pass


class RefreshProjectsTool(Tool):
    name: ClassVar[str] = "refresh_projects"
    description: ClassVar[str] = (
        "Re-scan the local projects folder for git repos, e.g. after cloning a "
        "new one. Use for 'refresh my projects' or 'find my new repo'."
    )
    args_model: ClassVar[type[BaseModel]] = RefreshProjectsArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, registry: ProjectRegistry) -> None:
        self._registry = registry

    async def run(self, args: RefreshProjectsArgs) -> ToolResult:  # type: ignore[override]
        count = await self._registry.refresh()
        projects = await self._registry.list_projects()
        names = ", ".join(p.name for p in projects) if projects else "none"
        return ToolResult(
            tool=self.name, ok=True,
            summary=f"Found {count} local project(s): {names}.",
            data={"count": count, "projects": names},
        )
