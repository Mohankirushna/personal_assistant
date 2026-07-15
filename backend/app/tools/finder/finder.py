"""Finder tools: search, list, create, move/rename, delete, compress, extract.

Deletion goes to the Trash via Finder (recoverable) but is still classified
DESTRUCTIVE so it always requires confirmation, per the product spec.
Search uses Spotlight (mdfind) — instant, and no filesystem walking.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import applescript_quote, expand_path, run_command, run_osascript
from app.tools.base import Tool

_MAX_LISTED = 50


def _describe_entries(entries: list[Path]) -> str:
    return "\n".join(f"{'[dir] ' if entry.is_dir() else ''}{entry.name}" for entry in entries)


class SearchArgs(BaseModel):
    query: str = Field(description="What to search for (file name or content keywords).")
    folder: str | None = Field(
        default=None, description="Folder to search in; omit to search everywhere."
    )


class FinderSearchTool(Tool):
    name: ClassVar[str] = "finder_search"
    description: ClassVar[str] = (
        "Search for files on this Mac by name or content, using Spotlight."
    )
    args_model: ClassVar[type[BaseModel]] = SearchArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: SearchArgs) -> ToolResult:  # type: ignore[override]
        argv = ["/usr/bin/mdfind"]
        if args.folder:
            argv += ["-onlyin", str(expand_path(args.folder))]
        argv.append(args.query)
        output = await run_command(argv)
        if not output.ok:
            return ToolResult.failure(self.name, f"search failed: {output.combined()}")
        hits = [line for line in output.stdout.splitlines() if line.strip()]
        shown = hits[:_MAX_LISTED]
        summary = (
            f"Found {len(hits)} result(s) for {args.query!r}"
            + (f" in {args.folder}" if args.folder else "")
            + (":\n" + "\n".join(shown) if shown else ".")
        )
        return ToolResult(tool=self.name, ok=True, summary=summary, data={"paths": shown})


class ListFolderArgs(BaseModel):
    path: str = Field(description="Folder to list, e.g. '~/Downloads'.")


class FinderListTool(Tool):
    name: ClassVar[str] = "finder_list"
    description: ClassVar[str] = "List the contents of a folder."
    args_model: ClassVar[type[BaseModel]] = ListFolderArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: ListFolderArgs) -> ToolResult:  # type: ignore[override]
        folder = expand_path(args.path)
        if not folder.is_dir():
            return ToolResult.failure(self.name, f"{folder} is not a folder")
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        shown = entries[:_MAX_LISTED]
        more = f"\n… and {len(entries) - len(shown)} more" if len(entries) > len(shown) else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"{folder} contains {len(entries)} item(s):\n"
            + _describe_entries(shown)
            + more,
            data={"entries": [str(entry) for entry in shown]},
        )


class CreateFolderArgs(BaseModel):
    path: str = Field(description="Folder to create, e.g. '~/Documents/Reports'.")


class FinderCreateFolderTool(Tool):
    name: ClassVar[str] = "finder_create_folder"
    description: ClassVar[str] = "Create a new folder (including parents)."
    args_model: ClassVar[type[BaseModel]] = CreateFolderArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: CreateFolderArgs) -> ToolResult:  # type: ignore[override]
        folder = expand_path(args.path)
        if folder.exists():
            return ToolResult.failure(self.name, f"{folder} already exists")
        folder.mkdir(parents=True)
        return ToolResult(tool=self.name, ok=True, summary=f"Created folder {folder}")


class MoveArgs(BaseModel):
    source: str = Field(description="File or folder to move/rename.")
    destination: str = Field(description="New path. Same folder with a new name = rename.")


class FinderMoveTool(Tool):
    name: ClassVar[str] = "finder_move"
    description: ClassVar[str] = "Move or rename a file or folder."
    args_model: ClassVar[type[BaseModel]] = MoveArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    async def run(self, args: MoveArgs) -> ToolResult:  # type: ignore[override]
        source = expand_path(args.source)
        destination = expand_path(args.destination)
        if not source.exists():
            return ToolResult.failure(self.name, f"{source} does not exist")
        if destination.exists():
            return ToolResult.failure(self.name, f"{destination} already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return ToolResult(tool=self.name, ok=True, summary=f"Moved {source} → {destination}")


class DeleteArgs(BaseModel):
    path: str = Field(description="File or folder to delete (goes to the Trash).")


class FinderDeleteTool(Tool):
    name: ClassVar[str] = "finder_delete"
    description: ClassVar[str] = (
        "Delete a file or folder. It is moved to the Trash (recoverable), "
        "but this always requires the user's confirmation."
    )
    args_model: ClassVar[type[BaseModel]] = DeleteArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.DESTRUCTIVE

    async def run(self, args: DeleteArgs) -> ToolResult:  # type: ignore[override]
        target = expand_path(args.path)
        if not target.exists():
            return ToolResult.failure(self.name, f"{target} does not exist")
        script = (
            'tell application "Finder" to delete '
            f"(POSIX file {applescript_quote(str(target))} as alias)"
        )
        output = await run_osascript(script)
        if not output.ok:
            hint = ""
            if "-1743" in output.stderr or "not authorized" in output.stderr.lower():
                hint = (
                    " — grant Automation permission (Finder) to Jarvis/your terminal in "
                    "System Settings → Privacy & Security → Automation"
                )
            return ToolResult.failure(self.name, f"could not delete: {output.combined()}{hint}")
        return ToolResult(tool=self.name, ok=True, summary=f"Moved {target} to the Trash")


class CompressArgs(BaseModel):
    path: str = Field(description="File or folder to compress.")
    archive: str | None = Field(
        default=None, description="Output .zip path; default is alongside the source."
    )


class FinderCompressTool(Tool):
    name: ClassVar[str] = "finder_compress"
    description: ClassVar[str] = "Compress a file or folder into a .zip archive."
    args_model: ClassVar[type[BaseModel]] = CompressArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: CompressArgs) -> ToolResult:  # type: ignore[override]
        source = expand_path(args.path)
        if not source.exists():
            return ToolResult.failure(self.name, f"{source} does not exist")
        archive = (
            expand_path(args.archive)
            if args.archive
            else source.with_suffix(source.suffix + ".zip")
        )
        if archive.exists():
            return ToolResult.failure(self.name, f"{archive} already exists")
        output = await run_command(
            ["/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", str(source), str(archive)],
            timeout=120,
        )
        if not output.ok:
            return ToolResult.failure(self.name, f"compress failed: {output.combined()}")
        return ToolResult(tool=self.name, ok=True, summary=f"Compressed {source} → {archive}")


class ExtractArgs(BaseModel):
    archive: str = Field(description="The .zip file to extract.")
    destination: str | None = Field(
        default=None, description="Folder to extract into; default is alongside the archive."
    )


class FinderExtractTool(Tool):
    name: ClassVar[str] = "finder_extract"
    description: ClassVar[str] = "Extract a .zip archive."
    args_model: ClassVar[type[BaseModel]] = ExtractArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: ExtractArgs) -> ToolResult:  # type: ignore[override]
        archive = expand_path(args.archive)
        if not archive.is_file():
            return ToolResult.failure(self.name, f"{archive} is not a file")
        destination = (
            expand_path(args.destination) if args.destination else archive.parent / archive.stem
        )
        output = await run_command(
            ["/usr/bin/ditto", "-x", "-k", str(archive), str(destination)], timeout=120
        )
        if not output.ok:
            return ToolResult.failure(self.name, f"extract failed: {output.combined()}")
        return ToolResult(tool=self.name, ok=True, summary=f"Extracted {archive} → {destination}")
