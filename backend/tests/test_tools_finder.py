"""Finder tools against a temp directory (no user data touched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.finder.finder import (
    FinderCompressTool,
    FinderCreateFolderTool,
    FinderDeleteTool,
    FinderExtractTool,
    FinderListTool,
    FinderMoveTool,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("deep")
    return tmp_path


async def test_list(workdir: Path) -> None:
    result = await FinderListTool().execute({"path": str(workdir)})
    assert result.ok
    assert "notes.txt" in result.summary
    assert "[dir] sub" in result.summary


async def test_list_rejects_file(workdir: Path) -> None:
    result = await FinderListTool().execute({"path": str(workdir / "notes.txt")})
    assert not result.ok


async def test_create_folder(workdir: Path) -> None:
    target = workdir / "new" / "nested"
    result = await FinderCreateFolderTool().execute({"path": str(target)})
    assert result.ok and target.is_dir()
    # Creating again fails cleanly.
    result = await FinderCreateFolderTool().execute({"path": str(target)})
    assert not result.ok


async def test_move_and_rename(workdir: Path) -> None:
    source = workdir / "notes.txt"
    dest = workdir / "renamed.txt"
    result = await FinderMoveTool().execute(
        {"source": str(source), "destination": str(dest)}
    )
    assert result.ok and dest.exists() and not source.exists()


async def test_move_refuses_overwrite(workdir: Path) -> None:
    result = await FinderMoveTool().execute(
        {"source": str(workdir / "notes.txt"), "destination": str(workdir / "sub")}
    )
    assert not result.ok


async def test_compress_extract_roundtrip(workdir: Path) -> None:
    archive = workdir / "sub.zip"
    result = await FinderCompressTool().execute(
        {"path": str(workdir / "sub"), "archive": str(archive)}
    )
    assert result.ok and archive.exists()

    out = workdir / "restored"
    result = await FinderExtractTool().execute(
        {"archive": str(archive), "destination": str(out)}
    )
    assert result.ok
    assert (out / "deep.txt").read_text() == "deep"


async def test_delete_missing_fails_before_confirmation_matters(workdir: Path) -> None:
    result = await FinderDeleteTool().execute({"path": str(workdir / "ghost.txt")})
    assert not result.ok


@pytest.mark.integration
async def test_delete_moves_to_trash(tmp_path: Path) -> None:
    """Talks to the real Finder; leaves one small file in the Trash."""
    victim = tmp_path / "jarvis-test-delete-me.txt"
    victim.write_text("bye")
    result = await FinderDeleteTool().execute({"path": str(victim)})
    if not result.ok and (
        "Automation permission" in result.summary or "timed out" in result.summary
    ):
        # Either TCC denied, or the consent dialog is waiting for a human.
        pytest.skip("host process lacks Automation (Finder) permission")
    assert result.ok, result.summary
    assert not victim.exists()
