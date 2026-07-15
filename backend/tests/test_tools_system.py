"""System + clipboard tools. Mutating tests are integration-marked; the
clipboard test saves and restores the user's pasteboard."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.tools.clipboard.clipboard import ClipboardReadTool, ClipboardWriteTool
from app.tools.registry import ToolRegistry
from app.tools.system.system import ScreenshotTool, VolumeTool


def test_discovery_finds_the_whole_suite() -> None:
    registry = ToolRegistry()
    registry.discover()
    names = {tool.name for tool in registry.list()}
    expected = {
        "clock",
        "finder_search", "finder_list", "finder_create_folder", "finder_move",
        "finder_delete", "finder_compress", "finder_extract",
        "terminal_run", "git", "vscode_open",
        "clipboard_read", "clipboard_write",
        "open_app", "quit_app", "list_running_apps", "volume", "screenshot",
        "media_control", "window_arrange", "brightness",
        "roll_dice",  # the example plugin — proves plugin discovery works
    }
    missing = expected - names
    assert not missing, f"tools not discovered: {missing}"
    # Service-dependent tools are NOT discovered; app.main injects them.
    assert "look_at_screen" not in names


@pytest.mark.integration
async def test_volume_read() -> None:
    result = await VolumeTool().execute({})
    assert result.ok, result.summary
    assert 0 <= result.data["level"] <= 100


@pytest.mark.integration
async def test_screenshot(tmp_path: Path) -> None:
    target = tmp_path / "shot.png"
    result = await ScreenshotTool().execute({"path": str(target)})
    if not result.ok and "Screen Recording" in result.summary:
        pytest.skip("host process lacks Screen Recording permission")
    assert result.ok, result.summary
    assert target.exists() and target.stat().st_size > 0


@pytest.mark.integration
async def test_clipboard_roundtrip_preserves_user_data() -> None:
    # Save whatever is on the clipboard now.
    saved = await asyncio.create_subprocess_exec(
        "/usr/bin/pbpaste", stdout=asyncio.subprocess.PIPE
    )
    original, _ = await saved.communicate()
    try:
        write = await ClipboardWriteTool().execute({"text": "jarvis-clipboard-test"})
        assert write.ok
        read = await ClipboardReadTool().execute({})
        assert read.ok
        assert "jarvis-clipboard-test" in read.summary
    finally:
        restore = await asyncio.create_subprocess_exec(
            "/usr/bin/pbcopy", stdin=asyncio.subprocess.PIPE
        )
        await restore.communicate(original)
