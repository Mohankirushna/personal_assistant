"""Browser tools. Risk classification is unit-tested; everything touching a
real Chromium/network is integration-marked."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.planner.schemas import RiskLevel

# Check for optional dependencies directly
try:
    import ddgs  # noqa: F401
    import playwright  # noqa: F401
    _has_browser_deps = True
except ImportError:
    _has_browser_deps = False

# Import browser tools (which only lazily import playwright/ddgs at runtime)
from app.tools.browser.browser import (
    BrowserDownloadTool,
    BrowserFillTool,
    BrowserOpenTool,
    BrowserSearchTool,
    BrowserSession,
)

pytestmark = pytest.mark.skipif(not _has_browser_deps, reason="playwright/ddgs not installed")


def test_fill_risk_classification() -> None:
    tool = BrowserFillTool(BrowserSession())
    normal = tool.parse_args({"selector": "#search", "value": "hi"})
    password = tool.parse_args({"selector": "input#Password", "value": "s3cret"})
    assert normal is not None and password is not None
    assert tool.assess_risk(normal) is RiskLevel.SENSITIVE
    assert tool.assess_risk(password) is RiskLevel.DESTRUCTIVE


@pytest.fixture
async def session(tmp_path: Path):  # noqa: ANN201
    browser_session = BrowserSession(downloads_dir=tmp_path)
    yield browser_session
    await browser_session.close()


@pytest.mark.integration
async def test_open_page(session: BrowserSession) -> None:
    result = await BrowserOpenTool(session).execute({"url": "https://example.com"})
    assert result.ok, result.summary
    assert "Example Domain" in result.summary
    assert result.data["status"] == 200


@pytest.mark.integration
async def test_open_adds_scheme(session: BrowserSession) -> None:
    result = await BrowserOpenTool(session).execute({"url": "example.com"})
    assert result.ok
    assert result.data["url"].startswith("https://")


@pytest.mark.integration
async def test_search_returns_results(session: BrowserSession) -> None:
    result = await BrowserSearchTool(session).execute({"query": "playwright python"})
    assert result.ok, result.summary
    assert len(result.data["results"]) >= 3
    assert all(r["url"].startswith("http") for r in result.data["results"])


@pytest.mark.integration
async def test_download(session: BrowserSession, tmp_path: Path) -> None:
    result = await BrowserDownloadTool(session).execute(
        {"url": "https://example.com/", "filename": "example.html"}
    )
    assert result.ok, result.summary
    saved = tmp_path / "example.html"
    assert saved.exists()
    assert b"Example Domain" in saved.read_bytes()
