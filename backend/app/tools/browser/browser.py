"""Browser tools (Playwright/Chromium): search, open pages, read them, fill
forms, download files.

A single shared BrowserSession lazily launches headless Chromium on first
use and keeps one page alive between tool calls, so "search X" followed by
"open the second result" works naturally. RAM matters more than isolation
here (8GB target), hence one browser, one context, one page.

Search goes through the ddgs library (DuckDuckGo's API endpoints) — the
HTML endpoints CAPTCHA-wall headless browsers. Form filling is SENSITIVE
(approved once per exact action); password fields are classified
DESTRUCTIVE so credential entry is always confirmed.

These tool classes require the shared session, so app.main registers them
explicitly (discovery skips service-dependent constructors).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools.base import Tool

logger = logging.getLogger(__name__)

_MAX_TEXT = 4000


class BrowserSession:
    """Lazily started shared Chromium instance."""

    def __init__(self, downloads_dir: Path | None = None) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._downloads_dir = downloads_dir or Path("~/Downloads").expanduser()

    async def page(self) -> Any:
        if self._page is None:
            from playwright.async_api import async_playwright

            logger.info("Launching headless Chromium")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            context = await self._browser.new_context(accept_downloads=True)
            self._page = await context.new_page()
        return self._page

    @property
    def downloads_dir(self) -> Path:
        return self._downloads_dir

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


async def _page_text(page: Any) -> str:
    text = await page.inner_text("body")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT] + f"\n… (truncated, {len(text)} chars total)"
    return text


class SearchWebArgs(BaseModel):
    query: str = Field(description="What to search the web for.")


class BrowserSearchTool(Tool):
    """Web search via the ddgs library rather than scraping: DuckDuckGo now
    CAPTCHA-walls headless browsers, and ddgs speaks the API endpoints
    properly (still free, keyless, local)."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Search the web and return the top results with URLs."
    args_model: ClassVar[type[BaseModel]] = SearchWebArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, session: BrowserSession) -> None:
        self._session = session  # kept for interface symmetry with the other tools

    async def run(self, args: SearchWebArgs) -> ToolResult:  # type: ignore[override]
        import asyncio

        from ddgs import DDGS

        def _search() -> list[dict[str, str]]:
            return list(DDGS().text(args.query, max_results=8))

        raw = await asyncio.to_thread(_search)
        results = [
            {"title": item.get("title", ""), "url": item.get("href", "")}
            for item in raw
            if item.get("href")
        ]
        if not results:
            return ToolResult.failure(self.name, f"no results for {args.query!r}")
        lines = [f"{i + 1}. {r['title']} — {r['url']}" for i, r in enumerate(results)]
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Top results for {args.query!r}:\n" + "\n".join(lines),
            data={"results": results},
        )


class OpenUrlArgs(BaseModel):
    url: str = Field(description="The URL to open, e.g. 'https://example.com'.")


class BrowserOpenTool(Tool):
    name: ClassVar[str] = "browser_open"
    description: ClassVar[str] = (
        "Open a web page in the assistant's browser and return its title and text "
        "(also used to read/summarize pages)."
    )
    args_model: ClassVar[type[BaseModel]] = OpenUrlArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    async def run(self, args: OpenUrlArgs) -> ToolResult:  # type: ignore[override]
        url = args.url if re.match(r"^https?://", args.url) else f"https://{args.url}"
        page = await self._session.page()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        status = response.status if response else "?"
        title = await page.title()
        text = await _page_text(page)
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Opened {url} (HTTP {status}) — {title!r}\n\n{text}",
            data={"url": url, "title": title, "status": status},
        )


class FillFieldArgs(BaseModel):
    selector: str = Field(description="CSS selector of the input field.")
    value: str = Field(description="Text to type into the field.")
    submit: bool = Field(default=False, description="Press Enter after filling.")


class BrowserFillTool(Tool):
    name: ClassVar[str] = "browser_fill"
    description: ClassVar[str] = (
        "Fill a form field on the currently open page (optionally submitting). "
        "Login/password fields always require user approval."
    )
    args_model: ClassVar[type[BaseModel]] = FillFieldArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    def assess_risk(self, args: BaseModel) -> RiskLevel:
        assert isinstance(args, FillFieldArgs)
        if "password" in args.selector.lower() or "passwd" in args.selector.lower():
            return RiskLevel.DESTRUCTIVE  # always confirm credential entry
        return RiskLevel.SENSITIVE

    async def run(self, args: FillFieldArgs) -> ToolResult:  # type: ignore[override]
        page = await self._session.page()
        await page.fill(args.selector, args.value, timeout=10_000)
        if args.submit:
            await page.press(args.selector, "Enter")
            await page.wait_for_load_state("domcontentloaded")
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Filled {args.selector}"
            + (" and submitted" if args.submit else "")
            + f" on {page.url}",
        )


class DownloadArgs(BaseModel):
    url: str = Field(description="Direct URL of the file to download.")
    filename: str | None = Field(default=None, description="Optional file name to save as.")


class BrowserDownloadTool(Tool):
    name: ClassVar[str] = "browser_download"
    description: ClassVar[str] = "Download a file from a URL into the Downloads folder."
    args_model: ClassVar[type[BaseModel]] = DownloadArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    async def run(self, args: DownloadArgs) -> ToolResult:  # type: ignore[override]
        page = await self._session.page()
        api_request = page.context.request
        response = await api_request.get(args.url, timeout=60_000)
        if not response.ok:
            return ToolResult.failure(
                self.name, f"download failed: HTTP {response.status} for {args.url}"
            )
        name = args.filename or args.url.rstrip("/").rsplit("/", 1)[-1] or "download"
        target = self._session.downloads_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await response.body())
        size = target.stat().st_size
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Downloaded {args.url} → {target} ({size} bytes)",
            data={"path": str(target), "bytes": size},
        )
