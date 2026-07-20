"""System tools: apps, volume, screenshots, media, windows, brightness.

Everything goes through AppleScript / built-in CLIs — no compiled helper
needed. Window management requires the host process to have Accessibility
permission (System Settings → Privacy & Security → Accessibility); the tool
reports that clearly when missing rather than failing cryptically.
"""

from __future__ import annotations

import asyncio
import ctypes
import html
import json
import re
from datetime import datetime
from typing import ClassVar, Literal
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from app.planner.schemas import RiskLevel, ToolResult
from app.tools._common import applescript_quote, expand_path, run_command, run_osascript
from app.tools.base import Tool


class OpenAppArgs(BaseModel):
    name: str = Field(description="Application name, e.g. 'Safari' or 'Notes'.")


class OpenAppTool(Tool):
    name: ClassVar[str] = "open_app"
    description: ClassVar[str] = "Open (or bring to front) a macOS application by name."
    args_model: ClassVar[type[BaseModel]] = OpenAppArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: OpenAppArgs) -> ToolResult:  # type: ignore[override]
        output = await run_command(["/usr/bin/open", "-a", args.name])
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not open {args.name!r}: {output.combined()}"
            )
        return ToolResult(tool=self.name, ok=True, summary=f"Opened {args.name}")


# Common site shorthands so "open youtube" resolves without a full URL.
_SITE_SHORTCUTS = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "maps": "https://maps.google.com",
    "github": "https://github.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "chatgpt": "https://chat.openai.com",
    "spotify": "https://open.spotify.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.com",
    "whatsapp": "https://web.whatsapp.com",
}


class OpenUrlArgs(BaseModel):
    target: str = Field(
        description="A website to open in a browser: a full URL "
        "(https://example.com), a domain (example.com), or a well-known site "
        "name ('youtube', 'gmail')."
    )
    browser: str | None = Field(
        default=None,
        description="Browser app to use, e.g. 'Google Chrome', 'Safari', "
        "'Firefox'. Omit for the system default browser.",
    )


class OpenUrlTool(Tool):
    name: ClassVar[str] = "open_url"
    description: ClassVar[str] = (
        "Open a website or URL in the user's real, visible browser (optionally "
        "a specific one like Chrome or Safari). Use this for 'open YouTube', "
        "'open gmail in Chrome', 'go to github.com' — NOT open_app."
    )
    args_model: ClassVar[type[BaseModel]] = OpenUrlArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _normalize(target: str) -> str:
        cleaned = target.strip().strip("/")
        key = cleaned.lower().removeprefix("www.")
        if key in _SITE_SHORTCUTS:
            return _SITE_SHORTCUTS[key]
        if cleaned.startswith(("http://", "https://")):
            return cleaned
        return f"https://{cleaned}"

    async def run(self, args: OpenUrlArgs) -> ToolResult:  # type: ignore[override]
        url = self._normalize(args.target)
        argv = ["/usr/bin/open", url]
        if args.browser:
            argv = ["/usr/bin/open", "-a", args.browser, url]
        output = await run_command(argv)
        if not output.ok:
            hint = ""
            if args.browser and "Unable to find application" in output.combined():
                hint = f" (is {args.browser!r} installed? try without specifying a browser)"
            return ToolResult.failure(self.name, f"could not open {url}: {output.combined()}{hint}")
        where = f" in {args.browser}" if args.browser else ""
        return ToolResult(tool=self.name, ok=True, summary=f"Opened {url}{where}")


class BrowserSearchArgs(BaseModel):
    query: str = Field(description="Words or a question to search for.")
    engine: Literal["google", "wikipedia"] = Field(
        default="google",
        description="Use 'wikipedia' only when the user explicitly asks to search Wikipedia.",
    )
    browser: str | None = Field(
        default=None,
        description="Browser app to use, such as 'Google Chrome', 'Safari', or 'Firefox'.",
    )


class BrowserSearchTool(Tool):
    """Open a search in the user's real browser, without a headless-browser dependency."""

    name: ClassVar[str] = "browser_search"
    description: ClassVar[str] = (
        "Search Google or Wikipedia in the user's visible browser. Use for requests such as "
        "'search volcanoes in Chrome' or 'search Wikipedia for Ada Lovelace'."
    )
    args_model: ClassVar[type[BaseModel]] = BrowserSearchArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _url(query: str, engine: str) -> str:
        if engine == "wikipedia":
            return "https://en.wikipedia.org/w/index.php?search=" + quote_plus(query)
        return "https://www.google.com/search?q=" + quote_plus(query)

    async def run(self, args: BrowserSearchArgs) -> ToolResult:  # type: ignore[override]
        url = self._url(args.query, args.engine)
        argv = ["/usr/bin/open", url]
        if args.browser:
            argv = ["/usr/bin/open", "-a", args.browser, url]
        output = await run_command(argv)
        if not output.ok:
            hint = ""
            if args.browser and "Unable to find application" in output.combined():
                hint = f" (is {args.browser!r} installed?)"
            return ToolResult.failure(
                self.name, f"could not open the search: {output.combined()}{hint}"
            )
        where = f" in {args.browser}" if args.browser else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Searching {args.engine} for {args.query!r}{where}.",
            data={"query": args.query, "engine": args.engine, "url": url},
        )


class BraveSearchOpenFirstArgs(BaseModel):
    query: str = Field(description="Words or a question to search for in Brave Search.")


class BraveSearchOpenFirstTool(Tool):
    """Find and open Brave Search's first ordinary web result in Brave."""

    name: ClassVar[str] = "brave_search_open_first"
    description: ClassVar[str] = (
        "Search Brave Search and open the first non-sponsored web result in the Brave browser, "
        "visible on screen. This is the DEFAULT choice whenever the user names a topic, asks a "
        "factual question, or says 'search for <topic>' — even a bare topic with no verb, e.g. "
        "'amazon forest' or 'ironman'. Prefer this over web_search unless the user explicitly "
        "asks for a written summary or list of results instead of a page to look at."
    )
    args_model: ClassVar[type[BaseModel]] = BraveSearchOpenFirstArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _first_result_url(page: str) -> str | None:
        """Extract the first external result, ignoring Brave navigation and ads."""
        for raw_url in re.findall(r'<a[^>]+href=["\']([^"\']+)', page, flags=re.IGNORECASE):
            candidate = html.unescape(raw_url)
            parsed = urlparse(candidate)
            if parsed.netloc.endswith("search.brave.com") and parsed.path == "/redirect":
                candidate = unquote(parse_qs(parsed.query).get("url", [""])[0])
                parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"}:
                continue
            if parsed.netloc.endswith("brave.com"):
                continue
            return candidate
        return None

    _BROWSER_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    @classmethod
    async def _fetch_search_page(cls, search_url: str) -> str:
        # Python's urllib is blocked by Brave's anti-bot layer (HTTP 429)
        # regardless of headers — it fingerprints the TLS/HTTP client itself,
        # not just the User-Agent string. curl's fingerprint isn't blocked, so
        # shell out to it instead of using urlopen. --fail is load-bearing:
        # without it a 429 block page comes back as "success", and its first
        # external link (a Tor support page) gets opened as "the result".
        output = await run_command([
            "/usr/bin/curl", "-s", "-L", "--fail", "--max-time", "15",
            "-A", cls._BROWSER_USER_AGENT,
            "-H", "Accept-Language: en-US,en;q=0.9",
            search_url,
        ])
        if not output.ok or not output.stdout.strip():
            raise RuntimeError(output.combined() or "empty response from Brave Search")
        return output.stdout

    @staticmethod
    def _ddgs_first_url(query: str) -> str | None:
        try:
            from ddgs import DDGS
        except ImportError:
            return None
        for result in DDGS().text(query, max_results=5):
            url = result.get("href", "")
            if url.startswith(("http://", "https://")):
                return url
        return None

    async def run(self, args: BraveSearchOpenFirstArgs) -> ToolResult:  # type: ignore[override]
        search_url = "https://search.brave.com/search?q=" + quote_plus(args.query)
        # Primary: the ddgs library (DuckDuckGo's API endpoints) — reliable
        # and not rate-limited the way scraping a results page is. Fallback:
        # scrape Brave's results page. Last resort: open the search itself,
        # which still gets the user real results — only actually failing to
        # launch Brave Browser is a hard failure.
        result_url: str | None = None
        try:
            result_url = await asyncio.to_thread(self._ddgs_first_url, args.query)
        except Exception:  # noqa: BLE001 - fall through to the Brave scrape
            result_url = None
        if result_url is None:
            try:
                page = await self._fetch_search_page(search_url)
                result_url = self._first_result_url(page)
            except Exception:  # noqa: BLE001 - any scrape failure falls back below
                result_url = None

        target_url = result_url or search_url
        output = await run_command(["/usr/bin/open", "-a", "Brave Browser", target_url])
        if not output.ok:
            return ToolResult.failure(
                self.name,
                f"could not open Brave: {output.combined()} (is Brave Browser installed?)",
            )
        if result_url is not None:
            summary = f"Opened the first search result for {args.query!r} in Brave."
        else:
            summary = (
                f"The search results didn't load directly, so I opened "
                f"the search for {args.query!r} in Brave instead."
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=summary,
            data={"query": args.query, "search_url": search_url, "url": target_url},
        )


class WebAnswerArgs(BaseModel):
    query: str = Field(description="The question or topic to look up on the web.")


class WebAnswerTool(Tool):
    """Search the web and return readable text (top page + result snippets)
    so the planner can answer a factual question in its own words.

    This is the read-and-answer counterpart to brave_search_open_first, which
    only opens a page. Doing the search + fetch in one deterministic tool call
    (rather than making the small model chain search -> open -> read itself)
    is what makes answering reliable on a 3B model.
    """

    name: ClassVar[str] = "web_answer"
    description: ClassVar[str] = (
        "Search the web, read the top result, and return its text so you can ANSWER the "
        "user's factual question in your own words — prices, facts, definitions, current "
        "events, and who/what/when/where/how questions. Returns text to read aloud, not a "
        "page to open; use brave_search_open_first instead when the user asks to open or "
        "visit a page."
    )
    args_model: ClassVar[type[BaseModel]] = WebAnswerArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    _MAX_PAGE_CHARS = 2000
    _MAX_SNIPPET_CHARS = 1200

    @staticmethod
    def _ddgs_results(query: str) -> list[dict[str, str]]:
        from ddgs import DDGS

        return list(DDGS().text(query, max_results=5))

    @staticmethod
    def _html_to_text(page: str) -> str:
        # Drop non-content elements, strip remaining tags, decode entities,
        # and collapse whitespace into a compact block the 3B model can read.
        page = re.sub(
            r"(?is)<(script|style|head|nav|footer|header|noscript)[^>]*>.*?</\1>", " ", page
        )
        page = re.sub(r"(?s)<[^>]+>", " ", page)
        return re.sub(r"\s+", " ", html.unescape(page)).strip()

    @classmethod
    async def _fetch_page_text(cls, url: str) -> str | None:
        output = await run_command([
            "/usr/bin/curl", "-s", "-L", "--fail", "--max-time", "12",
            "-A", BraveSearchOpenFirstTool._BROWSER_USER_AGENT,
            "-H", "Accept-Language: en-US,en;q=0.9",
            url,
        ])
        if not output.ok or not output.stdout.strip():
            return None
        return cls._html_to_text(output.stdout) or None

    async def run(self, args: WebAnswerArgs) -> ToolResult:  # type: ignore[override]
        try:
            results = await asyncio.to_thread(self._ddgs_results, args.query)
        except Exception as exc:  # noqa: BLE001 - network errors need a clear reply
            return ToolResult.failure(self.name, f"could not search the web: {exc}")
        if not results:
            return ToolResult.failure(self.name, f"no web results for {args.query!r}.")

        snippets = [
            f"- {(r.get('title') or '').strip()}: {(r.get('body') or '').strip()}"
            for r in results[:5]
            if (r.get("body") or "").strip()
        ]
        snippet_block = "\n".join(snippets)[: self._MAX_SNIPPET_CHARS]
        top_url = results[0].get("href", "")

        # Lead with the search-result snippets: they are concise and answer-
        # oriented, which a 3B model reads more reliably than a full page's
        # navigation/markup noise. The page text follows as supporting detail.
        parts: list[str] = []
        if snippet_block:
            parts.append(f"Search results:\n{snippet_block}")
        page_text = await self._fetch_page_text(top_url) if top_url else None
        if page_text:
            parts.append(f"From the top result ({top_url}):\n{page_text[: self._MAX_PAGE_CHARS]}")
        content = "\n\n".join(parts)
        if not content:
            return ToolResult.failure(
                self.name, f"found results for {args.query!r} but no readable text."
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=content,
            data={
                "query": args.query,
                "url": top_url,
                "results": [
                    {"title": r.get("title", ""), "url": r.get("href", "")} for r in results
                ],
            },
        )


class YouTubePlayArgs(BaseModel):
    query: str = Field(description="Song, artist, or video to find and play on YouTube.")
    browser: str | None = Field(
        default=None,
        description="Browser app to use, e.g. 'Brave Browser' or 'Google Chrome'.",
    )


class YouTubePlayTool(Tool):
    """Find the first matching YouTube video and open it in a visible browser.

    YouTube does not offer a public desktop-control API. Resolving a watch URL
    before opening it is substantially more reliable than opening a search
    page and hoping a fragile UI script clicks the intended result.
    """

    name: ClassVar[str] = "youtube_play"
    description: ClassVar[str] = (
        "Find a requested song/video on YouTube, open the matching video in the user's "
        "visible browser, and request autoplay. Use for 'play <song> on YouTube'."
    )
    args_model: ClassVar[type[BaseModel]] = YouTubePlayArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _find_video_id(query: str) -> str | None:
        url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed HTTPS origin
            page = response.read().decode("utf-8", errors="ignore")
        match = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', page)
        return match.group(1) if match else None

    async def run(self, args: YouTubePlayArgs) -> ToolResult:  # type: ignore[override]
        try:
            video_id = await asyncio.to_thread(self._find_video_id, args.query)
        except Exception as exc:  # noqa: BLE001 - network errors need a clear tool result
            return ToolResult.failure(self.name, f"could not search YouTube: {exc}")
        if video_id is None:
            return ToolResult.failure(
                self.name, f"could not find a YouTube video for {args.query!r}"
            )

        url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
        argv = ["/usr/bin/open", url]
        if args.browser:
            argv = ["/usr/bin/open", "-a", args.browser, url]
        output = await run_command(argv)
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not open the YouTube video: {output.combined()}"
            )
        where = f" in {args.browser}" if args.browser else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Opened a YouTube result for {args.query!r}{where} and requested playback.",
            data={"url": url, "query": args.query, "video_id": video_id},
        )


class MusicPlatformPromptArgs(BaseModel):
    query: str = Field(description="The song, artist, or music request the user wants to play.")


class MusicPlatformPromptTool(Tool):
    """Return a deterministic clarification instead of silently choosing a service."""

    name: ClassVar[str] = "music_platform_prompt"
    description: ClassVar[str] = (
        "Ask which music platform to use when the user requested music without naming one. "
        "Offer YouTube, Spotify, and Apple Music."
    )
    args_model: ClassVar[type[BaseModel]] = MusicPlatformPromptArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: MusicPlatformPromptArgs) -> ToolResult:  # type: ignore[override]
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=(
                f"Where would you like to play {args.query}: YouTube, Spotify, or Apple Music?"
            ),
            data={"query": args.query, "platforms": ["YouTube", "Spotify", "Apple Music"]},
        )


class SpotifyPlayArgs(BaseModel):
    query: str = Field(description="Song, artist, album, or playlist to find in Spotify.")


_SPOTIFY_TRACK_ID = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)")


class SpotifyPlayTool(Tool):
    """Resolve a song to a Spotify track and play it via Spotify's own
    AppleScript API (`play track "spotify:track:…"`).

    Spotify's desktop app has no scriptable *search*, but playback of a known
    track URI is first-class scripting — far sturdier than the previous
    approach of typing into its search UI and hunting for a Play button,
    which needed Accessibility permission and broke on UI changes. The track
    is found by a web search restricted to open.spotify.com/track pages.
    """

    name: ClassVar[str] = "spotify_play"
    description: ClassVar[str] = (
        "Find a requested song on Spotify and play it in the Spotify desktop app. "
        "Use when the user specifically names Spotify."
    )
    args_model: ClassVar[type[BaseModel]] = SpotifyPlayArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _ddgs_track_search(query: str) -> str | None:
        try:
            from ddgs import DDGS
        except ImportError:
            return None
        for result in DDGS().text(f"site:open.spotify.com/track {query}", max_results=5):
            match = _SPOTIFY_TRACK_ID.search(result.get("href", ""))
            if match:
                return match.group(1)
        return None

    @staticmethod
    async def _find_track_id(query: str) -> str | None:
        # Primary: the ddgs library (same engine as web_search) — reliable
        # and not rate-limited the way scraping results pages is.
        try:
            track_id = await asyncio.to_thread(SpotifyPlayTool._ddgs_track_search, query)
        except Exception:  # noqa: BLE001 - fall through to the scrape fallback
            track_id = None
        if track_id is not None:
            return track_id
        # Fallback: scrape Brave Search (can be intermittently 429'd).
        search_url = "https://search.brave.com/search?q=" + quote_plus(
            f"site:open.spotify.com/track {query}"
        )
        output = await run_command([
            "/usr/bin/curl", "-s", "-L", "--max-time", "15",
            "-A", BraveSearchOpenFirstTool._BROWSER_USER_AGENT,
            "-H", "Accept-Language: en-US,en;q=0.9",
            search_url,
        ])
        if not output.ok:
            return None
        match = _SPOTIFY_TRACK_ID.search(output.stdout)
        return match.group(1) if match else None

    async def run(self, args: SpotifyPlayArgs) -> ToolResult:  # type: ignore[override]
        track_id = await self._find_track_id(args.query)
        if track_id is None:
            # Couldn't resolve a specific track — open Spotify's search UI so
            # the user can pick, and say so honestly rather than pretending.
            uri = "spotify:search:" + quote(args.query, safe="")
            output = await run_command(["/usr/bin/open", "-a", "Spotify", uri])
            if not output.ok:
                return ToolResult.failure(
                    self.name,
                    f"could not open Spotify: {output.combined()} (is Spotify installed?)",
                )
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=(
                    f"I couldn't pin down an exact track for {args.query!r}, so I opened "
                    "the Spotify search — pick the one you want."
                ),
                data={"query": args.query, "uri": uri},
            )

        script = f'''tell application "Spotify"
    activate
    play track "spotify:track:{track_id}"
    delay 1
    set trackInfo to (name of current track) & "||" & (artist of current track)
    return trackInfo & "||" & (player state as string)
end tell'''
        playback = await run_osascript(script)
        if not playback.ok:
            hint = ""
            if "Not authorized" in playback.combined() or "-1743" in playback.combined():
                hint = (
                    " Approve the 'Jarvis wants to control Spotify' popup, or enable it in "
                    "System Settings → Privacy & Security → Automation → Jarvis → Spotify."
                )
            return ToolResult.failure(
                self.name,
                f"could not control Spotify: {playback.combined()}{hint}",
            )
        name, _, rest = playback.stdout.strip().partition("||")
        artist, _, state = rest.partition("||")
        if state.strip() != "playing":
            return ToolResult.failure(
                self.name,
                f"Spotify accepted the track but reports state {state.strip()!r}, not playing.",
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Playing {name} by {artist} on Spotify.",
            data={
                "query": args.query, "track_id": track_id,
                "track": name, "artist": artist,
            },
        )


class SpotifyOpenPlaylistArgs(BaseModel):
    playlist: str = Field(description="Name of a playlist in the user's Spotify library.")


class SpotifyOpenPlaylistTool(Tool):
    name: ClassVar[str] = "spotify_open_playlist"
    description: ClassVar[str] = (
        "Open a named playlist in the user's Spotify library. Use when the user asks to "
        "open a playlist in Spotify, especially a personal playlist."
    )
    args_model: ClassVar[type[BaseModel]] = SpotifyOpenPlaylistArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: SpotifyOpenPlaylistArgs) -> ToolResult:  # type: ignore[override]
        output = await run_command(["/usr/bin/open", "-a", "Spotify"])
        if not output.ok:
            return ToolResult.failure(
                self.name,
                f"could not open Spotify: {output.combined()} (is Spotify installed?)",
            )
        # Spotify does not expose a scriptable playlist library. Search inside
        # the signed-in desktop app, then activate the matching playlist result
        # through Accessibility rather than relying on public web search.
        script = f'''\
tell application "Spotify" to activate
delay 0.8
tell application "System Events"
    tell process "Spotify"
        keystroke "l" using command down
        delay 0.3
        keystroke {applescript_quote(args.playlist)}
        delay 0.8
        key code 36
        delay 1.5
        set searchWindow to window 1
        set {{windowX, windowY}} to position of searchWindow
        set {{windowWidth, windowHeight}} to size of searchWindow
        set resultLeft to windowX + (windowWidth * 1 / 10)
        set resultTop to windowY + (windowHeight * 1 / 8)
        set resultBottom to windowY + (windowHeight * 3 / 4)
        repeat with uiElement in entire contents of searchWindow
            try
                set {{elementX, elementY}} to position of uiElement
                if elementX > resultLeft then
                    if elementY > resultTop and elementY < resultBottom then
                        set elementText to ""
                        try
                            set elementText to name of uiElement as text
                        end try
                        try
                            set elementDescription to description of uiElement as text
                            set elementText to elementText & " " & elementDescription
                        end try
                        try
                            set elementText to elementText & " " & (value of uiElement as text)
                        end try
                        if elementText contains {applescript_quote(args.playlist)} then
                            click uiElement
                            return "opened"
                        end if
                    end if
                end if
            end try
        end repeat
    end tell
end tell
return "not found"'''
        playlist_result = await run_osascript(script)
        if not playlist_result.ok:
            return ToolResult.failure(
                self.name,
                "Spotify opened, but Jarvis could not inspect your library. "
                "Grant Accessibility permission to Jarvis (or the Terminal running the backend) "
                "in System Settings → Privacy & Security → Accessibility. "
                + playlist_result.combined(),
            )
        if playlist_result.stdout.strip() != "opened":
            return ToolResult.failure(
                self.name,
                f"Spotify searched for the playlist {args.playlist!r}, but could not open a "
                "matching result. Use the playlist's exact displayed name.",
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Opened your Spotify playlist {args.playlist!r}.",
            data={"playlist": args.playlist},
        )


class NewsSearchArgs(BaseModel):
    query: str = Field(description="Topic to search for in recent news.")
    browser: str | None = Field(
        default=None,
        description="Browser app to use, e.g. 'Brave Browser' or 'Google Chrome'.",
    )


class NewsSearchTool(Tool):
    name: ClassVar[str] = "news_search"
    description: ClassVar[str] = (
        "Open a Google News search for recent articles about a topic in the user's visible "
        "browser. Only use when the request itself signals news/current-events — words like "
        "'news', 'recent', 'latest', 'happening now', 'updates'. A bare topic or general "
        "question with none of those words (e.g. 'amazon forest', 'ironman') is NOT a news "
        "request — use brave_search_open_first for that instead."
    )
    args_model: ClassVar[type[BaseModel]] = NewsSearchArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: NewsSearchArgs) -> ToolResult:  # type: ignore[override]
        url = "https://news.google.com/search?q=" + quote_plus(args.query)
        argv = ["/usr/bin/open", url]
        if args.browser:
            argv = ["/usr/bin/open", "-a", args.browser, url]
        output = await run_command(argv)
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not open the news search: {output.combined()}"
            )
        where = f" in {args.browser}" if args.browser else ""
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Opened recent news about {args.query!r}{where}.",
            data={"url": url, "query": args.query},
        )


class QuitAppArgs(BaseModel):
    name: str = Field(description="Application to quit, e.g. 'Safari'.")


class QuitAppTool(Tool):
    name: ClassVar[str] = "quit_app"
    description: ClassVar[str] = (
        "Quit a running application (it may prompt to save unsaved work)."
    )
    args_model: ClassVar[type[BaseModel]] = QuitAppArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SENSITIVE

    async def run(self, args: QuitAppArgs) -> ToolResult:  # type: ignore[override]
        script = f"tell application {applescript_quote(args.name)} to quit"
        output = await run_osascript(script)
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not quit {args.name!r}: {output.combined()}"
            )
        return ToolResult(tool=self.name, ok=True, summary=f"Quit {args.name}")


class ListAppsArgs(BaseModel):
    pass


class ListAppsTool(Tool):
    name: ClassVar[str] = "list_running_apps"
    description: ClassVar[str] = "List the applications currently running (visible ones)."
    args_model: ClassVar[type[BaseModel]] = ListAppsArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: BaseModel) -> ToolResult:
        script = (
            'tell application "System Events" to get name of every process '
            "whose background only is false"
        )
        output = await run_osascript(script)
        if not output.ok:
            return ToolResult.failure(self.name, output.combined())
        apps = [name.strip() for name in output.stdout.split(",") if name.strip()]
        return ToolResult(
            tool=self.name,
            ok=True,
            summary="Running apps: " + ", ".join(apps),
            data={"apps": apps},
        )


class BluetoothDevicesArgs(BaseModel):
    pass


class BluetoothDevicesTool(Tool):
    """List devices currently connected to the Mac over Bluetooth."""

    name: ClassVar[str] = "list_bluetooth_devices"
    description: ClassVar[str] = (
        "List Bluetooth devices that are currently connected to this Mac."
    )
    args_model: ClassVar[type[BaseModel]] = BluetoothDevicesArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _connected_device_names(payload: object) -> list[str]:
        """Extract connected devices from system_profiler's nested JSON output."""
        names: list[str] = []

        def visit(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return

            connected = str(value.get("device_connected", "")).lower() == "yes"
            if connected:
                name = next(
                    (
                        str(value[key]).strip()
                        for key in ("device_name", "_name", "name")
                        if value.get(key)
                    ),
                    "Unknown Bluetooth device",
                )
                if name not in names:
                    names.append(name)
            for child in value.values():
                visit(child)

        visit(payload)
        return names

    async def run(self, args: BluetoothDevicesArgs) -> ToolResult:  # type: ignore[override]
        output = await run_command(["/usr/sbin/system_profiler", "SPBluetoothDataType", "-json"])
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not inspect Bluetooth devices: {output.combined()}"
            )
        try:
            devices = self._connected_device_names(json.loads(output.stdout))
        except json.JSONDecodeError:
            return ToolResult.failure(
                self.name, "could not read Bluetooth device information from macOS"
            )
        if not devices:
            return ToolResult(
                tool=self.name,
                ok=True,
                summary="No Bluetooth devices are currently connected.",
                data={"devices": []},
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            summary="Connected Bluetooth devices: " + ", ".join(devices),
            data={"devices": devices},
        )


class VolumeArgs(BaseModel):
    level: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Absolute volume 0-100 to set; omit for a relative change or to just read it.",
    )
    direction: Literal["up", "down"] | None = Field(
        default=None,
        description="Adjust volume relatively ('turn the volume up') instead of an absolute level.",
    )
    amount: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Percentage points to change by when direction is set.",
    )
    muted: bool | None = Field(
        default=None,
        description="Mute (true) or unmute (false) output. Preserves the volume level, unlike "
        "setting level to 0.",
    )


class VolumeTool(Tool):
    name: ClassVar[str] = "volume"
    description: ClassVar[str] = (
        "Get, set, or relatively adjust (up/down) the system output volume (0-100), "
        "or mute/unmute without changing the level."
    )
    args_model: ClassVar[type[BaseModel]] = VolumeArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    async def _current_level() -> int | None:
        output = await run_osascript("output volume of (get volume settings)")
        if not output.ok:
            return None
        return int(output.stdout.strip())

    @staticmethod
    async def _current_muted() -> bool | None:
        output = await run_osascript("output muted of (get volume settings)")
        if not output.ok:
            return None
        return output.stdout.strip() == "true"

    async def run(self, args: VolumeArgs) -> ToolResult:  # type: ignore[override]
        if args.muted is not None:
            output = await run_osascript(
                f"set volume output muted {'true' if args.muted else 'false'}"
            )
            if not output.ok:
                return ToolResult.failure(self.name, output.combined())
            state = "Muted" if args.muted else "Unmuted"
            return ToolResult(
                tool=self.name, ok=True, summary=f"{state} the volume.",
                data={"muted": args.muted},
            )

        if args.level is None and args.direction is None:
            level = await self._current_level()
            muted = await self._current_muted()
            if level is None:
                return ToolResult.failure(self.name, "could not read the current volume.")
            summary = f"Volume is {level}%" + (" (muted)" if muted else "")
            return ToolResult(
                tool=self.name, ok=True, summary=summary,
                data={"level": level, "muted": bool(muted)},
            )

        target = args.level
        if target is None:
            current = await self._current_level()
            if current is None:
                return ToolResult.failure(self.name, "could not read the current volume.")
            delta = args.amount if args.direction == "up" else -args.amount
            target = max(0, min(100, current + delta))

        output = await run_osascript(f"set volume output volume {target}")
        if not output.ok:
            return ToolResult.failure(self.name, output.combined())
        return ToolResult(
            tool=self.name, ok=True, summary=f"Volume set to {target}%", data={"level": target}
        )


class BatteryStatusArgs(BaseModel):
    pass


class BatteryStatusTool(Tool):
    name: ClassVar[str] = "battery_status"
    description: ClassVar[str] = "Get the Mac's current battery charge percentage and power state."
    args_model: ClassVar[type[BaseModel]] = BatteryStatusArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    @staticmethod
    def _parse_status(output: str) -> tuple[int, str] | None:
        match = re.search(r"(\d{1,3})%;\s*([^;\n]+)", output)
        if not match:
            return None
        return int(match.group(1)), match.group(2).strip()

    async def run(self, args: BatteryStatusArgs) -> ToolResult:  # type: ignore[override]
        output = await run_command(["/usr/bin/pmset", "-g", "batt"])
        if not output.ok:
            return ToolResult.failure(
                self.name, f"could not read battery status: {output.combined()}"
            )
        parsed = self._parse_status(output.stdout)
        if parsed is None:
            return ToolResult.failure(self.name, "could not find a battery charge percentage")
        percentage, state = parsed
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"Your Mac battery is at {percentage}% ({state}).",
            data={"percentage": percentage, "state": state},
        )


class SystemPowerArgs(BaseModel):
    action: Literal["restart", "shutdown"] = Field(
        description="Whether to restart or shut down the Mac."
    )


class SystemPowerTool(Tool):
    """Restart or shut down macOS after the safety gate receives approval."""

    name: ClassVar[str] = "system_power"
    description: ClassVar[str] = (
        "Restart or shut down the Mac. This always requires explicit user confirmation."
    )
    args_model: ClassVar[type[BaseModel]] = SystemPowerArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.DESTRUCTIVE

    async def run(self, args: SystemPowerArgs) -> ToolResult:  # type: ignore[override]
        command = "restart" if args.action == "restart" else "shut down"
        output = await run_osascript(f'tell application "System Events" to {command}')
        if not output.ok:
            return ToolResult.failure(
                self.name,
                f"could not {args.action} the Mac: {output.combined()}",
            )
        verb = "Restarting" if args.action == "restart" else "Shutting down"
        return ToolResult(
            tool=self.name,
            ok=True,
            summary=f"{verb} your Mac now.",
            data={"action": args.action},
        )


class ScreenshotArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Where to save; default is a timestamped file on the Desktop."
    )


class ScreenshotTool(Tool):
    name: ClassVar[str] = "screenshot"
    description: ClassVar[str] = "Take a screenshot of the whole screen and save it."
    args_model: ClassVar[type[BaseModel]] = ScreenshotArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: ScreenshotArgs) -> ToolResult:  # type: ignore[override]
        if args.path:
            target = expand_path(args.path)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
            target = expand_path(f"~/Desktop/Screenshot {stamp}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        output = await run_command(["/usr/sbin/screencapture", "-x", str(target)])
        if not output.ok or not target.exists():
            hint = ""
            if "could not create image" in output.stderr:
                hint = (
                    " — grant Screen Recording permission to Jarvis/your terminal in "
                    "System Settings → Privacy & Security → Screen Recording"
                )
            return ToolResult.failure(
                self.name, f"screencapture failed: {output.combined()}{hint}"
            )
        return ToolResult(
            tool=self.name, ok=True, summary=f"Screenshot saved to {target}",
            data={"path": str(target)},
        )


class MediaArgs(BaseModel):
    action: Literal["play", "pause", "next", "previous"] = Field(
        description="Media action. Use 'pause' for stop/pause/quiet, 'play' "
        "for play/resume/continue, 'next' to skip, 'previous' to go back."
    )


class MediaTool(Tool):
    name: ClassVar[str] = "media_control"
    description: ClassVar[str] = (
        "Control music playback in Music or Spotify: play, pause (also for "
        "'stop'), skip to next, or go to the previous track."
    )
    args_model: ClassVar[type[BaseModel]] = MediaArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    # Explicit verbs, never the `playpause` toggle: "pause then stop" must not
    # flip playback back on. Each verb is idempotent for its intent.
    _ACTIONS: ClassVar[dict[str, str]] = {
        "play": "play",
        "pause": "pause",
        "next": "next track",
        "previous": "previous track",
    }

    async def _running(self, app: str) -> bool:
        output = await run_osascript(f"application {applescript_quote(app)} is running")
        return output.ok and output.stdout.strip() == "true"

    @staticmethod
    def _expected_state(action: str) -> str | None:
        return {"play": "playing", "pause": "paused"}.get(action)

    async def _player_state(self, quoted_player: str) -> str | None:
        output = await run_osascript(
            f"tell application {quoted_player} to player state"
        )
        return output.stdout.strip() if output.ok else None

    async def run(self, args: MediaArgs) -> ToolResult:  # type: ignore[override]
        verb = self._ACTIONS[args.action]
        for player in ("Spotify", "Music"):
            if not await self._running(player):
                continue
            quoted = applescript_quote(player)
            output = await run_osascript(f"tell application {quoted} to {verb}")
            if not output.ok:
                return ToolResult.failure(
                    self.name, f"could not control {player}: {output.combined()}"
                )
            expected_state = self._expected_state(args.action)
            if expected_state is not None:
                # Spotify occasionally acknowledges `pause` without changing
                # its state. Confirm the result before reporting success. A
                # play/pause fallback is safe only after observing "playing".
                await asyncio.sleep(0.15)
                state = await self._player_state(quoted)
                if args.action == "pause" and state == "playing":
                    fallback = await run_osascript(
                        f"tell application {quoted} to playpause"
                    )
                    if fallback.ok:
                        await asyncio.sleep(0.15)
                        state = await self._player_state(quoted)
                if state != expected_state:
                    current = state or "an unreadable state"
                    return ToolResult.failure(
                        self.name,
                        f"{player} is still {current}; it did not {args.action}.",
                    )
            # Skipping implies the user wants to *hear* the result, so force
            # playback — a bare `next track` can leave the player paused/silent
            # in some states. Give the player a moment to switch tracks before
            # reading back, so the reported title isn't the previous one.
            if args.action in ("next", "previous"):
                await run_osascript(f"tell application {quoted} to play")
                await asyncio.sleep(0.4)
            # Read the real resulting state AND current track, so the reply is
            # grounded in what actually happened — "next" that didn't advance
            # can't be reported as success, and the model can't invent a title.
            state = await self._player_state(quoted) or "unknown"
            track_out = await run_osascript(
                f'tell application {quoted} to name of current track'
            )
            artist_out = await run_osascript(
                f'tell application {quoted} to artist of current track'
            )
            track = track_out.stdout.strip() if track_out.ok else ""
            artist = artist_out.stdout.strip() if artist_out.ok else ""
            now = f"'{track}'" + (f" by {artist}" if artist else "") if track else "nothing"
            verb_past = {"play": "playing", "pause": "paused",
                         "next": "skipped to", "previous": "went back to"}[args.action]
            summary = (
                f"{player} is {state}."
                if args.action in ("play", "pause")
                else f"{player} {verb_past} {now} ({state})."
            )
            return ToolResult(
                tool=self.name,
                ok=True,
                summary=summary,
                data={"player": player, "state": state, "track": track, "artist": artist},
            )
        return ToolResult.failure(self.name, "Neither Music nor Spotify is running.")


class WindowArgs(BaseModel):
    app: str = Field(description="Application whose front window to arrange, e.g. 'Safari'.")
    position: Literal["left_half", "right_half", "maximize", "center"] = Field(
        description="Where to place the window."
    )


class WindowArrangeTool(Tool):
    name: ClassVar[str] = "window_arrange"
    description: ClassVar[str] = (
        "Move/resize an app's front window: left half, right half, maximize, or center."
    )
    args_model: ClassVar[type[BaseModel]] = WindowArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    async def run(self, args: WindowArgs) -> ToolResult:  # type: ignore[override]
        bounds_output = await run_osascript(
            'tell application "Finder" to get bounds of window of desktop'
        )
        if not bounds_output.ok:
            return ToolResult.failure(self.name, bounds_output.combined())
        try:
            _x, _y, width, height = (int(v.strip()) for v in bounds_output.stdout.split(","))
        except ValueError:
            return ToolResult.failure(
                self.name, f"could not parse screen bounds: {bounds_output.stdout!r}"
            )

        frames = {
            "left_half": (0, 25, width // 2, height - 25),
            "right_half": (width // 2, 25, width // 2, height - 25),
            "maximize": (0, 25, width, height - 25),
            "center": (width // 6, height // 8, width * 2 // 3, height * 3 // 4),
        }
        x, y, w, h = frames[args.position]
        app_name = applescript_quote(args.app)
        script = (
            f'tell application "System Events" to tell process {app_name}\n'
            f"  set position of front window to {{{x}, {y}}}\n"
            f"  set size of front window to {{{w}, {h}}}\n"
            "end tell"
        )
        output = await run_osascript(script)
        if not output.ok:
            hint = ""
            if "assistive access" in output.stderr.lower() or "1002" in output.stderr:
                hint = (
                    " (grant Accessibility permission to your terminal/Jarvis in "
                    "System Settings → Privacy & Security → Accessibility)"
                )
            return ToolResult.failure(self.name, f"could not arrange window: "
                                                 f"{output.combined()}{hint}")
        return ToolResult(
            tool=self.name, ok=True,
            summary=f"Placed {args.app}'s front window at {args.position}",
        )


class BrightnessArgs(BaseModel):
    level: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Absolute brightness 0.0-1.0 to set; omit for a relative change "
        "or to just read it.",
    )
    direction: Literal["up", "down"] | None = Field(
        default=None,
        description="Adjust brightness relatively ('turn the brightness up') instead of an "
        "absolute level.",
    )
    amount: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="How much to change brightness by (0.0-1.0 scale) when direction is set.",
    )


class BrightnessTool(Tool):
    """Read/set display brightness via the private DisplayServices framework.

    The public `brightness` CLI relies on DDC/IOKit APIs that Apple has
    blocked for the built-in display on Apple Silicon Macs — it silently
    does nothing there even when installed. DisplayServicesGetBrightness /
    DisplayServicesSetBrightness are the same private calls System Settings'
    own brightness slider uses, and work on both Intel and Apple Silicon.
    """

    name: ClassVar[str] = "brightness"
    description: ClassVar[str] = (
        "Get, set, or relatively adjust (up/down) the display brightness (0.0 to 1.0)."
    )
    args_model: ClassVar[type[BaseModel]] = BrightnessArgs
    risk_level: ClassVar[RiskLevel] = RiskLevel.SAFE

    _CORE_GRAPHICS_PATH = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    _DISPLAY_SERVICES_PATH = (
        "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices"
    )

    @classmethod
    def _display_services(cls) -> tuple[ctypes.CDLL, int]:
        core_graphics = ctypes.CDLL(cls._CORE_GRAPHICS_PATH)
        core_graphics.CGMainDisplayID.restype = ctypes.c_uint32
        display_id = core_graphics.CGMainDisplayID()

        display_services = ctypes.CDLL(cls._DISPLAY_SERVICES_PATH)
        display_services.DisplayServicesGetBrightness.restype = ctypes.c_int
        display_services.DisplayServicesGetBrightness.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
        ]
        display_services.DisplayServicesSetBrightness.restype = ctypes.c_int
        display_services.DisplayServicesSetBrightness.argtypes = [ctypes.c_uint32, ctypes.c_float]
        return display_services, display_id

    @classmethod
    def _get_brightness(cls) -> float | None:
        display_services, display_id = cls._display_services()
        value = ctypes.c_float()
        if display_services.DisplayServicesGetBrightness(display_id, ctypes.byref(value)) != 0:
            return None
        return value.value

    @classmethod
    def _set_brightness(cls, level: float) -> bool:
        display_services, display_id = cls._display_services()
        return display_services.DisplayServicesSetBrightness(display_id, ctypes.c_float(level)) == 0

    async def run(self, args: BrightnessArgs) -> ToolResult:  # type: ignore[override]
        try:
            if args.level is None and args.direction is None:
                level = await asyncio.to_thread(self._get_brightness)
                if level is None:
                    return ToolResult.failure(self.name, "could not read the display brightness.")
                return ToolResult(
                    tool=self.name, ok=True, summary=f"Brightness is {level:.0%}",
                    data={"level": level},
                )

            target = args.level
            if target is None:
                current = await asyncio.to_thread(self._get_brightness)
                if current is None:
                    return ToolResult.failure(self.name, "could not read the display brightness.")
                delta = args.amount if args.direction == "up" else -args.amount
                target = max(0.0, min(1.0, current + delta))

            ok = await asyncio.to_thread(self._set_brightness, target)
        except OSError as exc:
            return ToolResult.failure(self.name, f"could not change brightness: {exc}")
        if not ok:
            return ToolResult.failure(self.name, "could not change the display brightness.")
        return ToolResult(
            tool=self.name, ok=True, summary=f"Brightness set to {target:.0%}",
            data={"level": target},
        )
