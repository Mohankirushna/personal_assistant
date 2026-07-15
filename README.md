# Jarvis for macOS

A local-first, 100% free and open-source AI desktop assistant for macOS. Control your
computer with natural voice commands — open/close apps, manage files, run terminal
commands, automate the browser, and more — powered entirely by local, open models.

Optimized for a MacBook Air M2 (8GB RAM). No paid APIs. Works offline for most tasks.

## Status

**All 10 phases complete — feature-complete v0.1.** A local voice assistant that
hears "hey jarvis", transcribes with Whisper, plans with a local LLM, drives 26
safety-gated tools (files, terminal, git, apps, clipboard, browser, screen vision),
remembers past interactions, and replies out loud — with a SwiftUI menu-bar app and
a Python backend, entirely on-device.

Quickstart in [backend/README.md](backend/README.md) · design in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · API in [docs/API.md](docs/API.md).

```bash
scripts/setup.sh --all --with-models     # deps + Chromium + models (~9GB)
cd backend && uv run jarvis-backend       # start the backend
# then build/run the menu-bar app: scripts/make_app.sh && open frontend/dist/Jarvis.app
```

## Why

Existing "AI assistant" projects either require a cloud API key, don't actually control
your desktop, or aren't tuned for low-RAM Apple Silicon hardware. This project aims to
be a real, safe, local alternative: every destructive action requires confirmation,
every tool call goes through an explicit planner (the LLM never executes commands
directly), and the model footprint is actively managed to fit in 8GB of RAM.

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Qwen2.5 3B Instruct (Ollama, local) |
| Speech-to-text | whisper.cpp, Metal-accelerated |
| Wake word | openWakeWord ("Jarvis") |
| Text-to-speech | Piper TTS |
| Vision (optional) | Qwen2.5-VL via Ollama, on-demand |
| Backend | FastAPI (Python, async) |
| macOS app | SwiftUI |
| Automation | AppleScript, Accessibility API (PyObjC), Playwright |
| Memory | SQLite + ChromaDB (local embeddings) |
| Package management | uv |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full rationale on every decision.

## Project structure

```
jarvis_v2/
├── backend/     # FastAPI app: planner, tools, memory, speech, tts, vision
├── frontend/    # SwiftUI macOS app (Phase 3+)
├── docs/        # architecture docs and ADRs
└── scripts/     # setup / model install scripts
```

## Roadmap

1. ~~Architecture & scaffolding~~ ✅
2. ~~Backend (FastAPI, Ollama integration, first `/chat` endpoint)~~ ✅
3. ~~macOS application (SwiftUI shell, permissions, backend IPC)~~ ✅
4. ~~Voice (wake word, STT, TTS pipeline)~~ ✅
5. ~~Planner (intent → structured tool-call plan)~~ ✅
6. ~~Tools (Finder, Terminal, Git, VS Code, Clipboard, System) + plugins~~ ✅
7. ~~Memory (SQLite + vector store, semantic recall)~~ ✅
8. ~~Vision (on-demand screen understanding)~~ ✅
9. ~~Automation (Playwright browser control end-to-end)~~ ✅
10. ~~Testing (unit + integration test suite, CI)~~ ✅

## License

MIT — see [LICENSE](LICENSE).
