# Jarvis for macOS

A local-first, 100% free and open-source AI desktop assistant for macOS. Control your
computer with natural voice commands — open/close apps, manage files, run terminal
commands, automate the browser, and more — powered entirely by local, open models.

Optimized for a MacBook Air M2 (8GB RAM). No paid APIs. Works offline for most tasks.

## Status

**Phase 4 of 10 — Voice online.** Backend chat API + SwiftUI menu-bar app + a full
local voice loop: "hey jarvis" wake word (openWakeWord), Whisper STT, LLM reply,
spoken TTS response. See [backend/README.md](backend/README.md) for the quickstart
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

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
5. ~~Planner (intent → structured tool-call plan)~~ ✅ ← you are here
6. Tools (Finder, Terminal, Browser, Git, VS Code, Clipboard, System)
7. Memory (SQLite + vector store, semantic recall)
8. Vision (on-demand screen understanding)
9. Automation (Playwright browser control end-to-end)
10. Testing (unit + integration test suite)

## License

MIT — see [LICENSE](LICENSE).
