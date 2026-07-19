# Frontend (macOS App)

SwiftUI menu-bar app for Jarvis, built as a Swift Package (no Xcode project
required — builds with Command Line Tools alone).

## Status

**Phase 3 complete.** The app:

- lives in the menu bar (`MenuBarExtra`) with a status indicator and a chat
  window,
- attaches to a running backend, or **spawns one itself** (`uv run
  jarvis-backend`) with a generated per-session bearer token passed via
  `JARVIS_AUTH_TOKEN` — so a backend started by the app only answers to the
  app,
- streams replies token-by-token over `WS /ws/chat`.

Voice UI (mic streaming, push-to-talk) lands in Phase 4.

## Layout

```
JarvisApp/
├── Package.swift
└── Sources/
    ├── JarvisAppKit/        # non-UI: wire types, REST+WS client, process manager
    ├── JarvisApp/           # SwiftUI app (menu bar, chat window)
    └── JarvisAppSelfTest/   # assertion-based checks (see Testing below)
```

## Build & run

```bash
cd frontend/JarvisApp
swift build                     # compile
swift run jarvis-app-selftest   # run the checks

# Bundle a real Jarvis.app (needed for mic/automation permission prompts):
../../scripts/make_app.sh
open ../dist/Jarvis.app
```

The backend directory is resolved from `JARVIS_BACKEND_DIR`, the
`backendDirectory` user default, or `../../backend` relative to the app —
set the env var when running from a non-standard location:

```bash
JARVIS_BACKEND_DIR=~/Downloads/projects/jarvis_v2/backend open ../dist/Jarvis.app
```

## Testing

Command Line Tools ship neither XCTest nor swift-testing, and building
swift-testing from source (swift-syntax) is unreasonable on 8GB RAM. Wire
decoding and client behavior are therefore covered by the
`jarvis-app-selftest` executable — plain assertions, nonzero exit on failure,
CI-friendly. With full Xcode installed these can be promoted to a real test
target.

## Known limitations

- The bundle is ad-hoc signed; first launch may require right-click → Open.

(A force-killed app used to leak its spawned backend, which then blocked
the next session with an unknown auth token. The backend is now leashed to
the app via its stdin pipe — `JARVIS_EXIT_ON_STDIN_CLOSE` — and exits when
the app dies, however it dies; the app also refuses to attach to a backend
it cannot authenticate to, with an error naming the fix.)
