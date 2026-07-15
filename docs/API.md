# API Reference

The backend serves REST + WebSocket on `127.0.0.1:8765` (loopback only).
Interactive OpenAPI docs are at `http://127.0.0.1:8765/docs` when it's running.

## Authentication

If `JARVIS_AUTH_TOKEN` is set, every endpoint except `GET /health` requires
`Authorization: Bearer <token>` (HTTP header, or WebSocket handshake header).
Unset = no auth (local dev). The SwiftUI app generates a token per session
when it spawns the backend.

## REST

### `GET /health`
Liveness + Ollama reachability. Never requires auth.
```json
{"status": "ok", "version": "0.2.0",
 "ollama": {"available": true, "loaded_models": ["qwen2.5:3b-instruct-q4_K_M"]},
 "active_model": "qwen2.5:3b-instruct-q4_K_M"}
```

### `POST /chat`
One-shot chat. Runs through the planner (tools + safety gate) when tools are
registered. Cannot prompt for confirmation, so sensitive/destructive tool
calls are denied with an explanation — use the WebSocket for interactive
approval.
```
Request:  {"message": "what time is it?", "session_id": "optional"}
Response: {"session_id": "abc123", "reply": "It's 3:04 PM."}
Errors:   503 if Ollama is down or the model isn't pulled
```

### `GET /tools`
List registered tools and their risk levels.
```json
[{"name": "finder_delete", "description": "...", "risk_level": "destructive"}]
```

### `GET /memory/history?limit=20`
Recent command history (needs the `memory` extra).

### `GET /memory/search?q=...&k=5`
Semantic search over past turns.

### `POST /voice/transcribe` · `POST /voice/speak`
Debug helpers (need the `voice` extra): WAV upload → text, and text → WAV.

## WebSocket

### `WS /ws/chat`
Streaming chat with interactive confirmations.

Client → server:
```json
{"message": "delete old.txt", "session_id": "optional"}
{"type": "confirm_response", "approved": true}
```
Server → client:
```json
{"type": "token", "content": "It"}
{"type": "confirm_request", "tool": "finder_delete", "risk": "destructive",
 "action": "finder_delete {\"path\": \"~/old.txt\"}"}
{"type": "done", "session_id": "abc", "reply": "Deleted it."}
{"type": "error", "message": "..."}          // socket stays open
```
On a `confirm_request`, the exact `action` string shown is the exact call that
will run — reply with `confirm_response`.

### `WS /ws/voice`
Full voice loop (needs the `voice` extra).

Client → server: binary 16kHz mono PCM16 frames, or JSON control
`{"type": "start_listening"}` (push-to-talk) / `{"type": "say", "text": "..."}`.

Server → client:
```json
{"type": "wake", "score": 0.82}
{"type": "listening"}
{"type": "transcript", "text": "what time is it"}
{"type": "reply", "session_id": "abc", "text": "It's 3:04 PM."}
<binary WAV frame>                              // spoken reply
{"type": "audio_end"}
{"type": "nothing_heard"}
{"type": "error", "message": "..."}
```

## Tool risk levels

| Level | Policy |
|---|---|
| `safe` | runs immediately |
| `sensitive` | confirmed once per exact action per session |
| `destructive` | always confirmed |

`terminal_run` and `git` compute risk per command; `browser_fill` is
`destructive` on password fields. See [ARCHITECTURE.md](ARCHITECTURE.md) §5.
