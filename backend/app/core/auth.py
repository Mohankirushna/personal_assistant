"""Bearer-token auth middleware.

Pure ASGI middleware (not BaseHTTPMiddleware) so it covers WebSocket
handshakes as well as HTTP requests. When no token is configured
(local development) it passes everything through; /health stays open either
way so the SwiftUI app can probe liveness before it knows the token.
"""

from __future__ import annotations

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send

_EXEMPT_PATHS = frozenset({"/health"})


class TokenAuthMiddleware:
    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self._app = app
        self._token = token

    def _authorized(self, scope: Scope) -> bool:
        if self._token is None or scope["type"] not in ("http", "websocket"):
            return True
        if scope["path"] in _EXEMPT_PATHS:
            return True
        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        expected = f"Bearer {self._token}"
        return hmac.compare_digest(authorization, expected)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._authorized(scope):
            await self._app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Closing before accepting rejects the handshake (HTTP 403).
            await send({"type": "websocket.close", "code": 4401})
            return
        body = json.dumps({"detail": "Missing or invalid bearer token"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
