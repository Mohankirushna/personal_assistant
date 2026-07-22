"""In-process holder for the device's current city.

The backend runs as a bundle-less subprocess, and macOS Location Services
refuses to grant (or even prompt) a process without its own app-bundle
identity — so the backend cannot obtain location itself. The SwiftUI app,
which does have that identity, fetches the location and POSTs the resolved
city here; the morning briefing reads it for accurate local weather.

A reported location is ignored once stale (the user may have travelled and
closed the laptop), so the briefing falls back to IP geolocation rather than
naming yesterday's city.
"""

from __future__ import annotations

import threading
import time

_MAX_AGE_SECONDS = 6 * 3600  # a location older than this is treated as unknown

_lock = threading.Lock()
_city: str | None = None
_updated_at: float = 0.0


def set_city(city: str) -> None:
    global _city, _updated_at
    with _lock:
        _city = city.strip() or None
        _updated_at = time.monotonic()


def get_city() -> str | None:
    with _lock:
        if _city is None or time.monotonic() - _updated_at > _MAX_AGE_SECONDS:
            return None
        return _city


def _reset_for_tests() -> None:
    global _city, _updated_at
    with _lock:
        _city = None
        _updated_at = 0.0
