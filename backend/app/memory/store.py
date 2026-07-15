"""SQLite-backed structured memory: command history, preferences, projects.

Synchronous sqlite3 wrapped in asyncio.to_thread — operations are tiny and
WAL mode keeps readers unblocked; a dependency on an async driver isn't
warranted.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    utterance TEXT NOT NULL,
    reply TEXT NOT NULL,
    steps_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_ts ON command_history (ts DESC);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    last_used REAL NOT NULL
);
"""


@dataclass(frozen=True)
class HistoryEntry:
    id: int
    ts: float
    session_id: str
    utterance: str
    reply: str
    steps: list[dict[str, Any]]


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._lock = asyncio.Lock()

    async def record_turn(
        self,
        session_id: str,
        utterance: str,
        reply: str,
        steps: list[dict[str, Any]],
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._connection.execute,
                "INSERT INTO command_history (ts, session_id, utterance, reply, steps_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (time.time(), session_id, utterance, reply, json.dumps(steps)),
            )
            await asyncio.to_thread(self._connection.commit)

    async def history(self, limit: int = 20) -> list[HistoryEntry]:
        def _query() -> list[HistoryEntry]:
            rows = self._connection.execute(
                "SELECT id, ts, session_id, utterance, reply, steps_json"
                " FROM command_history ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                HistoryEntry(
                    id=row[0], ts=row[1], session_id=row[2],
                    utterance=row[3], reply=row[4], steps=json.loads(row[5]),
                )
                for row in rows
            ]

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def set_preference(self, key: str, value: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._connection.execute,
                "INSERT INTO preferences (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await asyncio.to_thread(self._connection.commit)

    async def get_preference(self, key: str) -> str | None:
        def _query() -> tuple[str] | None:
            return self._connection.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()

        async with self._lock:
            row = await asyncio.to_thread(_query)
        return row[0] if row else None

    async def touch_project(self, path: str, name: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._connection.execute,
                "INSERT INTO projects (path, name, last_used) VALUES (?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET last_used = excluded.last_used",
                (path, name, time.time()),
            )
            await asyncio.to_thread(self._connection.commit)

    async def recent_projects(self, limit: int = 10) -> list[tuple[str, str]]:
        def _query() -> list[tuple[str, str]]:
            rows = self._connection.execute(
                "SELECT path, name FROM projects ORDER BY last_used DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [(row[0], row[1]) for row in rows]

        async with self._lock:
            return await asyncio.to_thread(_query)

    def close(self) -> None:
        self._connection.close()
