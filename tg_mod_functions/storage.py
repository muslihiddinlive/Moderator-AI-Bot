"""Pluggable storage backends for stateful moderation data (warning counts).

Telegram's Bot API has no concept of "warnings" — it's not part of the
platform, so the bot has to remember it itself. This module defines a tiny
interface so you can back it with anything: a dict, a JSON file, Redis,
Postgres, or a Telegram channel/supergroup used as a database.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class WarnStorage(ABC):
    """Abstract interface for storing per-user warning counts."""

    @abstractmethod
    async def add_warn(self, chat_id: int, user_id: int) -> int:
        """Add one warning and return the new total count."""

    @abstractmethod
    async def get_warns(self, chat_id: int, user_id: int) -> int:
        """Return the current warning count (0 if none)."""

    @abstractmethod
    async def reset_warns(self, chat_id: int, user_id: int) -> None:
        """Reset warnings back to zero."""


class InMemoryWarnStorage(WarnStorage):
    """Default storage: a plain dict, lost on process restart.

    Fine for development or small single-process bots. For anything that
    needs to survive a restart, use ``JSONFileWarnStorage`` or implement
    ``WarnStorage`` with your own backend and pass it to
    ``ModerationToolkit(bot, warn_storage=...)``.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], int] = {}
        self._lock = asyncio.Lock()

    async def add_warn(self, chat_id: int, user_id: int) -> int:
        async with self._lock:
            key = (chat_id, user_id)
            self._data[key] = self._data.get(key, 0) + 1
            return self._data[key]

    async def get_warns(self, chat_id: int, user_id: int) -> int:
        return self._data.get((chat_id, user_id), 0)

    async def reset_warns(self, chat_id: int, user_id: int) -> None:
        async with self._lock:
            self._data[(chat_id, user_id)] = 0


class JSONFileWarnStorage(WarnStorage):
    """Persists warning counts to a local JSON file.

    Survives process restarts without needing an external database — a
    good fit for small single-instance deployments (e.g. Render's free
    tier), as long as the disk itself is persistent. Render's free tier
    uses an ephemeral filesystem, so the file is wiped on every redeploy;
    for a backend that survives redeploys on free-tier hosting, use
    something external (Telegram-channel-as-DB, a free Postgres/Redis
    add-on, etc.) instead.

    Writes are atomic (write-to-temp-file + rename) so a crash mid-write
    can't corrupt the file, and an in-process lock serializes access so
    concurrent handlers don't race on the same file.
    """

    def __init__(self, path: str | os.PathLike[str] = "warns.json") -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write({})

    def _read(self) -> dict[str, int]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict[str, int]) -> None:
        # Atomic write: build the new file next to the target, then rename
        # over it. A crash or restart mid-write leaves the old file intact
        # instead of a half-written, corrupt one.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent) or ".", prefix=self._path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self._path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _key(chat_id: int, user_id: int) -> str:
        return f"{chat_id}:{user_id}"

    async def add_warn(self, chat_id: int, user_id: int) -> int:
        async with self._lock:
            data = self._read()
            key = self._key(chat_id, user_id)
            data[key] = data.get(key, 0) + 1
            self._write(data)
            return data[key]

    async def get_warns(self, chat_id: int, user_id: int) -> int:
        data = self._read()
        return data.get(self._key(chat_id, user_id), 0)

    async def reset_warns(self, chat_id: int, user_id: int) -> None:
        async with self._lock:
            data = self._read()
            data[self._key(chat_id, user_id)] = 0
            self._write(data)
