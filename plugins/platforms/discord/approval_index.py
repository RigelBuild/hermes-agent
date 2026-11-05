"""Daily Discord thread and message references for exec-approval indexing."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_STATE_FILENAME = "discord_approval_index.json"
_STATE_SUBDIR = "gateway"
_SUMMARY_LIMIT = 80


class DiscordApprovalIndex:
    """Post pending and settled approval lines in one persisted daily thread."""

    def __init__(self, adapter: Any, channel_id: str) -> None:
        self._adapter = adapter
        self._channel_id = str(channel_id)
        self._thread_lock = asyncio.Lock()
        self._thread_date: str | None = None
        self._thread = None
        self._pending: dict[str, tuple[str, str]] = {}
        self._posting: set[str] = set()
        self._settled: dict[str, str] = {}

    def _state_path(self) -> Path:
        return get_hermes_home() / _STATE_SUBDIR / _STATE_FILENAME

    def _read_state(self) -> dict[str, str]:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(date): str(thread_id) for date, thread_id in data.items() if thread_id}

    async def _write_state(self, date: str, thread_id: str) -> None:
        await asyncio.to_thread(atomic_json_write, self._state_path(), {date: thread_id}, indent=None)

    async def _get_daily_thread(self):
        date = dt.date.today().isoformat()
        if self._thread_date == date and self._thread is not None:
            return self._thread
        async with self._thread_lock:
            if self._thread_date == date and self._thread is not None:
                return self._thread
            parent = await self._adapter._resolve_channel(self._channel_id)
            saved_id = self._read_state().get(date)
            thread = None
            if saved_id:
                try:
                    candidate = await self._adapter._resolve_channel(saved_id)
                    parent_id = getattr(candidate, "parent_id", None)
                    if parent_id is None:
                        parent_id = getattr(getattr(candidate, "parent", None), "id", None)
                    if parent_id is None or str(parent_id) == str(parent.id):
                        thread = candidate
                except Exception:
                    logger.debug("Stored Discord approval index thread %s is unavailable", saved_id)
            if thread is None:
                title = f"Approvals {date}"
                if self._adapter._is_forum_parent(parent):
                    created = await parent.create_thread(name=title, content=title)
                    thread = self._adapter._forum_thread_parts(created)[0]
                else:
                    starter = await parent.send(title)
                    thread = await starter.create_thread(name=title)
                await self._write_state(date, str(thread.id))
            self._thread_date, self._thread = date, thread
            return thread

    @staticmethod
    def _summary(command: str) -> str:
        text = " ".join(str(command or "").split())
        if len(text) > _SUMMARY_LIMIT:
            return text[:_SUMMARY_LIMIT - 1].rstrip() + "…"
        return text

    async def add_pending(self, request_id: str, command: str, jump_url: str) -> None:
        request_id = str(request_id)
        self._posting.add(request_id)
        try:
            thread = await self._get_daily_thread()
            message = await thread.send(f"🟡 pending · {self._summary(command)} · {jump_url}")
            reference = (str(thread.id), str(message.id))
            self._pending[request_id] = reference
            choice = self._settled.pop(request_id, None)
            if choice is not None:
                await self._edit(reference, choice)
        finally:
            self._posting.discard(request_id)

    async def resolve(self, request_id: str, choice: str | None) -> None:
        request_id = str(request_id)
        reference = self._pending.pop(request_id, None)
        if reference is None:
            if request_id in self._posting:
                self._settled[request_id] = choice or ""
            return
        await self._edit(reference, choice)

    async def _edit(self, reference: tuple[str, str], choice: str | None) -> None:
        status = {
            "once": "✅ approved · once",
            "session": "✅ approved · session",
            "always": "✅ approved · always",
            "deny": "❌ denied",
            "timeout": "⌛ timed out",
        }.get(choice)
        if status is None:
            return
        thread_id, message_id = reference
        thread = await self._adapter._resolve_channel(thread_id)
        message = await thread.fetch_message(int(message_id))
        await message.edit(content=status)
