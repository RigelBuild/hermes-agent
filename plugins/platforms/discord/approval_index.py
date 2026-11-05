"""Daily Discord thread and message references for exec-approval indexing."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_STATE_FILENAME = "discord_approval_index.json"
_STATE_SUBDIR = "gateway"
_SUMMARY_LIMIT = 80
_INDEX_OPERATION_TIMEOUT_SECONDS = 5.0
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()<>#+.!|~])")


class DiscordApprovalIndex:
    """Post pending and settled approval lines in one persisted daily thread."""

    def __init__(self, adapter: Any, channel_id: str) -> None:
        self._adapter = adapter
        self._channel_id = str(channel_id)
        self._thread_lock = asyncio.Lock()
        self._thread_date: str | None = None
        self._thread = None
        self._pending: dict[str, tuple[str, str, str, str]] = {}
        self._posting: set[str] = set()
        self._settled: dict[str, str] = {}
        self._reservations: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

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

    async def _write_state(self, date: str, thread_id: str | None) -> None:
        state = {date: thread_id} if thread_id else {}
        await asyncio.to_thread(atomic_json_write, self._state_path(), state, indent=None)

    @staticmethod
    def _thread_unavailable(error: Exception) -> bool:
        status = getattr(error, "status", None)
        code = getattr(error, "code", None)
        return status == 404 or code in {10003, 50083}

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
                    if (parent_id is None or str(parent_id) == str(parent.id)) and not (
                            getattr(candidate, "locked", False) or getattr(candidate, "archived", False)):
                        thread = candidate
                except Exception:
                    logger.debug("Stored Discord approval index thread %s is unavailable", saved_id)
            if thread is None:
                if saved_id:
                    await self._write_state(date, None)
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

    async def _invalidate_thread(self, thread) -> None:
        date = dt.date.today().isoformat()
        async with self._thread_lock:
            if self._thread is thread:
                self._thread_date = None
                self._thread = None
                await self._write_state(date, None)

    @staticmethod
    def _summary(command_preview: str) -> str:
        text = " ".join(str(command_preview or "").split())
        text = _MARKDOWN.sub(lambda match: "\\" + match.group(1), text).replace("@", "@\u200b")
        if len(text) > _SUMMARY_LIMIT:
            return text[:_SUMMARY_LIMIT - 1].rstrip() + "…"
        return text

    def reserve_pending(self, request_id: str) -> None:
        """Buffer terminal outcomes from before the prompt send is acknowledged."""
        self._reservations.add(str(request_id))

    def discard_pending_reservation(self, request_id: str) -> None:
        request_id = str(request_id)
        self._reservations.discard(request_id)
        self._settled.pop(request_id, None)

    def start_pending(self, request_id: str, command_preview: str, jump_url: str) -> None:
        request_id = str(request_id)
        task = asyncio.create_task(self._add_pending_bounded(request_id, command_preview, jump_url))
        self._reservations.discard(request_id)
        self._posting.add(request_id)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def start_resolve(self, request_id: str, choice: str | None) -> None:
        task = asyncio.create_task(self._resolve_bounded(str(request_id), choice))
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.warning(
                "Discord approval index background task failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _add_pending_bounded(self, request_id: str, command_preview: str, jump_url: str) -> None:
        try:
            await asyncio.wait_for(
                self.add_pending(request_id, command_preview, jump_url),
                timeout=_INDEX_OPERATION_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("Discord approval index post timed out for request %s", request_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Discord approval index could not record pending prompt", exc_info=True)
        finally:
            self._posting.discard(request_id)
            self._settled.pop(request_id, None)
            self._reservations.discard(request_id)

    async def _resolve_bounded(self, request_id: str, choice: str | None) -> None:
        try:
            await asyncio.wait_for(
                self.resolve(request_id, choice), timeout=_INDEX_OPERATION_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("Discord approval index edit timed out for request %s", request_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Discord approval index could not update resolved prompt", exc_info=True)

    async def add_pending(self, request_id: str, command_preview: str, jump_url: str) -> None:
        request_id = str(request_id)
        self._posting.add(request_id)
        try:
            summary = self._summary(command_preview)
            content = f"🟡 pending · {summary} · {jump_url}"
            thread = await self._get_daily_thread()
            try:
                message = await self._send_pending_line(thread, content)
            except Exception as error:
                if not self._thread_unavailable(error):
                    raise
                await self._invalidate_thread(thread)
                thread = await self._get_daily_thread()
                message = await self._send_pending_line(thread, content)
            reference = (str(thread.id), str(message.id), summary, jump_url)
            choice = self._settled.pop(request_id, None)
            if choice is None:
                self._pending[request_id] = reference
            else:
                await self._edit(reference, choice)
        except BaseException:
            self._settled.pop(request_id, None)
            self._pending.pop(request_id, None)
            raise
        finally:
            self._posting.discard(request_id)

    async def _send_pending_line(self, thread, content: str):
        import discord
        return await thread.send(
            content, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
        )

    async def resolve(self, request_id: str, choice: str | None) -> None:
        request_id = str(request_id)
        reference = self._pending.pop(request_id, None)
        if reference is None:
            if request_id in self._posting or request_id in self._reservations:
                self._settled[request_id] = choice or ""
            return
        await self._edit(reference, choice)

    async def _edit(self, reference: tuple[str, str, str, str], choice: str | None) -> None:
        status = {
            "once": "✅ approved · once",
            "session": "✅ approved · session",
            "always": "✅ approved · always",
            "deny": "❌ denied",
            "timeout": "⌛ timed out",
            "cancelled": "⛔ cancelled",
        }.get(choice)
        if status is None:
            return
        thread_id, message_id, summary, jump_url = reference
        thread = await self._adapter._resolve_channel(thread_id)
        message = await thread.fetch_message(int(message_id))
        await message.edit(content=f"{status} · {summary} · {jump_url}")
