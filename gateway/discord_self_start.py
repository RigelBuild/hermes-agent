"""File-backed provenance for Discord threads explicitly opened as sessions."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from utils import atomic_json_write

TTL_SECONDS = 600
_MAX_ENTRIES = 500
_THREAD_LOCK_PATH = ".gateway_created_threads.lock"


def _path(home: Path | str | None = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    return Path(home) / "state" / "gateway_created_threads.json"


@contextmanager
def _locked(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """Serialize registry read/modify/write across gateway and sender processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / _THREAD_LOCK_PATH
    with lock_path.open("a+b") as lock_file:
        from pm.filesystem import lock_fd, unlock_fd

        if not lock_fd(lock_file.fileno(), wait=True, timeout=timeout):
            raise TimeoutError("Discord self-start registry lock timed out")
        try:
            yield
        finally:
            unlock_fd(lock_file.fileno())


def _read(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("threads") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {
        str(thread_id): item
        for thread_id, item in entries.items()
        if isinstance(item, dict)
        and isinstance(item.get("parent_id"), str)
        and isinstance(item.get("created_at"), (int, float))
        and math.isfinite(item["created_at"])
    }


def _live(entry: dict[str, Any], now: float) -> bool:
    created_at = entry["created_at"]
    age = now - created_at
    return 0 <= age < TTL_SECONDS


def record_created_thread(
    thread_id: str,
    parent_id: str,
    *,
    created_at: float | None = None,
    home: Path | str | None = None,
) -> None:
    """Record one explicitly opted-in thread, retaining only live bounded entries."""
    thread_id, parent_id = str(thread_id).strip(), str(parent_id).strip()
    if not thread_id or not parent_id:
        return
    now = time.time() if created_at is None else float(created_at)
    if not math.isfinite(now):
        return
    path = _path(home)
    with _locked(path):
        entries = _read(path)
        entries = {key: value for key, value in entries.items() if _live(value, now)}
        entries[thread_id] = {"parent_id": parent_id, "created_at": now}
        if len(entries) > _MAX_ENTRIES:
            entries = dict(
                sorted(entries.items(), key=lambda item: item[1]["created_at"])[
                    -_MAX_ENTRIES:
                ]
            )
        atomic_json_write(path, {"threads": entries}, indent=None, mode=0o600)


def consume_created_thread(
    thread_id: str,
    parent_id: str,
    *,
    now: float | None = None,
    home: Path | str | None = None,
) -> bool:
    """Atomically consume the live thread record only when its parent matches."""
    thread_id, parent_id = str(thread_id).strip(), str(parent_id).strip()
    if not thread_id or not parent_id:
        return False
    current_time = time.time() if now is None else float(now)
    if not math.isfinite(current_time):
        return False
    path = _path(home)
    with _locked(path, timeout=0.25):
        entries = _read(path)
        entry = entries.get(thread_id)
        live = bool(
            entry and entry["parent_id"] == parent_id and _live(entry, current_time)
        )
        retained = {
            key: value for key, value in entries.items() if _live(value, current_time)
        }
        if live:
            del retained[thread_id]
        if retained != entries:
            atomic_json_write(path, {"threads": retained}, indent=None, mode=0o600)
        return live
