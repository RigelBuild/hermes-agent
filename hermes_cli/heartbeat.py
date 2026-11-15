"""Session heartbeats — recurring re-entry prompts for the current session.

Session-scoped and in-process (CLI or gateway must be running); durable cross-process scheduling stays
``hermes cron``. Injection is a plain user message with no system-prompt mutation or toolset swap.
Heartbeats default to idle-only; gateway sessions may opt into busy-turn steering."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_INTERVAL_SECONDS = 60  # floor: re-entering more often than once a minute is a busy-loop, not a heartbeat
POLL_SECONDS = 5.0  # how often drivers poll for due heartbeats; not user-facing

HEARTBEAT_PROMPT_TEMPLATE = (
    "[Heartbeat — recurring instruction, fires every {interval}]\n{prompt}\n\n"
    "If there is nothing meaningful to do or report for this instruction "
    "right now, reply briefly that nothing has changed and stop — do not invent work."
)

_INTERVAL_RE = re.compile(
    r"^\s*(?:every\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)\s*$", re.IGNORECASE)

_UNIT_SECONDS = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), 1),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), 60),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 3600),
    **dict.fromkeys(("d", "day", "days"), 86400),
}

# field -> (coercer, default used when the stored value is missing/falsy)
_STATE_FIELDS = {
    "prompt": (str, ""), "interval_seconds": (int, 0), "status": (str, "active"),
    "created_at": (float, 0.0), "last_fired_at": (float, 0.0), "fire_count": (int, 0),
    "delivery_mode": (str, "steer"),
    "last_error": (str, ""),
}


def parse_interval(text: str) -> Optional[int]:
    """Parse ``10m`` / ``every 2h`` / ``every 90 minutes`` into seconds.

    None when not an interval; below ``MIN_INTERVAL_SECONDS`` returns -1 so callers can tell "too small" apart.
    """
    m = _INTERVAL_RE.match(text) if text else None
    if not m:
        return None
    seconds = int(float(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()])
    return -1 if seconds < MIN_INTERVAL_SECONDS else seconds


def parse_heartbeat_set(text: str) -> tuple[Optional[int], str, str]:
    """Parse the interval and optional leading delivery flag without rewriting prompt text."""
    match = re.match(r"^\s*(?:every\s+)?(\S+)(?:\s+(.*))?$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None, "", "steer"
    interval = parse_interval(match.group(1))
    prompt = match.group(2) or ""
    flag = re.match(r"^--(idle|steer)(?:\s+|$)", prompt)
    return interval, prompt[flag.end():] if flag else prompt, flag.group(1) if flag else "steer"


def format_interval(seconds: int) -> str:
    """Human-readable interval (``600`` → ``10m``)."""
    seconds = int(seconds)
    units = ((86400, "d"), (3600, "h"), (60, "m"))
    return next((f"{seconds // unit}{suffix}" for unit, suffix in units if seconds % unit == 0), f"{seconds}s")


@dataclass
class HeartbeatState:
    """Serializable per-session heartbeat."""

    prompt: str
    interval_seconds: int
    status: str = "active"          # active | paused | cleared
    created_at: float = 0.0
    last_fired_at: float = 0.0
    fire_count: int = 0
    delivery_mode: str = "steer"  # idle | steer
    last_error: str = ""

    def to_json(self) -> str:
        from hermes_cli.debug_redaction import redact_debug_support_text
        data = asdict(self)
        data["last_error"] = redact_debug_support_text(self.last_error, max_chars=500)
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> HeartbeatState:
        data = json.loads(raw)
        return cls(**{name: coerce(data.get(name) or default) for name, (coerce, default) in _STATE_FIELDS.items()})

    def is_due(self, now: Optional[float] = None) -> bool:
        if self.status != "active" or not self.prompt or self.interval_seconds <= 0:
            return False
        return (time.time() if now is None else now) - (self.last_fired_at or self.created_at) >= self.interval_seconds

    def render_prompt(self) -> str:
        return HEARTBEAT_PROMPT_TEMPLATE.format(interval=format_interval(self.interval_seconds), prompt=self.prompt)


def _get_session_db() -> Optional[Any]:
    """Persistence goes through the goals module's per-HERMES_HOME cached SessionDB (one shared connection)."""
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
        return _goals_db()
    except Exception as exc:  # pragma: no cover
        logger.debug("HeartbeatManager: SessionDB bootstrap failed (%s)", exc)
        return None


_META_PREFIX = "heartbeat:"
_ROUTE_PREFIX = _META_PREFIX + "route:"
_CLAIM_LOCK = threading.Lock()  # every driver of a session's heartbeat runs in its owning gateway process


_ROUTING_SCOPE_CACHE: dict[str, str] = {}
_ROUTING_SCOPE_LOCK = threading.Lock()



def _default_routing_context() -> tuple[Any, Optional[str]]:
    """Return the root routing DB and its active scope, or no routing context."""
    try:
        from hermes_constants import (
            get_routing_process_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home = get_routing_process_hermes_home()
        home_key = str(home.resolve())
        with _ROUTING_SCOPE_LOCK:
            scope = _ROUTING_SCOPE_CACHE.get(home_key)
        token = set_hermes_home_override(str(home))
        try:
            from hermes_cli.goals import _get_session_db

            if scope is None:
                from gateway.config import load_gateway_config
                from gateway.session import SessionStore

                config = load_gateway_config()
                store = object.__new__(SessionStore)
                store.sessions_dir = config.sessions_dir
                scope = store._routing_scope()
                with _ROUTING_SCOPE_LOCK:
                    _ROUTING_SCOPE_CACHE[home_key] = scope
            db = _get_session_db()
            return (db, scope) if db is not None else (None, None)
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        logger.debug("HeartbeatManager: routing context unavailable: %s", exc)
        return None, None


def _legacy_route(
    session_id: str,
    db: Any,
    routing_db: Any = None,
    routing_scope: Optional[str] = None,
) -> tuple[Optional[str], bool]:
    """Resolve a session only when the scoped routing index confirms its current owner."""
    if not session_id:
        return None, False
    if routing_db is None:
        return None, False
    try:
        row = db.get_session(session_id) if db is not None else None
        route = row.get("session_key") if row else None
        if not route:
            return None, False
        if routing_scope is None:
            return None, True
        scope = "" if ":local:" in route else routing_scope
        entries = routing_db._read_all(
            "SELECT entry_json FROM gateway_routing WHERE scope = ? AND session_key = ?",
            (scope, route),
        )
        for entry in entries:
            try:
                data = json.loads(entry["entry_json"])
            except (ValueError, TypeError, RecursionError):
                continue
            if (isinstance(data, dict) and data.get("session_key") == route
                    and data.get("session_id") == session_id and not data.get("suspended")):
                return route, False
        return None, True
    except Exception as exc:
        logger.debug("HeartbeatManager: route lookup failed: %s", exc)
        return None, True

def load_heartbeat(session_id: str) -> Optional[HeartbeatState]:
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    try:
        raw = db.get_meta(_META_PREFIX + session_id)
    except Exception as exc:
        logger.debug("HeartbeatManager: get_meta failed: %s", exc)
        return None
    try:
        state = HeartbeatState.from_json(raw) if raw else None
    except Exception as exc:
        logger.warning("HeartbeatManager: could not parse stored heartbeat for %s: %s", session_id, exc)
        return None
    return None if state is None or state.status == "cleared" else state



def store_has_active_heartbeat(db: Any) -> bool:
    """True when *db* holds an ACTIVE ``heartbeat:*`` row — or one that cannot be parsed (unknown, so
    the caller keeps its full sweep). ``clear``/``pause`` keep their rows (status ``cleared``/``paused``),
    so key existence alone is not "active". Read errors propagate: "unavailable" is the caller's call."""
    for _key, raw in db.list_meta_prefix(_META_PREFIX):
        try:
            if HeartbeatState.from_json(raw).status == "active":
                return True
        except Exception:
            return True
    return False


def save_heartbeat(session_id: str, state: HeartbeatState) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write
        _warn_dropped_write("HeartbeatManager", "heartbeat", session_id)
        return
    try:
        db.set_meta(_META_PREFIX + session_id, state.to_json())
    except Exception as exc:
        logger.debug("HeartbeatManager: set_meta failed: %s", exc)


def load_route_heartbeat(route_key: str, db: Any = None) -> Optional[HeartbeatState]:
    db = db if db is not None else _get_session_db()
    if not route_key or db is None:
        return None
    try:
        raw = db.get_meta(_ROUTE_PREFIX + route_key)
        state = HeartbeatState.from_json(raw) if raw else None
        return None if state is None or state.status == "cleared" else state
    except Exception as exc:
        logger.warning("HeartbeatManager: could not load route heartbeat: %s", exc)
        return None


def save_route_heartbeat(route_key: str, state: HeartbeatState, db: Any = None) -> bool:
    db = db if db is not None else _get_session_db()
    if not route_key or db is None:
        return False
    try:
        db.set_meta(_ROUTE_PREFIX + route_key, state.to_json())
        return True
    except Exception as exc:
        logger.debug("HeartbeatManager: route set_meta failed: %s", exc)
        return False


def list_heartbeats(db: Any = None) -> list[tuple[str, HeartbeatState]]:
    db = db if db is not None else _get_session_db()
    if db is None:
        return []
    result = []
    for key, raw in db.list_meta_prefix(_ROUTE_PREFIX):
        try:
            state = HeartbeatState.from_json(raw)
            if state.status in {"active", "paused"} and state.prompt.strip() and state.interval_seconds >= MIN_INTERVAL_SECONDS:
                result.append((key[len(_ROUTE_PREFIX):], state))
        except (ValueError, TypeError, AttributeError):
            continue
    return sorted(result, key=lambda item: item[0])


def migrate_heartbeat_to_route(session_id: str, route_key: str, db: Any = None) -> bool:
    db = db if db is not None else _get_session_db()
    if not session_id or not route_key or db is None:
        return False
    with _CLAIM_LOCK:
        try:
            def migrate(conn):
                legacy_key = _META_PREFIX + session_id
                route_meta = _ROUTE_PREFIX + route_key
                legacy = conn.execute("SELECT value FROM state_meta WHERE key = ?", (legacy_key,)).fetchone()
                if not legacy:
                    return False
                data = json.loads(legacy[0])
                if not isinstance(data, dict) or data.get("status") not in {"active", "paused"}:
                    return False
                if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
                    return False
                if HeartbeatState.from_json(legacy[0]).interval_seconds < MIN_INTERVAL_SECONDS:
                    return False
                existing = conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (route_meta,)).fetchone()
                if existing is None:
                    conn.execute("INSERT INTO state_meta(key, value) VALUES (?, ?)", (route_meta, legacy[0]))
                conn.execute("DELETE FROM state_meta WHERE key = ?", (legacy_key,))
                return existing is None
            return bool(db._execute_write(migrate))
        except Exception as exc:
            logger.debug("HeartbeatManager: route migration failed: %s", exc)
            return False


class HeartbeatManager:
    """Per-session heartbeat state + due-tick decisions; the surface CLI + gateway talk to.

    Drivers (CLI thread / gateway task) call :meth:`due_prompt` on a poll cadence while the session is
    idle; a non-None return is the user-role message to inject.
    """

    def __init__(
        self,
        session_id: str,
        *,
        route_key: Optional[str] = None,
        db: Any = None,
        routing_db: Any = None,
        routing_scope: Optional[str] = None,
    ):
        self.session_id = session_id
        self.route_key = route_key
        self.db = db
        if not route_key and routing_db is None and routing_scope is None:
            routing_db, routing_scope = _default_routing_context()
        self.routing_db = routing_db
        self.routing_scope = routing_scope
        if not self.route_key:
            route, archived = self._storage()
            if route and not archived:
                migrate_heartbeat_to_route(self.session_id, route, db=self.db)
        self._state: Optional[HeartbeatState] = self._load()
        self._last_claim: Optional[tuple[float, int]] = None

    def _storage(self) -> tuple[Optional[str], bool]:
        if self.route_key:
            return self.route_key, False
        db = self.db if self.db is not None else _get_session_db()
        return _legacy_route(self.session_id, db, self.routing_db, self.routing_scope)

    def _load(self) -> Optional[HeartbeatState]:
        route, archived = self._storage()
        if archived:
            return None
        return load_route_heartbeat(route, self.db) if route else load_heartbeat(self.session_id)

    def _save(self, state: HeartbeatState) -> None:
        route, archived = self._storage()
        if archived:
            return
        if route:
            save_route_heartbeat(route, state, self.db)
        else:
            save_heartbeat(self.session_id, state)

    @property
    def state(self) -> Optional[HeartbeatState]:
        return self._state

    def has_heartbeat(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def status_line(self) -> str:
        s = self._state
        if s is None:
            return "No heartbeat. Set one with /heartbeat every <interval> <prompt>."
        every = format_interval(s.interval_seconds)
        mode = "steer (gateway only)" if s.delivery_mode == "steer" else s.delivery_mode
        fired = f", fired {s.fire_count}×" if s.fire_count else ""
        if s.status == "active":
            next_in = max(0, int((s.last_fired_at or s.created_at) + s.interval_seconds - time.time()))
            return f"♥ Heartbeat (every {every}, next in ~{next_in}s{fired}, mode {mode}): {s.prompt}"
        icon = "⏸ " if s.status == "paused" else ""
        return f"{icon}Heartbeat ({s.status}, every {every}{fired}, mode {mode}): {s.prompt}"

    def set(self, prompt: str, interval_seconds: int, *, delivery_mode: str = "steer") -> HeartbeatState:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("heartbeat prompt is empty")
        interval_seconds = int(interval_seconds)
        if interval_seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(f"interval must be at least {MIN_INTERVAL_SECONDS}s")
        if delivery_mode not in {"idle", "steer"}:
            raise ValueError("delivery mode must be idle or steer")
        with _CLAIM_LOCK:
            self._state = HeartbeatState(prompt=prompt, interval_seconds=interval_seconds, status="active",
                                         created_at=time.time(), delivery_mode=delivery_mode)
            self._save(self._state)
        return self._state

    def _set_status(self, status: str, *, reanchor: bool = False) -> Optional[HeartbeatState]:
        with _CLAIM_LOCK:
            self._state = self._load()
            if not self._state:
                return None
            self._state.status = status
            if reanchor:
                self._state.last_fired_at = time.time()
            self._save(self._state)
        return self._state

    def pause(self) -> Optional[HeartbeatState]:
        return self._set_status("paused")

    def resume(self) -> Optional[HeartbeatState]:
        # Re-anchor so resuming doesn't instantly fire a stale tick.
        return self._set_status("active", reanchor=True)

    def clear(self) -> bool:
        cleared = self._set_status("cleared") is not None
        self._state = None
        return cleared

    def due_prompt(self, now: Optional[float] = None) -> Optional[str]:
        """Return the injection prompt if the heartbeat is due, else None.

        The fire is recorded immediately (before the turn runs) so overlapping polls or a long turn can never
        double-fire the same tick. Missed ticks coalesce: the anchor resets to NOW, not the theoretical
        schedule.
        """
        with _CLAIM_LOCK:
            s = self._state
            if s is None or not s.is_due(now):
                return None
            # Compare-and-set on the stored row: the gateway poller and a viewer's due_prompt RPC each hold
            # their own manager over it, and only one may take the tick. A row cleared or paused since is lost too.
            stored = self._load()
            if stored is None or stored.status != "active" or (
                    stored.last_fired_at, stored.fire_count) != (s.last_fired_at, s.fire_count):
                self._state = stored
                return None
            self._last_claim = (s.last_fired_at, s.fire_count)
            s.last_fired_at = now if now is not None else time.time()
            s.fire_count += 1
            self._save(s)
            return s.render_prompt()

    def abandon_fire(self) -> bool:
        """Rewind the fire recorded by the last :meth:`due_prompt` whose turn never started, so the tick stays
        due for the next poll instead of being silently consumed. Mirrors ``LoopManager.abandon_tick``. Skipped
        (False) when the persisted state moved on — a pause/resume/clear that landed in between wins."""
        with _CLAIM_LOCK:
            claim, s = self._last_claim, self._state
            if claim is None or s is None:
                return False
            current = self._load()
            if current is None or current.status != "active" or (current.last_fired_at, current.fire_count) != (
                    s.last_fired_at, s.fire_count):
                return False
            current.last_fired_at, current.fire_count = claim
            self._last_claim = None
            self._state = current
            self._save(current)
            return True

    def record_error(self, error: object) -> None:
        from hermes_cli.debug_redaction import redact_debug_support_text
        with _CLAIM_LOCK:
            current = self._load()
            if current is None:
                self._state = None
                return
            current.last_error = redact_debug_support_text(error, max_chars=500) if error else ""
            self._state = current
            self._save(current)


def migrate_heartbeat_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry a heartbeat across a compression session rotation (copy to child, archive parent, never raise).

    Same shape as ``goals.migrate_goal_to_session``.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_heartbeat(old_session_id)
        if state is None or load_heartbeat(new_session_id) is not None:
            return False
        save_heartbeat(new_session_id, state)
        state.status = "cleared"
        save_heartbeat(old_session_id, state)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("HeartbeatManager: migration failed: %s", exc)
        return False


__all__ = [
    "HEARTBEAT_PROMPT_TEMPLATE",
    "MIN_INTERVAL_SECONDS",
    "POLL_SECONDS",
    "HeartbeatManager",
    "HeartbeatState",
    "format_interval",
    "load_heartbeat",
    "load_route_heartbeat",
    "save_route_heartbeat",
    "migrate_heartbeat_to_route",
    "list_heartbeats",
    "migrate_heartbeat_to_session",
    "parse_heartbeat_set",
    "parse_interval",
    "save_heartbeat",
]
