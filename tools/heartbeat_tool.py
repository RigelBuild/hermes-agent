"""`heartbeat` agent tool: set, clear, status and list route heartbeats.

A target is a routed session id, or ``discord:<channel_id>`` matched against the gateway routing index.
A target with no routed session is refused, and so is a route with no native admission: the tool never
creates a session. Every set and clear is logged.
"""

import json
import logging
import sys
from dataclasses import asdict
from typing import Any, Callable, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger("tools.heartbeat_tool")

_ACTIONS = ("set", "clear", "status", "list")


def _route_entries(routing_db: Any, routing_scope: Optional[str]) -> list[dict]:
    """Routing entries in this scope; unreadable rows are skipped."""
    if routing_db is None or routing_scope is None:
        return []
    rows = routing_db._read_all(
        "SELECT entry_json FROM gateway_routing WHERE scope = ?", (routing_scope,))
    out = []
    for row in rows:
        try:
            data = json.loads(row["entry_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("session_key") and data.get("session_id"):
            out.append(data)
    return out


def _resolve_target(target: str, entries: list[dict]) -> Optional[dict]:
    """The one live routed entry for *target*, or None. A suspended route never matches."""
    live = [data for data in entries if not data.get("suspended")]
    if target.startswith("discord:"):
        chat_id = target[len("discord:"):].strip()
        matches = [data for data in live
                   if data.get("platform") == "discord" and (data.get("origin") or {}).get("chat_id") == chat_id]
    else:
        matches = [data for data in live if data.get("session_id") == target]
    return matches[0] if len(matches) == 1 else None


def _state_dict(state: Any) -> Optional[dict]:
    return asdict(state) if state is not None else None


def heartbeat_action(
    args: dict,
    *,
    db: Any = None,
    routing_db: Any = None,
    routing_scope: Optional[str] = None,
    admission_check: Callable[[dict], bool],
) -> str:
    """Run one heartbeat action and return a JSON string.

    *db* holds the ``heartbeat:route:`` rows; *routing_db* and *routing_scope* locate the routed sessions.
    *admission_check* must return True only when the route has native admission (see ``_native_admission``).
    """
    from hermes_cli.heartbeat import (
        HeartbeatManager, MIN_INTERVAL_SECONDS, list_heartbeats, load_route_heartbeat, parse_interval,
    )

    action = str(args.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        return tool_error(f"unknown action {action!r}; use one of: {', '.join(_ACTIONS)}", success=False)
    if action == "list":
        heartbeats = [{"route": key, **asdict(state)} for key, state in list_heartbeats(db)]
        return json.dumps({"success": True, "heartbeats": heartbeats})

    target = str(args.get("target") or "").strip()
    if not target:
        return tool_error("target is required: a session id or discord:<channel_id>", success=False)
    entry = _resolve_target(target, _route_entries(routing_db, routing_scope))
    if entry is None:
        return tool_error(f"no routed session for {target}", success=False)
    route = entry["session_key"]
    manager = HeartbeatManager(entry["session_id"], route_key=route, db=db)

    if action == "status":
        return json.dumps({"success": True, "route": route,
                           "heartbeat": _state_dict(load_route_heartbeat(route, db))})

    if action == "clear":
        cleared = manager.clear()
        logger.info("heartbeat clear route=%s target=%s", route, target)
        return json.dumps({"success": True, "route": route, "cleared": cleared})

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required for set", success=False)
    raw_interval = str(args.get("interval") or "").strip()
    seconds = parse_interval(raw_interval)
    if seconds is None or seconds < 0:
        return tool_error(
            f"interval {raw_interval!r} must look like 10m or 2h and be at least {MIN_INTERVAL_SECONDS}s",
            success=False)
    if not admission_check(entry):
        return tool_error(
            "this route has no native admission yet; send a normal message in the channel first", success=False)
    state = manager.set(prompt, seconds)
    logger.info("heartbeat set route=%s target=%s interval=%ss", route, target, seconds)
    return json.dumps({"success": True, "route": route, "heartbeat": asdict(state)})


def _native_admission(runner: Any) -> Callable[[dict], bool]:
    """Admission check backed by the live gateway: the route's committed native envelope must exist."""
    def check(entry: dict) -> bool:
        from gateway.session import SessionEntry
        from gateway.session_automation import committed_route_envelope
        from gateway.session_authorities import active_authority
        from datetime import datetime

        authority = active_authority(runner)
        if authority is None:
            return False
        now = datetime.now()
        stub = SessionEntry(entry["session_key"], entry["session_id"], now, now)
        return committed_route_envelope(authority, stub) is not None
    return check


def _handler(args, **kw):
    runner_ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
    runner = runner_ref() if callable(runner_ref) else None
    if runner is None:
        return tool_error("heartbeat needs a running gateway", success=False)
    store = runner.session_store
    return heartbeat_action(
        args,
        db=store._routing_db,
        routing_db=store._routing_db,
        routing_scope=store._routing_scope(),
        admission_check=_native_admission(runner),
    )


def check_heartbeat_requirements() -> bool:
    """Only a gateway session can own a route heartbeat."""
    from utils import env_var_enabled
    return env_var_enabled("HERMES_GATEWAY_SESSION")


HEARTBEAT_SCHEMA = {
    "name": "heartbeat",
    "description": (
        "Set, clear, check or list recurring heartbeats. A heartbeat re-enters a session with a prompt on an "
        "interval. The target is a session id, or discord:<channel_id> for a channel that already has a "
        "session. A target with no session is refused."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS), "description": "set, clear, status or list."},
            "target": {"type": "string", "description": "A session id, or discord:<channel_id>."},
            "interval": {"type": "string", "description": "For set: e.g. 10m or 2h. At least 60s."},
            "prompt": {"type": "string", "description": "For set: the heartbeat prompt."},
        },
        "required": ["action"],
    },
}

registry.register(
    name="heartbeat",
    toolset="heartbeat",
    schema=HEARTBEAT_SCHEMA,
    handler=_handler,
    check_fn=check_heartbeat_requirements,
    emoji="♥",
)
