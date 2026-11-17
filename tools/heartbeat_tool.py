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


def _distinct_dbs(dbs: list[Any]) -> list[Any]:
    """The non-None stores in *dbs*, one per database file."""
    out: dict[str, Any] = {}
    for db in dbs:
        if db is not None:
            out.setdefault(str(getattr(db, "db_path", id(db))), db)
    return list(out.values())


def heartbeat_action(
    args: dict,
    *,
    db_for_key: Callable[[Optional[str]], Any],
    routing_db: Any = None,
    routing_scope: Optional[str] = None,
    admission_check: Callable[[dict], bool],
) -> str:
    """Run one heartbeat action and return a JSON string.

    *db_for_key* returns the store holding a route's rows (the gateway's ``SessionStore._db_for_key``), so a
    named-profile route is read and written in its own profile's store. *routing_db* and *routing_scope*
    locate the routed sessions. *admission_check* must return True only when the route has validated native
    admission.
    """
    from hermes_cli.heartbeat import (
        HeartbeatManager, MIN_INTERVAL_SECONDS, list_heartbeats, load_route_heartbeat, parse_interval,
    )

    action = str(args.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        return tool_error(f"unknown action {action!r}; use one of: {', '.join(_ACTIONS)}", success=False)
    entries = _route_entries(routing_db, routing_scope)
    if action == "list":
        stores = _distinct_dbs([routing_db, *(db_for_key(entry["session_key"]) for entry in entries)])
        found: dict[str, Any] = {}
        for store in stores:
            for key, state in list_heartbeats(store):
                found.setdefault(key, state)
        heartbeats = [{"route": key, **asdict(found[key])} for key in sorted(found)]
        return json.dumps({"success": True, "heartbeats": heartbeats})

    target = str(args.get("target") or "").strip()
    if not target:
        return tool_error("target is required: a session id or discord:<channel_id>", success=False)
    entry = _resolve_target(target, entries)
    if entry is None:
        return tool_error(f"no routed session for {target}", success=False)
    route = entry["session_key"]
    db = db_for_key(route)
    if db is None:
        return tool_error(f"no session store for {target}; retry", success=False)
    manager = HeartbeatManager(entry["session_id"], route_key=route, db=db)

    if action == "status":
        state = load_route_heartbeat(route, db)
        return json.dumps({"success": True, "route": route,
                           "heartbeat": asdict(state) if state else None})

    if action == "clear":
        cleared = manager.clear()
        logger.info("heartbeat clear route=%s target=%s cleared=%s", route, target, cleared)
        # Verify the row: a write that did not land leaves an active heartbeat behind.
        if load_route_heartbeat(route, db) is not None:
            return tool_error("heartbeat clear did not persist; retry", success=False)
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
    # Compare the whole state, not only the prompt: a failed write leaves the previous row behind.
    persisted = load_route_heartbeat(route, db)
    if persisted is None or persisted.to_json() != state.to_json():
        return tool_error("heartbeat set did not persist; retry", success=False)
    logger.info("heartbeat set route=%s target=%s interval=%ss", route, target, seconds)
    return json.dumps({"success": True, "route": route, "heartbeat": asdict(persisted)})


def _native_admission(runner: Any) -> Callable[[dict], bool]:
    """Admission from the live gateway: the same validated committed-envelope check the poller uses.

    The authority is the one owning the route's profile, so a named-profile route is checked against its own
    store. The route's source and adapter come from the stored routing entry's origin, so the envelope is
    checked against this route and not merely for the presence of a provenance key. Any failure denies.
    """
    def check(entry: dict) -> bool:
        from datetime import datetime

        from gateway.session import SessionEntry, SessionSource
        from gateway.session_automation import validated_committed_route_envelope
        from gateway.session_detached import _route_authority

        try:
            authority = _route_authority(runner, runner.session_store, entry.get("session_key"))
            origin = entry.get("origin")
            if authority is None or not origin:
                return False
            source = SessionSource.from_dict(origin)
            adapter = runner._delivery_adapter_for(source)
            if adapter is None:
                return False
            now = datetime.now()
            stored = SessionEntry(entry["session_key"], entry["session_id"], now, now,
                                  origin=source, platform=source.platform)
            validated_committed_route_envelope(authority, adapter, source, stored)
            return True
        except Exception as exc:
            logger.debug("heartbeat admission denied for %s: %s", entry.get("session_key"), exc)
            return False
    return check


def _gateway_runner() -> Optional[Any]:
    """The live gateway runner, or None. Only the gateway process has one: a child that inherited
    ``_HERMES_GATEWAY`` never imports ``gateway.run``, so it gets None."""
    module = sys.modules.get("gateway.run")
    runner_ref = getattr(module, "_gateway_runner_ref", None) if module is not None else None
    return runner_ref() if callable(runner_ref) else None


def _handler(args, **kw):
    runner = _gateway_runner()
    if runner is None:
        return tool_error("heartbeat needs a running gateway", success=False)
    store = runner.session_store
    return heartbeat_action(
        args,
        db_for_key=store._db_for_key,
        routing_db=store._routing_db,
        routing_scope=store._routing_scope(),
        admission_check=_native_admission(runner),
    )


def check_heartbeat_requirements() -> bool:
    """Advertised only while a live gateway runner exists. ``_HERMES_GATEWAY`` alone is not enough: it is
    inherited by gateway-spawned children, which have no runner."""
    return _gateway_runner() is not None


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
