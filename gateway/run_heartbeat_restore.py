"""Recover heartbeat watches from the gateway's canonical persisted routing index."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("gateway.run")


async def restore_heartbeat_watches(runner) -> None:
    """Retryable startup/poll scan; failed reads never prune existing watches.

    SessionStore owns one routing index across profiles. Its origin and exact key,
    rather than a second heartbeat routing snapshot, also cover pre-upgrade state.
    Run all storage work off-loop so a cold profile DB cannot block adapters.
    """
    from gateway.run import _profile_runtime_scope, _multiplex_profile_homes
    from gateway.run_idle_gates import profile_has_active_heartbeat
    from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, migrate_heartbeat_to_route
    from hermes_cli.goals import _get_session_db
    from hermes_constants import get_hermes_home
    store = runner.session_store

    def scan():
        restored = []
        # The poller may have been spawned by a named profile's /heartbeat command.
        # Anchor even default origins to the gateway home, not inherited context.
        home = getattr(store, "_routing_home", None) or get_hermes_home()
        watched_homes = [home]
        if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
            watched_homes.extend(profile_home for _name, profile_home in _multiplex_profile_homes(runner.config))
        if not any(profile_has_active_heartbeat(profile_home) for profile_home in dict.fromkeys(watched_homes)):
            return restored
        with _profile_runtime_scope(home):
            # Enter each profile's scope once per scan; it hydrates secrets and terminal policy.
            # Sources are read through _restored_source so the persisted receiving bot is re-pinned.
            routing_db = getattr(store, "_routing_db", None)
            routing_scope = store._routing_scope()
            try:
                routing_entries = routing_db.load_gateway_routing_entries(scope=routing_scope)
            except Exception:
                logger.debug("heartbeat routing index unavailable; skipping legacy migration", exc_info=True)
                routing_entries = {}
            route_by_session = {}
            for route, raw_entry in routing_entries.items():
                try:
                    data = json.loads(raw_entry)
                except (ValueError, TypeError, RecursionError):
                    continue
                if (isinstance(data, dict) and data.get("session_id")
                        and data.get("session_key") == route and not data.get("suspended")):
                    route_by_session[data["session_id"]] = route
            by_scope: dict = {}
            for entry in store.list_sessions():
                if entry.origin is None or not entry.session_id or entry.suspended:
                    continue
                try:
                    source = runner._restored_source(entry)
                    by_scope.setdefault(runner._profile_scope_key_for_source(source), []).append((entry, source))
                except Exception:
                    logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
            multiplex = bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False))
            if multiplex:
                for _name, profile_home in _multiplex_profile_homes(runner.config):
                    by_scope.setdefault(profile_home, [])
            else:
                by_scope.setdefault(None, [])
            warned = getattr(runner, "_heartbeat_unresolved_legacy", None)
            if warned is None:
                warned = runner._heartbeat_unresolved_legacy = set()
            for scope, group in by_scope.items():
                try:
                    with _profile_runtime_scope(scope or home):
                        db = _get_session_db()
                        if db is None:
                            continue
                        # Current-owner legacy schedules take precedence over archived history.
                        current_session_ids = {entry.session_id for entry, _source in group}
                        legacy = []
                        for meta_key, raw in db.list_meta_prefix("heartbeat:"):
                            if meta_key.startswith("heartbeat:route:"):
                                continue
                            session_id = meta_key[len("heartbeat:"):]
                            try:
                                data = json.loads(raw)
                                if not isinstance(data, dict) or data.get("status") not in {"active", "paused"}:
                                    continue
                                state = HeartbeatState.from_json(raw)
                                if not state.prompt.strip() or state.interval_seconds < 60:
                                    continue
                            except (ValueError, TypeError, AttributeError):
                                continue
                            route = route_by_session.get(session_id)
                            row = db.get_session(session_id)
                            if route is None or row is None or row.get("session_key") != route:
                                unresolved_key = (str(scope or home), session_id)
                                if unresolved_key not in warned:
                                    warned.add(unresolved_key)
                                    logger.warning("heartbeat legacy session %s has no resolvable route", session_id)
                                continue
                            legacy.append((session_id in current_session_ids, state.created_at, session_id, route))
                        for _current, _created_at, session_id, route in sorted(legacy, reverse=True):
                            migrate_heartbeat_to_route(session_id, route, db=db)
                        for entry, source in group:
                            try:
                                if HeartbeatManager(entry.session_id, route_key=entry.session_key, db=db).has_heartbeat():
                                    restored.append((scope, entry.session_key, source, entry.session_id))
                            except Exception:
                                logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
                except Exception:
                    logger.debug("heartbeat restore scope for %s failed", scope or home, exc_info=True)
        return restored

    try:
        candidates = await runner._run_in_executor_with_context(scan)
        for scope in dict.fromkeys(item[0] for item in candidates):
            group = [item for item in candidates if item[0] == scope]
            with runner._profile_scope_for_source(group[0][2]):
                for _scope, key, source, session_id in group:
                    # A reset/compression may have published a new owner during the executor hop.
                    if store.peek_session_id(key) == session_id:
                        db = store._db_for_key(key)
                        if HeartbeatManager(session_id, route_key=key, db=db).is_active():
                            runner._register_heartbeat_watch(key, source, session_id)
    except Exception:
        logger.debug("heartbeat restore scan failed; retrying on next poll", exc_info=True)
