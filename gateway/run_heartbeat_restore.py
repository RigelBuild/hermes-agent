"""Recover heartbeat watches from the gateway's canonical persisted routing index."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("gateway.run")


async def restore_heartbeat_watches(runner) -> None:
    """Retryable startup/poll scan; failed reads never prune existing watches.

    All storage reads and profile scope entries run in the executor.
    """
    from gateway.run import _profile_runtime_scope, _multiplex_profile_homes
    from gateway.run_idle_gates import profile_has_active_heartbeat
    from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, _legacy_route, migrate_heartbeat_to_route
    from hermes_cli.goals import _get_session_db
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override

    store = runner.session_store

    def scan():
        home = getattr(store, "_routing_home", None) or get_hermes_home()
        multiplex = bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False))
        profile_homes = list(_multiplex_profile_homes(runner.config)) if multiplex else []
        watched_homes = [home, *(profile_home for _name, profile_home in profile_homes)]
        if not any(profile_has_active_heartbeat(path) for path in dict.fromkeys(watched_homes)):
            return []
        groups = {}
        token = set_hermes_home_override(str(home))
        try:
            for entry in store.list_sessions():
                if entry.origin is None or not entry.session_id or entry.suspended:
                    continue
                try:
                    source = runner._restored_source(entry)
                    groups.setdefault(runner._profile_scope_key_for_source(source), []).append((entry, source))
                except Exception:
                    logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
        finally:
            reset_hermes_home_override(token)
        for _name, profile_home in profile_homes:
            groups.setdefault(profile_home, [])
        if not multiplex:
            groups.setdefault(None, [])
        warned = getattr(runner, "_heartbeat_unresolved_legacy", None)
        if warned is None:
            warned = runner._heartbeat_unresolved_legacy = set()
        restored = []
        for scope, group in groups.items():
            try:
                context = (runner._profile_scope_for_source(group[0][1]) if group else
                           _profile_runtime_scope(scope) if scope is not None else runner._standalone_launch_scope())
                with context:
                    db = _get_session_db()
                    if db is None:
                        continue
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
                            route, archived = _legacy_route(session_id, db, store._routing_db, store._routing_scope())
                            if route and not archived:
                                migrate_heartbeat_to_route(session_id, route, db=db)
                                continue
                        except (ValueError, TypeError, AttributeError, RuntimeError):
                            pass
                        unresolved_key = (str(scope or home), session_id)
                        if unresolved_key not in warned:
                            warned.add(unresolved_key)
                            logger.warning("heartbeat legacy session %s has no resolvable route", session_id)
                    for entry, source in group:
                        try:
                            if (store.peek_session_id(entry.session_key) == entry.session_id and
                                    HeartbeatManager(entry.session_id, route_key=entry.session_key, db=db).is_active()):
                                restored.append((entry.session_key, source, entry.session_id))
                        except Exception:
                            logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
            except Exception:
                logger.debug("heartbeat restore scope for %s failed", scope or home, exc_info=True)
        return restored

    try:
        for key, source, session_id in await runner._run_in_executor_with_context(scan):
            runner._register_heartbeat_watch(key, source, session_id)
    except Exception:
        logger.debug("heartbeat restore scan failed; retrying on next poll", exc_info=True)
