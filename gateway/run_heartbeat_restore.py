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
    from gateway.run import _profile_runtime_scope
    from gateway.session import SessionSource
    from hermes_cli.heartbeat import HeartbeatManager, migrate_heartbeat_to_route
    from hermes_constants import get_hermes_home

    store = runner.session_store

    def scan():
        restored = []
        # The poller may have been spawned by a named profile's /heartbeat command.
        # Anchor even default origins to the gateway home, not inherited context.
        home = getattr(store, "_routing_home", None) or get_hermes_home()
        with _profile_runtime_scope(home):
            # Enter each profile's scope once per scan, not once per routed session: a scope entry
            # hydrates the secret scope and terminal policy, so N sessions cost N parses otherwise.
            # Sources are read through _restored_source so the persisted receiving bot is re-pinned
            # before the scope key is derived; the same source object is what gets registered.
            by_scope: dict = {}
            for entry in store.list_sessions():
                if entry.origin is None or not entry.session_id or entry.suspended:
                    continue
                try:
                    source = runner._restored_source(entry)
                    by_scope.setdefault(runner._profile_scope_key_for_source(source), []).append((entry, source))
                except Exception:
                    logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
            for group in by_scope.values():
                try:
                    with runner._profile_scope_for_source(group[0][1]):
                        db = store._db_for_key(group[0][0].session_key)
                        # Current owners take precedence over older rows for the same route.
                        for entry, _source in group:
                            migrate_heartbeat_to_route(entry.session_id, entry.session_key, db=db)
                        for meta_key, _raw in db.list_meta_prefix("heartbeat:"):
                            if meta_key.startswith("heartbeat:route:"):
                                continue
                            session_id = meta_key[len("heartbeat:"):]
                            row = db.get_session(session_id)
                            if not row:
                                continue
                            route = row.get("session_key")
                            if not route and row.get("origin_json"):
                                try:
                                    origin = SessionSource.from_dict(json.loads(row["origin_json"]))
                                    route = runner._session_key_for_source(origin)
                                except (ValueError, TypeError, KeyError):
                                    continue
                            if route:
                                migrate_heartbeat_to_route(session_id, route, db=db)
                        for entry, source in group:
                            try:
                                db = store._db_for_key(entry.session_key)
                                if HeartbeatManager(entry.session_id, route_key=entry.session_key, db=db).has_heartbeat():
                                    restored.append((entry.session_key, source, entry.session_id))
                            except Exception:
                                logger.debug("heartbeat restore for %s failed", entry.session_key, exc_info=True)
                except Exception:
                    logger.debug("heartbeat restore scope for %s failed", group[0][0].session_key, exc_info=True)
        return restored

    try:
        candidates = await runner._run_in_executor_with_context(scan)
        for key, source, session_id in candidates:
            # A reset/compression may have published a new owner during the executor hop.
            if store.peek_session_id(key) == session_id:
                with runner._profile_scope_for_source(source):
                    db = store._db_for_key(key)
                    if HeartbeatManager(session_id, route_key=key, db=db).has_heartbeat():
                        runner._register_heartbeat_watch(key, source, session_id)
    except Exception:
        logger.debug("heartbeat restore scan failed; retrying on next poll", exc_info=True)
