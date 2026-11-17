"""The `heartbeat` agent tool: set, clear, status and list heartbeats on a routed session or Discord channel."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli import goals
from hermes_cli.heartbeat import list_heartbeats, load_route_heartbeat
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import heartbeat_tool

CHANNEL = "111222333"
OTHER_CHANNEL = "999888777"


@pytest.fixture
def ctx():
    goals._DB_CACHE.clear()
    db = goals._get_session_db()
    scope = SessionStore(GatewayConfig().sessions_dir, GatewayConfig())._routing_scope()
    yield db, scope
    goals._DB_CACHE.clear()


def _route(db, scope, chat_id, session_id, *, suspended=False):
    """Create a routed Discord channel session the way the gateway does (routing index + session row)."""
    source = SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="group", user_id="u1")
    key = build_session_key(source)
    db.create_session(session_id, "discord", session_key=key)
    now = datetime.now()
    entry = SessionEntry(key, session_id, now, now, origin=source, platform=source.platform, chat_type="group")
    data = entry.to_dict()
    data["suspended"] = suspended
    db.save_gateway_routing_entry(key, json.dumps(data), scope=scope)
    return key


def _call(ctx, args, *, admitted=True):
    """Run the tool through its public handler, with a stub gateway runner over *ctx*'s store."""
    db, scope = ctx
    store = SimpleNamespace(_db_for_key=lambda _key: db, _routing_db=db, _routing_scope=lambda: scope)
    runner = SimpleNamespace(session_store=store)
    with patch.object(heartbeat_tool, "_gateway_runner", return_value=runner), \
            patch.object(heartbeat_tool, "_native_admission", return_value=lambda _entry: admitted):
        return json.loads(heartbeat_tool._handler(args))


def test_set_on_routed_discord_channel_stores_route_heartbeat(ctx):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-overseer")

    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "check the board"})

    assert out["success"] is True
    assert out["route"] == key
    state = load_route_heartbeat(key, db)
    assert state.interval_seconds == 600
    assert state.prompt == "check the board"
    assert state.status == "active"


def test_set_accepts_a_session_id_target(ctx):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-secretary")

    out = _call(ctx, {"action": "set", "target": "sess-secretary", "interval": "30m", "prompt": "summarise"})

    assert out["success"] is True
    assert load_route_heartbeat(key, db).interval_seconds == 1800


def test_status_reports_the_heartbeat_on_a_channel(ctx):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-arbiter")
    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "keep going"})

    out = _call(ctx, {"action": "status", "target": f"discord:{CHANNEL}"})

    assert out["success"] is True
    assert out["heartbeat"]["interval_seconds"] == 600
    assert out["heartbeat"]["prompt"] == "keep going"


def test_clear_removes_the_channel_heartbeat(ctx):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-overseer")
    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})

    out = _call(ctx, {"action": "clear", "target": f"discord:{CHANNEL}"})

    assert out["success"] is True
    assert load_route_heartbeat(key, db) is None


def test_list_returns_every_active_route_heartbeat(ctx):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-a")
    _route(db, scope, OTHER_CHANNEL, "sess-b")
    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "a"})
    _call(ctx, {"action": "set", "target": f"discord:{OTHER_CHANNEL}", "interval": "1h", "prompt": "b"})

    out = _call(ctx, {"action": "list"})

    assert out["success"] is True
    assert sorted(item["prompt"] for item in out["heartbeats"]) == ["a", "b"]
    assert {k for k, _ in list_heartbeats(db)}


def test_target_without_a_routed_session_is_refused(ctx):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer")

    out = _call(ctx, {"action": "set", "target": f"discord:{OTHER_CHANNEL}", "interval": "10m", "prompt": "x"})

    assert out["success"] is False
    assert "no routed session" in out["error"]


def test_set_refused_when_route_has_no_native_admission(ctx):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-overseer")

    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"}, admitted=False)

    assert out["success"] is False
    assert "admission" in out["error"]
    assert load_route_heartbeat(key, db) is None


def test_suspended_route_is_not_a_target(ctx):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer", suspended=True)

    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})

    assert out["success"] is False


@pytest.mark.parametrize("interval", ["5s", "bogus", ""])
def test_bad_interval_is_refused(ctx, interval):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer")

    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": interval, "prompt": "x"})

    assert out["success"] is False


def test_set_and_clear_are_logged(ctx, caplog):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer")
    caplog.set_level(logging.INFO, logger="tools.heartbeat_tool")

    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})
    _call(ctx, {"action": "clear", "target": f"discord:{CHANNEL}"})

    messages = [r.getMessage() for r in caplog.records if r.name == "tools.heartbeat_tool"]
    assert any("set" in m and CHANNEL in m for m in messages)
    assert any("clear" in m and CHANNEL in m for m in messages)


def test_unknown_action_is_an_error(ctx):
    out = _call(ctx, {"action": "explode", "target": f"discord:{CHANNEL}"})
    assert out["success"] is False


def test_tool_is_registered_only_with_a_live_gateway_runner(monkeypatch):
    import sys
    from types import SimpleNamespace

    from tools import heartbeat_tool as mod

    monkeypatch.setenv("_HERMES_GATEWAY", "1")  # inherited by gateway-spawned children
    monkeypatch.delitem(sys.modules, "gateway.run", raising=False)
    assert mod.check_heartbeat_requirements() is False
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_gateway_runner_ref=lambda: SimpleNamespace()))
    assert mod.check_heartbeat_requirements() is True
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_gateway_runner_ref=lambda: None))
    assert mod.check_heartbeat_requirements() is False


def test_handler_refuses_outside_the_gateway(monkeypatch):
    from tools import heartbeat_tool as mod

    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    out = json.loads(mod._handler({"action": "list"}))
    assert out["success"] is False
    assert "gateway" in out["error"]


def test_native_admission_denies_when_no_authority(ctx):
    from tools import heartbeat_tool as mod

    class _Runner:
        session_authority = None
        session_authorities = None
        session_store = SimpleNamespace(_named_profile_for_key=lambda _key: None)

    check = mod._native_admission(_Runner())
    assert check({"session_key": "k", "session_id": "s", "origin": {"platform": "discord", "chat_id": "1"}}) is False


def test_native_admission_denies_without_origin(ctx):
    from tools import heartbeat_tool as mod

    class _Runner:
        session_authority = object()
        session_authorities = None
        session_store = SimpleNamespace(_named_profile_for_key=lambda _key: None)

    check = mod._native_admission(_Runner())
    assert check({"session_key": "k", "session_id": "s"}) is False


def test_set_reports_failure_when_the_write_does_not_persist(ctx, monkeypatch):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer")

    def _lost(*_args, **_kwargs):
        return False

    monkeypatch.setattr("hermes_cli.heartbeat.save_route_heartbeat", _lost)
    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})

    assert out["success"] is False
    assert "did not persist" in out["error"]


def test_set_verify_rejects_a_stale_row_when_the_write_fails(ctx, monkeypatch):
    db, scope = ctx
    _route(db, scope, CHANNEL, "sess-overseer")
    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})

    monkeypatch.setattr("hermes_cli.heartbeat.save_route_heartbeat", lambda *_args, **_kwargs: False)
    out = _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "30m", "prompt": "x"})

    assert out["success"] is False
    assert "did not persist" in out["error"]


def test_clear_with_nothing_stored_reports_cleared_false(ctx):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-overseer")

    out = _call(ctx, {"action": "clear", "target": f"discord:{CHANNEL}"})

    assert out["success"] is True
    assert out["cleared"] is False
    assert load_route_heartbeat(key, db) is None


def test_clear_reports_failure_when_the_row_survives(ctx, monkeypatch):
    db, scope = ctx
    key = _route(db, scope, CHANNEL, "sess-overseer")
    _call(ctx, {"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"})

    monkeypatch.setattr("hermes_cli.heartbeat.save_route_heartbeat", lambda *_args, **_kwargs: False)
    out = _call(ctx, {"action": "clear", "target": f"discord:{CHANNEL}"})

    assert out["success"] is False
    assert "did not persist" in out["error"]
    assert load_route_heartbeat(key, db) is not None


def _admission_case(monkeypatch, *, has_envelope=True, restored_chat_id=CHANNEL, route_adapter=None):
    """A routed Discord channel on the launch profile with a stub runner and authority.

    Only the admissions read and the connector restore are stubbed. The real route checks in
    ``validated_committed_route_envelope`` run. Returns ``(runner, entry)``.
    """
    from types import SimpleNamespace

    source = SessionSource(platform=Platform.DISCORD, chat_id=CHANNEL, chat_type="group", user_id="u1")
    key = build_session_key(source)
    adapter = object()
    store = SimpleNamespace(_named_profile_for_key=lambda _key: None, _generate_session_key=lambda _source: key)
    authority = SimpleNamespace(db=None, logical_owner=lambda session_id: session_id, runner=None)
    runner = SimpleNamespace(
        session_authority=authority, session_authorities=None, session_store=store,
        _delivery_adapter_for=lambda _source: adapter,
        _adapter_for_source=lambda _source: route_adapter or adapter,
    )
    authority.runner = runner
    envelope = {"provenance": {"connector": "discord"}, "source": source.to_dict()}
    rows = [{"payload": {"native_text_v1": envelope}}] if has_envelope else []
    restored = SessionSource(platform=Platform.DISCORD, chat_id=restored_chat_id, chat_type="group", user_id="u1")
    monkeypatch.setattr("gateway.session_automation.list_session_admissions",
                        lambda _db, session_id, pending_only=True: rows)
    monkeypatch.setattr("gateway.session_envelope.restore_native",
                        lambda _payload, runner=None: SimpleNamespace(source=restored))
    entry = {"session_key": key, "session_id": "sess-admit", "origin": source.to_dict(), "platform": "discord"}
    return runner, entry


def test_native_admission_admits_a_matching_committed_envelope(monkeypatch):
    from tools import heartbeat_tool as mod

    runner, entry = _admission_case(monkeypatch)

    assert mod._native_admission(runner)(entry) is True


def test_native_admission_denies_a_mismatched_chat_id(monkeypatch):
    from tools import heartbeat_tool as mod

    runner, entry = _admission_case(monkeypatch, restored_chat_id=OTHER_CHANNEL)

    assert mod._native_admission(runner)(entry) is False


def test_native_admission_denies_a_mismatched_adapter(monkeypatch):
    from tools import heartbeat_tool as mod

    runner, entry = _admission_case(monkeypatch, route_adapter=object())

    assert mod._native_admission(runner)(entry) is False


def test_native_admission_denies_a_missing_envelope(monkeypatch):
    from tools import heartbeat_tool as mod

    runner, entry = _admission_case(monkeypatch, has_envelope=False)

    assert mod._native_admission(runner)(entry) is False


@pytest.fixture
def multiplex_store(tmp_path, monkeypatch):
    """A launch home and a ``fitness`` profile home, served by one multiplexing store."""
    from unittest.mock import patch

    import hermes_state

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")  # identity marker
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    goals._DB_CACHE.clear()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig(multiplex_profiles=True))
    store._loaded = True
    yield store, root, profile
    goals._DB_CACHE.clear()


def _fitness_route(store):
    """A routed Discord channel owned by ``fitness``: its row in that profile's store, its entry in the launch index."""
    source = SessionSource(platform=Platform.DISCORD, chat_id=CHANNEL, chat_type="group", user_id="u1", profile="fitness")
    key = store._generate_session_key(source)
    profile_db = store._db_for_key(key)
    profile_db.create_session("sess-fitness", "discord", session_key=key)
    now = datetime.now()
    data = SessionEntry(key, "sess-fitness", now, now, origin=source, platform=source.platform, chat_type="group").to_dict()
    store._routing_db.save_gateway_routing_entry(key, json.dumps(data), scope=store._routing_scope())
    return key, source, profile_db


@contextmanager
def _active_home(home):
    """Run the body with *home* as the active profile scope, as a profile turn binds it."""
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def fitness_scope(multiplex_store):
    _store, _root, profile = multiplex_store
    with _active_home(profile):
        yield profile


def test_heartbeat_on_a_named_profile_route_lands_in_the_profile_store(multiplex_store, fitness_scope, monkeypatch):
    from types import SimpleNamespace

    from hermes_cli.heartbeat import HeartbeatManager
    from tools import heartbeat_tool as mod

    store, _root, _profile = multiplex_store
    key, _source, profile_db = _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)

    out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "check fitness"}))

    assert out["success"] is True
    assert load_route_heartbeat(key, profile_db).prompt == "check fitness"
    assert load_route_heartbeat(key, store._routing_db) is None
    assert HeartbeatManager("sess-fitness", route_key=key, db=store._db_for_key(key)).state.prompt == "check fitness"
    listed = json.loads(mod._handler({"action": "list"}))
    assert [item["route"] for item in listed["heartbeats"]] == [key]


def test_admission_uses_the_routed_profile_authority(multiplex_store, fitness_scope, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from tools import heartbeat_tool as mod

    store, _root, profile = multiplex_store
    _key, source, profile_db = _fitness_route(store)
    adapter = object()
    fitness = SimpleNamespace(db=profile_db, logical_owner=lambda sid: sid, runner=None)
    launch = SimpleNamespace(db=store._db_for_key(None), logical_owner=lambda sid: sid, runner=None)
    registry = SimpleNamespace(
        for_home=lambda home: fitness if Path(home).resolve() == profile.resolve() else None,
        active=lambda: launch,
    )
    runner = SimpleNamespace(
        session_store=store, session_authorities=registry, session_authority=launch,
        _delivery_adapter_for=lambda _source: adapter, _adapter_for_source=lambda _source: adapter,
    )
    fitness.runner = launch.runner = runner
    envelope = {"provenance": {"connector": "discord"}, "source": source.to_dict()}

    def _rows(db, session_id, pending_only=True):
        return [{"payload": {"native_text_v1": envelope}}] if Path(db.db_path) == Path(profile_db.db_path) else []

    monkeypatch.setattr("gateway.session_automation.list_session_admissions", _rows)
    monkeypatch.setattr("gateway.session_envelope.restore_native", lambda _payload, runner=None: SimpleNamespace(source=source))
    monkeypatch.setattr(mod, "_gateway_runner", lambda: runner)

    out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "check fitness"}))

    assert out["success"] is True


def test_cross_profile_set_is_refused_and_writes_nothing(multiplex_store, monkeypatch):
    from tools import heartbeat_tool as mod

    store, root, _profile = multiplex_store
    key, _source, profile_db = _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)

    with _active_home(root):  # the launch profile's turn, not the route's
        out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "steal"}))

    assert out["success"] is False
    assert load_route_heartbeat(key, profile_db) is None
    assert load_route_heartbeat(key, store._routing_db) is None


def test_cross_profile_clear_is_refused_and_leaves_the_row(multiplex_store, fitness_scope, monkeypatch):
    from tools import heartbeat_tool as mod

    store, root, _profile = multiplex_store
    key, _source, profile_db = _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)
    set_out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "keep"}))
    assert set_out["success"] is True

    with _active_home(root):
        out = json.loads(mod._handler({"action": "clear", "target": f"discord:{CHANNEL}"}))

    assert out["success"] is False
    assert load_route_heartbeat(key, profile_db).prompt == "keep"


def test_cross_profile_status_is_refused(multiplex_store, fitness_scope, monkeypatch):
    from tools import heartbeat_tool as mod

    store, root, _profile = multiplex_store
    _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)
    set_out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "private"}))
    assert set_out["success"] is True

    with _active_home(root):
        out = json.loads(mod._handler({"action": "status", "target": f"discord:{CHANNEL}"}))

    assert out["success"] is False
    assert "heartbeat" not in out


def test_list_omits_another_profiles_routes(multiplex_store, fitness_scope, monkeypatch):
    from tools import heartbeat_tool as mod

    store, root, _profile = multiplex_store
    _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)
    set_out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "fitness"}))
    assert set_out["success"] is True

    with _active_home(root):
        listed = json.loads(mod._handler({"action": "list"}))

    assert listed == {"success": True, "heartbeats": []}


def test_multiplexed_call_with_no_active_scope_fails_closed(multiplex_store, monkeypatch):
    from tools import heartbeat_tool as mod

    store, _root, _profile = multiplex_store
    key, _source, profile_db = _fitness_route(store)
    monkeypatch.setattr(mod, "_gateway_runner", lambda: SimpleNamespace(session_store=store))
    monkeypatch.setattr(mod, "_native_admission", lambda _runner: lambda _entry: True)

    set_out = json.loads(mod._handler({"action": "set", "target": f"discord:{CHANNEL}", "interval": "10m", "prompt": "x"}))
    listed = json.loads(mod._handler({"action": "list"}))

    assert set_out["success"] is False
    assert load_route_heartbeat(key, profile_db) is None
    assert listed == {"success": True, "heartbeats": []}
