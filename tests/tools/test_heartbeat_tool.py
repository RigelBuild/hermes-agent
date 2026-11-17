"""The `heartbeat` agent tool: set, clear, status and list heartbeats on a routed session or Discord channel."""

import json
import logging
from datetime import datetime

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli import goals
from hermes_cli.heartbeat import list_heartbeats, load_route_heartbeat
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
    db, scope = ctx
    raw = heartbeat_tool.heartbeat_action(
        args, db=db, routing_db=db, routing_scope=scope, admission_check=lambda entry: admitted,
    )
    return json.loads(raw)


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
