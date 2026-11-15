"""Heartbeats follow the routed conversation through compression and reset boundaries."""

import asyncio

import pytest
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_cli.heartbeat import (
    HeartbeatManager,
    HeartbeatState,
    load_heartbeat,
    migrate_heartbeat_to_session,
    save_heartbeat,
)


def test_reset_carries_only_the_departing_conversations_heartbeat(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="reset", user_id="owner")
    entry = store.get_or_create_session(source)
    key, parent = entry.session_key, entry.session_id
    child = parent + "-compressed"
    state = HeartbeatState(prompt="check deploy", interval_seconds=60, created_at=1)
    save_heartbeat(parent, state)
    save_heartbeat("unrelated", state)
    db = store._db_for_key(key)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="telegram",
        require_compression_lease=False, model="offline", model_config={},
        system_prompt="offline", messages=[{"role": "user", "content": "retained"}],
    )
    assert migrate_heartbeat_to_session(parent, child)
    assert store.advance_compression_session(key, parent, child)
    assert load_heartbeat(parent) is None
    assert HeartbeatManager(child).due_prompt(now=120)

    replacement = store.reset_session(key)

    assert replacement.session_id != child
    assert load_heartbeat(child) is None
    assert HeartbeatManager(replacement.session_id).due_prompt(now=120)
    assert HeartbeatManager("unrelated").due_prompt(now=120)
    # Resuming the archived conversation must not resurrect its old schedule.
    store.switch_session(key, child)
    assert load_heartbeat(child) is None
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_heartbeat_remains_on_route_through_reset_and_fresh_runner_restore(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.run_heartbeat_restore import restore_heartbeat_watches

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="reset", user_id="owner")
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source)
    manager = HeartbeatManager(entry.session_id)
    state = manager.set("check deploy", 60)
    state.created_at = 1
    save_heartbeat(entry.session_id, state)

    replacement = store.reset_session(entry.session_key)
    assert replacement.session_id != entry.session_id
    assert load_heartbeat(entry.session_id) is None

    store.close_all_db_handles()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = SessionStore(tmp_path / "sessions", config)
    runner._heartbeat_watch = {}
    runner._start_heartbeat_poller = lambda: None
    runner._profile_name_for_source = lambda source, adapter_profile=None: source.profile
    runner._delivery_adapter_for = lambda source: object()
    runner._run_in_executor_with_context = asyncio.to_thread

    try:
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {
            replacement.session_key: (source, replacement.session_id),
        }
        assert HeartbeatManager(replacement.session_id).due_prompt(now=120)
        assert HeartbeatManager(entry.session_id).due_prompt(now=120) is None
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_restore_migrates_archived_session_heartbeat_to_current_route(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.run_heartbeat_restore import restore_heartbeat_watches

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="archive", user_id="owner")
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    entry = store.get_or_create_session(source)
    old_id, key = entry.session_id, entry.session_key
    replacement = store.reset_session(key)
    db = store._db_for_key(key)
    db.set_meta("heartbeat:" + old_id, HeartbeatState(
        prompt="check archived", interval_seconds=60, created_at=1,
    ).to_json())
    store.close_all_db_handles()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = SessionStore(tmp_path / "sessions", config)
    runner._heartbeat_watch = {}
    runner._start_heartbeat_poller = lambda: None
    runner._profile_name_for_source = lambda source, adapter_profile=None: source.profile
    runner._delivery_adapter_for = lambda source: object()
    runner._run_in_executor_with_context = asyncio.to_thread
    try:
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {key: (source, replacement.session_id)}
        db = runner.session_store._db_for_key(key)
        assert HeartbeatManager(replacement.session_id, route_key=key, db=db).due_prompt(now=120)
        assert HeartbeatManager(old_id).clear()
        assert not HeartbeatManager(replacement.session_id, route_key=key, db=db).has_heartbeat()
    finally:
        runner.session_store.close_all_db_handles()

