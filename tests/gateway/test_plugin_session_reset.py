"""Tests for plugin-triggered gateway session resets."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import hermes_yaml as yaml
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _plugin_context(tmp_path, monkeypatch, *, allowed):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "entries": {
                        "notify-plugin": {
                            "allow_gateway_injection": allowed,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager(scope_key=str(tmp_path))
    return PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    ), manager


def _runner(tmp_path):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._plugin_session_key_admissible = MagicMock(return_value=True)
    runner._hm_stop_owner_turn = AsyncMock()
    runner._invalidate_session_run_generation = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._cached_agent_for = lambda _key: None
    runner._evict_cached_agent = MagicMock()
    runner._clear_conversation_scope = MagicMock()
    runner._finalize_session_off_loop = AsyncMock(return_value=[])
    runner._deliver_session_end_messages = AsyncMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    return runner, entry


def test_session_store_reset_refuses_stale_expected_id_after_rotation(tmp_path):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="store-race", user_id="store-race")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    original = store.get_or_create_session(source)

    successor = store.reset_session(original.session_key, expected_session_id=original.session_id)
    assert successor is not None and successor.session_id != original.session_id

    stale = store.reset_session(original.session_key, expected_session_id=original.session_id)

    current = store.lookup_by_session_key(original.session_key)
    assert stale is None
    assert current is not None and current.session_id == successor.session_id


@pytest.mark.asyncio
async def test_plugin_reset_releases_occupied_running_agent_slot(tmp_path, monkeypatch):
    runner, entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)
    agent = object()
    runner._session_state(entry.session_key).turn.agent = agent
    runner._session_state(entry.session_key).turn.started_ts = 1.0
    assert runner._is_session_running(entry.session_key)

    runner._persist_active_agents = MagicMock()
    runner._running_agents[entry.session_key] = agent
    runner._running_agents_ts[entry.session_key] = 1.0
    runner._release_running_agent_state = GatewayRunner._release_running_agent_state.__get__(runner)
    assert context.reset_session(entry.session_id, "refusal") is True
    await asyncio.gather(*tuple(runner._background_tasks), return_exceptions=True)

    assert not runner._is_session_running(entry.session_key)
    assert entry.session_key not in runner._running_agents_ts


def test_plugin_reset_requires_gateway_injection_permission(tmp_path, monkeypatch):
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=False)
    resetter = MagicMock(return_value=True)
    manager.set_gateway_session_resetter(object(), resetter)

    assert context.reset_session("session-42", "refusal") is False
    resetter.assert_not_called()


@pytest.mark.asyncio
async def test_plugin_reset_by_session_id_rotates_the_session(tmp_path, monkeypatch):
    runner, entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)
    reset = runner.session_store.reset_session
    with patch.object(
        runner.session_store, "reset_session", wraps=reset
    ) as reset_session:
        accepted = context.reset_session(
            entry.session_id, "reasoning_extraction refusal"
        )
        assert accepted is True
        await asyncio.gather(*tuple(runner._background_tasks), return_exceptions=True)

    current = runner.session_store.lookup_by_session_key(entry.session_key)
    assert current is not None and current.session_id != entry.session_id
    reset_session.assert_called_once_with(entry.session_key, expected_session_id=entry.session_id)
    runner._hm_stop_owner_turn.assert_awaited_once_with(entry.session_key)
    assert runner._plugin_session_key_admissible.call_count == 2
    runner._plugin_session_key_admissible.assert_any_call(
        entry.session_key, plugin_home=tmp_path, plugin_id="notify-plugin"
    )


@pytest.mark.asyncio
async def test_plugin_reset_refuses_stale_id_after_concurrent_reset(tmp_path, monkeypatch):
    runner, entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)
    first_entered_stop = asyncio.Event()
    release_first = asyncio.Event()
    stop_calls = 0

    async def stop_owner_turn(_session_key):
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            first_entered_stop.set()
            await release_first.wait()

    runner._hm_stop_owner_turn = stop_owner_turn
    assert context.reset_session(entry.session_id, "request A") is True
    first_task = next(iter(runner._background_tasks))
    await asyncio.wait_for(first_entered_stop.wait(), timeout=2)

    assert context.reset_session(entry.session_id, "request B") is True
    second_tasks = runner._background_tasks - {first_task}
    await asyncio.wait_for(asyncio.gather(*tuple(second_tasks)), timeout=2)
    successor = runner.session_store.lookup_by_session_key(entry.session_key)
    assert successor is not None and successor.session_id != entry.session_id

    release_first.set()
    assert await asyncio.wait_for(first_task, timeout=2) is False
    current = runner.session_store.lookup_by_session_key(entry.session_key)
    assert current is not None and current.session_id == successor.session_id


@pytest.mark.asyncio
async def test_plugin_reset_unknown_session_id_is_refused(tmp_path, monkeypatch):
    runner, _entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)

    assert context.reset_session("unknown-session", "refusal") is False
    runner._hm_stop_owner_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_reset_refuses_a_session_owned_by_another_profile(
    tmp_path, monkeypatch
):
    runner, entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)
    runner._plugin_session_key_admissible.return_value = False

    assert context.reset_session(entry.session_id, "refusal") is False
    runner._plugin_session_key_admissible.assert_called_once_with(
        entry.session_key, plugin_home=tmp_path, plugin_id="notify-plugin"
    )
    runner._hm_stop_owner_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_reset_schedules_from_a_non_loop_thread(tmp_path, monkeypatch):
    runner, entry = _runner(tmp_path)
    context, manager = _plugin_context(tmp_path, monkeypatch, allowed=True)
    manager.set_gateway_session_resetter(runner, runner._schedule_plugin_session_reset)
    reset_started = asyncio.Event()

    async def stop_owner_turn(_session_key):
        reset_started.set()

    runner._hm_stop_owner_turn = AsyncMock(side_effect=stop_owner_turn)
    accepted = await asyncio.wait_for(
        asyncio.to_thread(context.reset_session, entry.session_id, "refusal"), timeout=2
    )
    assert accepted is True
    await asyncio.wait_for(reset_started.wait(), timeout=2)
    runner._hm_stop_owner_turn.assert_awaited_once_with(entry.session_key)
