"""Busy heartbeat delivery through the gateway's existing steer path."""
import asyncio

from gateway.platforms.event import MessageEvent
from tests.gateway.test_heartbeat_poller import poller
import pytest

from hermes_cli.heartbeat import HeartbeatManager


@pytest.mark.asyncio
async def test_authority_busy_steer_coalesces_until_turn_ends(monkeypatch):
    from hermes_cli import heartbeat
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway
    from gateway.run_heartbeat_acceptance import admit_heartbeat

    runner, adapter, _authority, source, sent, _, _ = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, "seed provenance")
    key = runner._session_key_for_source(source)
    entry = await runner.async_session_store.get_or_create_session(source)
    manager = HeartbeatManager(entry.session_id)
    manager.set("check status", 60, delivery_mode="steer")
    now = manager.state.created_at + 61
    monkeypatch.setattr(heartbeat, "time", type("Clock", (), {"time": staticmethod(lambda: now)}))
    from tests.agent.test_steer import _bare_agent
    agent = _bare_agent()
    runner._session_state(key).turn.agent = agent
    await admit_heartbeat(runner, adapter, source, entry.session_id, key)
    messages = [{"role": "tool", "tool_call_id": "long", "content": "tool result"}]
    agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
    injected = [messages[-1]["content"]] if len(messages) > 1 else []
    assert len(injected) == 1
    assert "Heartbeat" in injected[0] and "check status" in injected[0]
    assert HeartbeatManager(entry.session_id).state.fire_count == 1
    now += 60
    await admit_heartbeat(runner, adapter, source, entry.session_id, key)
    assert agent._pending_steer is None
    assert HeartbeatManager(entry.session_id).state.fire_count == 1


@pytest.mark.asyncio
async def test_authority_refuses_busy_steer_without_committed_route(monkeypatch):
    from gateway.run_heartbeat_acceptance import admit_heartbeat
    from hermes_cli import heartbeat
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_authority_lane_moves_on import _gateway

    runner, adapter, _, source, _, _, _ = await _gateway(monkeypatch)
    entry = await runner.async_session_store.get_or_create_session(source)
    key = runner._session_key_for_source(source)
    manager = HeartbeatManager(entry.session_id)
    manager.set("check status", 60, delivery_mode="steer")
    now = manager.state.created_at + 61
    monkeypatch.setattr(heartbeat, "time", type("Clock", (), {"time": staticmethod(lambda: now)}))
    agent = type("Agent", (), {"steer": lambda self, text: pytest.fail("unvalidated steer")})()
    runner._session_state(key).turn.agent = agent
    with pytest.raises(RuntimeStoreError):
        await admit_heartbeat(runner, adapter, source, entry.session_id, key)
    assert HeartbeatManager(entry.session_id).state.fire_count == 0


@pytest.mark.asyncio
async def test_authority_default_busy_heartbeat_runs_after_turn(monkeypatch):
    from gateway.run_heartbeat_acceptance import admit_heartbeat
    from hermes_cli import heartbeat
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway
    from tests.waiting import async_wait_until

    runner, adapter, authority, source, sent, ran, gates = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, "seed provenance")
    entry = await runner.async_session_store.get_or_create_session(source)
    key = runner._session_key_for_source(source)
    manager = HeartbeatManager(entry.session_id)
    manager.set("check status", 60)
    now = manager.state.created_at + 61
    monkeypatch.setattr(heartbeat, "time", type("Clock", (), {"time": staticmethod(lambda: now)}))
    gates["long turn"] = asyncio.Event()
    await adapter.handle_message(MessageEvent(text="long turn", source=source, message_id="long"))
    await async_wait_until(lambda: any(row[0] == "long turn" for row in ran), what="long turn starts")
    try:
        await admit_heartbeat(runner, adapter, source, entry.session_id, key)
        pending = list_session_admissions(authority.db, session_id=authority.logical_owner(entry.session_id))
        assert any(row['payload'].get('native_text_v1', {}).get('automation', {}).get('heartbeat')
                   for row in pending)
        assert not any("[Heartbeat" in row[0] for row in ran)
        gates["long turn"].set()
        await async_wait_until(lambda: any("[Heartbeat" in row[0] for row in ran), what="heartbeat runs")
        assert sum("[Heartbeat" in row[0] for row in ran) == 1
    finally:
        gates["long turn"].set()

@pytest.mark.asyncio
async def test_steer_mode_heartbeat_reaches_busy_tool_turn_once(poller):
    runner, adapter, watch, key, clock = poller
    started, release = asyncio.Event(), asyncio.Event()
    received = []

    class RunningAgent:
        def __init__(self):
            self.steered = []

        def steer(self, text):
            self.steered.append(text)
            return True

    agent = RunningAgent()

    async def handler(event):
        event._heartbeat_execution_started = True
        received.append(event)
        started.set()
        await release.wait()

    adapter.set_message_handler(handler)
    runner._session_state(key).turn.agent = agent
    manager = HeartbeatManager("heartbeat-session")
    manager.set("check status", 60, delivery_mode="steer")
    try:
        # Start an ordinary long turn; the heartbeat fires while it owns the slot.
        adapter._session_tasks[key] = asyncio.create_task(handler(MessageEvent(text="long tool", source=watch[key][0])))
        adapter._active_sessions[key] = asyncio.Event()
        clock.now += 60
        await runner._heartbeat_poll_once(watch)
        assert len(agent.steered) == 1
        assert "check status" in agent.steered[0]
        assert HeartbeatManager("heartbeat-session").state.fire_count == 1

        # Another due interval in the same turn is consumed by the pending steer, not queued again.
        clock.now += 60
        await runner._heartbeat_poll_once(watch)
        assert len(agent.steered) == 1
        assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    finally:
        release.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        task = adapter._session_tasks.pop(key, None)
        if task is not None:
            await task
        adapter._active_sessions.pop(key, None)
        runner._session_state(key).turn.agent = None


def test_heartbeat_legacy_state_and_prompt_parsing():
    from hermes_cli.heartbeat import HeartbeatState, parse_heartbeat_set

    state = HeartbeatState.from_json('{"prompt":"check","interval_seconds":60}')
    assert state.delivery_mode == "idle"
    assert parse_heartbeat_set("every 10m --steer check   --steer  later") == (
        600, "check   --steer  later", "steer"
    )
    assert parse_heartbeat_set("10m check   --steer  later") == (
        600, "check   --steer  later", "idle"
    )


def test_heartbeat_status_names_gateway_only_mode(poller):
    manager = HeartbeatManager("heartbeat-session")
    manager.set("check status", 60, delivery_mode="steer")
    assert "steer (gateway only)" in manager.status_line()
