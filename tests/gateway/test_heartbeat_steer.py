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
async def test_authority_idle_busy_heartbeat_runs_after_turn(monkeypatch):
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
    manager.set("check status", 60, delivery_mode="idle")
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
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    started, release = asyncio.Event(), asyncio.Event()
    received = []

    class RunningAgent:
        def __init__(self):
            self.steered = []

        def steer(self, text, *, ticket=None):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,authorized", [("idle", True), ("steer", False)])
async def test_legacy_busy_holds_unsteerable_heartbeat(poller, mode, authorized):
    from tests.agent.test_steer import _bare_agent

    runner, adapter, watch, key, clock = poller
    runner._is_user_authorized_for_source = lambda source, **kwargs: authorized
    received = []
    async def handler(event):
        event._heartbeat_execution_started = True
        received.append(event)
    adapter.set_message_handler(handler)
    agent = _bare_agent()
    runner._session_state(key).turn.agent = agent
    HeartbeatManager("heartbeat-session").set("check status", 60, delivery_mode=mode)
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    assert agent._pending_steer is None
    assert HeartbeatManager("heartbeat-session").state.fire_count == 0
    runner._session_state(key).turn.agent = None
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    assert len(received) == 1 and received[0]._heartbeat_session_id == "heartbeat-session"
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1


@pytest.mark.asyncio
async def test_unconsumed_ticket_is_refunded_and_runs_as_heartbeat(poller):
    from tests.agent.test_steer import _bare_agent

    runner, adapter, watch, key, clock = poller
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    runner._persist_active_agents = lambda: None
    received = []
    async def handler(event):
        event._heartbeat_execution_started = True
        received.append(event)
    adapter.set_message_handler(handler)
    agent = _bare_agent()
    runner._session_state(key).turn.agent = agent
    HeartbeatManager("heartbeat-session").set("check status", 60, delivery_mode="steer")
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    assert agent._pending_steer and agent._drain_leftover_steer() is None
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    runner._release_running_agent_state(key)
    assert agent._pending_steer is None
    assert HeartbeatManager("heartbeat-session").state.fire_count == 0
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    assert len(received) == 1 and received[0]._heartbeat_session_id == "heartbeat-session"
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1


def test_follow_up_resets_steer_flag():
    from gateway.session_state import TurnState

    turn = TurnState(heartbeat_steered=True)
    with turn.running_follow_up(object()):
        assert not turn.heartbeat_steered
    assert not turn.heartbeat_steered


@pytest.mark.asyncio
async def test_follow_up_refunds_outer_before_claiming_again(poller):
    from tests.agent.test_steer import _bare_agent

    runner, adapter, watch, key, clock = poller
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    adapter.set_message_handler(lambda event: None)
    agent = _bare_agent()
    turn = runner._session_state(key).turn
    turn.agent = agent
    HeartbeatManager("heartbeat-session").set("check status", 60, delivery_mode="steer")
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    with turn.running_follow_up(object()):
        assert agent._pending_steer is None
        assert HeartbeatManager("heartbeat-session").state.fire_count == 0
        await runner._heartbeat_poll_once(watch)
        assert len(agent._pending_steer_parts) == 1
        assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    assert agent._pending_steer is None
    assert HeartbeatManager("heartbeat-session").state.fire_count == 0


def test_refund_failure_does_not_hold_session_slot(poller, caplog):
    runner, _, _, key, _ = poller
    runner._persist_active_agents = lambda: None
    class Agent:
        def withdraw_steer(self, ticket):
            return True
    class Manager:
        def abandon_fire(self):
            raise RuntimeError("storage failed")
    turn = runner._session_state(key).turn
    turn.agent = Agent()
    turn.heartbeat_claim = (turn.agent, "ticket", Manager())
    assert runner._release_running_agent_state(key)
    assert turn.agent is None and turn.heartbeat_claim is None
    assert "Failed to settle heartbeat steer" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["rotated", "draining", "sentinel", "already_steered"])
async def test_authority_busy_steer_fences(monkeypatch, condition):
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.run_heartbeat_acceptance import admit_heartbeat
    from hermes_cli import heartbeat
    from hermes_state_runtime import RuntimeStoreError
    from tests.agent.test_steer import _bare_agent
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway

    runner, adapter, _, source, sent, _, _ = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, "seed provenance")
    entry = await runner.async_session_store.get_or_create_session(source)
    key = runner._session_key_for_source(source)
    manager = HeartbeatManager(entry.session_id)
    manager.set("check status", 60, delivery_mode="steer")
    now = manager.state.created_at + 61
    monkeypatch.setattr(heartbeat, "time", type("Clock", (), {"time": staticmethod(lambda: now)}))
    agent = _bare_agent()
    turn = runner._session_state(key).turn
    turn.agent = _AGENT_PENDING_SENTINEL if condition == "sentinel" else agent
    turn.heartbeat_steered = condition == "already_steered"
    if condition == "draining":
        runner._draining = True
    if condition == "rotated":
        async def validate_then_rotate(fn, *args):
            result = fn(*args)
            monkeypatch.setattr(runner.session_store, "peek_session_id", lambda route: "other-session")
            return result
        monkeypatch.setattr(runner, "_run_in_executor_with_context", validate_then_rotate)
    if condition == "draining":
        with pytest.raises(RuntimeStoreError):
            await admit_heartbeat(runner, adapter, source, entry.session_id, key)
    else:
        await admit_heartbeat(runner, adapter, source, entry.session_id, key)
    assert agent._pending_steer is None
    assert HeartbeatManager(entry.session_id).state.fire_count == 0


@pytest.mark.asyncio
async def test_next_turn_can_steer_parent_without_fanout(poller):
    from tests.agent.test_steer import _bare_agent

    runner, adapter, watch, key, clock = poller
    runner._is_user_authorized_for_source = lambda source, **kwargs: True
    runner._persist_active_agents = lambda: None
    adapter.set_message_handler(lambda event: None)
    parent, child = _bare_agent(), _bare_agent()
    parent._active_children = [child]
    HeartbeatManager("heartbeat-session").set("check status", 60, delivery_mode="steer")
    runner._session_state(key).turn.agent = parent
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    assert parent._pending_steer and child._pending_steer is None
    parent._drain_pending_steer()
    runner._release_running_agent_state(key)
    next_agent = _bare_agent()
    runner._session_state(key).turn.agent = next_agent
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    assert next_agent._pending_steer
    assert HeartbeatManager("heartbeat-session").state.fire_count == 2


def test_heartbeat_legacy_state_and_prompt_parsing():
    from hermes_cli.heartbeat import HeartbeatState, parse_heartbeat_set

    state = HeartbeatState.from_json('{"prompt":"check","interval_seconds":60}')
    assert state.delivery_mode == "steer"
    assert parse_heartbeat_set("every 10m --steer check   --steer  later") == (
        600, "check   --steer  later", "steer"
    )
    assert parse_heartbeat_set("10m check   --steer  later") == (
        600, "check   --steer  later", "steer"
    )
    assert parse_heartbeat_set("10m --idle check") == (600, "check", "idle")


def test_heartbeat_status_names_gateway_only_mode(poller):
    manager = HeartbeatManager("heartbeat-session")
    manager.set("check status", 60, delivery_mode="steer")
    assert "steer (gateway only)" in manager.status_line()
