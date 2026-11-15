"""Heartbeat polls wake idle adapters, never accumulate deferred ticks."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND, display_kind_for_event
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli.heartbeat import HeartbeatManager


class _HeartbeatAdapter(BasePlatformAdapter):
    """Real adapter lifecycle with an in-memory wire transport, no bot/model."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="wire-1")


@pytest.fixture
def poller(monkeypatch):
    from hermes_cli import goals, heartbeat

    goals._DB_CACHE.clear()
    goals._get_session_db()
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(heartbeat, "time", SimpleNamespace(time=lambda: clock.now))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm", message_id="watch-cmd")
    key = build_session_key(source)
    adapter = _HeartbeatAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._run_in_executor_with_context = asyncio.to_thread
    watch = {key: (source, "heartbeat-session")}
    HeartbeatManager("heartbeat-session").set("check status", 60)
    yield runner, adapter, watch, key, clock
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_idle_wake_coalesces_intervals_while_adapter_owns_turn(poller):
    runner, adapter, watch, key, clock = poller
    started, release = asyncio.Event(), asyncio.Event()
    received = []

    async def handler(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary
        received.append(event)
        started.set()
        await release.wait()

    adapter.set_message_handler(handler)
    # A remembered topic must not redirect the watched session or introduce an
    # executor yield between the idle check and the adapter's session claim.
    adapter._topic_recovery_fn = lambda source: "another-topic"
    try:
        clock.now += 60
        await runner._heartbeat_poll_once(watch)
        assert key in adapter._active_sessions, "idle poll must start a turn, not park it in FIFO"
        await asyncio.wait_for(started.wait(), 2)
        for _ in range(14):
            clock.now += 60
            await runner._heartbeat_poll_once(watch)
        assert len(received) == 1
        assert not received[0].internal  # authorization and emergency-stop still apply
        # Gateway-stamped provenance: the turn may end silently (#113031) and is not a reply to
        # the message that registered the watch (#112149).
        assert display_kind_for_event(received[0]) == INTERNAL_NOTIFICATION_DISPLAY_KIND
        assert received[0].source.message_id is None
        assert runner._queue_depth(key, adapter=adapter) == 0
        assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    finally:
        release.set()
        await asyncio.gather(*adapter._background_tasks)


@pytest.mark.asyncio
async def test_unavailable_or_busy_session_leaves_persisted_tick_due(poller):
    runner, adapter, watch, key, clock = poller
    clock.now += 900
    original = HeartbeatManager("heartbeat-session").state.to_json()
    user = MessageEvent(text="user follow-up", source=watch[key][0])
    # Missing adapter and missing handler must not consume a due tick.
    runner._delivery_adapter_for = lambda source: None
    await runner._heartbeat_poll_once(watch)
    runner._delivery_adapter_for = lambda source: adapter
    await runner._heartbeat_poll_once(watch)

    async def handler(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary

    adapter.set_message_handler(handler)
    for state in (runner._running_agents, adapter._active_sessions, adapter._pending_messages):
        state[key] = user
        await runner._heartbeat_poll_once(watch)
        assert state[key] is user
        state.pop(key)
    assert HeartbeatManager("heartbeat-session").state.to_json() == original
    await runner._heartbeat_poll_once(watch)
    assert key in adapter._active_sessions
    await asyncio.gather(*adapter._background_tasks)
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_id", ["1571155392069763142", "1571157949362081882", "1571263020523327488"]
)
@pytest.mark.parametrize(
    "command", ["/heartbeat every 1m check status", "/heartbeat resume"]
)
async def test_discord_group_slash_only_heartbeat_is_refused(
    monkeypatch, chat_id, command
):
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway
    from hermes_state_runtime import list_session_admissions

    user_id = "1385361904591044669"
    runner, adapter, authority, _, sent, _, _ = await _gateway(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        scope_id="1477066497896742955",
        guild_id="1477066497896742955",
    )
    key = runner._session_key_for_source(source)
    if command == "/heartbeat resume":
        session_entry = await runner.async_session_store.get_or_create_session(source)
        seeded_manager = HeartbeatManager(session_entry.session_id)
        seeded_manager.set("check status", 60)
        seeded_manager.pause()
        before_resume = seeded_manager.state
    monkeypatch.setattr(runner, "_start_heartbeat_poller", lambda: None)

    await _chat(adapter, source, sent, command)

    session_id = runner.session_store.peek_session_id(key)
    state = HeartbeatManager(session_id).state
    if command == "/heartbeat resume":
        assert state.status == "paused"
        assert (state.created_at, state.fire_count) == (
            before_resume.created_at,
            before_resume.fire_count,
        )
    else:
        assert state is None
    assert not list_session_admissions(
        authority.db, session_id=session_id, pending_only=False
    )
    from agent.i18n import t

    assert sent[-1] == t("gateway.heartbeat.needs_admission")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_id", ["1571155392069763142", "1571157949362081882", "1571263020523327488"]
)
@pytest.mark.parametrize(
    "command", ["/heartbeat every 1m check status", "/heartbeat resume"]
)
async def test_discord_group_heartbeat_is_allowed_after_native_message(
    monkeypatch, chat_id, command
):
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway

    user_id = "1385361904591044669"
    runner, adapter, _, _, sent, _, _ = await _gateway(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        scope_id="1477066497896742955",
        guild_id="1477066497896742955",
    )
    await _chat(adapter, source, sent, "check status")

    if command == "/heartbeat resume":
        session_id = runner.session_store.peek_session_id(
            runner._session_key_for_source(source)
        )
        manager = HeartbeatManager(session_id)
        manager.set("check status", 60)
        manager.pause()

    monkeypatch.setattr(runner, "_start_heartbeat_poller", lambda: None)
    await _chat(adapter, source, sent, command)

    session_id = runner.session_store.peek_session_id(
        runner._session_key_for_source(source)
    )
    state = HeartbeatManager(session_id).state
    assert state is not None and state.status == "active"
    assert "check status" in sent[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_id", ["1571155392069763142", "1571157949362081882", "1571263020523327488"]
)
async def test_discord_group_heartbeat_without_admission_reports_missing_provenance(
    monkeypatch, caplog, chat_id
):
    from tests.gateway.test_authority_lane_moves_on import _gateway
    from hermes_cli import heartbeat
    from hermes_state_runtime import list_session_admissions

    user_id = "1385361904591044669"
    runner, adapter, authority, _, _, _, _ = await _gateway(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        scope_id="1477066497896742955",
        guild_id="1477066497896742955",
    )
    key = runner._session_key_for_source(source)
    session_entry = await runner.async_session_store.get_or_create_session(source)
    session_id = session_entry.session_id
    HeartbeatManager(session_id).set("check status", 60)
    monkeypatch.setattr(runner, "_start_heartbeat_poller", lambda: None)
    monkeypatch.setattr(
        heartbeat,
        "time",
        SimpleNamespace(
            time=lambda: HeartbeatManager(session_id).state.created_at + 61
        ),
    )
    assert not list_session_admissions(
        authority.db, session_id=session_id, pending_only=False
    )
    caplog.set_level("DEBUG", logger="gateway.run")

    watch = {key: (source, session_id)}
    await runner._heartbeat_poll_once(watch)
    await runner._heartbeat_poll_once(watch)

    records = [
        record
        for record in caplog.records
        if key in record.message and "route provenance" in record.message
    ]
    assert [record.levelname for record in records] == ["WARNING", "DEBUG"]
    assert all("not_found" in record.message for record in records)
    assert not list_session_admissions(
        authority.db, session_id=session_id, pending_only=False
    )
    assert HeartbeatManager(session_id).state.fire_count == 0


@pytest.mark.asyncio
async def test_poll_watch_exception_is_warning(poller, monkeypatch, caplog):
    runner, _, watch, key, _ = poller

    async def fail(*args):
        raise RuntimeError("poll failed")

    monkeypatch.setattr(runner, "_heartbeat_poll_watch", fail)
    await runner._heartbeat_poll_once(watch)
    assert any(
        record.levelname == "WARNING" and key in record.message and "poll failed" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_poll_warning_window_is_per_route_and_reason(poller, monkeypatch, caplog):
    from gateway import run_goals

    runner, _, watch, key, _ = poller
    clock = SimpleNamespace(now=0.0, reason="first")
    monkeypatch.setattr(run_goals, "time", SimpleNamespace(monotonic=lambda: clock.now))

    async def fail(*args):
        raise RuntimeError(clock.reason)

    monkeypatch.setattr(runner, "_heartbeat_poll_watch", fail)
    caplog.set_level("DEBUG", logger="gateway.run")
    await runner._heartbeat_poll_once(watch)
    await runner._heartbeat_poll_once(watch)
    clock.reason = "second"
    await runner._heartbeat_poll_once(watch)
    clock.reason = "first"
    await runner._heartbeat_poll_once({"another": watch[key]})
    clock.now = 3600.0
    await runner._heartbeat_poll_once(watch)
    assert [record.levelname for record in caplog.records if "heartbeat poll for" in record.message] == [
        "WARNING",
        "DEBUG",
        "WARNING",
        "WARNING",
        "WARNING",
    ]


@pytest.mark.asyncio
async def test_emergency_stop_holds_due_tick_without_messaging_until_resume(poller, monkeypatch):
    """`hermes pause` must not turn a due heartbeat into a chat message on every poll (prod: ~10k
    identical paused notices in 5h); the tick fires normally after `hermes resume`."""
    from agent import estop

    runner, adapter, watch, _key, clock = poller
    runner._is_user_authorized = lambda source: True
    sent, received = [], []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="wire-1")

    async def real_gateway(event):
        received.append(event)
        return await runner._handle_message(event)

    monkeypatch.setattr(adapter, "send", send)
    adapter.set_message_handler(real_gateway)
    clock.now += 900
    original = HeartbeatManager("heartbeat-session").state.to_json()
    estop.engage(reason="maintenance")
    for _ in range(3):
        await runner._heartbeat_poll_once(watch)
        await asyncio.gather(*adapter._background_tasks)
    assert (received, sent) == ([], [])
    assert HeartbeatManager("heartbeat-session").state.to_json() == original

    async def agent_turn(event):
        event._heartbeat_execution_started = True  # fake agent execution boundary
        received.append(event)

    estop.disengage()
    adapter.set_message_handler(agent_turn)
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    assert len(received) == 1
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1


@pytest.mark.asyncio
async def test_emergency_stop_gate_drops_heartbeat_silently_but_tells_real_users(poller, monkeypatch):
    from agent import estop

    runner, adapter, watch, key, _clock = poller
    source = watch[key][0]
    runner._is_user_authorized = lambda source: True
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="wire-1")

    monkeypatch.setattr(adapter, "send", send)
    adapter.set_message_handler(runner._handle_message)
    estop.engage(reason="maintenance")
    heartbeat = runner._synthetic_prompt_event(source, "check status")
    heartbeat._heartbeat_session_id = "heartbeat-session"
    await adapter.handle_message(heartbeat)
    await asyncio.gather(*adapter._background_tasks)
    assert sent == []

    await adapter.handle_message(MessageEvent(text="are you there?", source=source))
    await asyncio.gather(*adapter._background_tasks)
    assert len(sent) == 1 and estop.paused_reply() in sent[0]
