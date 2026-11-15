"""Heartbeat polls wake idle adapters, never accumulate deferred ticks."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND, display_kind_for_event
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
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
    db = goals._get_session_db()
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(heartbeat, "time", SimpleNamespace(time=lambda: clock.now))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm", message_id="watch-cmd")
    key = build_session_key(source)
    config = GatewayConfig()
    runner = object.__new__(GatewayRunner)
    runner.session_store = SessionStore(config.sessions_dir, config)
    db.create_session("heartbeat-session", "telegram", session_key=key)
    now = datetime.now()
    entry = SessionEntry(key, "heartbeat-session", now, now, origin=source, platform=source.platform)
    db.save_gateway_routing_entry(
        key, json.dumps(entry.to_dict()), scope=runner.session_store._routing_scope(),
    )
    adapter = _HeartbeatAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._run_in_executor_with_context = asyncio.to_thread
    watch = {key: (source, "heartbeat-session")}
    HeartbeatManager("heartbeat-session").set("check status", 60)
    yield runner, adapter, watch, key, clock
    goals._DB_CACHE.clear()


def test_legacy_manager_uses_current_route_without_session_duplicate(poller):
    from hermes_cli import goals
    from hermes_cli.heartbeat import load_route_heartbeat

    _, _, _, key, clock = poller
    db = goals._get_session_db()
    manager = HeartbeatManager('heartbeat-session')
    assert db.get_meta('heartbeat:heartbeat-session') is None
    assert manager.pause().status == 'paused'
    assert load_route_heartbeat(key, db).status == 'paused'
    manager.resume()
    clock.now += 60
    assert manager.due_prompt() is not None
    assert load_route_heartbeat(key, db).fire_count == 1
    assert manager.abandon_fire()
    manager.record_error('failure')
    assert load_route_heartbeat(key, db).last_error == 'failure'
    assert manager.clear()
    assert load_route_heartbeat(key, db) is None
    assert db.get_meta('heartbeat:heartbeat-session') is None

def test_legacy_manager_uses_only_scoped_routing_index(poller):
    from hermes_cli import goals
    from hermes_cli.heartbeat import load_route_heartbeat

    runner, _, _, key, _ = poller
    db = goals._get_session_db()
    scope = runner.session_store._routing_scope()
    manager = HeartbeatManager("heartbeat-session", routing_db=db, routing_scope=scope)
    assert manager.state is not None
    assert manager.pause().status == "paused"
    assert load_route_heartbeat(key, db).status == "paused"


def test_legacy_manager_does_not_match_other_routing_scope(poller):
    from hermes_cli import goals
    from hermes_cli.heartbeat import load_route_heartbeat

    runner, _, _, key, _ = poller
    db = goals._get_session_db()
    other_key = key + ":other-scope"
    db.create_session("other-scope-session", "telegram", session_key=other_key)
    other_scope = str(db.db_path.parent / "other-sessions")
    now = datetime.now()
    entry = SessionEntry(other_key, "other-scope-session", now, now, platform=Platform.TELEGRAM)
    db.save_gateway_routing_entry(other_key, json.dumps(entry.to_dict()), scope=other_scope)
    with pytest.raises(RuntimeError, match="route lookup failed"):
        HeartbeatManager("other-scope-session", routing_db=db, routing_scope=runner.session_store._routing_scope())
    assert load_route_heartbeat(other_key, db) is None


def test_route_lookup_failure_is_not_reported_as_success(poller, monkeypatch):
    from hermes_cli import goals

    runner, _, _, _, _ = poller
    db = goals._get_session_db()

    def fail(*args, **kwargs):
        raise OSError("routing unavailable")
    monkeypatch.setattr(db, "_read_all", fail)
    with pytest.raises(RuntimeError, match="route lookup failed"):
        HeartbeatManager("heartbeat-session", routing_db=db, routing_scope=runner.session_store._routing_scope()).set("x", 60)


def test_heartbeat_list_reports_route_and_error(poller, capsys):
    from hermes_cli.subcommands.heartbeat import _list_heartbeats

    _, _, _, key, _ = poller
    manager = HeartbeatManager("heartbeat-session")
    manager.record_error("failed")
    _list_heartbeats(None)
    output = capsys.readouterr().out
    assert f"Route: {key}" in output
    assert "Interval: 1m" in output
    assert "Last fired: never" in output
    assert "Fire count: 0" in output
    assert "Last error: failed" in output


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
async def test_clearing_route_heartbeat_stops_future_fires(poller):
    runner, adapter, watch, key, clock = poller

    async def handler(event):
        event._heartbeat_execution_started = True

    adapter.set_message_handler(handler)
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1

    HeartbeatManager("heartbeat-session").clear()
    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)

    state = HeartbeatManager("heartbeat-session").state
    assert state is None
    assert not adapter._background_tasks
    assert not adapter._pending_messages
    assert not watch


@pytest.mark.asyncio
async def test_due_heartbeat_steers_an_active_turn_instead_of_waiting_for_idle(poller):
    runner, adapter, watch, key, clock = poller
    active = asyncio.Event()
    HeartbeatManager("heartbeat-session").set("check status", 60)
    steered = []

    class RunningAgent:
        def steer(self, text, *, ticket=None):
            steered.append(text)
            active.set()
            return True

    adapter.set_message_handler(lambda _event: None)
    adapter._active_sessions[key] = asyncio.Event()
    runner._busy_input_mode = "steer"
    runner._is_user_authorized_for_source = lambda source, allow_adapter_delegation=True: True
    runner._admit_bot_message_for_source = lambda source: True
    runner._admit_bot_chain = lambda event: True
    runner._effective_busy_input_mode = lambda source: "steer"
    runner._route_plaintext_approval_while_busy = lambda *args: asyncio.sleep(0, result=False)
    runner._busy_skill_turn = lambda *args: asyncio.sleep(0, result=(False, None))
    runner._effective_busy_text_mode = lambda source: "steer"
    turn = SimpleNamespace(agent=RunningAgent(), heartbeat_steered=False, heartbeat_claim=None)
    runner._peek_session_state = lambda session_key: SimpleNamespace(turn=turn)
    runner._resolve_busy_steer_or_redirect = lambda *args: asyncio.sleep(
        0, result=SimpleNamespace(
            effective_mode="steer", steered=True, redirected=False,
            demoted_for_subagents=False, demoted_for_compression=False,
        ),
    )
    runner._busy_interrupt_and_ack = lambda *args: asyncio.sleep(0)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    clock.now += 60
    await runner._heartbeat_poll_once(watch)
    await asyncio.wait_for(active.wait(), 2)

    assert len(steered) == 1
    assert "check status" in steered[0]
    assert "Heartbeat" in steered[0]
    assert key not in adapter._pending_messages
    assert HeartbeatManager("heartbeat-session").state.fire_count == 1



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
    from agent.i18n import t
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway

    user_id = "1385361904591044669"
    runner, adapter, authority, _, sent, _, _ = await _gateway(monkeypatch)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        guild_id="1477066497896742955",
    )
    key = runner._session_key_for_source(source)
    assert key == f"agent:main:discord:group:{chat_id}:{user_id}"
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
    assert sent[-1] == t("gateway.heartbeat.needs_admission")


@pytest.mark.asyncio
async def test_discord_group_heartbeat_resume_without_paused_state_reports_no_resume(
    monkeypatch
):
    from agent.i18n import t
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway

    _runner, adapter, _, _, sent, _, _ = await _gateway(monkeypatch)
    user_id = "1385361904591044669"
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id="1571155392069763142",
        chat_type="group",
        user_id=user_id,
        guild_id="1477066497896742955",
    )
    await _chat(adapter, source, sent, "/heartbeat resume")

    assert sent[-1] == t("gateway.heartbeat.no_resume")


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
        guild_id="1477066497896742955",
    )
    key = runner._session_key_for_source(source)
    assert key == f"agent:main:discord:group:{chat_id}:{user_id}"
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
async def test_discord_group_active_heartbeat_resume_reanchors_after_native_message(monkeypatch):
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway

    runner, adapter, _, _, sent, _, _ = await _gateway(monkeypatch)
    user_id = "1385361904591044669"
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", user_id)
    source = adapter.build_source(
        chat_id="1571155392069763142",
        chat_type="group",
        user_id=user_id,
        guild_id="1477066497896742955",
    )
    await _chat(adapter, source, sent, "check status")
    session_id = runner.session_store.peek_session_id(runner._session_key_for_source(source))
    manager = HeartbeatManager(session_id)
    before = manager.set("check status", 60)
    monkeypatch.setattr(runner, "_start_heartbeat_poller", lambda: None)

    await _chat(adapter, source, sent, "/heartbeat resume")

    after = HeartbeatManager(session_id).state
    assert after.status == "active"
    assert after.created_at == before.created_at
    assert after.fire_count == before.fire_count
    assert after.last_fired_at is not None
    assert "check status" in sent[-1]


@pytest.mark.asyncio
async def test_poller_skips_envelope_validation_until_heartbeat_is_due(monkeypatch):
    from tests.gateway.test_authority_lane_moves_on import _chat, _gateway
    from gateway import session_automation

    runner, adapter, _, source, sent, _, _ = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, "check status")
    session_id = runner.session_store.peek_session_id(runner._session_key_for_source(source))
    state = HeartbeatManager(session_id).set("check status", 3600)
    validated = False

    def reject_early_validation(*args):
        nonlocal validated
        validated = True
        raise AssertionError("envelope validation ran before a tick was due")

    monkeypatch.setattr(session_automation, "validated_committed_route_envelope", reject_early_validation)
    watch = {runner._session_key_for_source(source): (source, session_id)}

    await runner._heartbeat_poll_once(watch)

    after = HeartbeatManager(session_id).state
    assert not validated
    assert after.fire_count == state.fire_count
    assert after.last_fired_at == state.last_fired_at


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/heartbeat every 1m x", "/heartbeat resume"])
async def test_legacy_heartbeat_commands_remain_available(monkeypatch, command):
    from agent.i18n import t
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin

    class LegacyRunner(GatewayGoalCommandsMixin):
        session_authority = None

        def __init__(self, adapter):
            self.adapter = adapter

        async def _get_heartbeat_manager_for_event(self, event):
            return HeartbeatManager("legacy-heartbeat"), SimpleNamespace(session_id="legacy-heartbeat")

        def _session_key_for_source(self, source):
            return "legacy-route"

        def _delivery_adapter_for(self, source):
            return self.adapter

        def _register_heartbeat_watch(self, *args):
            pass

    adapter = _HeartbeatAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    adapter.set_message_handler(lambda event: None)
    runner = LegacyRunner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="legacy", user_id="legacy")
    if command == "/heartbeat resume":
        HeartbeatManager("legacy-heartbeat").set("x", 60)
        HeartbeatManager("legacy-heartbeat").pause()
    result = await runner._handle_heartbeat_command(MessageEvent(text=command, source=source))
    assert result != t("gateway.heartbeat.needs_admission")
    assert HeartbeatManager("legacy-heartbeat").state.status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/heartbeat every 1m x", "/heartbeat resume"])
async def test_local_heartbeat_commands_remain_available(tmp_path, monkeypatch, command):
    from agent.i18n import t
    from gateway.platforms.event import MessageEvent
    from hermes_cli.heartbeat import HeartbeatManager
    from tests.gateway.test_session_end_delegations import _local

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    source = authority.sessions[ref.session_id].source
    monkeypatch.setattr(runner, "_register_heartbeat_watch", lambda *args: None)
    if command == "/heartbeat resume":
        manager = HeartbeatManager(ref.session_id)
        manager.set("x", 60)
        manager.pause()

    result = await runner._handle_heartbeat_command(MessageEvent(text=command, source=source))

    assert result != t("gateway.heartbeat.needs_admission")
    assert HeartbeatManager(ref.session_id).state.status == "active"


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
        guild_id="1477066497896742955",
    )
    key = runner._session_key_for_source(source)
    assert key == f"agent:main:discord:group:{chat_id}:{user_id}"
    session_entry = await runner.async_session_store.get_or_create_session(source)
    session_id = session_entry.session_id
    created_at = HeartbeatManager(session_id).set("check status", 60).created_at
    monkeypatch.setattr(runner, "_start_heartbeat_poller", lambda: None)
    monkeypatch.setattr(
        heartbeat,
        "time",
        SimpleNamespace(
            time=lambda: created_at + 61
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
