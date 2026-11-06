"""Durable bootstrap, revalidation, and recovery for plugin-origin Discord threads."""
from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from weakref import ref as weakref_ref
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import hermes_yaml as yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_state_runtime import RuntimeStoreError, get_session_admission, list_session_admissions
from tests.waiting import async_wait_until

PLUGIN = "desk"
THREAD = "9001"


class _StubDiscordAdapter:
    def __init__(self, runner, *, owned=True):
        self.gateway_runner = runner
        self.authority = None
        self.owned = owned
        self.thread_owned_by_self = AsyncMock(side_effect=self._owns_thread)
        self.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="reply"))
        self._threads = SimpleNamespace(mark_async=AsyncMock())

    async def _owns_thread(self, thread_id):
        return self.owned and str(thread_id) == THREAD

    async def _send_final_text(self, event, session_key, text_content, metadata,
                              is_ephemeral_response, ephemeral_ttl, record_delivery):
        record_delivery(SimpleNamespace(message_id="reply"))

    async def _deliver_attachments(self, event, extracted, metadata, *, anything_sent, record_delivery):
        return None

    async def handle_message(self, event):
        if not self.gateway_runner._is_user_authorized_for_source(
                event.source, allow_adapter_delegation=False):
            return False
        # Real ingress binds connector provenance, which later keyed plugin notes borrow.
        from gateway.session_ingress_context import native_callback
        with native_callback(self.gateway_runner, event, self.authority.profile_id):
            await self.authority.admit_native(event)
        return True

    def _unwrap_ephemeral(self, response):
        return response, 0

    async def _extract_response_content(self, text, event, session_key, is_ephemeral_response=False):
        return SimpleNamespace(text_content=text, attachments=[])


@pytest_asyncio.fixture
async def origin_env(tmp_path, monkeypatch):
    from gateway.session_authority import SessionAuthority, initialize_session_authority

    home = tmp_path / "home"
    home.mkdir()
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Conftest pins DEFAULT_DB_PATH once hermes_state is imported; the authority must own this home.
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = _StubDiscordAdapter(runner)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="plugin-origin")
    adapter.authority = authority
    from gateway.session_ingress_context import register_transport_home
    register_transport_home(runner, None, home)
    from gateway.session import SessionSource
    source = SessionSource(
        platform=Platform.DISCORD, chat_id=THREAD, chat_name="thread", chat_type="thread",
        thread_id=THREAD, parent_chat_id="8001", user_id="operator-id", user_name="operator",
        scope_id="7001")
    source._transport_adapter_ref = weakref_ref(adapter)
    from gateway.session_identity import resolve_identity
    resolve_identity(source, runner=runner, adapter=adapter)
    plugin_home = home

    def config(*, origin=True, gateway=True, thread_per_user=False):
        entry = {"allow_gateway_injection": gateway}
        if origin is not None:
            entry["allow_origin_sessions"] = origin
        value = {"plugins": {"entries": {PLUGIN: entry}}, "thread_sessions_per_user": thread_per_user,
                 "platform_toolsets": {"cotal": ["discord"], "discord": ["discord_admin"]}}
        config_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    config()
    manager = PluginManager(scope_key=str(plugin_home))
    context = PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    origin_dispatches = []
    dispatch_origin = runner._dispatch_plugin_origin_injection

    async def record_origin_dispatch(**kwargs):
        try:
            result = await dispatch_origin(**kwargs)
        except Exception as exc:
            origin_dispatches.append(exc)
            raise
        origin_dispatches.append(result)
        return result

    monkeypatch.setattr(runner, "_dispatch_plugin_origin_injection", record_origin_dispatch)
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)

    executed = []
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setattr("tools.discord_tool._load_allowed_actions_config", lambda: None)
    monkeypatch.setattr("tools.discord_tool._load_post_channels_config", lambda: {"500"})
    monkeypatch.setattr("tools.discord_tool._discord_request",
                        lambda *args, **kwargs: {"id": "900", "type": 0})

    async def turn(event, _source, _quick_key, _generation):
        import json
        from gateway.session import build_session_context
        from gateway.session_ingress import admission_author, plugin_origin_turn
        from gateway.session_results import execution_result
        from gateway.run_turn import GatewayTurnMixin
        from tools.cronjob_job_args import _origin_from_env
        from tools.discord_tool import discord_core
        marker = plugin_origin_turn.get()
        toolsets = GatewayTurnMixin._resolve_enabled_toolsets_for_source(
            runner, {"platform_toolsets": {"cotal": ["discord"], "discord": ["discord_admin"]}},
            event.source, "discord")
        entry = runner.session_store.lookup_by_session_key(runner._session_key_for_source(event.source))
        # The session env _hmwa_prepare_turn binds for tools, from the turn's restored source.
        with runner._session_env_scope(build_session_context(event.source, runner.config, entry)):
            cron_origin = _origin_from_env()
            post = json.loads(await asyncio.to_thread(
                discord_core, action="post_message", channel_id="900", content="x"))
        executed.append(SimpleNamespace(text=event.text, source=event.source, marker=marker,
                                        author=admission_author.get(), toolsets=toolsets,
                                        cron_origin=cron_origin, post=post))
        captured = execution_result.get()
        if captured is not None:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source, home=home,
                          plugin_home=plugin_home, config=config, context=context, manager=manager,
                          executed=executed, config_path=config_path, origin_dispatches=origin_dispatches)

    for live in authority.sessions.values():
        if live.task is not None and not live.task.done():
            live.task.cancel()
    await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task is not None),
                         return_exceptions=True)
    await asyncio.to_thread(authority.db.close)


def _origin_source(source):
    return source.to_dict()


def _rows(authority, sid):
    return list_session_admissions(authority.db, session_id=sid, pending_only=False)


def _plugin_row(authority, sid, text):
    return next(row for row in _rows(authority, sid) if row["payload"].get("text") == text)


async def _wait_tasks(env):
    await async_wait_until(lambda: all(live.task is None or live.task.done()
                                       for live in env.authority.sessions.values()),
                           timeout=10, what="session drains")
async def _inject_origin(env, text, *, origin=None):
    before = len(env.origin_dispatches)
    accepted = await asyncio.to_thread(
        env.context.inject_message, text, origin=origin or _origin_source(env.source))
    if accepted:
        await async_wait_until(lambda: len(env.origin_dispatches) > before,
                               timeout=10, what="plugin-origin dispatch")
    return accepted, env.origin_dispatches[-1] if accepted else None


@pytest.mark.asyncio
async def test_bootstrap_uses_plugin_principal_and_narrow_turn_without_human_allowlist(origin_env):
    env = origin_env
    from gateway.plugin_origin_session import eligible
    from gateway.session_authorities import authority_for_home
    assert authority_for_home(env.runner, env.home) is env.authority
    assert eligible(env.runner, env.source), "fixture route should have no committed human input"
    accepted, outcome = await _inject_origin(env, "first note")
    assert accepted is True
    assert outcome is True, f"plugin-origin dispatch did not admit: {outcome!r}"
    await async_wait_until(lambda: env.executed, timeout=10, what="plugin turn")
    await _wait_tasks(env)

    entries = list(env.runner.session_store._entries.values())
    assert len(entries) == 1
    entry = entries[0]
    row = _plugin_row(env.authority, entry.session_id, "first note")
    assert set(row["payload"]) == {"text", "plugin_origin_v1"}
    assert row["payload"]["plugin_origin_v1"]["plugin_id"] == PLUGIN
    assert entry.origin.user_id == f"plugin:{PLUGIN}"
    assert entry.origin.is_bot is True
    assert env.executed[0].source.user_id == f"plugin:{PLUGIN}"
    assert env.executed[0].source.is_bot is True
    assert env.executed[0].marker is True
    assert env.executed[0].author == {"id": f"plugin:{PLUGIN}", "is_bot": True}
    assert "discord_admin" not in env.executed[0].toolsets
    assert env.executed[0].toolsets == ["discord"]
    assert env.adapter.thread_owned_by_self.await_count >= 1
    assert env.executed[0].cron_origin["user_id"] == f"plugin:{PLUGIN}"
    assert "outside discord.post_channels" in env.executed[0].post["error"]
    env.adapter._threads.mark_async.assert_awaited_with(THREAD)

    session_key = env.runner._session_key_for_source(env.source)
    assert await asyncio.to_thread(env.context.inject_message, "second note", session_key=session_key)
    await async_wait_until(lambda: len(env.executed) == 2, timeout=10, what="keyed second note")
    await _wait_tasks(env)
    assert env.executed[1].marker is True
    assert env.executed[1].source.user_id == f"plugin:{PLUGIN}"
    assert "discord_admin" not in env.executed[1].toolsets
    assert len(env.runner.session_store._entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing_key", "non_thread", "not_owned", "per_user", "other_profile", "estop"])
async def test_bootstrap_refusals_create_no_session(origin_env, monkeypatch, case):
    env = origin_env
    origin = _origin_source(env.source)
    if case == "missing_key":
        env.config(origin=None)
    elif case == "non_thread":
        origin.update(chat_type="group", thread_id=None)
    elif case == "not_owned":
        env.adapter.owned = False
    elif case == "per_user":
        env.config(thread_per_user=True)
    elif case == "other_profile":
        origin["profile"] = "other"
    elif case == "estop":
        monkeypatch.setattr("agent.estop.check_paused", lambda *_args, **_kwargs: True)
    accepted, outcome = await _inject_origin(env, "refused", origin=origin)
    if case in {"missing_key", "other_profile"}:
        assert accepted is False
    else:
        assert accepted is True
        assert outcome is False
    await _wait_tasks(env)
    assert env.runner.session_store._entries == {}
    assert env.executed == []


@pytest.mark.asyncio
async def test_existing_session_plugin_and_operator_inputs_use_their_own_authorization(origin_env, monkeypatch):
    env = origin_env
    from gateway.platforms.event import MessageType

    accepted, outcome = await _inject_origin(env, "plugin one")
    assert accepted is True and outcome is True
    await async_wait_until(lambda: len(env.executed) == 1, timeout=10, what="bootstrap turn")
    await _wait_tasks(env)
    session_key = env.runner._session_key_for_source(env.source)
    entry = env.runner.session_store.lookup_by_session_key(session_key)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", env.source.user_id)
    await env.adapter.handle_message(MessageEvent(text="operator reply", message_type=MessageType.TEXT,
                                                  source=env.source, message_id="operator-1"))
    await async_wait_until(lambda: len(env.executed) == 2, timeout=10, what="operator turn")
    await _wait_tasks(env)
    assert env.executed[1].marker is False
    assert env.executed[1].source.user_id == "operator-id"

    assert await asyncio.to_thread(env.context.inject_message, "plugin two", session_key=session_key)
    await async_wait_until(lambda: len(env.executed) == 3, timeout=10, what="keyed plugin turn")
    await _wait_tasks(env)
    assert env.executed[2].marker is True
    assert "discord_admin" not in env.executed[2].toolsets
    assert entry.session_id == env.runner.session_store.lookup_by_session_key(session_key).session_id

    other = PluginContext(PluginManifest(name="other", key="other", source="user"), env.manager)
    config = yaml.safe_load(env.config_path.read_text(encoding="utf-8"))
    config["plugins"]["entries"]["other"] = {"allow_gateway_injection": True}
    env.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert await asyncio.to_thread(other.inject_message, "other plugin", session_key=session_key)
    await async_wait_until(lambda: len(env.executed) == 4, timeout=10, what="other plugin turn")
    await _wait_tasks(env)
    assert env.executed[3].marker is False
    assert "discord_admin" in env.executed[3].toolsets

    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    assert await asyncio.to_thread(env.context.inject_message, "unreadable config", session_key=session_key)
    await async_wait_until(lambda: any(row["payload"].get("text") == "unreadable config"
                                       for row in _rows(env.authority, entry.session_id)),
                           timeout=10, what="queued keyed note")
    env.config_path.write_text("plugins: [", encoding="utf-8")
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
    env.authority._schedule(SessionRef(env.authority.profile_id, entry.session_id))
    await async_wait_until(lambda: len(env.executed) == 5, timeout=10, what="unreadable-config turn")
    await _wait_tasks(env)
    assert env.executed[4].marker is True


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revoked", "thread_per_user", "unreadable", "estop"])
async def test_refused_queued_plugin_row_is_cancelled_and_operator_reply_runs(origin_env, monkeypatch, change):
    env = origin_env
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "before claim")
    assert accepted is True and outcome is True
    route = env.runner._session_key_for_source(env.source)
    entry = env.runner.session_store.lookup_by_session_key(route)
    row = _plugin_row(env.authority, entry.session_id, "before claim")
    if change == "revoked":
        env.config(origin=False)
    elif change == "thread_per_user":
        env.config(thread_per_user=True)
    elif change == "unreadable":
        env.config_path.write_text("plugins: [", encoding="utf-8")
    else:
        monkeypatch.setattr("agent.estop.check_paused", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
    env.authority._schedule(SessionRef(env.authority.profile_id, entry.session_id))
    await async_wait_until(lambda: get_session_admission(env.authority.db, admission_id=row["admission_id"])["outcome"]
                           == "cancelled", timeout=10, what="cancelled plugin admission")
    await _wait_tasks(env)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", env.source.user_id)
    await env.adapter.handle_message(MessageEvent(text="operator after refusal", source=env.source,
                                                  message_id="operator-after-refusal"))
    await async_wait_until(lambda: env.executed, timeout=10, what="operator turn after refusal")
    await _wait_tasks(env)
    assert env.executed[0].text == "operator after refusal"
    assert env.executed[0].marker is False
    assert get_session_admission(env.authority.db, admission_id=row["admission_id"])["status"] == "terminal"


@pytest.mark.asyncio
async def test_unknown_descriptor_key_pauses_non_local_fifo_with_permission_denied(origin_env, monkeypatch):
    env = origin_env
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import admit_session_input

    route = env.runner._session_key_for_source(env.source)
    ref = env.authority._publish_route(env.source, *await tracked_write(
        env.authority, partial(env.authority._reserve_route, env.source)))
    env.authority.sessions[ref.session_id] = LiveSession(env.source, route)
    await tracked_write(env.authority, partial(
        admit_session_input, env.authority.db, epoch=env.authority.epoch,
        principal_id="automation:" + route, session_id=ref.session_id, request_id="unknown-plugin",
        payload={"text": "unknown", "unknown_descriptor_v1": {}},
        _authorize_write=env.authority._admission_gate()))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        await env.authority._claim_next(ref, env.authority.sessions[ref.session_id])


@pytest.mark.asyncio
async def test_recovery_executes_queued_plugin_row_and_refuses_revoked_plugin(origin_env, monkeypatch):
    env = origin_env
    from gateway.session_authority import SessionAuthority
    schedule = SessionAuthority._schedule

    async def restart(**config):
        env.authority.sessions.clear()
        env.config(**config)
        monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
        assert await env.authority.recover_native_sessions(
            [(entry.session_id, env.source, env.adapter)]) == {entry.session_id: "ready"}

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "recover me")
    assert accepted is True and outcome is True
    route = env.runner._session_key_for_source(env.source)
    entry = env.runner.session_store.lookup_by_session_key(route)
    row = _plugin_row(env.authority, entry.session_id, "recover me")
    await restart()
    await async_wait_until(lambda: len(env.executed) == 1, timeout=10, what="recovered plugin turn")
    await _wait_tasks(env)
    assert get_session_admission(env.authority.db, admission_id=row["admission_id"])["status"] == "terminal"
    assert env.executed[0].marker is True

    for text, config in (("revoked across restart", {"origin": False}),
                         ("per-user across restart", {"thread_per_user": True})):
        env.config()
        monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
        accepted, outcome = await _inject_origin(env, text)
        assert accepted is True and outcome is True
        refused = _plugin_row(env.authority, entry.session_id, text)
        await restart(**config)
        await async_wait_until(lambda: get_session_admission(
            env.authority.db, admission_id=refused["admission_id"])["outcome"] == "cancelled",
            timeout=10, what=f"recovered row settled: {text}")
        await _wait_tasks(env)
    assert len(env.executed) == 1


@pytest.mark.asyncio
async def test_two_plugin_injections_racing_operator_message_preserve_fifo(origin_env, monkeypatch):
    env = origin_env
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", env.source.user_id)
    gate = asyncio.Event()
    entered = asyncio.Event()
    original = env.adapter._owns_thread
    calls = 0

    async def owned(thread_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await gate.wait()
        return await original(thread_id)

    def route_waiters():
        turn = env.authority._route_turns.get(env.runner.session_store._generate_session_key(env.source))
        return turn[1] if turn is not None else 0

    env.adapter.thread_owned_by_self.side_effect = owned
    first = asyncio.to_thread(env.context.inject_message, "first", origin=_origin_source(env.source))
    first_task = asyncio.create_task(first)
    await asyncio.wait_for(entered.wait(), timeout=10)
    # The first note holds the route turn; each later input must queue behind it before the next starts.
    second_task = asyncio.create_task(asyncio.to_thread(env.context.inject_message, "second",
                                                       origin=_origin_source(env.source)))
    await async_wait_until(lambda: route_waiters() == 2, timeout=10, what="second note queued")
    native_task = asyncio.create_task(env.adapter.handle_message(
        MessageEvent(text="operator", source=env.source, message_id="operator-race")))
    await async_wait_until(lambda: route_waiters() == 3, timeout=10, what="operator message queued")
    gate.set()
    assert await first_task is True
    assert await second_task is True
    assert await native_task is True
    await async_wait_until(lambda: len(env.executed) == 3, timeout=10, what="ordered turns")
    await _wait_tasks(env)
    entry = env.runner.session_store.lookup_by_session_key(env.runner._session_key_for_source(env.source))
    rows = _rows(env.authority, entry.session_id)
    assert [row["payload"]["text"] for row in rows] == ["first", "second", "operator"]


@pytest.mark.asyncio
async def test_plugin_origin_refusal_and_descriptor_revalidation(origin_env, monkeypatch):
    env = origin_env
    from gateway.plugin_origin_session import PluginOriginRefused, check
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef

    assert env.runner._plugin_origin_admissible(env.source, plugin_home=env.home, plugin_id=PLUGIN)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "check me")
    assert accepted is True and outcome is True
    entry = env.runner.session_store.lookup_by_session_key(env.runner._session_key_for_source(env.source))
    row = _plugin_row(env.authority, entry.session_id, "check me")
    env.config(origin=False)
    with pytest.raises(PluginOriginRefused, match="authorization"):
        await check(env.authority, SessionRef(str(env.home), entry.session_id), row,
                    source=env.source, adapter=env.adapter)
@pytest.mark.asyncio
async def test_managed_plugin_admission_passes_cotal_toolsets_and_origin_marker(origin_env, monkeypatch):
    import json
    from dataclasses import replace
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission, plugin_origin_turn
    from gateway.session_local import _bypass_policy

    env = origin_env
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "managed plugin note")
    assert accepted is True and outcome is True
    entry = env.runner.session_store.lookup_by_session_key(env.runner._session_key_for_source(env.source))
    row = _plugin_row(env.authority, entry.session_id, "managed plugin note")
    ref = SessionRef(env.authority.profile_id, entry.session_id)
    policy = _bypass_policy({"cwd": str(env.home), "model": "fixture", "provider": "custom",
                             "base_url": "http://127.0.0.1:9/v1", "ignore_user_config": True},
                            private_secrets={})
    config = policy.config()
    config["platform_toolsets"] = {"cotal": ["discord"], "cli": ["discord_admin"]}
    policy = replace(policy, config_json=json.dumps(config), toolsets=("discord_admin",))
    observed = []

    async def managed(authority, ref, row, policy, *, plugin_origin=False):
        observed.append((plugin_origin_turn.get(), policy.toolsets, plugin_origin))
        return "managed reply"

    monkeypatch.setattr("gateway.session_completion_notes.note_if_unviewed", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda *_args: None)
    monkeypatch.setattr("gateway.session_managed_worker.managed_policy", lambda *_args: policy)
    monkeypatch.setattr("gateway.session_managed_worker.execute_managed", managed)

    assert await execute_admission(env.authority, ref, row) == "managed reply"
    assert observed == [(True, ("discord",), True)]
    assert plugin_origin_turn.get() is False


@pytest.mark.asyncio
async def test_revoked_plugin_origin_is_dropped_after_claim(origin_env, monkeypatch, caplog):
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef

    env = origin_env
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "revoke between claim and run")
    assert accepted is True and outcome is True
    entry = env.runner.session_store.lookup_by_session_key(env.runner._session_key_for_source(env.source))
    row = _plugin_row(env.authority, entry.session_id, "revoke between claim and run")
    original_execute = env.authority._execute_claimed

    async def revoke_after_claim(ref, live, claimed_row):
        env.config(origin=False)
        return await original_execute(ref, live, claimed_row)

    monkeypatch.setattr(env.authority, "_execute_claimed", revoke_after_claim)
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
    with caplog.at_level("WARNING", logger="gateway.session_ingress"):
        env.authority._schedule(SessionRef(env.authority.profile_id, entry.session_id))
        await async_wait_until(lambda: get_session_admission(
            env.authority.db, admission_id=row["admission_id"])["status"] == "terminal",
            timeout=10, what="revoked plugin-origin row to be dropped")
    await _wait_tasks(env)

    settled = get_session_admission(env.authority.db, admission_id=row["admission_id"])
    assert settled["outcome"] == "failed"
    assert env.executed == []
    assert any("refused at execution" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_each_claimed_row_uses_a_fresh_context(origin_env, monkeypatch):
    from contextvars import ContextVar
    from gateway.platforms.event import MessageType
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import plugin_origin_turn

    env = origin_env
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    accepted, outcome = await _inject_origin(env, "fresh plugin context row")
    assert accepted is True and outcome is True
    entry = env.runner.session_store.lookup_by_session_key(env.runner._session_key_for_source(env.source))
    ref = SessionRef(env.authority.profile_id, entry.session_id)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", env.source.user_id)
    operator_task = asyncio.create_task(env.adapter.handle_message(MessageEvent(
        text="operator context row", message_type=MessageType.TEXT, source=env.source,
        message_id="operator-context-row")))
    await async_wait_until(lambda: len(_rows(env.authority, entry.session_id)) == 2,
                           timeout=10, what="operator row admission")

    drain_value = ContextVar("plugin_origin_drain_value", default=None)
    row_context = []
    post_row_drain_context = []
    claim = env.authority._claim_with_retry

    async def claim_with_drain_context(ref, live):
        result = await claim(ref, live)
        if result is not None and isinstance(result[0], dict):
            post_row_drain_context.append((drain_value.get(), plugin_origin_turn.get()))
            drain_value.set("set by drain")
        return result

    async def execute_with_row_marker(ref, live, claimed_row):
        row_context.append((claimed_row["payload"]["text"], drain_value.get(), plugin_origin_turn.get()))
        plugin_origin_turn.set(True)
        return "reply", "completed"

    monkeypatch.setattr(env.authority, "_claim_with_retry", claim_with_drain_context)
    monkeypatch.setattr(env.authority, "_execute_claimed", execute_with_row_marker)
    monkeypatch.setattr(SessionAuthority, "_schedule", schedule)
    env.authority._schedule(ref)
    await operator_task
    await _wait_tasks(env)

    assert row_context == [
        ("fresh plugin context row", None, True),
        ("operator context row", None, False),
    ]
    assert post_row_drain_context == [(None, False), ("set by drain", False)]


def test_local_plugin_origin_display_keeps_cotal_toolsets(origin_env, monkeypatch):
    from contextlib import contextmanager
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal
    from gateway.session_ingress import plugin_origin_turn
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import local_adapter_map
    from gateway.session_policy import LocalSessionPolicy

    env = origin_env
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    actor = Principal("test-owner", env.authority.profile_id,
                      frozenset({"session:create", "session:submit", "session:read"}), "fixture")
    ref = create_local_session(env.authority, actor, {
        "request_id": "local-plugin-toolsets", "source": "cli", "cwd": str(env.home),
        "model": "fixture", "toolsets": ["terminal"],
    })
    adapter = local_adapter_map(env.authority)[Platform.LOCAL]
    policy = adapter.policies[ref.session_id]
    runner = env.runner
    original_adapters = runner.adapters
    runner.adapters = {Platform.LOCAL: adapter}
    source = env.authority.sessions[ref.session_id].source

    @contextmanager
    def plugin_turn():
        token = plugin_origin_turn.set(True)
        try:
            yield
        finally:
            plugin_origin_turn.reset(token)

    try:
        assert isinstance(policy, LocalSessionPolicy)
        with plugin_turn():
            display = GatewayRunner._run_agent_display_settings(runner, source)
        assert display.enabled_toolsets == ["discord"]
    finally:
        runner.adapters = original_adapters

