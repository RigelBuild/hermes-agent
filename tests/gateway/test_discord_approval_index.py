"""Behavioral tests for the Discord daily exec-approval index."""

import asyncio
import datetime as dt
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.gateway.conftest import _ensure_discord_mock

_ensure_discord_mock()

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class _Message:
    def __init__(self, channel, message_id, content="", **kwargs):
        self.channel = channel
        self.id = message_id
        self.content = content
        self.embeds = kwargs.get("embeds", [])
        self.guild = channel.guild
        self.edits = []
        self.edit_kwargs = []
    async def create_thread(self, *, name, **kwargs):
        return await self.channel.create_thread(name=name, **kwargs)

    async def edit(self, *, content=None, **kwargs):
        self.content = content
        self.edits.append(content)
        self.edit_kwargs.append(kwargs)


class _Channel:
    def __init__(self, client, channel_id, *, guild_id="777", channel_type=0, parent=None):
        self.client = client
        self.id = int(channel_id)
        self.guild = SimpleNamespace(id=guild_id)
        self.type = SimpleNamespace(value=channel_type)
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.locked = False
        self.archived = False
        self.sent = []
        self.threads = []
        self.messages = {}
        self.name = None

    async def send(self, content=None, **kwargs):
        message = self.client.message(self, content or "", **kwargs)
        self.sent.append((content, kwargs, message))
        self.messages[message.id] = message
        return message

    async def create_thread(self, *, name, content=None, **kwargs):
        thread = self.client.channel(name=name, parent=self, channel_type=11)
        self.threads.append(thread)
        if content:
            await thread.send(content=content)
        return thread

    async def fetch_message(self, message_id):
        return self.messages[int(message_id)]


class _Client:
    def __init__(self):
        self.channels = {}
        self.next_id = 1000

    def channel(self, channel_id=None, *, name=None, parent=None, channel_type=0, guild_id="777"):
        self.next_id += 1
        channel = _Channel(
            self, channel_id or self.next_id, guild_id=guild_id,
            channel_type=channel_type, parent=parent,
        )
        channel.name = name
        self.channels[channel.id] = channel
        return channel

    def message(self, channel, content="", **kwargs):
        self.next_id += 1
        return _Message(channel, self.next_id, content, **kwargs)

    def get_channel(self, channel_id):
        return self.channels.get(int(channel_id))

    async def fetch_channel(self, channel_id):
        return self.channels[int(channel_id)]


def _adapter(home, *, channel_id="900", client=None, forum=False, extra=None):
    home.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["HERMES_HOME"] = str(home)
    client = client or _Client()
    prompt_channel = client.channel("555")
    index_channel = client.channel(channel_id, channel_type=15 if forum else 0) if channel_id else None
    adapter = DiscordAdapter(PlatformConfig(
        enabled=True, token="***", extra={"approval_index_channel": channel_id or "", **(extra or {})},
    ))
    adapter._client = client
    return adapter, client, prompt_channel, index_channel


async def _send_prompt(adapter, request_id="approval-1", command="make release"):
    return await adapter.send_exec_approval(
        chat_id="555", command=command, description="release task", session_key="discord:555",
        request_id=request_id,
    )


async def _wait_for_index(adapter):
    if adapter._approval_index is None:
        return
    tasks = tuple(adapter._approval_index._tasks)
    if tasks:
        await asyncio.gather(*tasks)


def _assert_index_temporary_state_empty(index):
    assert index._pending == {}
    assert index._posting == set()
    assert index._settled == {}
    assert index._reservations == set()


@pytest.mark.asyncio
async def test_stalled_settlement_edit_does_not_delay_approval_wait(tmp_path, monkeypatch, caplog):
    from tools import approval as approval_module

    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    request_id = "stalled-settlement"
    await _send_prompt(adapter, request_id)
    await _wait_for_index(adapter)
    started = asyncio.Event()

    async def stall_edit(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(index, "_edit", stall_edit)
    monkeypatch.setattr(
        "plugins.platforms.discord.approval_index._INDEX_OPERATION_TIMEOUT_SECONDS", 0.5,
    )
    runner = _TurnRunner(_StatusAdapter(adapter), "discord:555")
    loop = asyncio.get_running_loop()
    runner._schedule = lambda coroutine, _label: asyncio.run_coroutine_threadsafe(coroutine, loop)
    wait_task = await _approval_wait(
        "discord:555", {"request_id": request_id, "command": "make release"}, runner,
    )

    assert approval_module.resolve_gateway_approval(
        "discord:555", "once", request_id=request_id,
    ) == 1
    decision = await asyncio.wait_for(wait_task, timeout=2)
    await asyncio.wait_for(started.wait(), timeout=2)
    settle_task = next(
        task for task in index._tasks
        if task.get_coro().__name__ == "_resolve_bounded"
    )
    assert decision["resolved"] and decision["choice"] == "once"
    assert settle_task in index._tasks
    await asyncio.gather(settle_task, return_exceptions=True)
    assert "edit timed out" in caplog.text
    assert index._tasks == set()
    _assert_index_temporary_state_empty(index)
    approval_module._gateway_queues.pop("discord:555", None)


def _line_messages(index_channel):
    statuses = ("🟡 pending ·", "✅ approved", "❌ denied", "⌛ timed out", "⛔ cancelled")
    return [message for thread in index_channel.threads for message in thread.messages.values()
            if any(status in message.content for status in statuses)]


@pytest.mark.asyncio
async def test_approval_prompt_posts_a_line_with_the_sent_messages_jump_url(tmp_path, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    fixed_now = datetime(2026, 11, 5, 23, 30, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("hermes_time.now", lambda: fixed_now)
    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")

    result = await _send_prompt(adapter)

    await _wait_for_index(adapter)
    assert result.success is True
    prompt = prompt_channel.sent[0][2]
    line = _line_messages(index_channel)[0]
    assert line.content == (
        f"🟡 pending · make release · https://discord.com/channels/777/{prompt_channel.id}/{prompt.id}"
    )
    assert index_channel.threads[0].name == "Thu Nov 5"
    assert json.loads((tmp_path / "home" / "gateway" / "discord_approval_index.json").read_text()) == {
        fixed_now.date().isoformat(): str(index_channel.threads[0].id)
    }

@pytest.mark.asyncio
async def test_approval_index_reuses_its_daily_thread_and_persisted_thread(tmp_path, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    local_now = datetime(2026, 11, 5, 23, 30, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr("hermes_time.now", lambda: local_now)
    home = tmp_path / "home"
    adapter, client, _prompt_channel, index_channel = _adapter(home)

    first = await _send_prompt(adapter, "approval-1")
    second = await _send_prompt(adapter, "approval-2", "git push")
    await _wait_for_index(adapter)

    assert first.success and second.success
    assert len(index_channel.threads) == 1
    state_path = home / "gateway" / "discord_approval_index.json"
    persisted = json.loads(state_path.read_text())
    date_key = local_now.date().isoformat()
    thread_id = str(index_channel.threads[0].id)
    assert persisted == {date_key: thread_id}
    assert index_channel.threads[0].name == "Thu Nov 5"

    restarted, _same_client, _prompt_channel, _index_channel = _adapter(home, client=client)
    third = await _send_prompt(restarted, "approval-3", "git status")
    await _wait_for_index(restarted)

    assert third.success is True
    assert len(index_channel.threads) == 1
    assert len(_line_messages(index_channel)) == 3


@pytest.mark.asyncio
async def test_approval_index_reservation_buffers_cancel_during_prompt_send(tmp_path, monkeypatch):
    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    request_id = "cancel-during-send"
    sending = asyncio.Event()
    continue_send = asyncio.Event()
    original_send = prompt_channel.send

    async def delayed_send(*args, **kwargs):
        sending.set()
        await continue_send.wait()
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(prompt_channel, "send", delayed_send)
    prompt_task = asyncio.create_task(_send_prompt(adapter, request_id))
    await asyncio.wait_for(sending.wait(), timeout=2)
    assert request_id in index._reservations

    await index.resolve(request_id, "cancelled")
    assert index._settled[request_id] == "cancelled"
    continue_send.set()
    result = await prompt_task
    await _wait_for_index(adapter)

    assert result.success
    assert _line_messages(index_channel)[0].content.startswith("⛔ cancelled")
    _assert_index_temporary_state_empty(index)


@pytest.mark.asyncio
async def test_failed_prompt_send_discards_index_reservation(tmp_path, monkeypatch):
    adapter, _client, prompt_channel, _index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    request_id = "failed-prompt-send"
    sending = asyncio.Event()
    continue_send = asyncio.Event()

    async def fail_after_pause(*_args, **_kwargs):
        sending.set()
        await continue_send.wait()
        raise RuntimeError("Discord send failed")

    monkeypatch.setattr(prompt_channel, "send", fail_after_pause)
    prompt_task = asyncio.create_task(_send_prompt(adapter, request_id))
    await asyncio.wait_for(sending.wait(), timeout=2)
    assert request_id in index._reservations
    await index.resolve(request_id, "cancelled")
    continue_send.set()
    result = await prompt_task
    await _wait_for_index(adapter)

    assert not result.success
    _assert_index_temporary_state_empty(index)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_fields", [{"status": 403}, {"code": 50013}])
async def test_approval_index_keeps_cached_thread_on_permission_error(tmp_path, monkeypatch, error_fields):
    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    thread = await index._get_daily_thread()

    async def deny_send(*_args, **_kwargs):
        error = RuntimeError("missing permissions")
        for key, value in error_fields.items():
            setattr(error, key, value)
        raise error

    monkeypatch.setattr(thread, "send", deny_send)
    with pytest.raises(RuntimeError, match="missing permissions"):
        await index.add_pending("permission-denied", "make release", "https://example.test")

    assert len(index_channel.threads) == 1
    assert index._thread is thread
    _assert_index_temporary_state_empty(index)


@pytest.mark.asyncio
async def test_interrupt_after_committed_choice_settles_index_as_cancelled(tmp_path, monkeypatch):
    from tools import approval as approval_module

    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    request_id = "committed-choice-interrupt"
    await _send_prompt(adapter, request_id)
    await _wait_for_index(adapter)
    runner = _TurnRunner(_StatusAdapter(adapter), "discord:555")

    def commit_then_interrupt(_event, session_key, *, interrupt_log):
        assert approval_module.resolve_gateway_approval(
            session_key, "once", request_id=request_id,
        ) == 1
        return "interrupted"

    monkeypatch.setattr("tools.approval_gateway_wait._poll_event", commit_then_interrupt)
    wait_task = await _approval_wait(
        "discord:555", {"request_id": request_id, "command": "make release"}, runner,
    )
    decision = await wait_task
    await runner.scheduled.pop(0)
    await _wait_for_index(adapter)

    assert decision["cancelled"]
    assert decision["choice"] == "deny"
    assert _line_messages(index_channel)[0].content.startswith("⛔ cancelled")
    _assert_index_temporary_state_empty(adapter._approval_index)
    approval_module._gateway_queues.pop("discord:555", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("config_yaml,expected_channel,expected_target", [
    ("discord:\n  enabled: true\n  approval_index_channel: 901\n  approval_mentions_target: index\n", "901", "index"),
    ("platforms:\n  discord:\n    enabled: true\n    extra:\n      approval_index_channel: 902\n      approval_mentions_target: index\n", "902", "index"),
])
async def test_approval_index_channel_config_enables_adapter(
    tmp_path, monkeypatch, config_yaml, expected_channel, expected_target,
):
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    discover_plugins()

    config = load_gateway_config()
    adapter = DiscordAdapter(config.platforms[Platform.DISCORD])

    assert config.platforms[Platform.DISCORD].extra["approval_index_channel"] == expected_channel
    assert config.platforms[Platform.DISCORD].extra["approval_mentions_target"] == expected_target
    assert adapter._approval_index._channel_id == expected_channel


@pytest.mark.asyncio
async def test_approval_index_creates_a_forum_post_for_the_day(tmp_path):
    adapter, _client, _prompt_channel, forum = _adapter(tmp_path / "home", forum=True)

    result = await _send_prompt(adapter)
    await _wait_for_index(adapter)

    assert result.success is True
    assert len(forum.threads) == 1
    from hermes_time import now, safe_strftime
    today = now().date()
    title = f"{safe_strftime(today, '%a %b')} {today.day}"
    assert forum.threads[0].name == title
    assert forum.threads[0].sent[0][0] == title
    assert _line_messages(forum)[0].content.startswith("🟡 pending · make release · ")


@pytest.mark.asyncio
async def test_approval_index_is_off_when_channel_is_unset(tmp_path):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home", channel_id="")

    result = await _send_prompt(adapter)
    await _wait_for_index(adapter)

    assert result.success is True
    assert adapter._approval_index is None


@pytest.mark.asyncio
async def test_approval_index_applies_settlement_during_pending_post(tmp_path, monkeypatch):
    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    thread = await index._get_daily_thread()
    original_send = thread.send
    posting = asyncio.Event()
    continue_post = asyncio.Event()

    async def delayed_send(*args, **kwargs):
        posting.set()
        await continue_post.wait()
        return await original_send(*args, **kwargs)

    monkeypatch.setattr(thread, "send", delayed_send)
    post_task = asyncio.create_task(index.add_pending(
        "fast-approval", "make release", "https://discord.com/channels/777/555/1234",
    ))
    await posting.wait()
    await index.resolve("fast-approval", "once")
    continue_post.set()
    await post_task
    assert _line_messages(index_channel)[0].content.startswith(
        "✅ approved · once · make release · https://discord.com/channels/777/555/1234"
    )
    _assert_index_temporary_state_empty(adapter._approval_index)


@pytest.mark.asyncio
async def test_approval_index_drops_settlement_without_pending_or_posting(tmp_path):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")

    await adapter._approval_index.resolve("unknown-approval", "once")

    assert adapter._approval_index._settled == {}

@pytest.mark.asyncio
async def test_approval_index_failure_does_not_fail_prompt(tmp_path, monkeypatch):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")

    async def fail(*_args, **_kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(adapter._approval_index, "add_pending", fail)
    result = await _send_prompt(adapter)
    await _wait_for_index(adapter)

    assert result.success is True


@pytest.mark.asyncio
async def test_index_send_stall_does_not_delay_successful_prompt(tmp_path, monkeypatch, caplog):
    adapter, _client, prompt_channel, _index_channel = _adapter(tmp_path / "home")
    started = asyncio.Event()
    stalled = asyncio.Event()

    async def stall(*_args, **_kwargs):
        started.set()
        await stalled.wait()

    monkeypatch.setattr(adapter._approval_index, "add_pending", stall)
    result = await asyncio.wait_for(_send_prompt(adapter), timeout=2)
    await asyncio.wait_for(started.wait(), timeout=2)

    assert result.success
    assert len(prompt_channel.sent) == 1
    tasks = tuple(adapter._approval_index._tasks)
    stalled.set()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert "could not start pending record" not in caplog.text


@pytest.mark.asyncio
async def test_index_send_timeout_cleans_up_and_does_not_raise(tmp_path, monkeypatch, caplog):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")
    started = asyncio.Event()
    stalled = asyncio.Event()

    async def stall(*_args, **_kwargs):
        started.set()
        await stalled.wait()

    monkeypatch.setattr("plugins.platforms.discord.approval_index._INDEX_OPERATION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(adapter._approval_index, "add_pending", stall)
    result = await _send_prompt(adapter, request_id="timed-out-index")
    await asyncio.wait_for(started.wait(), timeout=2)
    await _wait_for_index(adapter)

    assert result.success
    assert "post timed out" in caplog.text
    assert adapter._approval_index._posting == set()
    assert adapter._approval_index._settled == {}
    assert adapter._approval_index._tasks == set()


@pytest.mark.asyncio
async def test_cancelling_index_task_does_not_cancel_approval_wait(tmp_path, monkeypatch):
    from tools import approval as approval_module
    from tools import approval_gateway_wait

    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")
    started = asyncio.Event()
    stalled = asyncio.Event()

    async def stall(*_args, **_kwargs):
        started.set()
        await stalled.wait()

    monkeypatch.setattr(adapter._approval_index, "add_pending", stall)
    monkeypatch.setattr(approval_gateway_wait, "_poll_event", lambda *_a, **_kw: "timeout")
    request_id = "independent-wait"
    result = await _send_prompt(adapter, request_id=request_id)
    await asyncio.wait_for(started.wait(), timeout=2)
    runner = _TurnRunner(_StatusAdapter(adapter), "discord:555")
    wait_task = await _approval_wait(
        "discord:555", {"request_id": request_id, "command": "make release"}, runner,
    )
    index_task = next(iter(adapter._approval_index._tasks))
    index_task.cancel()
    await asyncio.gather(index_task, return_exceptions=True)
    decision = await wait_task

    assert result.success
    assert not decision["resolved"]
    assert index_task not in adapter._approval_index._tasks
    assert adapter._approval_index._settled == {}
    assert runner.scheduled
    await asyncio.gather(*runner.scheduled)
    approval_module._gateway_queues.pop("discord:555", None)

class _TurnRunner:
    def __init__(self, adapter, session_key):
        self.scheduled = []
        self._ctx = SimpleNamespace(
            session_key=session_key,
            _run_still_current=lambda: True,
            _status_adapter=adapter,
            _status_chat_id="555",
            _status_thread_metadata=None,
        )

    def _schedule(self, coroutine, _label):
        self.scheduled.append(coroutine)


class _StatusAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.sent = []

    async def resolve_approval_index(self, request_id, choice):
        await self.adapter.resolve_approval_index(request_id, choice)

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SimpleNamespace(success=True)


class _LoopFuture:
    def __init__(self, future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)


async def _approval_wait(session_key, approval_data, runner, *, poll=None):
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    from tools.approval_gateway_wait import _await_gateway_decision

    notified = threading.Event()

    def notify(data):
        register_timeout_notice(
            runner, data, command=data["command"], card_message_id=None,
        )
        from gateway.run_turn_runner_approval_settle import register_approval_index_settle
        status_adapter = runner._ctx._status_adapter
        if status_adapter is not None:
            register_approval_index_settle(runner, data, status_adapter.adapter)
        notified.set()
    task = asyncio.create_task(asyncio.to_thread(_await_gateway_decision, session_key, notify, approval_data))
    assert await asyncio.to_thread(notified.wait, 5)
    return task


@pytest.mark.asyncio
async def test_button_approval_edits_its_index_line(tmp_path):
    from tools import approval as approval_module
    import plugins.platforms.discord.adapter as discord_adapter

    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")
    request_id = "button-approval"
    session_key = "discord:555"
    await _send_prompt(adapter, request_id)
    await _wait_for_index(adapter)
    view = prompt_channel.sent[0][1]["view"]
    view._check_auth = lambda _interaction: True
    runner = _TurnRunner(_StatusAdapter(adapter), session_key)
    approval_data = {"request_id": request_id, "command": "make release"}
    wait_task = await _approval_wait(session_key, approval_data, runner)
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1, display_name="alice"),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
        message=SimpleNamespace(embeds=[SimpleNamespace(set_footer=Mock(), color=None)]),
    )

    await view._resolve(interaction, "once", discord_adapter.discord.Color.green(), "platform.discord.approval.resolved_once")
    await wait_task
    await runner.scheduled.pop(0)
    await _wait_for_index(adapter)

    assert _line_messages(index_channel)[0].content.startswith("✅ approved")
    approval_module._gateway_queues.pop(session_key, None)


@pytest.mark.asyncio
async def test_typed_deny_edits_its_index_line(tmp_path):
    from tests.gateway.test_approve_deny_commands import _make_event, _make_runner, _make_source
    from tools import approval as approval_module

    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    slash_runner = _make_runner()
    session_key = slash_runner._session_key_for_source(_make_source())
    await adapter.send_exec_approval(
        chat_id="555", command="rm build", session_key=session_key, request_id="typed-deny",
    )
    await asyncio.gather(*adapter._approval_index._tasks)
    settle_runner = _TurnRunner(_StatusAdapter(adapter), session_key)
    wait_task = await _approval_wait(
        session_key, {"request_id": "typed-deny", "command": "rm build"}, settle_runner,
    )

    result = await slash_runner._handle_deny_command(_make_event("/deny wrong directory"))
    await wait_task
    await settle_runner.scheduled.pop(0)
    await _wait_for_index(adapter)

    assert "wrong directory" in result
    assert _line_messages(index_channel)[0].content.startswith("❌ denied")
    approval_module._gateway_queues.pop(session_key, None)


@pytest.mark.asyncio
async def test_timeout_edits_its_index_line(tmp_path, monkeypatch):
    from tools import approval as approval_module
    from tools import approval_gateway_wait

    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    request_id = "timed-out"
    session_key = "discord:555"
    await _send_prompt(adapter, request_id)
    await _wait_for_index(adapter)
    status_adapter = _StatusAdapter(adapter)
    runner = _TurnRunner(status_adapter, session_key)
    monkeypatch.setattr(approval_gateway_wait, "_poll_event", lambda *_a, **_kw: "timeout")
    task = await _approval_wait(
        session_key, {"request_id": request_id, "command": "make release"}, runner,
    )

    decision = await task
    await runner.scheduled.pop(0)
    await runner.scheduled.pop(0)

    await _wait_for_index(adapter)
    assert decision["resolved"] is False
    assert _line_messages(index_channel)[0].content.startswith("⌛ timed out")
    assert status_adapter.sent


@pytest.mark.asyncio
async def test_settled_index_line_keeps_summary_and_prompt_link(tmp_path):
    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    await _send_prompt(adapter, command="make release")
    await _wait_for_index(adapter)
    index = adapter._approval_index

    await index.resolve("approval-1", "once")

    line = _line_messages(index_channel)[0]
    assert line.content.startswith("✅ approved · once · make release · ")
    assert "https://discord.com/channels/777/555/" in line.content


@pytest.mark.asyncio
async def test_unavailable_cached_thread_is_recreated_once(tmp_path):
    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    await index._get_daily_thread()
    stale = index._thread
    original_send = stale.send
    failures = 0

    async def unavailable_once(*args, **kwargs):
        nonlocal failures
        failures += 1
        if failures == 1:
            error = RuntimeError("channel is unavailable")
            error.status = 404
            raise error
        return await original_send(*args, **kwargs)

    stale.send = unavailable_once
    await index.add_pending("after-delete", "make release", "https://example.test/prompt")

    assert failures == 1
    assert len(index_channel.threads) == 2
    assert index._thread is index_channel.threads[-1]
    assert len(_line_messages(index_channel)) == 1


@pytest.mark.asyncio
async def test_index_summary_uses_prompt_preview_before_normalizing(tmp_path):
    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")
    private_suffix = "PRIVATE-ARGUMENT"
    await _send_prompt(adapter, command=" " * 2500 + private_suffix)
    await _wait_for_index(adapter)

    assert private_suffix not in prompt_channel.sent[0][2].content
    assert private_suffix not in _line_messages(index_channel)[0].content


@pytest.mark.asyncio
async def test_index_pending_line_pings_only_allowed_users_and_settle_edit_does_not_ping(tmp_path, monkeypatch):
    from tests.gateway.conftest import _ensure_discord_mock
    _ensure_discord_mock()
    import discord

    class _AllowedMentions:
        def __init__(self, *, users=True, roles=True, everyone=True, replied_user=True):
            self.users, self.roles, self.everyone, self.replied_user = users, roles, everyone, replied_user

        @classmethod
        def none(cls):
            return cls(users=False, roles=False, everyone=False, replied_user=False)

    monkeypatch.setattr(discord, "AllowedMentions", _AllowedMentions)
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    adapter, _client, prompt_channel, index_channel = _adapter(
        tmp_path / "home", extra={"approval_mentions": True, "approval_mentions_target": "index"},
    )
    adapter._allowed_user_ids = {"222", "111", "alice"}
    adapter._allowed_role_ids = {"999"}

    result = await _send_prompt(adapter, request_id="mention-index")
    await _wait_for_index(adapter)

    assert result.success
    assert "<@111>" not in prompt_channel.sent[0][0]
    line = _line_messages(index_channel)[0]
    pending_kwargs = index_channel.threads[0].sent[-1][1]
    assert line.content.startswith("<@111> <@222> 🟡 pending")
    assert pending_kwargs["allowed_mentions"].users is True
    assert pending_kwargs["allowed_mentions"].roles is False
    assert pending_kwargs["allowed_mentions"].everyone is False
    assert pending_kwargs["allowed_mentions"].replied_user is False
    await adapter._approval_index.resolve("mention-index", "once")
    edit_kwargs = index_channel.threads[0].messages[int(line.id)].edit_kwargs[-1]
    assert edit_kwargs["allowed_mentions"].users is False
    assert edit_kwargs["allowed_mentions"].roles is False
    assert edit_kwargs["allowed_mentions"].everyone is False
    assert edit_kwargs["allowed_mentions"].replied_user is False


@pytest.mark.asyncio
async def test_truncated_approval_attaches_full_command(tmp_path, monkeypatch):
    from tests.gateway.conftest import _ensure_discord_mock
    _ensure_discord_mock()
    import discord

    import plugins.platforms.discord.adapter as discord_adapter
    captured_file = []
    monkeypatch.setattr(discord_adapter.discord, "File", lambda *args, **kwargs: (captured_file.append((args, kwargs)) or SimpleNamespace(filename=kwargs["filename"])))
    adapter, _client, prompt_channel, _index_channel = _adapter(tmp_path / "home", channel_id="")
    command = "printf 'complete command'\n" + "x" * 5000

    result = await adapter.send_exec_approval(
        chat_id="555", command=command, description="review this command", session_key="discord:555",
    )

    assert result.success
    content, kwargs, _message = prompt_channel.sent[0]
    assert "truncated" in content.lower()
    assert "command.txt" in content
    assert kwargs["files"][0].filename == "command.txt"
    args, file_kwargs = captured_file[0]
    assert file_kwargs["filename"] == "command.txt"
    assert args[0].read().decode() == command
@pytest.mark.asyncio
async def test_untruncated_approval_has_no_command_attachment(tmp_path, monkeypatch):
    from tests.gateway.conftest import _ensure_discord_mock
    _ensure_discord_mock()
    import discord

    monkeypatch.setattr(discord, "File", lambda fp, filename: SimpleNamespace(fp=fp, filename=filename))
    adapter, _client, prompt_channel, _index_channel = _adapter(tmp_path / "home", channel_id="")
    result = await adapter.send_exec_approval(
        chat_id="555", command="echo ok", description="short command", session_key="discord:555",
    )
    assert result.success
    assert "files" not in prompt_channel.sent[0][1]

@pytest.mark.asyncio
async def test_resolve_during_failed_post_clears_temporary_state(tmp_path, monkeypatch):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home")
    index = adapter._approval_index
    thread = await index._get_daily_thread()
    posting = asyncio.Event()
    fail_post = asyncio.Event()

    async def delayed_failure(*_args, **_kwargs):
        posting.set()
        await fail_post.wait()
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(thread, "send", delayed_failure)
    post = asyncio.create_task(index.add_pending("failed-fast", "make release", "https://example.test"))
    await posting.wait()
    await index.resolve("failed-fast", "once")
    fail_post.set()
    with pytest.raises(RuntimeError, match="index unavailable"):
        await post

    assert index._posting == set()
    assert index._settled == {}
    _assert_index_temporary_state_empty(index)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_path", ["interrupt", "withdrawal", "teardown"])
async def test_terminal_cancelled_paths_update_index_and_clear_pending(tmp_path, monkeypatch, terminal_path):
    from tools import approval as approval_module
    from tools import approval_gateway_wait

    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    request_id = f"withdrawn-{terminal_path}"
    await _send_prompt(adapter, request_id, command="make release")
    await asyncio.gather(*adapter._approval_index._tasks)
    runner = _TurnRunner(_StatusAdapter(adapter), "discord:555")
    if terminal_path == "interrupt":
        monkeypatch.setattr(approval_gateway_wait, "_poll_event", lambda *_a, **_kw: "interrupted")
    wait_task = await _approval_wait(
        "discord:555", {"request_id": request_id, "command": "make release"}, runner,
    )
    if terminal_path == "withdrawal":
        assert approval_module.withdraw_gateway_approval(
            "discord:555", request_id, "the session ended before the prompt was answered",
        )
    elif terminal_path == "teardown":
        approval_module.unregister_gateway_notify("discord:555")

    decision = await wait_task
    await runner.scheduled.pop(0)
    await _wait_for_index(adapter)

    assert decision["cancelled"]
    assert _line_messages(index_channel)[0].content.startswith("⛔ cancelled · make release · ")
    assert request_id not in adapter._approval_index._pending
    assert request_id not in adapter._approval_index._settled
    _assert_index_temporary_state_empty(adapter._approval_index)
    approval_module._gateway_queues.pop("discord:555", None)


@pytest.mark.asyncio
async def test_runner_to_discord_redacts_credential_from_prompt_and_index(tmp_path):
    from gateway.run_turn_runner import TurnRunner
    from tools import approval as approval_module

    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")
    ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="555", _status_thread_metadata=None,
        session_key="discord:555", source=SimpleNamespace(platform=SimpleNamespace(value="discord")),
        _loop_for_step=asyncio.get_running_loop(), stream_consumer_holder=[None],
    )
    runner = TurnRunner(SimpleNamespace(), ctx)
    loop = asyncio.get_running_loop()
    runner._schedule = lambda coro, _label: _LoopFuture(asyncio.run_coroutine_threadsafe(coro, loop))
    runner._close_native_stream_boundary = lambda _why: None
    fake_token = "ghp_" + "X" * 36
    approval = {
        "request_id": "credential-approval",
        "command": f"curl -H 'Authorization: token {fake_token}' https://example.test",
    }

    await asyncio.to_thread(runner._approval_notify_sync, approval)
    await asyncio.gather(*adapter._approval_index._tasks)

    prompt = prompt_channel.sent[0][2]
    line = _line_messages(index_channel)[0]
    assert fake_token not in prompt.content
    assert fake_token not in line.content
    assert "curl" in prompt.content and "curl" in line.content
    approval_module._gateway_queues.pop("discord:555", None)
