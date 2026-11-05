"""Behavioral tests for the Discord daily exec-approval index."""

import asyncio
import datetime as dt
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

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

    async def create_thread(self, *, name, **kwargs):
        return await self.channel.create_thread(name=name, **kwargs)

    async def edit(self, *, content=None, **kwargs):
        self.content = content
        self.edits.append(content)


class _Channel:
    def __init__(self, client, channel_id, *, guild_id="777", channel_type=0, parent=None):
        self.client = client
        self.id = int(channel_id)
        self.guild = SimpleNamespace(id=guild_id)
        self.type = SimpleNamespace(value=channel_type)
        self.parent = parent
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


def _adapter(home, *, channel_id="900", client=None, forum=False):
    home.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["HERMES_HOME"] = str(home)
    client = client or _Client()
    prompt_channel = client.channel("555")
    index_channel = client.channel(channel_id, channel_type=15 if forum else 0) if channel_id else None
    adapter = DiscordAdapter(PlatformConfig(
        enabled=True, token="***", extra={"approval_index_channel": channel_id or ""},
    ))
    adapter._client = client
    return adapter, client, prompt_channel, index_channel


async def _send_prompt(adapter, request_id="approval-1", command="make release"):
    return await adapter.send_exec_approval(
        chat_id="555", command=command, description="release task", session_key="discord:555",
        metadata={"_approval_request_id": request_id},
    )


def _line_messages(index_channel):
    return [message for thread in index_channel.threads for message in thread.messages.values()
            if message.content.startswith(("🟡 pending", "✅ approved", "❌ denied", "⌛ timed out"))]


@pytest.mark.asyncio
async def test_approval_prompt_posts_a_line_with_the_sent_messages_jump_url(tmp_path):
    adapter, _client, prompt_channel, index_channel = _adapter(tmp_path / "home")

    result = await _send_prompt(adapter)

    assert result.success is True
    prompt = prompt_channel.sent[0][2]
    line = _line_messages(index_channel)[0]
    assert line.content == (
        f"🟡 pending · make release · https://discord.com/channels/777/{prompt_channel.id}/{prompt.id}"
    )
    assert index_channel.threads[0].name == f"Approvals {dt.date.today().isoformat()}"


@pytest.mark.asyncio
async def test_approval_index_reuses_its_daily_thread_and_persisted_thread(tmp_path):
    home = tmp_path / "home"
    adapter, client, _prompt_channel, index_channel = _adapter(home)

    first = await _send_prompt(adapter, "approval-1")
    second = await _send_prompt(adapter, "approval-2", "git push")

    assert first.success and second.success
    assert len(index_channel.threads) == 1
    state_path = home / "gateway" / "discord_approval_index.json"
    persisted = json.loads(state_path.read_text())
    date_key = dt.date.today().isoformat()
    thread_id = str(index_channel.threads[0].id)
    assert persisted == {date_key: thread_id}

    restarted, _same_client, _prompt_channel, _index_channel = _adapter(home, client=client)
    third = await _send_prompt(restarted, "approval-3", "git status")

    assert third.success is True
    assert len(index_channel.threads) == 1
    assert len(_line_messages(index_channel)) == 3


@pytest.mark.asyncio
async def test_approval_index_creates_a_forum_post_for_the_day(tmp_path):
    adapter, _client, _prompt_channel, forum = _adapter(tmp_path / "home", forum=True)

    result = await _send_prompt(adapter)

    assert result.success is True
    assert len(forum.threads) == 1
    assert forum.threads[0].name == f"Approvals {dt.date.today().isoformat()}"
    assert forum.threads[0].sent[0][0] == f"Approvals {dt.date.today().isoformat()}"
    assert _line_messages(forum)[0].content.startswith("🟡 pending · make release · ")


@pytest.mark.asyncio
async def test_approval_index_is_off_when_channel_is_unset(tmp_path):
    adapter, _client, _prompt_channel, _index_channel = _adapter(tmp_path / "home", channel_id="")

    result = await _send_prompt(adapter)

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

    assert _line_messages(index_channel)[0].content == "✅ approved · once"

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

    assert result.success is True


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


async def _approval_wait(session_key, approval_data, runner, *, poll=None):
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    from tools.approval_gateway_wait import _await_gateway_decision

    notified = threading.Event()

    def notify(data):
        register_timeout_notice(
            runner, data, command=data["command"], card_message_id=None,
        )
        from gateway.run_turn_runner_approval_settle import register_approval_index_settle
        register_approval_index_settle(runner, data, runner._ctx._status_adapter.adapter)
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
    view = prompt_channel.sent[0][1]["view"]
    view._check_auth = lambda _interaction: True
    runner = _TurnRunner(_StatusAdapter(adapter), session_key)
    approval_data = {"request_id": request_id, "command": "make release"}
    wait_task = await _approval_wait(session_key, approval_data, runner)
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1, display_name="alice"),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
        message=SimpleNamespace(embeds=[SimpleNamespace(set_footer=AsyncMock(), color=None)]),
    )

    await view._resolve(interaction, "once", discord_adapter.discord.Color.green(), "platform.discord.approval.resolved_once")
    await wait_task
    await runner.scheduled.pop(0)

    assert _line_messages(index_channel)[0].content.startswith("✅ approved")
    approval_module._gateway_queues.pop(session_key, None)


@pytest.mark.asyncio
async def test_typed_deny_edits_its_index_line(tmp_path):
    from tests.gateway.test_approve_deny_commands import _make_event, _make_runner, _make_source
    from tools import approval as approval_module

    adapter, _client, _prompt_channel, index_channel = _adapter(tmp_path / "home")
    slash_runner = _make_runner()
    session_key = slash_runner._session_key_for_source(_make_source())
    request_id = "typed-deny"
    await adapter.send_exec_approval(
        chat_id="555", command="rm build", session_key=session_key,
        metadata={"_approval_request_id": request_id},
    )
    settle_runner = _TurnRunner(_StatusAdapter(adapter), session_key)
    wait_task = await _approval_wait(
        session_key, {"request_id": request_id, "command": "rm build"}, settle_runner,
    )

    result = await slash_runner._handle_deny_command(_make_event("/deny wrong directory"))
    await wait_task
    await settle_runner.scheduled.pop(0)

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
    status_adapter = _StatusAdapter(adapter)
    runner = _TurnRunner(status_adapter, session_key)
    monkeypatch.setattr(approval_gateway_wait, "_poll_event", lambda *_a, **_kw: "timeout")
    task = await _approval_wait(
        session_key, {"request_id": request_id, "command": "make release"}, runner,
    )

    decision = await task
    await runner.scheduled.pop(0)
    await runner.scheduled.pop(0)

    assert decision["resolved"] is False
    assert _line_messages(index_channel)[0].content.startswith("⌛ timed out")
    assert status_adapter.sent
    approval_module._gateway_queues.pop(session_key, None)
