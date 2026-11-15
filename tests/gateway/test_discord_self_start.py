"""A Discord bot's own forum opener may start one opt-in session (RIG-10198)."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import Mock

import discord

from gateway.platforms.helpers import MessageDeduplicator
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
THREAD_ID = 700
PARENT_ID = 42


def _thread():
    channel = Mock(spec=discord.Thread)
    channel.id = THREAD_ID
    channel.parent_id = PARENT_ID
    return channel


def _message(channel=None):
    return SimpleNamespace(
        id=THREAD_ID,
        author=SimpleNamespace(id=BOT_ID, bot=True),
        channel=channel or _thread(),
        content="start this task",
        mentions=[],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
    )


def _adapter(forums=(str(PARENT_ID),)):
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra={"self_start_forums": list(forums)})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._get_parent_channel_id = Mock(return_value=str(PARENT_ID))
    adapter._discord_free_response_channels = Mock(return_value=set())
    adapter._discord_channel_keys = Mock(return_value={str(THREAD_ID)})
    adapter._discord_thread_require_mention = Mock(return_value=False)
    adapter._threads = set()
    return adapter


def test_configured_gateway_created_opener_is_admitted_once(tmp_path, monkeypatch):
    from gateway.discord_self_start import record_created_thread

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_created_thread(str(THREAD_ID), str(PARENT_ID), home=tmp_path)
    adapter = _adapter()

    assert adapter._discord_message_admission(_message(), claim=False) == (True, True)
    assert adapter._discord_message_admission(_message(), claim=False) == (False, False)


def test_registered_opener_reaches_gateway_handler(tmp_path, monkeypatch):
    import asyncio
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock

    from gateway import discord_self_start

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json")
    discord_self_start.record_created_thread(str(THREAD_ID), str(PARENT_ID))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake", extra={"self_start_forums": [str(PARENT_ID)]}))
    bot = SimpleNamespace(id=BOT_ID, bot=True, name="bot", display_name="bot")
    message = _message()
    message.author = bot
    message.attachments = []
    message.reference = None
    message.created_at = datetime.now(UTC)
    adapter._client = SimpleNamespace(user=bot)
    adapter._ready_event.set()
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter.handle_message = AsyncMock()
    assert asyncio.run(adapter._dispatch_discord_message(message)) is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "start this task"
    assert event.source.role_authorized is True
    assert event.source.thread_id == str(THREAD_ID)


def test_later_bot_message_cannot_consume_starter_grant(tmp_path, monkeypatch):
    from gateway import discord_self_start

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json")
    discord_self_start.record_created_thread(str(THREAD_ID), str(PARENT_ID))
    adapter = _adapter()
    later = _message()
    later.id = THREAD_ID + 1
    assert adapter._discord_message_admission(later, claim=False) == (False, False)
    assert adapter._discord_message_admission(_message(), claim=False) == (True, True)


def test_registry_written_after_event_is_admitted_once(tmp_path, monkeypatch):
    import asyncio

    from gateway import discord_self_start
    from plugins.platforms.discord import adapter as discord_adapter

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json")
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        if sum(sleeps) == 1:
            discord_self_start.record_created_thread(str(THREAD_ID), str(PARENT_ID))

    monkeypatch.setattr(discord_adapter.asyncio, "sleep", sleep)
    adapter = _adapter()
    message = _message()
    assert adapter._discord_message_admission(message, claim=False) == (False, False)
    assert asyncio.run(adapter._await_gateway_created_opener(message)) == (True, True)
    assert sleeps == [0.5, 0.5]
    assert adapter._discord_message_admission(message, claim=False) == (False, False)


def test_registry_miss_times_out_after_five_seconds(tmp_path, monkeypatch):
    import asyncio

    from gateway import discord_self_start
    from plugins.platforms.discord import adapter as discord_adapter

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json")
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(discord_adapter.asyncio, "sleep", sleep)
    assert asyncio.run(_adapter()._await_gateway_created_opener(_message())) == (False, False)
    assert sleeps == [0.5] * 10


def test_unconfigured_parent_does_not_admit_created_opener(tmp_path, monkeypatch):
    from gateway.discord_self_start import record_created_thread

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_created_thread(str(THREAD_ID), str(PARENT_ID), home=tmp_path)
    adapter = _adapter(forums=("999",))

    assert adapter._discord_message_admission(_message(), claim=False) == (False, False)


def test_unrecorded_thread_is_not_admitted(tmp_path, monkeypatch):
    from gateway import discord_self_start

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "state" / "registry.json")
    adapter = _adapter()
    assert adapter._discord_message_admission(_message(), claim=False) == (False, False)


def test_unrecorded_opener_has_no_gateway_auth_grant(tmp_path, monkeypatch):
    import asyncio

    from gateway import discord_self_start
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import Platform, SessionSource

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "state" / "registry.json")
    adapter = _adapter()
    admitted, role_authorized = adapter._discord_message_admission(_message(), claim=False)
    assert (admitted, role_authorized) == (False, False)

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_pre_gateway_dispatch_hook = lambda event, _source: asyncio.sleep(0, result=event)
    runner._is_user_authorized_for_source = lambda source, **_kwargs: source.role_authorized is True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._admit_bot_chain = lambda _event: True
    source = SessionSource(
        platform=Platform.DISCORD, chat_id=str(THREAD_ID), chat_type="thread", user_id=str(BOT_ID),
        thread_id=str(THREAD_ID), parent_chat_id=str(PARENT_ID), is_bot=True,
        role_authorized=role_authorized,
    )
    event = MessageEvent(text="start this task", message_id="800", source=source)
    assert asyncio.run(runner._hm_admit_event(event)) is None


def test_expired_created_thread_is_not_admitted(tmp_path, monkeypatch):
    from gateway import discord_self_start
    from gateway.discord_self_start import record_created_thread

    monkeypatch.setattr(discord_self_start, "_path", lambda _home=None: tmp_path / "state" / "registry.json")
    record_created_thread(str(THREAD_ID), str(PARENT_ID), created_at=time.time() - 700)
    adapter = _adapter()
    assert adapter._discord_message_admission(_message(), claim=False) == (False, False)


def test_opener_and_later_human_reply_share_thread_session_key():
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    opener = SessionSource(
        platform=Platform.DISCORD, chat_id=str(THREAD_ID), chat_type="thread", user_id=str(BOT_ID),
        thread_id=str(THREAD_ID), parent_chat_id=str(PARENT_ID), is_bot=True,
    )
    reply = SessionSource(
        platform=Platform.DISCORD, chat_id=str(THREAD_ID), chat_type="thread", user_id="123",
        thread_id=str(THREAD_ID), parent_chat_id=str(PARENT_ID),
    )

    assert build_session_key(opener) == build_session_key(reply)
