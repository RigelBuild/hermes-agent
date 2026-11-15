"""Discord forum opener admission uses one adapter-minted capability."""

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.discord_self_start import record_created_thread
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_ingress_context import native_callback, register_transport_home
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter

BOT = 123456789012345678
THREAD = 123456789012345679
PARENT = 123456789012345680


def message():
    channel = Mock(spec=discord.Thread)
    channel.id = THREAD
    channel.parent_id = PARENT
    bot = SimpleNamespace(id=BOT, bot=True, name="bot", display_name="bot")
    return SimpleNamespace(
        id=THREAD,
        author=bot,
        channel=channel,
        content="start this task",
        mentions=[],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
        reference=None,
        created_at=datetime.now(UTC),
    )


def adapter():
    result = DiscordAdapter(
        PlatformConfig(
            enabled=True, token="fake", extra={"self_start_forums": [str(PARENT)]}
        )
    )
    result._client = SimpleNamespace(user=message().author)
    result._ready_event.set()
    result._text_batch_delay_seconds = 0
    result._discord_history_backfill = lambda: False
    return result


@pytest.mark.asyncio
async def test_opener_commits_native_admission_and_unrecorded_is_denied(
    tmp_path, monkeypatch
):
    from gateway import discord_self_start

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(
        discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json"
    )
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    discord_adapter = adapter()
    discord_adapter.gateway_runner, runner.adapters = (
        runner,
        {Platform.DISCORD: discord_adapter},
    )
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="self-start"
    )
    authority._schedule = lambda _ref: None
    runner._wire_adapter_handlers(discord_adapter)
    register_transport_home(runner, None, tmp_path)
    receipts = []

    async def handler(event):
        with native_callback(runner, event, tmp_path):
            receipts.append(await authority.admit_native(event))

    discord_adapter.handle_message = handler
    try:
        record_created_thread(str(THREAD), str(PARENT))
        assert await discord_adapter._dispatch_discord_message(message())
        (receipt,) = receipts
        (row,) = list_session_admissions(
            authority.db, session_id=receipt.ref.session_id, pending_only=False
        )
        assert row["payload"]["text"] == "start this task"
        assert row["payload"]["native_text_v1"]["provenance"]["self_start"]
        from gateway.session_envelope import check_native_route

        restored, _ = await check_native_route(
            runner,
            row["payload"],
            receipt.ref.session_id,
            discord_adapter.build_source(
                chat_id=str(THREAD),
                thread_id=str(THREAD),
                parent_chat_id=str(PARENT),
                chat_type="thread",
                user_id="42",
                message_id="later",
            ),
            discord_adapter,
        )
        assert restored.self_start_token
        source = discord_adapter.build_source(
            chat_id=str(THREAD),
            thread_id=str(THREAD),
            parent_chat_id=str(PARENT),
            chat_type="thread",
            user_id=str(BOT),
            message_id=str(THREAD),
            is_bot=True,
        )
        unrecorded = MessageEvent(
            text="unrecorded", source=source, message_id=str(THREAD)
        )
        with (
            native_callback(runner, unrecorded, tmp_path),
            pytest.raises(RuntimeStoreError),
        ):
            await authority.admit_native(unrecorded)
        source.self_start_token = "forged"
        event = MessageEvent(text="forged", source=source, message_id=str(THREAD))
        with native_callback(runner, event, tmp_path), pytest.raises(RuntimeStoreError):
            await authority.admit_native(event)
        assert not await discord_adapter._dispatch_discord_message(message())
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_delayed_write_and_five_second_miss(tmp_path, monkeypatch):
    from gateway import discord_self_start
    from plugins.platforms.discord import adapter as module

    monkeypatch.setattr(
        discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json"
    )
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        if sum(sleeps) == 1:
            record_created_thread(str(THREAD), str(PARENT))

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    instance = adapter()
    assert await instance._admit_gateway_created_opener(message())
    assert sleeps == [0.5, 0.5]
    sleeps.clear()

    async def miss(delay):
        sleeps.append(delay)

    monkeypatch.setattr(module.asyncio, "sleep", miss)
    assert await instance._admit_gateway_created_opener(message()) is None
    assert sleeps == [0.5] * 10


@pytest.mark.asyncio
async def test_wrong_channel_later_bot_and_expired_are_dropped(tmp_path, monkeypatch):
    from gateway import discord_self_start

    monkeypatch.setattr(
        discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json"
    )
    instance = adapter()
    original = message()
    record_created_thread(str(THREAD), str(PARENT))
    later = message()
    later.id += 1
    assert await instance._admit_gateway_created_opener(later) is None
    wrong = message()
    wrong.channel.parent_id += 1
    assert await instance._admit_gateway_created_opener(wrong) is None
    assert await instance._admit_gateway_created_opener(original)
    record_created_thread(str(THREAD), str(PARENT), created_at=time.time() - 700)
    monkeypatch.setattr("plugins.platforms.discord.adapter.asyncio.sleep", AsyncMock())
    assert await instance._admit_gateway_created_opener(original) is None
