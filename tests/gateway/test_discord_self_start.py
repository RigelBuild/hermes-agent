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

    async def handler(event):
        with native_callback(runner, event, tmp_path):
            await runner._handle_message(event)

    discord_adapter.handle_message = handler
    try:
        record_created_thread(str(THREAD), str(PARENT))
        task = asyncio.create_task(discord_adapter._dispatch_discord_message(message()))
        from tests.waiting import async_wait_until

        await async_wait_until(lambda: bool(authority.sessions))
        sid = next(iter(authority.sessions))
        await async_wait_until(
            lambda: bool(
                list_session_admissions(
                    authority.db, session_id=sid, pending_only=False
                )
            )
        )
        (row,) = list_session_admissions(
            authority.db, session_id=sid, pending_only=False
        )
        assert row["payload"]["text"] == "start this task"
        assert row["payload"]["native_text_v1"]["provenance"]["self_start"]
        from gateway.run_turn_runner import TurnRunner
        from gateway.session_contract import SessionRef

        turns = []

        def run_sync(turn):
            turns.append(turn._ctx.message)
            result = {"final_response": "done", "messages": [], "completed": True}
            turn._ctx.result_holder[0] = result
            return result

        monkeypatch.setattr(TurnRunner, "run_sync", run_sync)
        ref = SessionRef(authority.profile_id, sid)
        live = authority.sessions[sid]
        claimed, _ = await authority._claim_with_retry(ref, live)
        assert claimed and claimed["admission_id"] == row["admission_id"]
        await authority._execute_claimed(ref, live, claimed)
        assert turns == ["[bot] start this task"]
        from gateway.session_envelope import check_native_route

        restored, _ = await check_native_route(
            runner,
            row["payload"],
            sid,
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
        assert restored.self_start_token is None
        assert restored._self_start_marker == {
            "thread": str(THREAD),
            "message": str(THREAD),
        }

        from gateway.session_envelope import restore_native
        from hermes_state_runtime import get_session_admission
        from plugins.platforms.discord import adapter as discord_module

        clock = time.monotonic()
        monkeypatch.setattr(discord_module.time, "monotonic", lambda: clock + 700)
        wake = MessageEvent(
            text="delegate completed",
            source=restored,
            internal=True,
            metadata={
                "gateway_session_key": runner.session_store._generate_session_key(
                    restored
                ),
                "gateway_session_id": sid,
            },
        )
        wake_receipt = await authority.admit_automation(
            discord_adapter, wake, "completion-1"
        )
        wake_row = get_session_admission(
            authority.db, admission_id=wake_receipt.admission_id
        )
        assert restore_native(wake_row["payload"], runner).text == "delegate completed"
        replacement = adapter()
        replacement.gateway_runner = runner
        runner.adapters = {Platform.DISCORD: replacement}
        bindings = [(sid, restored, replacement)]
        assert (await authority.recover_native_sessions(bindings))[sid] == "ready"
        forged_payload = __import__("copy").deepcopy(row["payload"])
        forged_payload["native_text_v1"]["source"]["user_id"] = "42"
        with pytest.raises(RuntimeStoreError):
            restore_native(forged_payload, runner)
        wrong_marker = __import__("copy").deepcopy(row["payload"])
        wrong_marker["native_text_v1"]["provenance"]["self_start"]["thread"] = "999"
        with pytest.raises(RuntimeStoreError):
            restore_native(wrong_marker, runner)
        wrong_thread = __import__("copy").deepcopy(row["payload"])
        wrong_thread["native_text_v1"]["source"]["thread_id"] = "999"
        wrong_thread["native_text_v1"]["source"]["chat_id"] = "999"
        with pytest.raises(RuntimeStoreError):
            restore_native(wrong_thread, runner)
        replacement.config.extra["self_start_forums"] = []
        with pytest.raises(RuntimeStoreError):
            restore_native(row["payload"], runner)
        runner.adapters = {Platform.DISCORD: discord_adapter}
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
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
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


@pytest.mark.parametrize("failure", ["human", "message", "parent", "expired"])
def test_nonce_redeem_checks_source_and_expiry(failure):
    instance = adapter()
    source = instance.build_source(
        chat_id=str(THREAD),
        thread_id=str(THREAD),
        parent_chat_id=str(PARENT),
        chat_type="thread",
        user_id=str(BOT),
        message_id=str(THREAD),
    )
    source.self_start_token = "nonce"
    instance._self_start_tokens[(str(THREAD), "nonce")] = time.monotonic() + 60
    if failure == "human":
        source.user_id = "42"
    elif failure == "message":
        source.message_id = "other"
    elif failure == "parent":
        source.parent_chat_id = "other"
    else:
        instance._self_start_tokens[(str(THREAD), "nonce")] = time.monotonic() - 1
    assert not instance.redeem_self_start(source)


def test_nonce_redeems_once():
    instance = adapter()
    source = instance.build_source(
        chat_id=str(THREAD),
        thread_id=str(THREAD),
        parent_chat_id=str(PARENT),
        chat_type="thread",
        user_id=str(BOT),
        message_id=str(THREAD),
    )
    source.self_start_token = "nonce"
    instance._self_start_tokens[(str(THREAD), "nonce")] = time.monotonic() + 60
    assert instance.redeem_self_start(source)
    assert not instance.redeem_self_start(source)


@pytest.mark.asyncio
async def test_self_authored_message_starts_no_session_or_reply(tmp_path, monkeypatch):
    from gateway import discord_self_start

    monkeypatch.setattr(
        discord_self_start, "_path", lambda _home=None: tmp_path / "registry.json"
    )
    instance = adapter()
    bot = instance._client.user
    channel = Mock(spec=discord.TextChannel)
    channel.id = PARENT
    own = message()
    own.id = THREAD
    own.author = bot
    own.channel = channel
    handle = AsyncMock()
    send = AsyncMock()
    instance.handle_message = handle
    instance.send = send

    assert instance._discord_message_admission(own, claim=False) == (False, False)
    assert not await instance._dispatch_discord_message(own)
    handle.assert_not_awaited()
    send.assert_not_awaited()
