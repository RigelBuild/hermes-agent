"""A runtime-draining refusal stays eligible for Discord missed-message backfill."""

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest_plugins = ("tests.gateway.test_discord_missed_message_backfill",)


from gateway.platforms.event import ProcessingOutcome
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.gateway import test_discord_missed_message_backfill as discord_tests
from tests.gateway.test_wake_during_restart_drain_authority import (
    Hold,
    admissions,
    authority_route,
)


async def _drain_harness(monkeypatch, tmp_path, *, hold=None):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    executed = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "42")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "42")
    runner, adapter, authority, source = await authority_route(monkeypatch, executed, hold)
    processing_outcomes = []
    on_processing_complete = adapter.on_processing_complete

    async def capture_processing_outcome(event, outcome):
        processing_outcomes.append(outcome)
        await on_processing_complete(event, outcome)

    monkeypatch.setattr(adapter, "on_processing_complete", capture_processing_outcome)
    adapter._text_batch_delay_seconds = 0
    monkeypatch.setattr(adapter, "send", DiscordAdapter.send.__get__(adapter))
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999, bot=True, display_name="Hermes", name="hermes"),
        get_channel=lambda _channel_id: None,
    )
    adapter._start_typing_refresh = lambda *args: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    channel = discord_tests.FakeChannel(channel_id=42)
    adapter._client.get_channel = lambda _channel_id: channel
    replies_sent = asyncio.Event()

    async def send(*, content, reference, **_kwargs):
        reply = SimpleNamespace(
            id=100 + len(channel._history_messages),
            content=content,
            author=adapter._client.user,
            reference=SimpleNamespace(message_id=getattr(reference, "message_id", None)),
        )
        channel._history_messages.append(reply)
        replies_sent.set()
        return reply

    channel.send = send
    monkeypatch.setattr(adapter, "_resolve_channel", AsyncMock(return_value=channel))
    monkeypatch.setattr(
        adapter,
        "_reply_reference_for_send",
        lambda reply_to, *_args: SimpleNamespace(message_id=reply_to),
    )

    def message(message_id, content):
        result = discord_tests.make_message(
            message_id=message_id, content=content, channel=channel
        )
        channel._history_messages.append(result)
        return result

    return SimpleNamespace(
        runner=runner,
        adapter=adapter,
        authority=authority,
        executed=executed,
        source=source,
        processing_outcomes=processing_outcomes,
        channel=channel,
        message=message,
        replies_sent=replies_sent,
    )


@pytest.mark.asyncio
async def test_runtime_draining_notice_leaves_discord_message_for_backfill(
    monkeypatch, caplog, tmp_path
):
    from agent.i18n import t
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    executed = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    runner, adapter, authority, _ = await authority_route(monkeypatch, executed)
    processing_outcomes = []
    on_processing_complete = adapter.on_processing_complete

    async def capture_processing_outcome(event, outcome):
        processing_outcomes.append(outcome)
        await on_processing_complete(event, outcome)

    monkeypatch.setattr(adapter, "on_processing_complete", capture_processing_outcome)
    adapter._text_batch_delay_seconds = 0
    monkeypatch.setattr(adapter, "send", DiscordAdapter.send.__get__(adapter))
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999, bot=True, display_name="Hermes", name="hermes"),
        get_channel=lambda _channel_id: None,
    )
    adapter._start_typing_refresh = lambda *args: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._record_delivery_obligation = AsyncMock(return_value=None)

    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "123")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "123")
    channel = discord_tests.FakeChannel()
    message = discord_tests.make_message(
        message_id=91, content="please answer", channel=channel
    )
    channel._history_messages.append(message)
    adapter._client.get_channel = lambda _channel_id: channel

    async def send(*, content, reference, **_kwargs):
        reply = SimpleNamespace(
            id=100 + len(channel._history_messages),
            content=content,
            author=adapter._client.user,
            reference=SimpleNamespace(message_id=message.id),
        )
        channel._history_messages.append(reply)
        return reply

    channel.send = send
    monkeypatch.setattr(adapter, "_resolve_channel", AsyncMock(return_value=channel))
    monkeypatch.setattr(
        adapter,
        "_reply_reference_for_send",
        lambda *args: SimpleNamespace(message_id=message.id),
    )
    captured_sources = []
    admit_native = authority.admit_native

    async def capture_admission_source(event):
        captured_sources.append(event.source)
        return await admit_native(event)

    monkeypatch.setattr(authority, "admit_native", capture_admission_source)
    runner._draining = True
    with caplog.at_level(logging.WARNING, logger="gateway.session_ingress"):
        assert await adapter._dispatch_discord_message(message)
        await asyncio.gather(*tuple(adapter._background_tasks))

    assert [reply.content for reply in channel._history_messages[1:]] == [
        t(
            "gateway.busy.drain_rejected_new_work",
            action=runner._status_action_gerund(),
        ),
    ]
    assert any("runtime_draining" in record.getMessage() for record in caplog.records)
    assert executed == []
    assert processing_outcomes == [ProcessingOutcome.FAILURE]
    assert processing_outcomes[0] != ProcessingOutcome.SUCCESS
    assert captured_sources and admissions(runner, authority, captured_sources[0]) == []
    assert not adapter._discord_message_is_persistently_complete(str(message.id))
    assert await adapter._should_backfill_discord_message(message)

    # Clear process-local dedup as a restart would; retain durable ledger and Discord history.
    runner._draining = False
    adapter._dedup.discard(str(message.id))
    await adapter._run_missed_message_backfill()
    await asyncio.gather(*tuple(adapter._background_tasks))

    assert executed == [message.content]
    assert adapter._discord_message_is_persistently_complete(str(message.id))
    assert processing_outcomes == [ProcessingOutcome.FAILURE, ProcessingOutcome.SUCCESS]
    assert len(channel._history_messages) == 3
    assert not await adapter._should_backfill_discord_message(message)
    authority.db.close()


@pytest.mark.asyncio
async def test_busy_drain_refusal_remains_eligible_for_discord_backfill(monkeypatch, tmp_path):
    hold = Hold()
    h = await _drain_harness(monkeypatch, tmp_path, hold=hold)
    first = h.message(201, "A")
    refused = h.message(202, "busy follow-up")
    await h.adapter._dispatch_discord_message(first)
    await asyncio.wait_for(hold.running.wait(), 10)
    h.runner._draining = True
    await h.adapter._dispatch_discord_message(refused)
    assert not h.adapter._discord_message_is_persistently_complete(str(refused.id))
    assert await h.adapter._should_backfill_discord_message(refused)
    h.runner._draining = False
    hold.finish.set()
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    await h.adapter._run_missed_message_backfill()
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    assert h.executed == ["A", "busy follow-up"]
    assert h.adapter._discord_message_is_persistently_complete(str(refused.id))
    h.authority.db.close()


@pytest.mark.asyncio
async def test_rewritten_discord_event_keeps_drain_refusal_retryable(monkeypatch, tmp_path):
    h = await _drain_harness(monkeypatch, tmp_path)
    message = h.message(203, "rewritten input")
    handler = h.adapter._message_handler

    async def rewrite(event):
        return await handler(replace(event, text="rewritten: " + event.text))

    h.adapter._message_handler = rewrite
    h.runner._draining = True
    await h.adapter._dispatch_discord_message(message)
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    assert h.processing_outcomes == [ProcessingOutcome.FAILURE]
    assert await h.adapter._should_backfill_discord_message(message)
    h.runner._draining = False
    await h.adapter._run_missed_message_backfill()
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    assert h.executed == ["rewritten: rewritten input"]
    assert h.adapter._discord_message_is_persistently_complete(str(message.id))
    h.authority.db.close()


@pytest.mark.asyncio
async def test_repeated_backfill_during_drain_preserves_message_and_attempts(monkeypatch, tmp_path):
    h = await _drain_harness(monkeypatch, tmp_path)
    message = h.message(204, "recover after drain")
    h.runner._draining = True
    await h.adapter._dispatch_discord_message(message)
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    for _ in range(4):
        await h.adapter._run_missed_message_backfill()
        await asyncio.gather(*tuple(h.adapter._background_tasks))
        assert await h.adapter._should_backfill_discord_message(message)
    attempts = h.adapter._with_discord_recovery_db(
        lambda conn: conn.execute(
            "SELECT attempts FROM discord_messages WHERE message_id=?", (str(message.id),)
        ).fetchone()[0]
    )
    assert attempts == 0
    assert len(h.channel._history_messages) == 6
    h.runner._draining = False
    await h.adapter._run_missed_message_backfill()
    await asyncio.gather(*tuple(h.adapter._background_tasks))
    assert h.executed == ["recover after drain"]
    assert h.adapter._discord_message_is_persistently_complete(str(message.id))
    assert not await h.adapter._should_backfill_discord_message(message)
    h.authority.db.close()
