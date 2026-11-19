"""Approval cards must release their view and preserve terminal state across edits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.i18n import t
from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


async def _pending_card(monkeypatch):
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    session = "agent:other:discord:555"
    entry = _ApprovalEntry({"request_id": "request-1", "command": "ls"})
    monkeypatch.setitem(approval._gateway_queues, session, [entry])
    message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[])
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        message.embeds = [kwargs["embed"]]
        return message

    channel = SimpleNamespace(send=send, get_partial_message=lambda _id: message)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key=session,
        request_id="request-1",
    )
    return adapter, sent["view"], entry, message


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_fails", [False, True])
async def test_click_stops_view_even_if_interaction_edit_fails(monkeypatch, edit_fails):
    adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    if edit_fails:
        message.edit = AsyncMock(side_effect=RuntimeError("Discord unavailable"))
    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    # A failed card edit is logged and never re-raised: the answer is already recorded.
    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")
    assert entry.result == "once"
    assert view.is_finished()
    interaction.response.defer.assert_awaited_once()
    footer = message.embeds[0].footer
    entry.observers[0]("resolved")
    await asyncio.sleep(0)
    await asyncio.gather(*adapter._background_tasks)
    assert view.is_finished()
    if edit_fails:
        # The recorded answer must not leave a pending-looking card: it is stamped through the message.
        assert message.edit.await_count == 2
        assert message.edit.await_args.kwargs["view"] is None
        footer = message.embeds[0].footer
        assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
            "platform.discord.approval.already_resolved")
        interaction.followup.send.assert_awaited_once()
    else:
        assert message.embeds[0].footer == footer
        assert message.edit.await_count == 1

@pytest.mark.asyncio
async def test_observer_that_fires_before_a_failed_click_edit_still_sees_the_card_retired(monkeypatch):
    """The waiter usually wakes before Discord rejects the edit, so the observer runs while the click
    still owns the card and skips it; the click itself must then retire the card."""
    adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    edits = {"n": 0}

    async def edit(**_kwargs):
        edits["n"] += 1
        if edits["n"] == 1:
            entry.observers[0]("resolved")
            await asyncio.sleep(0)
            await asyncio.gather(*adapter._background_tasks)
            assert message.edit.await_count == 1  # the observer skipped: the click owned the edit
            raise RuntimeError("Discord unavailable")

    message.edit = AsyncMock(side_effect=edit)
    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()))

    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")

    assert message.edit.await_count == 2
    assert message.edit.await_args.kwargs["view"] is None
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")


@pytest.mark.asyncio
@pytest.mark.parametrize("ui_timeout_first", [True, False])
async def test_ui_timeout_ends_wait_as_cancelled_and_later_answer_resolves_nothing(monkeypatch, ui_timeout_first):
    _adapter, view, entry, message = await _pending_card(monkeypatch)
    if ui_timeout_first:
        await view.on_timeout()
        # The card can no longer be answered, so the wait ends at once as withdrawn, not as a user deny.
        assert entry.event.is_set()
        assert entry.result is None
        assert entry.cancelled
        assert view.session_key not in approval._gateway_queues or entry not in approval._gateway_queues[view.session_key]
    assert approval.resolve_gateway_approval(view.session_key, "once", request_id=view.request_id) == (
        0 if ui_timeout_first else 1)
    await view.retire("resolved")
    if not ui_timeout_first:
        await view.on_timeout()
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")
    assert view.is_finished()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,footer_key", [
    ("resolved", "platform.discord.approval.already_resolved"),
    ("timeout", "platform.discord.prompt.expired_footer"),
    ("interrupted", "platform.discord.slash_confirm.resolved_cancel"),
    ("session_closed", "platform.discord.slash_confirm.resolved_cancel"),
    ("notify_failed", "platform.discord.slash_confirm.resolved_cancel"),
])
async def test_external_settlement_retires_card_and_tracks_edit(monkeypatch, reason, footer_key):
    adapter, view, entry, message = await _pending_card(monkeypatch)
    pending_color = message.embeds[0].color
    entered, release = asyncio.Event(), asyncio.Event()

    async def edit(**_kwargs):
        entered.set()
        await release.wait()

    message.edit.side_effect = edit
    entry.observers[0](reason)
    await asyncio.wait_for(entered.wait(), 5)
    assert view.is_finished()
    assert len(adapter._background_tasks) == 1
    task = next(iter(adapter._background_tasks))
    kwargs = message.edit.await_args.kwargs
    assert kwargs["view"] is None
    embed = kwargs["embed"]
    footer = embed.footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(footer_key)
    assert embed.color != pending_color
    release.set()
    await task
    await asyncio.sleep(0)
    assert not adapter._background_tasks


@pytest.mark.asyncio
async def test_inflight_ui_expiry_cannot_overwrite_backend_answer(monkeypatch):
    _adapter, view, _entry, message = await _pending_card(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    displayed = []

    async def edit(**kwargs):
        footer = kwargs["embed"].footer
        text = footer["text"] if isinstance(footer, dict) else footer.text
        if not entered.is_set():
            entered.set()
            await release.wait()
        displayed.append(text)

    message.edit.side_effect = edit
    ui_task = asyncio.create_task(view.on_timeout())
    await asyncio.wait_for(entered.wait(), 5)
    backend_task = asyncio.create_task(view.retire("resolved"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(ui_task, backend_task)
    assert displayed == [t("platform.discord.prompt.expired_footer"),
                         t("platform.discord.approval.already_resolved")]


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_last", [False, True])
async def test_timeout_edits_cannot_restore_buttons_or_pending_embed(monkeypatch, notice_last):
    adapter, view, _entry, message = await _pending_card(monkeypatch)
    pending_color = message.embeds[0].color
    entered = {name: asyncio.Event() for name in ("notice", "retire")}
    release = {name: asyncio.Event() for name in entered}
    state = {"content": "command", "view": view, "embed": message.embeds[0]}

    async def edit(**kwargs):
        name = "notice" if "content" in kwargs else "retire"
        entered[name].set()
        await release[name].wait()
        state.update(kwargs)

    message.edit.side_effect = edit
    # The gateway's timeout notice edits the card through the generic edit_message.
    notice_task = asyncio.create_task(adapter.edit_message("555", "1234", "Timed out"))
    retire_task = asyncio.create_task(view.retire("timeout"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 5)
    first, last = ("retire", "notice") if notice_last else ("notice", "retire")
    tasks = {"notice": notice_task, "retire": retire_task}
    release[first].set()
    await tasks[first]
    release[last].set()
    await tasks[last]
    assert state["content"] == "Timed out"
    assert state["view"] is None
    assert state["embed"].color != pending_color
    assert view.is_finished()


@pytest.mark.asyncio
async def test_external_answer_then_stale_click_preserves_resolved_observer(monkeypatch):
    """A no-op stale click must not suppress the externally resolved observer."""
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(discord_adapter.discord.Color, "dark_grey", lambda: 8, raising=False)
    adapter._allowed_user_ids = {"42"}
    session = "agent:other:discord:555"
    entry = _ApprovalEntry({"request_id": "request-1", "command": "ls"})
    monkeypatch.setitem(approval._gateway_queues, session, [entry])
    message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[])
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        message.embeds = [kwargs["embed"]]
        return message

    channel = SimpleNamespace(send=send, get_partial_message=lambda _id: message)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key=session,
        request_id="request-1",
    )
    view = sent["view"]

    async def interaction_edit(**kwargs):
        message.embeds = [kwargs["embed"]]

    interaction = SimpleNamespace(
        message=message,
        user=SimpleNamespace(id=42, display_name="tester"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    assert approval.resolve_gateway_approval(session, "once", request_id="request-1") == 1
    assert entry.result == "once"
    entry.observers[0]("resolved")  # the waiter thread schedules the resolved observer
    assert not view.resolved, "the resolved observer has not run; the old button is still active"

    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")
    await asyncio.sleep(0)  # let the observer's call_soon_threadsafe callback enqueue retirement
    await asyncio.gather(*adapter._background_tasks)

    footer = message.embeds[0].footer
    footer_text = footer["text"] if isinstance(footer, dict) else footer.text
    assert footer_text == t("platform.discord.approval.already_resolved")
    # The stale click only answers its clicker; the card edit belongs to the observer, so the
    # two edits can't race and leave "command was not run" on top.
    interaction.response.defer.assert_awaited_once()
    interaction.followup.send.assert_awaited_once_with(
        t("platform.discord.approval.no_longer_pending"), ephemeral=True)
    # The observer owns the one card edit; the stale click only answers its clicker.
    message.edit.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_click_keeps_a_pending_card_live(monkeypatch):
    """A click whose lookup raises recorded nothing: the card stays clickable, only the clicker
    hears about it, and the request's observer still settles the card later."""
    adapter, view, entry, message = await _pending_card(monkeypatch)
    session = view.session_key
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    real_resolve = approval.resolve_gateway_approval
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *a, **k: 1 / 0)
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        message=message)

    await view._resolve(interaction, "once", 1, "gateway.exec_approval.action_once")

    interaction.followup.send.assert_awaited_once_with(
        t("platform.discord.approval.click_failed"), ephemeral=True)
    message.edit.assert_not_awaited()
    assert not view.resolved and not view.is_finished()
    assert approval._gateway_queues[session] == [entry]  # still pending, nothing recorded

    assert real_resolve(session, "once", request_id="request-1") == 1  # answered by text later
    entry.observers[0]("resolved")
    await asyncio.sleep(0)
    await asyncio.gather(*adapter._background_tasks)
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")
    assert view.is_finished()


class _UnknownInteraction(Exception):
    """Discord's 10062: the press was not acknowledged within its ~3s window."""
    code = 10062


@pytest.mark.asyncio
async def test_press_is_acknowledged_before_a_slow_resolve_and_card_edited_after(monkeypatch):
    # Discord drops a press not acknowledged within ~3s. A slow resolve must run after the defer, and the
    # card must then be edited through its message, not through the expired response (10062).
    _adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    window = {"now": 0.0, "acked": False}
    order = []
    real_resolve = approval.resolve_gateway_approval_card

    def open_window_or_raise():
        if not window["acked"] and window["now"] >= 3.0:
            raise _UnknownInteraction("Unknown interaction")

    async def defer(**_kwargs):
        order.append("defer")
        open_window_or_raise()
        window["acked"] = True

    async def edit_message(**_kwargs):
        order.append("edit_message")
        open_window_or_raise()

    def slow_resolve(*args, **kwargs):
        order.append("resolve")
        window["now"] += 4.0  # the resolve outruns the ack window
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(approval, "resolve_gateway_approval_card", slow_resolve)
    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(
            defer=AsyncMock(side_effect=defer), edit_message=AsyncMock(side_effect=edit_message),
            send_message=AsyncMock()),
    )

    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")

    assert order == ["defer", "resolve"]
    interaction.response.edit_message.assert_not_awaited()
    message.edit.assert_awaited_once()
    assert message.edit.await_args.kwargs["view"] is view
    assert entry.result == "once"
    assert view.is_finished()



@pytest.mark.asyncio
async def test_failed_defer_still_resolves_the_approval_and_edits_the_card(monkeypatch):
    # A press whose ack fails with 10062 must still resolve: the answer is recorded and the card is edited
    # through its message, which needs no interaction token. The ack failure is logged, not raised.
    _adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(
            defer=AsyncMock(side_effect=_UnknownInteraction("Unknown interaction")),
            send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")

    interaction.response.defer.assert_awaited_once()
    assert entry.result == "once"
    message.edit.assert_awaited_once()
    assert message.edit.await_args.kwargs["view"] is view
    assert view.is_finished()


_FENCED_TABLE = "cat <<'EOF'\n```\n| item | " + "description " * 10 + "|\n|---|---|\n" + "| a | b |\n" * 30 + "```\nEOF"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["echo " + "x" * 5000, _FENCED_TABLE * 10, ("echo '" + "`" * 7 + "' ") * 300],
                         ids=["long", "fenced-table", "backtick-runs"])
async def test_timeout_notice_fits_the_command_into_one_in_place_edit(monkeypatch, command):
    """The gateway's notice reaches the real Discord edit as one message: the command is cut to
    the room left and cannot close the card's fence (Discord would rewrite a table after it into
    a longer bullet list), so no "(1/N)" marker and no streaming-preview state are left."""
    from gateway import run_turn_runner_approval_settle as settle_mod

    adapter, _view, _entry, message = await _pending_card(monkeypatch)
    adapter.send = AsyncMock()
    ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="555", _status_thread_metadata=None)

    await settle_mod._post_timeout_notice(ctx, command, "1234", 300)

    message.edit.assert_awaited_once()
    content = message.edit.await_args.kwargs["content"]
    assert content.endswith("...\n```") and len(content) == adapter.MAX_MESSAGE_LENGTH
    assert content.count("```") == 2  # only the card's own fence
    assert "(1/" not in content and adapter._last_overflow_preview == {}
    adapter.send.assert_not_awaited()
