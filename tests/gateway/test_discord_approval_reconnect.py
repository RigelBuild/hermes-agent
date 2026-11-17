"""An exec-approval press made after a Discord adapter reconnect must still reach its approval.

Each ``connect()`` builds a new ``commands.Bot``, and discord.py keeps a view's buttons in the
store of the client that sent it. A press on an older client's card is dropped by discord.py
with no reply, so the agent turn never resumes. The adapter answers these presses itself.
"""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.i18n import t
from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.fixture(autouse=True)
def _stubbed_discord_builds_exec_buttons(monkeypatch):
    """``tests/gateway/conftest.py`` stubs discord.py when it is not installed. The stub's ``ui.button`` returns
    the method unchanged and its View builds no items, so ``ExecApprovalView`` would have no children and no
    custom_ids. Give it what discord.py gives a real view: a Button item per decorated button, set on the
    instance and listed in ``children``, plus ``remove_item``. A no-op when the real library is installed."""
    if hasattr(discord_platform.discord, "__file__"):
        return
    view_base = discord_platform.discord.ui.View
    base_init = view_base.__init__

    def init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        if type(self).__name__ != "ExecApprovalView":
            return
        for name in ("allow_once", "allow_session", "allow_always", "deny"):
            item = discord_platform.discord.ui.Button(label=name)
            item.callback = getattr(self, name)
            setattr(self, name, item)
            self.add_item(item)

    def remove_item(self, item):
        self.children[:] = [child for child in self.children if child is not item]

    monkeypatch.setattr(view_base, "__init__", init)
    monkeypatch.setattr(view_base, "remove_item", remove_item, raising=False)


class _FakeBot:
    def __init__(self, *, intents, allowed_mentions=None, **_):
        self.intents = intents
        self.allowed_mentions = allowed_mentions
        self.application_id = 999
        self.user = SimpleNamespace(id=999, name="Hermes")
        self._events = {}
        self.tree = SimpleNamespace(
            sync=AsyncMock(return_value=[]),
            fetch_commands=AsyncMock(return_value=[]),
            get_commands=lambda *a, **k: [],
            command=lambda *a, **k: (lambda fn: fn),
        )
        self.http = SimpleNamespace(
            upsert_global_command=AsyncMock(), edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(),
        )

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()

    async def close(self):
        return None


async def _connect(adapter, monkeypatch, created):
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status_scoped_locks.release_scoped_lock", lambda scope, identity: None)
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False, members=False, voice_states=False))

    def factory(*, command_prefix, intents, allowed_mentions=None, **_):
        bot = _FakeBot(intents=intents, allowed_mentions=allowed_mentions)
        created.append(bot)
        return bot

    monkeypatch.setattr(discord_platform.commands, "Bot", factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    assert await adapter.connect() is True


def _capture_card(bot, message_id=1234):
    """Make ``bot`` send the card into a channel and record its kwargs (view, embed, message)."""
    card = {}

    async def send(**kwargs):
        message = SimpleNamespace(id=message_id, edit=AsyncMock(), embeds=[kwargs["embed"]])
        card.update(kwargs)
        card["message"] = message
        return message

    channel = SimpleNamespace(send=AsyncMock(side_effect=send))
    bot.get_channel = lambda _chat_id: channel
    return card


def _queue_entry(monkeypatch, session, request_id):
    """Put one pending gateway approval for ``request_id`` on ``session``'s queue."""
    entry = _ApprovalEntry({"request_id": request_id, "command": "rm x"})
    entry.authority = {"requester_id": "11111"}
    monkeypatch.setitem(approval._gateway_queues, session, [entry])
    return entry


def _press(custom_id, *, message=None, user_id="11111"):
    """A component press. ``is_done()`` turns True once the interaction is answered, as discord.py does."""
    answered = {"done": False}

    def _mark_answered(*_args, **_kwargs):
        answered["done"] = True

    response = SimpleNamespace(
        send_message=AsyncMock(side_effect=_mark_answered),
        edit_message=AsyncMock(side_effect=_mark_answered),
        is_done=lambda: answered["done"])
    interaction = SimpleNamespace(
        data={"custom_id": custom_id},
        user=SimpleNamespace(id=user_id, display_name="alice"),
        response=response,
        message=message,
        channel_id=555,
        guild=None,
        channel=None,
    )
    return interaction, response


@pytest.mark.asyncio
async def test_pending_approval_press_after_reconnect_resolves_request(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _ApprovalEntry({"request_id": request_id, "command": "rm x"})
    entry.authority = {"requester_id": "11111"}
    monkeypatch.setitem(approval._gateway_queues, session, [entry])

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    # The card was sent on the first client; a reconnect replaces it.
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    assert len(created) == 2

    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(f"hermes:exec:{request_id}:once", message=card["message"])
    await on_interaction(interaction)

    assert entry.result == "once"
    assert entry.event.is_set()
    response.edit_message.assert_awaited_once()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_approve_on_live_client_after_resolve_logs_no_unmatched_warning(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.set_level(logging.INFO)

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _ApprovalEntry({"request_id": request_id, "command": "rm x"})
    entry.authority = {"requester_id": "11111"}
    monkeypatch.setitem(approval._gateway_queues, session, [entry])

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    # The live Approve resolves the entry and finishes the card's view (``_finalize_embed`` stops it).
    custom_id = f"hermes:exec:{request_id}:once"
    interaction, response = _press(custom_id, message=card["message"])
    await card["view"].press(interaction, custom_id)
    assert entry.result == "once"
    assert card["view"].is_finished()
    response.edit_message.assert_awaited_once()
    caplog.clear()

    # discord.py hands the same answered interaction to the live client's on_interaction.
    on_interaction = created[0]._events["on_interaction"]
    await on_interaction(interaction)

    response.send_message.assert_not_awaited()
    assert response.edit_message.await_count == 1
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.INFO and "approval press already answered" in r.getMessage()
               and f"custom_id={custom_id}" in r.getMessage() for r in caplog.records)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_unmatched_press_gets_ephemeral_reply_and_log(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    caplog.set_level(logging.INFO)
    on_interaction = created[0]._events["on_interaction"]

    interaction, response = _press(f"hermes:exec:{uuid.uuid4().hex}:once")
    await on_interaction(interaction)

    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.no_longer_pending")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("custom_id=hermes:exec:" in r.getMessage() for r in warnings)

    caplog.set_level(logging.DEBUG)
    caplog.clear()
    other, other_response = _press("hermes:model:abc")
    await on_interaction(other)

    other_response.send_message.assert_not_awaited()
    assert any(r.levelno == logging.DEBUG and "custom_id=hermes:model:abc" in r.getMessage()
               for r in caplog.records)
    assert not any("custom_id=hermes:model:abc" in r.getMessage()
                   and r.levelno >= logging.INFO for r in caplog.records)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_live_finished_view_unanswered_press_gets_no_longer_pending(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.set_level(logging.INFO)

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    # The card went stale (finished) on the live client, and nobody answered it.
    await card["view"].retire("resolved")
    assert card["view"].is_finished()
    caplog.clear()

    custom_id = f"hermes:exec:{request_id}:once"
    on_interaction = created[0]._events["on_interaction"]
    interaction, response = _press(custom_id, message=card["message"])
    await on_interaction(interaction)

    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.no_longer_pending")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    assert any(r.levelno == logging.WARNING and f"custom_id={custom_id}" in r.getMessage()
               for r in caplog.records)
    assert entry.result is None
    assert not entry.event.is_set()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_card_buttons_carry_stable_ids_for_kept_choices_only(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    card = _capture_card(created[0])
    full_id = uuid.uuid4().hex
    full_session = "agent:full:discord:555"
    _queue_entry(monkeypatch, full_session, full_id)
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=full_session, request_id=full_id)
    assert result.success is True
    assert {c.custom_id for c in card["view"].children} == {
        f"hermes:exec:{full_id}:{choice}" for choice in ("once", "session", "always", "deny")}

    no_always_id = uuid.uuid4().hex
    no_always_session = "agent:noalways:discord:555"
    no_always_entry = _queue_entry(monkeypatch, no_always_session, no_always_id)
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=no_always_session, request_id=no_always_id,
        allow_permanent=False)
    assert result.success is True
    view = card["view"]
    assert {c.custom_id for c in view.children} == {
        f"hermes:exec:{no_always_id}:{choice}" for choice in ("once", "session", "deny")}

    always_id = f"hermes:exec:{no_always_id}:always"
    interaction, response = _press(always_id, message=card["message"])
    assert await view.press(interaction, always_id) is False
    assert no_always_entry.result is None
    assert not no_always_entry.event.is_set()
    response.send_message.assert_not_awaited()
    response.edit_message.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orphan_press_by_unlisted_user_is_refused_and_entry_stays_pending(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    # Reconnect: the card's view is now an orphan, so the adapter answers its presses.
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    assert len(created) == 2

    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(
        f"hermes:exec:{request_id}:once", message=card["message"], user_id="99999")
    await on_interaction(interaction)

    assert entry.result is None
    assert not entry.event.is_set()
    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == discord_platform._unauthorized()
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    response.edit_message.assert_not_awaited()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orphan_press_of_removed_button_logs_warning_and_gets_no_longer_pending(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.set_level(logging.INFO)

    session = "agent:noalways:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id, allow_permanent=False)
    assert result.success is True

    # Reconnect: the card is now an orphan, and it has no "always" button.
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.clear()

    custom_id = f"hermes:exec:{request_id}:always"
    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(custom_id, message=card["message"])
    await on_interaction(interaction)

    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.no_longer_pending")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    response.edit_message.assert_not_awaited()
    assert entry.result is None
    assert not entry.event.is_set()
    assert any(r.levelno == logging.WARNING and "orphaned approval press matched no live button" in r.getMessage()
               and f"custom_id={custom_id}" in r.getMessage() for r in caplog.records)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orphan_press_that_raises_gets_click_failed_and_warning(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.set_level(logging.INFO)

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.clear()

    async def boom(_interaction, _custom_id):
        raise RuntimeError("press failed")

    monkeypatch.setattr(card["view"], "press", boom)
    custom_id = f"hermes:exec:{request_id}:once"
    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(custom_id, message=card["message"])
    await on_interaction(interaction)

    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.click_failed")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    response.edit_message.assert_not_awaited()
    assert entry.result is None
    assert any(r.levelno == logging.WARNING and "orphaned approval press failed" in r.getMessage()
               and f"custom_id={custom_id}" in r.getMessage() for r in caplog.records)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orphan_press_whose_edit_fails_after_recording_answer_gets_already_resolved(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True

    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    # The real view.press runs. The answer is recorded, then the card edit fails without answering the
    # interaction, so the adapter's orphan path must still reply.
    custom_id = f"hermes:exec:{request_id}:once"
    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(custom_id, message=card["message"])
    response.edit_message = AsyncMock(side_effect=RuntimeError("edit failed after answer"))
    await on_interaction(interaction)

    assert entry.result == "once"
    response.edit_message.assert_awaited_once()
    card["message"].edit.assert_awaited()  # retire() stamped the card through its message
    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.already_resolved")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_card_sent_across_reconnect_is_resolved_by_the_new_client(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)
    card = {}

    async def send(**kwargs):
        # The reconnect lands while the card is still in flight on the old client.
        await _connect(adapter, monkeypatch, created)
        adapter._allowed_user_ids = {"11111"}
        message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[kwargs["embed"]])
        card.update(kwargs)
        card["message"] = message
        return message

    channel = SimpleNamespace(send=AsyncMock(side_effect=send))
    created[0].get_channel = lambda _chat_id: channel
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True
    assert len(created) == 2

    custom_id = f"hermes:exec:{request_id}:once"
    interaction, response = _press(custom_id, message=card["message"])
    await created[1]._events["on_interaction"](interaction)

    assert entry.result == "once"
    assert entry.event.is_set()
    response.edit_message.assert_awaited_once()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_orphan_press_naming_another_cards_message_is_refused(monkeypatch, caplog):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.set_level(logging.INFO)

    session_a = "agent:a:discord:555"
    session_b = "agent:b:discord:555"
    request_a = uuid.uuid4().hex
    request_b = uuid.uuid4().hex
    entry_a = _queue_entry(monkeypatch, session_a, request_a)
    _queue_entry(monkeypatch, session_b, request_b)

    card_a = _capture_card(created[0], message_id=1111)
    assert (await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session_a, request_id=request_a)).success is True
    card_b = _capture_card(created[0], message_id=2222)
    assert (await adapter.send_exec_approval(
        chat_id="555", command="rm y", session_key=session_b, request_id=request_b)).success is True

    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}
    caplog.clear()

    # Card A's button, pressed on card B's message.
    custom_id = f"hermes:exec:{request_a}:once"
    on_interaction = created[1]._events["on_interaction"]
    interaction, response = _press(custom_id, message=card_b["message"])
    await on_interaction(interaction)

    assert entry_a.result is None
    assert not entry_a.event.is_set()
    response.send_message.assert_awaited_once()
    assert response.send_message.await_args.args[0] == t("platform.discord.approval.no_longer_pending")
    assert response.send_message.await_args.kwargs.get("ephemeral") is True
    response.edit_message.assert_not_awaited()
    assert any(r.levelno == logging.WARNING and "component press matched no pending approval" in r.getMessage()
               and f"custom_id={custom_id}" in r.getMessage() for r in caplog.records)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_live_unfinished_view_press_is_left_to_discord(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    entry = _queue_entry(monkeypatch, session, request_id)

    card = _capture_card(created[0])
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)
    assert result.success is True
    assert not card["view"].is_finished()

    # The live client's store routes this press to the view, so the listener must not answer it.
    custom_id = f"hermes:exec:{request_id}:once"
    interaction, response = _press(custom_id, message=card["message"])
    await created[0]._events["on_interaction"](interaction)

    response.send_message.assert_not_awaited()
    response.edit_message.assert_not_awaited()
    assert entry.result is None
    assert not entry.event.is_set()
    await adapter.disconnect()


class _StubApprovalView:
    def __init__(self):
        self.done = False

    def is_finished(self):
        return self.done


def test_exec_approval_view_registry_prunes_finished_then_oldest_past_cap():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    cap = adapter._EXEC_APPROVAL_VIEW_CAP
    views: dict[str, _StubApprovalView] = {}

    def register(rid):
        views[rid] = _StubApprovalView()
        adapter._register_exec_approval_view(rid, views[rid])
        assert len(adapter._exec_approval_views) <= cap

    for i in range(cap):
        register(f"s{i}")
    settled = [f"s{i}" for i in range(10, 20)]
    for rid in settled:
        views[rid].done = True

    # Over the cap with settled views present: the settled ones are dropped first, so no unfinished
    # view is evicted yet.
    register("n0")
    assert not set(settled) & set(adapter._exec_approval_views)
    assert "s0" in adapter._exec_approval_views
    assert len(adapter._exec_approval_views) == cap - len(settled) + 1

    # With no settled views left, the oldest unfinished views go first.
    for i in range(1, 40):
        register(f"n{i}")
    assert len(adapter._exec_approval_views) == cap
    assert "s0" not in adapter._exec_approval_views
    assert "n39" in adapter._exec_approval_views


@pytest.mark.asyncio
async def test_failed_card_send_is_not_registered(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    created: list[_FakeBot] = []
    await _connect(adapter, monkeypatch, created)
    adapter._allowed_user_ids = {"11111"}

    session = "agent:other:discord:555"
    request_id = uuid.uuid4().hex
    _queue_entry(monkeypatch, session, request_id)

    channel = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("send failed")))
    created[0].get_channel = lambda _chat_id: channel
    result = await adapter.send_exec_approval(
        chat_id="555", command="rm x", session_key=session, request_id=request_id)

    assert result.success is False
    assert request_id not in adapter._exec_approval_views
    await adapter.disconnect()
