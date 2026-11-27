"""A chat message refused by a restart's after-turn wait is answered with the drain notice.

While ``request_restart`` holds the gateway in ``draining`` for an in-flight turn (up to
``agent.restart_after_turn_timeout``), the session authority's admission gate refuses every new
messaging input with ``runtime_draining`` (#152410). That refusal must reach the sender as the same
"Gateway is restarting" notice the in-process path gives, not escape ``_handle_message`` as an
exception the adapter turns into the generic "Something went wrong… /retry" turn error. Real
GatewayRunner, DiscordAdapter and SessionAuthority over SQLite.
"""

import pytest

from tests.gateway.test_wake_during_restart_drain_authority import admissions, authority_route


@pytest.mark.asyncio
async def test_message_during_restart_wait_gets_the_drain_notice_not_a_turn_error(monkeypatch):
    from agent.i18n import t
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from unittest.mock import AsyncMock

    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, executed)
    adapter._start_typing_refresh = lambda *args: None
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='sent')

    monkeypatch.setattr(adapter, 'send', send)
    adapter._run_processing_hook = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *args: None
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    runner._restart_requested = True
    runner._draining = True  # request_restart: refusing new turns while the in-flight one finishes

    event = MessageEvent(text='are you there?', source=source, message_id='B')
    await adapter.handle_message(event)
    session_key = adapter._event_session_key(event)
    await adapter._session_tasks[session_key]

    assert sent == [t('gateway.busy.drain_rejected_new_work', action=runner._status_action_gerund())]
    assert executed == [] and admissions(runner, authority, source) == []  # nothing half-admitted

@pytest.mark.asyncio
async def test_request_id_admission_conflict_is_propagated(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import RuntimeStoreError

    executed = []
    runner, _adapter, authority, source = await authority_route(monkeypatch, executed)

    async def conflict(_event):
        raise RuntimeStoreError('admission_conflict', field='request_id')

    monkeypatch.setattr(authority, 'admit_native', conflict)
    event = MessageEvent(text='changed resend', source=source, message_id='conflict')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await runner._handle_message(event)
    assert executed == [] and admissions(runner, authority, source) == []


@pytest.mark.asyncio
async def test_retiring_authority_uses_profile_unavailable_notice(monkeypatch):
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.parked_route_refusal import PARKED_ROUTE_NOTICE
    from unittest.mock import AsyncMock

    executed = []
    runner, adapter, authority, source = await authority_route(monkeypatch, executed)
    adapter._start_typing_refresh = lambda *args: None
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='sent')

    monkeypatch.setattr(adapter, 'send', send)
    adapter._run_processing_hook = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *args: None
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    runner._draining = False
    authority.retiring = True
    event = MessageEvent(text='please retry me', source=source, message_id='C')
    await adapter._process_message_background(event, adapter._event_session_key(event))
    assert sent == [PARKED_ROUTE_NOTICE]
    assert executed == []
    assert admissions(runner, authority, source) == []  # runtime_draining is refused before commit

    authority.retiring = False

    async def fail_with_other_error(_event):
        raise ValueError('unrelated failure')

    monkeypatch.setattr(authority, 'admit_native', fail_with_other_error)
    event = MessageEvent(text='another message', source=source, message_id='D')
    await adapter._process_message_background(event, adapter._event_session_key(event))
    from agent.i18n import t
    assert sent[-1] == t('gateway.notify.turn_error_detailed', error_type='ValueError',
                         error_detail='unrelated failure')
