"""Watch wakes in a bot-created Discord thread admit, and a final refusal is not requeued forever.

The session store keeps Discord auto-thread naming state (``auto_thread_created``,
``auto_thread_initial_name``) on the route origin, which a committed human envelope never carries.
``snapshot_automation`` compared the two, so every watch wake in such a thread was refused with
``admission_conflict`` and the drain requeued it every ~2 s across restarts.
"""
import logging
import queue

import pytest

from gateway.session_identity import replace_source
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_automation_notification_category import _native_owner


def _watch_event(metadata, event_id):
    return {'type': 'watch_match', 'session_id': 'proc_x', 'event_id': event_id,
            'session_key': metadata['gateway_session_key'], 'platform': 'discord', 'chat_id': '42',
            'chat_type': 'dm', 'pattern': 'ready', 'output': 'ready',
            'parent_session_id': metadata['gateway_session_id']}


@pytest.mark.asyncio
async def test_watch_wake_admits_in_a_bot_created_thread(tmp_path, monkeypatch):
    runner, authority, _adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        route = metadata['gateway_session_key']
        runner.session_store._entries[route].origin = replace_source(
            source, auto_thread_created=True, auto_thread_initial_name='first message ...')
        assert await runner._inject_watch_notification('[ready]', _watch_event(metadata, 'e1')) is True
        owner = authority.logical_owner(metadata['gateway_session_id'])
        assert [r for r in list_session_admissions(authority.db, session_id=owner)
                if (r['payload'].get('native_text_v1') or {}).get('automation')]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_drain_drops_a_finally_refused_watch_event(tmp_path, monkeypatch, caplog):
    runner, authority, _adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        route = metadata['gateway_session_key']
        # A wake whose source genuinely differs from the committed route is refused on every retry.
        runner.session_store._entries[route].origin = replace_source(source, user_name='renamed')
        pending = queue.Queue()
        pending.put(_watch_event(metadata, 'e1'))
        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)
        assert pending.empty()
        assert [r.getMessage() for r in caplog.records if 'Watch notification' in r.getMessage()] == [
            'Watch notification refused, dropping it for discord chat=42: admission_conflict']
    finally:
        authority.db.close()


@pytest.mark.parametrize('reason, requeued', [
    ('invalid_params', False), ('permission_denied', False), ('profile_mismatch', False),
    ('runtime_draining', True), ('queue_full', True), ('not_found', True),
])
@pytest.mark.asyncio
async def test_drain_requeues_only_owner_state_refusals(tmp_path, monkeypatch, reason, requeued):
    from gateway import session_automation
    from hermes_state_runtime import RuntimeStoreError
    runner, authority, _adapter, _source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')

    def refuse(*args, **kwargs):
        raise RuntimeStoreError(reason)
    monkeypatch.setattr(session_automation, 'snapshot_automation', refuse)
    try:
        pending = queue.Queue()
        pending.put(_watch_event(metadata, 'e1'))
        await runner._drain_watch_notifications(pending)
        assert pending.empty() is not requeued
    finally:
        authority.db.close()
