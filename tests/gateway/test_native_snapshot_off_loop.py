"""``prepare_native``'s legacy-retry read runs in a worker thread: a slow admissions read never stalls
the owner loop, and the read is bounded to the event's request."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_slow_admission_read_does_not_block_the_loop(tmp_path, monkeypatch):
    import hermes_state_runtime
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_envelope import prepare_native
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from plugins.platforms.a2a.adapter import A2AAdapter

    for key in ("A2A_ALLOWED_USERS", "A2A_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="off-loop")
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(chat_id="ctx-1", chat_name="a2a:peer", chat_type="dm", user_id="peer",
                                  user_name="peer")
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, message_id="m1", source=source)
    monkeypatch.setattr(runner.session_store, "lookup_by_session_key", lambda key: SimpleNamespace(session_id="s1"))
    calls = []
    loop = asyncio.get_running_loop()
    loop_ran = threading.Event()
    observed = []

    def blocking_list(db, **kwargs):
        assert db is authority.db
        calls.append(kwargs)
        # The loop sets loop_ran. It can run that callback only while this read is off the loop,
        # so the wait returns at once off the loop and times out on it.
        loop.call_soon_threadsafe(loop_ran.set)
        observed.append(loop_ran.wait(timeout=5))
        return []
    monkeypatch.setattr(hermes_state_runtime, "list_session_admissions", blocking_list)
    with native_callback(runner, event, get_hermes_home()):
        payload = await prepare_native(runner, event)
    assert payload["text"] == "hello"
    assert observed == [True], "the admissions read blocked the event loop: loop_ran was not set within 5 s"
    assert calls == [{"session_id": "s1", "pending_only": False, "request_id": "m1"}]


@pytest.mark.asyncio
async def test_event_without_message_id_skips_the_legacy_read(tmp_path, monkeypatch):
    """An event with no message id can match no legacy row, so it never reads the session's admissions."""
    import hermes_state_runtime
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_envelope import snapshot_native
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from plugins.platforms.a2a.adapter import A2AAdapter

    for key in ("A2A_ALLOWED_USERS", "A2A_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform("a2a"): adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="no-message-id")
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(chat_id="ctx-2", chat_name="a2a:peer", chat_type="dm", user_id="peer",
                                  user_name="peer")
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, message_id=None, source=source)
    monkeypatch.setattr(runner.session_store, "lookup_by_session_key", lambda key: SimpleNamespace(session_id="s1"))
    calls = []
    monkeypatch.setattr(hermes_state_runtime, "list_session_admissions", lambda db, **kwargs: calls.append(kwargs) or [])
    with native_callback(runner, event, get_hermes_home()):
        payload = snapshot_native(runner, event)
    assert payload["text"] == "hello"
    assert calls == []
