from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
from gateway.session_pending_controls import PendingControls
from hermes_cli.gateway_agent import GatewayAgent
from tools import terminal_tool


def test_reviewer_denial_reaches_cli_callback(monkeypatch):
    events = SimpleNamespace(lock=threading.RLock(), publish=Mock())
    controls = PendingControls(events)
    controls.register("session", "route", 1, {"request_id": "r", "smart_denied": True})
    payload = events.publish.call_args.args[1]
    callback = Mock(return_value="deny")
    monkeypatch.setattr(terminal_tool, "_get_approval_callback", lambda: callback)
    agent = SimpleNamespace(connection=SimpleNamespace(call=Mock()), session_id="session", _generation=1)
    GatewayAgent._on_approval(agent, payload, None, None)
    assert callback.call_args.kwargs["smart_denied"] is True


def test_callback_type_error_does_not_retry_without_deny_warning(monkeypatch):
    calls = []

    def callback(command, description, **kwargs):
        calls.append(kwargs)
        raise TypeError("callback bug")

    monkeypatch.setattr(terminal_tool, "_get_approval_callback", lambda: callback)
    agent = SimpleNamespace(connection=SimpleNamespace(call=Mock()), session_id="session", _generation=1)
    with pytest.raises(TypeError, match="callback bug"):
        GatewayAgent._on_approval(agent, {"choices": ["once", "deny"], "smart_denied": True}, None, None)
    assert len(calls) == 1
    assert calls[0]["smart_denied"] is True
