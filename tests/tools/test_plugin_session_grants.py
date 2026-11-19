"""Plugin keys: a session grant can expire and can refuse "Always".

A ``pre_tool_call`` approve directive may set ``session_ttl_seconds`` and ``allow_permanent``.
With neither set, the old behaviour is unchanged.
"""

import pytest

from tools import approval
from tools.approval import request_tool_approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "grant-session"
KEY = "plugin-x:cls:run"
PLUGIN_KEY = f"plugin_rule:{KEY}"


@pytest.fixture(autouse=True)
def _session(monkeypatch):
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="default": SESSION)
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (cb, True, False, False))
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval.approval_context, "is_plugin_origin_turn", lambda: False)
    approval._session_approved.pop(SESSION, None)
    approval._session_approval_expiry.pop(SESSION, None)
    yield
    approval._session_approved.pop(SESSION, None)
    approval._session_approval_expiry.pop(SESSION, None)
    with approval._lock:
        approval._permanent_set().discard(PLUGIN_KEY)


class _Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    # Expiry reads the wall clock (time.time), so a host sleep counts toward a grant's TTL.
    c = _Clock()
    monkeypatch.setattr(approval.time, "time", c)
    return c


def _prompt(monkeypatch, choice):
    """Answer the human prompt with ``choice``; record every prompt shown."""
    shown = []

    def fake_prompt(command, description, **kwargs):
        shown.append(kwargs)
        return choice

    monkeypatch.setattr(approval, "prompt_dangerous_approval", fake_prompt)
    return shown


def _reviewer(monkeypatch, verdict):
    calls = []
    monkeypatch.setattr(approval, "_smart_verdict", lambda *a, **k: calls.append(verdict) or verdict)
    return calls


def test_session_grant_with_ttl_skips_the_prompt_while_live(monkeypatch, clock):
    shown = _prompt(monkeypatch, "session")
    first = request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert first["approved"] is True
    assert len(shown) == 1

    clock.now += 3599
    second = request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert second["approved"] is True
    assert len(shown) == 1, "a live grant must not re-prompt"


def test_session_grant_expires_after_the_ttl(monkeypatch, clock):
    shown = _prompt(monkeypatch, "session")
    request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    clock.now += 3601
    again = request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert again["approved"] is True
    assert len(shown) == 2, "an expired grant must fall back to the human prompt"


def test_session_grant_expires_at_exactly_the_ttl(monkeypatch, clock):
    shown = _prompt(monkeypatch, "session")
    request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    clock.now += 3600
    request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert len(shown) == 2, "a grant is expired at exactly its TTL, not one second later"


def test_grant_without_ttl_never_expires(monkeypatch, clock):
    shown = _prompt(monkeypatch, "session")
    request_tool_approval("terminal", "run", rule_key=KEY)
    clock.now += 10 * 3600
    request_tool_approval("terminal", "run", rule_key=KEY)
    assert len(shown) == 1, "no TTL means the old, untimed behaviour"


def test_always_is_refused_when_allow_permanent_is_false(monkeypatch):
    shown = []

    def prompt(command, description, **kwargs):
        shown.append(kwargs)
        return "always"

    monkeypatch.setattr(approval, "prompt_dangerous_approval", prompt)
    result = request_tool_approval(
        "vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600, allow_permanent=False,
    )
    assert shown[0]["allow_permanent"] is False
    assert approval._is_permanently_approved(PLUGIN_KEY) is False
    assert result["approved"] is True  # the prompt's "always" downgrades to a session grant
    assert approval.is_approved(SESSION, PLUGIN_KEY) is True


def test_always_without_a_ttl_is_not_saved_when_allow_permanent_is_false(monkeypatch):
    _prompt(monkeypatch, "always")
    result = request_tool_approval("plugin-x", "run", rule_key=KEY, allow_permanent=False)
    assert result["approved"] is True
    assert approval._is_permanently_approved(PLUGIN_KEY) is False
    assert approval.is_approved(SESSION, PLUGIN_KEY) is True


def test_a_ttl_hides_always(monkeypatch):
    shown = _prompt(monkeypatch, "once")
    request_tool_approval("plugin-x", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert shown[0]["allow_permanent"] is False, "a timed grant must not offer a permanent save"


def test_persist_choice_never_saves_always_when_policy_forbids(monkeypatch):
    approval._persist_choice(SESSION, "always", [PLUGIN_KEY], allow_permanent=False)
    assert approval._is_permanently_approved(PLUGIN_KEY) is False
    assert approval.is_approved(SESSION, PLUGIN_KEY) is True


def test_typed_always_on_a_card_that_forbids_it_is_a_session_answer(monkeypatch):
    entry = _ApprovalEntry({
        "command": "run", "pattern_key": PLUGIN_KEY, "pattern_keys": [PLUGIN_KEY],
        "allow_permanent": False, "allow_session": True,
    })
    with approval._lock:
        approval._gateway_queues.setdefault(SESSION, []).append(entry)
    try:
        assert approval.resolve_gateway_approval(SESSION, "always", request_id=entry.data["request_id"]) == 1
        assert entry.result == "session"
    finally:
        with approval._lock:
            approval._gateway_queues.pop(SESSION, None)



def test_a_grant_is_not_reused_after_a_process_restart(monkeypatch, clock):
    """Grants live only in process memory. A fresh process with the same session id sees none."""
    import importlib

    _prompt(monkeypatch, "session")
    request_tool_approval("plugin-x", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert approval.is_approved(SESSION, PLUGIN_KEY) is True

    restarted = importlib.reload(approval)
    try:
        assert restarted.is_approved(SESSION, PLUGIN_KEY) is False, "a restarted process must not carry the grant"
    finally:
        importlib.reload(approval)

def test_always_is_offered_when_no_policy_is_given(monkeypatch):
    shown = _prompt(monkeypatch, "once")
    request_tool_approval("terminal", "run", rule_key=KEY)
    assert shown[0]["allow_permanent"] is True


def test_clear_session_also_drops_the_expiry(monkeypatch, clock):
    _prompt(monkeypatch, "session")
    request_tool_approval("plugin-x", "run", rule_key=KEY, session_ttl_seconds=3600)
    approval.clear_session(SESSION)
    assert SESSION not in approval._session_approval_expiry
    assert approval.is_approved(SESSION, PLUGIN_KEY) is False


def test_reviewer_deny_blocks_inside_a_live_grant(monkeypatch, clock):
    """Invariant 2: inside a live grant the reviewer runs, and a DENY blocks."""
    _prompt(monkeypatch, "session")
    request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)
    assert approval.is_approved(SESSION, PLUGIN_KEY) is True

    calls = _reviewer(monkeypatch, "deny")
    shown = _prompt(monkeypatch, "deny")
    result = request_tool_approval(
        "vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600, review=True,
    )
    assert calls == ["deny"], "the reviewer must run on every call inside a grant"
    assert result["approved"] is False, "a reviewer DENY must block even with a live grant"
    assert shown and shown[0]["smart_denied"] is True


def test_reviewer_allow_inside_grant_skips_the_prompt(monkeypatch, clock):
    """Invariant 2 (ALLOW side): the reviewer runs, and a live grant still skips the prompt."""
    _prompt(monkeypatch, "session")
    request_tool_approval("vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600)

    calls = _reviewer(monkeypatch, "approve")
    shown = _prompt(monkeypatch, "deny")
    result = request_tool_approval(
        "vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600, review=True,
    )
    assert calls == ["approve"]
    assert result["approved"] is True
    assert shown == [], "a reviewer ALLOW inside a live grant must not re-prompt"


def test_reviewer_allow_without_grant_still_asks_the_human(monkeypatch):
    """Invariant 1: with no grant, a reviewer ALLOW must not replace Matt's prompt."""
    calls = _reviewer(monkeypatch, "approve")
    shown = _prompt(monkeypatch, "once")
    result = request_tool_approval(
        "vault_secret", "run", rule_key=KEY, session_ttl_seconds=3600, review=True,
    )
    assert calls == ["approve"]
    assert len(shown) == 1, "no grant: the human prompt must still appear"
    assert result["approved"] is True
