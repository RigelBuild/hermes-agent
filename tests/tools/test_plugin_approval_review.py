"""Plugin approval rules: show the real request and let the smart reviewer gate first.

code block instead of ``<tool> (plugin approval rule)``) and ``review`` (run the smart reviewer
before the human: APPROVE and ESCALATE still ask; DENY asks for a one-operation owner override).
"""

import pytest

from tools import approval
import tools.approval_context as tools_approval_context
from tools.approval import request_tool_approval


@pytest.fixture(autouse=True)
def _gateway_session(monkeypatch):
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="default": "review-session")
    monkeypatch.setattr(tools_approval_context, "get_current_session_key", lambda default="default": "review-session")
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr(tools_approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    approval._denial_tally.pop("review-session", None)
    yield


def _capture_human(monkeypatch, choice="once"):
    seen = {}

    def _human(spec, **kwargs):
        seen.update(kwargs)
        return {"approved": choice != "deny", "message": None}

    monkeypatch.setattr(approval, "_human_decision", _human)
    return seen


def _verdict(monkeypatch, verdict):
    calls = []

    def _smart(command, description, pattern_key, pattern_keys, session_key):
        calls.append(command)
        return verdict

    monkeypatch.setattr(approval, "_smart_verdict", _smart)
    return calls


def test_display_replaces_the_synthetic_label(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval("terminal", "why", rule_key="k", display="sops set f.yaml '[\"k\"]'")
    assert seen["command"] == "sops set f.yaml '[\"k\"]'"


def test_without_display_the_label_is_unchanged(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval("terminal", "why", rule_key="k")
    assert seen["command"] == "<terminal> (plugin approval rule)"


def test_explicit_display_is_not_enriched_or_changed(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval(
        "browser_vault", "fill login", display="resolved title on https://example.test",
        args={"action": "fill", "handle": "vault_123"},
    )
    assert seen["command"] == "resolved title on https://example.test"



def test_reviewer_is_not_called_unless_requested(monkeypatch):
    _capture_human(monkeypatch)
    calls = _verdict(monkeypatch, "approve")
    request_tool_approval("terminal", "why", rule_key="k", display="cmd")
    assert calls == []


def test_reviewer_approve_still_asks_the_human(monkeypatch):
    seen = _capture_human(monkeypatch)
    _verdict(monkeypatch, "approve")
    result = request_tool_approval("terminal", "why", rule_key="k", display="cmd", review=True)
    assert result["approved"] is True
    assert seen["command"] == "cmd"
    assert seen["description"].startswith("why\n")
    assert "passed" in seen["description"]


def test_vault_metadata_is_in_reviewer_display_and_human_prompt(monkeypatch):

    monkeypatch.setattr("agent.vault_backends.describe_ref", lambda item, field=None: {
        "title": "Production login", "username": "owner@example.test",
        "allowed_origins": ["https://example.test"], "field_label": "API key",
    })
    seen = _capture_human(monkeypatch)
    calls = _verdict(monkeypatch, "approve")
    request_tool_approval("browser_vault", "fill login", args={"action": "fill", "handle": "vault_123"}, review=True)

    assert len(calls) == 1
    assert "Production login" in calls[0]
    assert "owner@example.test" in calls[0]
    assert "https://example.test" in calls[0]
    assert "Production login" in seen["command"]

def test_reviewer_escalate_still_asks_the_human(monkeypatch):
    seen = _capture_human(monkeypatch)
    _verdict(monkeypatch, "escalate")
    request_tool_approval("terminal", "why", rule_key="k", display="cmd", review=True)
    assert "carefully" in seen["description"]



def test_reviewer_runs_even_in_manual_mode(monkeypatch):
    _capture_human(monkeypatch)
    calls = _verdict(monkeypatch, "approve")
    request_tool_approval("terminal", "why", rule_key="k", display="cmd", review=True)
    assert calls == ["cmd"]
def test_terminal_approval_displays_full_command_without_600_character_clipping(monkeypatch, capsys):
    from tools import approval_prompt

    command = "x" * 1200
    monkeypatch.setattr(approval_prompt, "_read_choice", lambda *_args: "o")
    monkeypatch.setattr("agent.i18n.t", lambda key, **_kwargs: "Choose once or deny" if key.startswith("approval.choose") else "Once?")

    assert approval_prompt.prompt_dangerous_approval(command, "run command", timeout_seconds=1) == "once"
    assert command in capsys.readouterr().out


def test_directive_carries_display_and_review(monkeypatch):
    from hermes_cli.plugins import resolve_pre_tool_block

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **kwargs: [
            {"action": "approve", "message": "why", "rule_key": "k", "display": "cmd", "review": True},
        ],
    )
    seen = {}

    def _approve(tool_name, reason, **kwargs):
        seen.update(kwargs, tool_name=tool_name, reason=reason)
        return {"approved": True, "message": None}

    monkeypatch.setattr("tools.approval.request_tool_approval", _approve)
    assert resolve_pre_tool_block("terminal", {}) is None
    assert seen == {"tool_name": "terminal", "reason": "why", "rule_key": "k", "display": "cmd", "review": True,
                    "args": {}}


def test_directive_review_must_be_literal_true(monkeypatch):
    from hermes_cli.plugins import resolve_pre_tool_block

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "approve", "message": "why", "review": "yes", "display": "  "}],
    )
    seen = {}
    monkeypatch.setattr("tools.approval.request_tool_approval",
                        lambda tool_name, reason, **kw: seen.update(kw) or {"approved": True, "message": None})
    resolve_pre_tool_block("terminal", {})
    assert seen["review"] is False
    assert seen["display"] is None


def _verdict_with_note(monkeypatch, verdict, note):
    from tools import approval_smart

    def _smart(command, description, pattern_key, pattern_keys, session_key):
        approval_smart._last_note.text = note
        return verdict

    monkeypatch.setattr(approval, "_smart_verdict", _smart)


def test_reviewer_reason_reaches_the_human_prompt(monkeypatch):
    seen = _capture_human(monkeypatch)
    _verdict_with_note(monkeypatch, "escalate", "Runs a script by path; its contents are not shown.")
    request_tool_approval("vault_secret", "why", rule_key="k", display="cmd", review=True)
    assert seen["description"] == ("why\n**Reviewer:** unsure, check it carefully: "
                                   "Runs a script by path; its contents are not shown.")


@pytest.mark.parametrize(("choice", "approved", "outcome"), [
    ("once", True, None),
    ("deny", False, "denied"),
    ("timeout", False, "timeout"),
])
def test_reviewer_deny_reaches_human_and_final_choice_controls_result(
    monkeypatch, choice, approved, outcome,
):
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (cb, True, False, False))
    seen = {}

    def _approve(command, description, **kwargs):
        seen.update(command=command, description=description, **kwargs)
        return choice

    _verdict_with_note(monkeypatch, "deny", "Pipes the key to a remote host.")
    result = request_tool_approval(
        "vault_secret", "why", rule_key="k", display="cmd", review=True,
        approval_callback=_approve,
    )

    assert seen["command"] == "cmd"
    assert seen["description"] == (
        "why\n**Reviewer:** flagged as dangerous: Pipes the key to a remote host."
    )
    assert seen["allow_permanent"] is False
    assert seen["smart_denied"] is True
    assert result["approved"] is approved
    assert result.get("outcome") == outcome
    if choice == "deny":
        assert "**Reviewer:** flagged as dangerous: Pipes the key to a remote host." in result["message"]


def test_a_stale_reason_does_not_leak_into_the_next_prompt(monkeypatch):
    from tools import approval_smart

    approval_smart._last_note.text = "left over from an earlier call"
    seen = _capture_human(monkeypatch)
    _verdict(monkeypatch, "approve")
    request_tool_approval("terminal", "why", rule_key="k", display="cmd", review=True)
    assert "left over" not in seen["description"]


def test_without_display_the_call_and_its_arguments_are_shown(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval("browser_vault", "why", rule_key="k",
                          args={"action": "list", "url": "chrome://new-tab-page"})
    assert seen["command"] == 'browser_vault(action="list", url="chrome://new-tab-page")'


def test_rendered_call_masks_credentials_and_clips_long_values(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval("browser_vault", "why", rule_key="k",
                          args={"action": "enter_code", "code": "123456", "password": "hunter22",
                                "note": "x" * 1000, "opts": {"n": 1}})
    shown = seen["command"]
    assert "123456" not in shown and "hunter22" not in shown
    assert 'code=[REDACTED]' in shown and "(1000 chars)" in shown and 'opts={"n": 1}' in shown


def test_display_still_wins_over_the_rendered_call(monkeypatch):
    seen = _capture_human(monkeypatch)
    request_tool_approval("terminal", "why", rule_key="k", display="cmd", args={"command": "other"})
    assert seen["command"] == "cmd"


def test_the_hook_dispatch_passes_the_args_that_will_run(monkeypatch):
    from hermes_cli.plugins import resolve_pre_tool_block

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "approve", "message": "why", "rule_key": "k"}],
    )
    seen = {}
    monkeypatch.setattr("tools.approval.request_tool_approval",
                        lambda tool_name, reason, **kw: seen.update(kw) or {"approved": True, "message": None})
    resolve_pre_tool_block("browser_vault", {"action": "list", "url": "https://x.test"})
    assert seen["args"] == {"action": "list", "url": "https://x.test"}
