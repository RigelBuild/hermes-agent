"""A Cotal turn's chat is the peer agent that started it, and a peer cannot approve: every
approval raised on the cotal platform must fail closed and never reach the turn's notifier."""

import pytest

import tools.approval as approval_mod
from tools import approval_context
from tools.approval_prompt import request_elicitation_consent
from tools.file_tools_write_guards import _request_protected_instruction_approval

SESSION = "agent:main:cotal:dm:mesh"


@pytest.fixture(params=[(mode, interactive) for mode in ("deny", "approve") for interactive in (False, True)],
                ids=lambda p: f"{p[0]}-{'interactive' if p[1] else 'daemon'}")
def cotal_turn(monkeypatch, request):
    unattended_mode, interactive = request.param
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    for var in ("HERMES_CRON_SESSION", "HERMES_INTERACTIVE", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(var, raising=False)
    # What a gateway-run Cotal turn looks like: ask mode, a bound platform and session key.
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "cotal")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION)
    if interactive:
        # The legacy `cli.py --gateway` entry exports HERMES_INTERACTIVE before serving turns.
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    def no_cli_prompt(*a, **k):
        pytest.fail("Cotal approval reached the local CLI prompt")
    monkeypatch.setattr(approval_mod, "prompt_dangerous_approval", no_cli_prompt)
    monkeypatch.setattr("tools.approval_prompt.prompt_dangerous_approval", no_cli_prompt)
    # An operator who trusts webhook/API automation must not thereby trust mesh peers.
    real_binary = approval_context._binary_approval_mode
    monkeypatch.setattr(approval_context, "_binary_approval_mode",
                        lambda key: unattended_mode if key == "unattended_mode" else real_binary(key))
    # Before the fix the card parks on an unanswered gateway wait; keep that red fast, not 20 min.
    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 1)
    approval_mod._session_approved.pop(SESSION, None)
    sent = []
    approval_mod.register_gateway_notify(SESSION, sent.append)
    yield sent
    approval_mod.unregister_gateway_notify(SESSION)


def test_terminal_approval_is_blocked_without_prompting_the_peer(cotal_turn):
    result = approval_mod.check_all_command_guards("sudo systemctl restart nginx", "local")
    assert result["approved"] is False
    # No config value allows it, so the block must not suggest one.
    assert "peer cannot approve" in result["message"] and "unattended_mode" not in result["message"]
    assert cotal_turn == []


def test_execute_code_approval_is_blocked_without_prompting_the_peer(cotal_turn):
    result = approval_mod.check_execute_code_guard("import os; os.remove('x')", "local")
    assert result["approved"] is False and "unattended_mode" not in result["message"]
    assert cotal_turn == []


def test_plugin_requested_approval_is_blocked_without_prompting_the_peer(cotal_turn):
    # The nemo-secret-guard plugin escalates secret reads through this gate.
    result = approval_mod.request_tool_approval("terminal", "secret read", display="op read op://x")
    assert result["approved"] is False
    assert cotal_turn == []


def test_browser_vault_consent_declines_without_prompting_the_peer(cotal_turn):
    assert request_elicitation_consent("Fill payment card 'visa' on https://pay.example", "card fill") == "decline"
    assert cotal_turn == []


def test_protected_instruction_write_is_blocked_without_prompting_the_peer(cotal_turn):
    result = _request_protected_instruction_approval(["AGENTS.md"])
    assert result is not None and result.startswith("BLOCKED")
    assert cotal_turn == []


def test_discord_turn_still_reaches_its_own_notifier(monkeypatch):
    """Control: an attended chat platform keeps the human round-trip."""
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    assert approval_mod._presence()[2:] == (True, True)
