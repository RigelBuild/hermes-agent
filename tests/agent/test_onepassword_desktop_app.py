"""1Password desktop-app mode: the app's CLI integration authorises every ``op`` call, so an unlock
takes no master password, mints no token, and only opens this session's gate."""

from __future__ import annotations

import json
import stat
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.base import UnlockRequired

# A stand-in `op` that logs each argv and whether any OP_SESSION* variable reached it. An
# `app-locked` file next to it makes every call fail the way op does when the app is locked.
_FAKE_DESKTOP_OP = r'''#!/usr/bin/env python3
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
session = sorted(k for k in dict(os.environ) if k.startswith("OP_SESSION"))
with open(os.path.join(here, "op.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "session": session}) + "\n")
if os.path.exists(os.path.join(here, "app-locked")):
    sys.stderr.write("[ERROR] account is not signed in\n"); sys.exit(1)
if argv[:1] == ["whoami"]:
    print("URL: https://my.1password.com"); sys.exit(0)
if argv[:2] == ["item", "list"]:
    print(json.dumps([{"id": "item-a", "title": "A", "vault": {"id": "vault-a"},
                       "urls": [{"href": "https://example.com/login"}]}])); sys.exit(0)
if argv[:2] == ["item", "get"]:
    print("dummy-password"); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def fake_op_desktop_app(tmp_path):
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe = tmp_path / "op"
    exe.write_text(_FAKE_DESKTOP_OP, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    with patch("agent.secret_scope.get_secret", side_effect=lambda name, default="": default):
        yield (lambda: OnePasswordLoginBackend({"binary_path": str(exe), "desktop_app": True})), tmp_path
    unlock_mod.lock("onepassword")


def _calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / "op.log").read_text(encoding="utf-8").splitlines()]


def test_unlock_needs_no_password_or_token_and_lock_closes_the_gate(fake_op_desktop_app):
    new_backend, tmp_path = fake_op_desktop_app
    backend = new_backend()
    assert backend.list_items() == []  # locked: nothing listed and op is never asked for items
    backend.unlock("")
    assert backend.is_unlocked()
    assert new_backend().resolve_password("op:item-a") == "dummy-password"
    calls = _calls(tmp_path)
    assert [c["argv"][:2] for c in calls] == [["whoami"], ["item", "list"], ["item", "get"]]
    assert all(c["session"] == [] for c in calls)
    unlock_mod.lock("onepassword")
    locked = new_backend()
    assert not locked.is_unlocked()
    with pytest.raises(UnlockRequired):
        locked._run("item", "get", "item-a")
    assert len(_calls(tmp_path)) == 3  # a locked gate never reaches op


def test_unlock_fails_and_stays_locked_while_the_app_is_locked(fake_op_desktop_app):
    new_backend, tmp_path = fake_op_desktop_app
    (tmp_path / "app-locked").touch()
    backend = new_backend()
    with pytest.raises(RuntimeError, match="did not authorise"):
        backend.unlock("")
    assert not backend.is_unlocked()


def test_browser_vault_unlock_uses_the_app_without_a_prompt(fake_op_desktop_app):
    """No surface prompt is needed: the tool unlocks through the app even where nobody could type
    a master password."""
    from tools.browser_vault_tool import browser_vault_unlock

    new_backend, _ = fake_op_desktop_app
    with patch("agent.vault_backends.enabled_backends", return_value=[new_backend()]):
        result = json.loads(browser_vault_unlock("onepassword"))
    assert result == {"success": True, "backend": "onepassword", "desktop_app": True}
    assert new_backend().is_unlocked()
