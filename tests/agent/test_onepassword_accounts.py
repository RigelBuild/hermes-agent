"""1Password backend across several accounts that share a master password (``vault.onepassword.accounts``)."""

import json
import stat
from pathlib import Path

import pytest

# A stand-in `op` with two accounts. `signin --raw --account X` reads the password on stdin and prints
# a per-account token; other commands require OP_SESSION_<X> to hold X's token and list X's items.
_FAKE_OP = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
account = os.environ.get("OP_ACCOUNT", "")
with open(os.path.join(here, "op.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "account": account,
                          "sessions": sorted(k for k in os.environ if k.startswith("OP_SESSION"))}) + "\n")
if argv[:2] == ["signin", "--raw"]:
    acct = argv[argv.index("--account") + 1] if "--account" in argv else ""
    if not acct:
        sys.stderr.write('[ERROR] found multiple accounts for filter ""\n'); sys.exit(1)
    if sys.stdin.read().strip() != "pw" or os.path.exists(os.path.join(here, "reject-" + acct)):
        sys.stderr.write("[ERROR] Unauthorized: invalid credentials\n"); sys.exit(1)
    print("tok-" + acct); sys.exit(0)
if not account or os.environ.get("OP_SESSION_" + account) != "tok-" + account:
    sys.stderr.write("[ERROR] You are not currently signed in.\n"); sys.exit(1)
if argv[:2] == ["item", "list"]:
    print(json.dumps([{"id": account + "-item", "title": account, "vault": {"id": "v-" + account},
                       "urls": [{"href": "https://" + account + ".example.com/login"}]}])); sys.exit(0)
if argv[:2] == ["item", "get"]:
    if not argv[2].startswith(account + "-"):
        sys.stderr.write("[ERROR] item not found\n"); sys.exit(1)
    print("123456" if "--otp" in argv else "pw-of-" + account); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def fake_op(tmp_path, monkeypatch):
    from agent.vault_backends import unlock

    exe = tmp_path / "op"
    exe.write_text(_FAKE_OP, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    unlock.lock("onepassword")
    yield exe, tmp_path / "op.log"
    unlock.lock("onepassword")


def _backend(exe: Path, **cfg):
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    return OnePasswordLoginBackend({"binary_path": str(exe), **cfg})


def _log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_unlock_signs_in_to_every_account_and_lists_and_reads_across_them(fake_op):
    exe, log = fake_op
    backend = _backend(exe, accounts=["my", "rigelbuild"])
    backend.unlock("pw")

    signins = [e for e in _log(log) if e["argv"][:2] == ["signin", "--raw"]]
    assert [e["argv"][-1] for e in signins] == ["my", "rigelbuild"]

    items = backend.list_items()
    assert sorted(m.id for m in items) == ["op:my-item", "op:rigelbuild-item"]
    # Each read goes to the account that listed the item, with only that account's session.
    fresh = _backend(exe, accounts=["my", "rigelbuild"])
    assert fresh.resolve_password("op:rigelbuild-item") == "pw-of-rigelbuild"
    assert fresh.resolve_password("op:my-item") == "pw-of-my"
    assert fresh.resolve_otp("op:rigelbuild-item") == "123456"
    gets = [e for e in _log(log) if e["argv"][:2] == ["item", "get"]]
    assert [(e["argv"][2], e["account"], e["sessions"]) for e in gets] == [
        ("rigelbuild-item", "rigelbuild", ["OP_SESSION_rigelbuild"]),
        ("my-item", "my", ["OP_SESSION_my"]),
        ("rigelbuild-item", "rigelbuild", ["OP_SESSION_rigelbuild"]),
    ]


def test_unlock_is_all_or_nothing(fake_op):
    from agent.vault_backends import unlock

    exe, _ = fake_op
    (exe.parent / "reject-rigelbuild").touch()
    backend = _backend(exe, accounts=["my", "rigelbuild"])
    with pytest.raises(RuntimeError, match=r"unlock failed \(rigelbuild\): .*invalid credentials"):
        backend.unlock("pw")
    assert not unlock.is_unlocked("onepassword")


def test_single_account_config_is_unchanged(fake_op):
    from agent.vault_backends import unlock

    exe, log = fake_op
    backend = _backend(exe, account="rigelbuild")
    backend.unlock("pw")
    # One account still stores op's bare token, as before.
    assert unlock.get_session_token("onepassword") == "tok-rigelbuild"
    assert [m.id for m in backend.list_items()] == ["op:rigelbuild-item"]
    assert [e["argv"][-1] for e in _log(log) if e["argv"][0] == "signin"] == ["rigelbuild"]


def test_accounts_accepts_a_string_and_dedupes():
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    assert OnePasswordLoginBackend({"accounts": "my"})._accounts() == ["my"]
    assert OnePasswordLoginBackend({"accounts": ["my", " my ", "", "rigelbuild"]})._accounts() == ["my", "rigelbuild"]
    assert OnePasswordLoginBackend({"accounts": [], "account": "x"})._accounts() == ["x"]
    assert OnePasswordLoginBackend({})._accounts() == [""]


def test_a_stale_single_token_after_a_config_change_asks_to_unlock_again(fake_op):
    from agent.vault_backends import unlock
    from agent.vault_backends.base import UnlockRequired

    exe, _ = fake_op
    _backend(exe, account="my").unlock("pw")
    assert unlock.is_unlocked("onepassword")
    with pytest.raises(UnlockRequired):
        _backend(exe, accounts=["my", "rigelbuild"]).list_items()
    assert not unlock.is_unlocked("onepassword")
