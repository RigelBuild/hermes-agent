"""vault_secret: search any 1Password item, list field labels, and run a command with a field injected."""

import json
import stat
from pathlib import Path

import pytest

_AGE = "AGE-SECRET-KEY-1QQQQTESTVALUEONLYFORTHISSUITEQQQQQQQQQQQQQQQQQQQQQQQQQQQ"

# A stand-in `op` with one account per OP_ACCOUNT, holding a Secure Note with a sectioned field.
_FAKE_OP = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
account = os.environ.get("OP_ACCOUNT", "")
with open(os.path.join(here, "op.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "account": account}) + "\n")
if argv[:2] == ["signin", "--raw"]:
    sys.stdin.read(); print("tok-" + argv[-1]); sys.exit(0)
if os.environ.get("OP_SESSION_" + account) != "tok-" + account:
    sys.stderr.write("[ERROR] You are not currently signed in.\n"); sys.exit(1)
items = {"rigelbuild": [{"id": "sopsid", "title": "SOPS", "category": "SECURE_NOTE", "vault": {"name": "Development"},
                         "sections": [{"id": "op1", "label": "Operator"}],
                         "fields": [{"id": "notesPlain", "label": "notesPlain", "type": "STRING", "value": ""},
                                    {"id": "pk", "label": "private key", "type": "CONCEALED", "section": {"id": "op1"},
                                     "value": "__AGE__"}]}],
         "my": [{"id": "gmailid", "title": "Gmail", "category": "LOGIN", "vault": {"name": "Private"}, "fields": []}]}
if argv[:2] == ["item", "list"]:
    print(json.dumps([{k: v for k, v in i.items() if k not in ("fields", "sections")} for i in items.get(account, [])]))
    sys.exit(0)
if argv[:2] == ["item", "get"]:
    found = [i for i in items.get(account, []) if i["id"] == argv[2]]
    if not found:
        sys.stderr.write("[ERROR] isn't an item in any vault\n"); sys.exit(1)
    print(json.dumps(found[0])); sys.exit(0)
sys.exit(2)
'''.replace("__AGE__", _AGE)


@pytest.fixture
def unlocked(tmp_path, monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe = tmp_path / "op"
    exe.write_text(_FAKE_OP, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    unlock.lock("onepassword")
    cfg = {"binary_path": str(exe), "accounts": ["my", "rigelbuild"]}
    OnePasswordLoginBackend(cfg).unlock("pw")
    import tools.vault_secret_tool as vst

    monkeypatch.setattr(vst, "_backend", lambda: OnePasswordLoginBackend(cfg))
    yield vst, tmp_path
    unlock.lock("onepassword")


def _call(vst, **args) -> dict:
    return json.loads(vst._handle(args))


def test_search_and_fields_return_metadata_only(unlocked):
    vst, _ = unlocked
    found = _call(vst, action="search", query="sops")
    assert found["items"] == [{"item": "op:sopsid", "title": "SOPS", "category": "SECURE_NOTE",
                               "vault": "Development", "account": "rigelbuild"}]
    fields = _call(vst, action="fields", item="op:sopsid")
    assert {"field": "pk", "label": "private key", "section": "Operator", "type": "CONCEALED",
            "has_value": True} in fields["fields"]
    assert _AGE not in json.dumps(fields)


def test_run_injects_the_field_and_redacts_it_in_every_common_form(unlocked):
    vst, tmp = unlocked
    command = ('test "$K" = "' + _AGE + '" && echo match; echo "$K"; printf %s "$K" | base64; '
               'python3 -c "import json,os; print(json.dumps(os.environ[\'K\']))"; echo "$K" > out.txt; wc -c < out.txt')
    res = _call(vst, action="run", command=command, workdir=str(tmp),
                env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["success"] and res["exit_code"] == 0 and res["injected"] == ["K"]
    assert res["output"].splitlines()[0] == "match"
    assert _AGE not in res["output"] and "[REDACTED]" in res["output"]
    import base64
    assert base64.b64encode(_AGE.encode()).decode() not in res["output"]


def test_run_refuses_reserved_names_and_bad_references(unlocked):
    vst, _ = unlocked
    for env in ({"PATH": {"item": "op:sopsid", "field": "pk"}}, {"lower": {"item": "op:sopsid", "field": "pk"}},
                {"K": {"item": "op:sopsid"}}, {}):
        res = _call(vst, action="run", command="true", env=env)
        assert res["success"] is False
    res = _call(vst, action="run", command="true", env={"K": {"item": "op:sopsid", "field": "nope"}})
    assert res["success"] is False and "field not found" in res["error"]
    res = _call(vst, action="run", command="true", env={"K": {"item": "op:--help", "field": "pk"}})
    assert res["success"] is False


def test_run_reports_failure_and_timeout(unlocked):
    vst, _ = unlocked
    res = _call(vst, action="run", command='echo "$K" >&2; exit 3', env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["success"] is False and res["exit_code"] == 3 and _AGE not in res["output"]
    res = _call(vst, action="run", command="sleep 5", timeout=1, env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["exit_code"] is None and "timed out" in res["output"]


def test_locked_backend_asks_for_an_unlock(unlocked):
    from agent.vault_backends import unlock

    vst, _ = unlocked
    unlock.lock("onepassword")
    assert _call(vst, action="search", query="")["error_type"] == "locked"
    assert _call(vst, action="run", command="true",
                 env={"K": {"item": "op:sopsid", "field": "pk"}})["error_type"] == "locked"


def test_redact_handles_multiline_values():
    from tools.vault_secret_tool import redact

    secret = "line-one-secret\nline-two-secret"
    assert redact("a line-two-secret b", [secret]) == "a [REDACTED] b"
