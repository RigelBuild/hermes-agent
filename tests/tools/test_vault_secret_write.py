"""vault_secret create/edit: write a 1Password item whose values never enter the conversation."""

import base64
import json
import os
import stat

import pytest

_SECRET = "GOCSPX-test-client-secret-ONLY-FOR-THIS-SUITE-0123456789"
_OLD = "old-concealed-value-ONLY-FOR-THIS-SUITE"

# A stand-in `op` keeping items in store.json, logging argv and stdin to separate files.
_FAKE_OP = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
account = os.environ.get("OP_ACCOUNT", "")
stdin = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
with open(os.path.join(here, "argv.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "account": account}) + "\n")
with open(os.path.join(here, "stdin.log"), "a") as log:
    log.write(stdin + "\n")
if argv[:2] == ["signin", "--raw"]:
    print("tok-" + argv[-1]); sys.exit(0)
if os.environ.get("OP_SESSION_" + account) != "tok-" + account:
    sys.stderr.write("[ERROR] You are not currently signed in.\n"); sys.exit(1)
path = os.path.join(here, "store.json")
store = json.load(open(path)) if os.path.exists(path) else {"my": [], "rigelbuild": []}
def save():
    json.dump(store, open(path, "w"))
if argv[:3] == ["item", "template", "get"]:
    cat = argv[3]
    if cat == "API Credential":
        print(json.dumps({"title": "", "category": "API_CREDENTIAL", "fields": [
            {"id": "username", "type": "STRING", "label": "username", "value": ""},
            {"id": "credential", "type": "CONCEALED", "label": "credential", "value": ""}]}))
    else:
        print(json.dumps({"title": "", "category": "SECURE_NOTE", "fields": [
            {"id": "notesPlain", "type": "STRING", "purpose": "NOTES", "label": "notesPlain", "value": ""}]}))
    sys.exit(0)
if argv[:2] == ["item", "create"]:
    item = json.loads(stdin)
    if item.get("title") == "FAIL-ECHO":
        sys.stderr.write("[ERROR] invalid item: " + stdin + "\n"); sys.exit(1)
    item["id"] = "new%d" % (len(store[account]) + 1)
    item["vault"] = {"id": "v1", "name": argv[argv.index("--vault") + 1]}
    store[account].append(item); save()
    print(json.dumps(item)); sys.exit(0)
if argv[:2] == ["item", "list"]:
    print(json.dumps([{"id": i["id"], "title": i["title"], "category": i.get("category", ""),
                       "vault": i["vault"]} for i in store.get(account, [])])); sys.exit(0)
if argv[:2] == ["item", "get"]:
    found = [i for i in store.get(account, []) if i["id"] == argv[2]]
    if not found:
        sys.stderr.write("[ERROR] isn't an item in any vault\n"); sys.exit(1)
    print(json.dumps(found[0])); sys.exit(0)
if argv[:2] == ["item", "edit"]:
    new = json.loads(stdin)
    if any(f.get("label") == "fail-echo" for f in new.get("fields", [])):
        sys.stderr.write("[ERROR] rejected: " + stdin + "\n"); sys.exit(1)
    store[account] = [new if i["id"] == argv[2] else i for i in store[account]]; save()
    print(json.dumps(new)); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    # tests/conftest.py's hermetic fixture already blanks credential-shaped env vars.
    from agent.vault_backends import unlock
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe = tmp_path / "op"
    exe.write_text(_FAKE_OP, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    unlock.lock("onepassword")
    cfg = {"binary_path": str(exe), "accounts": ["my", "rigelbuild"], "allow_write": True}
    OnePasswordLoginBackend(cfg).unlock("pw")
    import tools.vault_secret_tool as vst

    backend = OnePasswordLoginBackend(cfg)
    monkeypatch.setattr(vst, "_backend", lambda: backend)
    monkeypatch.setattr(vst, "_scratch_root", lambda: os.path.realpath(scratch))
    yield vst, tmp_path, scratch, cfg
    unlock.lock("onepassword")


def _call(vst, **args) -> dict:
    return json.loads(vst._handle(args))


def _secret_file(scratch, name="secret.txt", value=_SECRET, mode=0o600):
    path = scratch / name
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _create(vst, fields, **extra):
    args = {"action": "create", "account": "rigelbuild", "vault": "Development", "title": "Google OAuth client",
            "category": "API Credential", "reason": "escrow the new OAuth client secret", "fields": fields}
    args.update(extra)
    return _call(vst, **args)


def _logs(tmp):
    argv = (tmp / "argv.log").read_text() if (tmp / "argv.log").exists() else ""
    stdin = (tmp / "stdin.log").read_text() if (tmp / "stdin.log").exists() else ""
    return argv, stdin


def _store(tmp):
    return json.loads((tmp / "store.json").read_text())


def test_create_from_file_and_literal_returns_handle_and_labels_only(env):
    vst, tmp, scratch, _ = env
    path = _secret_file(scratch)
    res = _create(vst, [{"label": "credential", "file": str(path)},
                        {"label": "username", "value": "client-id-123.apps.googleusercontent.com"}])
    assert res["success"] is True, res
    assert res["item"] == "op:new1" and res["vault"] == "Development" and res["account"] == "rigelbuild"
    assert [f["label"] for f in res["fields"]] == ["credential", "username"]
    assert _SECRET not in json.dumps(res)
    fields = {f["label"]: f for f in _store(tmp)["rigelbuild"][0]["fields"]}
    assert fields["credential"]["value"] == _SECRET and fields["credential"]["type"] == "CONCEALED"
    assert fields["username"]["value"] == "client-id-123.apps.googleusercontent.com"
    assert path.exists() and res["deleted_sources"] == []


def test_value_reaches_op_on_stdin_never_argv(env):
    vst, tmp, scratch, _ = env
    assert _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}])["success"] is True
    argv, stdin = _logs(tmp)
    assert _SECRET not in argv
    assert _SECRET in stdin


def test_delete_sources_removes_the_file_only_after_success(env):
    vst, _, scratch, _ = env
    path = _secret_file(scratch)
    failed = _create(vst, [{"label": "credential", "file": str(path)}], title="FAIL-ECHO", delete_sources=True)
    assert failed["success"] is False and path.exists()
    res = _create(vst, [{"label": "credential", "file": str(path)}], delete_sources=True)
    assert res["success"] is True and not path.exists() and res["deleted_sources"] == [str(path)]


def test_op_error_echoing_the_payload_is_redacted(env):
    vst, _, scratch, _ = env
    res = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}], title="FAIL-ECHO")
    assert res["success"] is False
    text = json.dumps(res)
    assert _SECRET not in text and base64.b64encode(_SECRET.encode()).decode() not in text


@pytest.mark.parametrize("field", [
    {"label": "credential", "type": "CONCEALED", "value": _SECRET},
    {"label": "client secret", "value": _SECRET},
    {"label": "API token", "value": _SECRET},
    {"label": "password", "value": _SECRET},
    {"label": "password", "type": "STRING", "value": _SECRET},
])
def test_inline_value_is_refused_for_secret_fields(env, field):
    vst, tmp, _, _ = env
    res = _create(vst, [field])
    assert res["success"] is False and "inline" in res["error"]
    assert '"create"' not in _logs(tmp)[0]


def test_inline_value_cannot_land_in_a_template_secret_field(env):
    # "credential" is CONCEALED in the template, so a literal typed STRING is still refused.
    vst, tmp, _, _ = env
    res = _create(vst, [{"label": "credential", "type": "STRING", "value": "not-really-plain"}])
    assert res["success"] is False and "secret field" in res["error"]
    assert '"create"' not in _logs(tmp)[0]


def test_plain_labels_that_contain_secret_words_still_take_literals(env):
    vst, _, _, _ = env
    res = _create(vst, [{"label": "shipping", "value": "none"}, {"label": "footprint", "value": "small"}],
                  category="Secure Note")
    assert res["success"] is True, res


def test_source_file_rules(env, tmp_path):
    vst, _, scratch, _ = env
    loose = _secret_file(scratch, "loose.txt", mode=0o644)
    assert "chmod 600" in _create(vst, [{"label": "credential", "file": str(loose)}])["error"]
    link = scratch / "link.txt"
    link.symlink_to(_secret_file(scratch, "target.txt"))
    assert "symlink" in _create(vst, [{"label": "credential", "file": str(link)}])["error"]
    outside = _secret_file(tmp_path, "outside.txt")
    assert "scratch dir" in _create(vst, [{"label": "credential", "file": str(outside)}])["error"]
    escape = scratch / ".." / "outside.txt"
    assert "scratch dir" in _create(vst, [{"label": "credential", "file": str(escape)}])["error"]
    short = _secret_file(scratch, "short.txt", value="ab")
    assert "shorter" in _create(vst, [{"label": "credential", "file": str(short)}])["error"]


def test_command_source_has_no_1password_session(env, monkeypatch):
    # The fixture unlocked the gateway (its token "tok-rigelbuild" lives in the unlock store), and a
    # stray session is also exported into the launch env: the value command must see neither.
    vst, tmp, _, _ = env
    monkeypatch.setenv("OP_SESSION_rigelbuild", "tok-rigelbuild")
    cmd = ('if env | grep -q -e "^OP_SESSION" -e "tok-rigelbuild"; then echo leaked-session-value; '
           'else echo ' + _SECRET + '; fi')
    res = _create(vst, [{"label": "credential", "command": cmd}])
    assert res["success"] is True, res
    value = next(f["value"] for f in _store(tmp)["rigelbuild"][0]["fields"] if f["label"] == "credential")
    assert value == _SECRET


def test_failing_command_source_reports_exit_code_only(env):
    vst, tmp, _, _ = env
    res = _create(vst, [{"label": "credential", "command": "echo " + _SECRET + " >&2; exit 3"}])
    assert res["success"] is False and res["error"] == "value command exited 3"
    assert '"create"' not in _logs(tmp)[0]


def test_edit_keeps_other_fields_and_redacts_existing_secrets(env):
    vst, tmp, scratch, _ = env
    created = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch, "old.txt", value=_OLD))},
                            {"label": "region", "section": "Meta", "command": "echo concealed-region"}])
    assert created["success"] is True, created
    res = _call(vst, action="edit", item=created["item"], reason="rotate the OAuth client secret",
                fields=[{"label": "credential", "file": str(_secret_file(scratch, "new.txt"))}])
    assert res["success"] is True, res
    fields = {f["label"]: f for f in _store(tmp)["rigelbuild"][0]["fields"]}
    assert fields["credential"]["value"] == _SECRET and fields["region"]["value"] == "concealed-region"
    # op rejects an edit and echoes the whole item: neither the new nor an existing value comes back.
    bad = _call(vst, action="edit", item=created["item"], reason="rotate the OAuth client secret",
                fields=[{"label": "fail-echo", "file": str(_secret_file(scratch, "x.txt", value=_OLD + "-new"))}])
    assert bad["success"] is False
    text = json.dumps(bad)
    assert _SECRET not in text and "concealed-region" not in text and _OLD not in text


def test_edit_cannot_turn_a_secret_field_plain(env):
    vst, _, scratch, _ = env
    created = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}])
    res = _call(vst, action="edit", item=created["item"], reason="change the credential field type",
                fields=[{"label": "credential", "type": "STRING", "file": str(_secret_file(scratch, "n.txt"))}])
    assert res["success"] is False and "secret field" in res["error"]


def test_edit_refuses_items_a_template_round_trip_could_damage(env):
    vst, tmp, scratch, _ = env
    created = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}])
    store = _store(tmp)
    store["rigelbuild"][0]["fields"].append({"id": "pk1", "type": "PASSKEY", "label": "passkey", "value": ""})
    (tmp / "store.json").write_text(json.dumps(store))
    res = _call(vst, action="edit", item=created["item"], reason="rotate the OAuth client secret",
                fields=[{"label": "credential", "file": str(_secret_file(scratch, "n.txt"))}])
    assert res["success"] is False and "PASSKEY" in res["error"]
    assert '"edit"' not in _logs(tmp)[0]


def test_errors_are_withheld_when_a_value_line_is_too_short_to_redact(env):
    vst, _, scratch, _ = env
    res = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch, value=_SECRET + "\nab"))}],
                  title="FAIL-ECHO")
    assert res["success"] is False and "withheld" in res["error"]
    assert "ab\\n" not in res["error"] and _SECRET not in res["error"]


def test_writes_are_off_unless_the_host_allows_them(env, monkeypatch):
    vst, _, scratch, cfg = env
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    monkeypatch.setattr(vst, "_backend", lambda: OnePasswordLoginBackend({**cfg, "allow_write": False}))
    res = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}])
    assert res["success"] is False and "allow_write" in res["error"]


def test_create_validates_its_inputs(env):
    vst, _, scratch, _ = env
    path = str(_secret_file(scratch))
    assert "account is required" in _create(vst, [{"label": "credential", "file": path}], account="")["error"]
    assert "not configured" in _create(vst, [{"label": "credential", "file": path}], account="other")["error"]
    assert "reason is required" in _create(vst, [{"label": "credential", "file": path}], reason="")["error"]
    assert "unknown category" in _create(vst, [{"label": "credential", "file": path}], category="Bogus")["error"]
    assert "exactly one" in _create(vst, [{"label": "credential", "file": path, "command": "true"}])["error"]
    assert "exactly one" in _create(vst, [{"label": "credential"}])["error"]
    assert "edit needs item" in _call(vst, action="edit", reason="some reason here",
                                      fields=[{"label": "credential", "file": path}])["error"]


def test_locked_backend_asks_for_an_unlock(env):
    vst, _, scratch, _ = env
    from agent.vault_backends import unlock

    unlock.lock("onepassword")
    res = _create(vst, [{"label": "credential", "file": str(_secret_file(scratch))}])
    assert res["success"] is False and res.get("error_type") == "locked"
