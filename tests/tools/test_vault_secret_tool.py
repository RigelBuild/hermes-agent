"""vault_secret: search any 1Password item, list field labels, and run a command with a field injected."""

import json
import os
import shlex
import stat
from types import SimpleNamespace

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
                                     "value": "__AGE__"},
                                    {"id": "pin", "label": "pin", "type": "CONCEALED", "value": "42"}]}],
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


def test_describe_ref_exposes_only_onepassword_metadata(unlocked, monkeypatch):
    from agent.vault_backends import describe_ref

    vst, _ = unlocked
    backend = vst._backend()
    monkeypatch.setattr(backend, "read_field", lambda *_args: pytest.fail("metadata lookup read a value"))
    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: [backend])

    metadata = describe_ref("op:sopsid", "pk")

    assert metadata == {
        "title": "SOPS", "vault": "Development", "account": "rigelbuild", "field_label": "private key",
    }
    assert _AGE not in json.dumps(metadata)

def test_describe_ref_fetches_one_item_without_listing(unlocked, monkeypatch):
    from agent.vault_backends import describe_ref

    vst, _ = unlocked
    backend = vst._backend()
    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: [backend])
    backend._any_accounts["sopsid"] = "rigelbuild"
    calls = []

    def record_run(*args, **kwargs):
        calls.append(args)
        return json.dumps({
            "id": "sopsid", "title": "SOPS", "vault": {"name": "Development"},
            "fields": [{"id": "pk", "label": "private key", "value": _AGE}],
        })

    monkeypatch.setattr(backend, "_run", record_run)
    metadata = describe_ref("op:sopsid", "pk")

    assert metadata["field_label"] == "private key"
    assert len(calls) == 1
    assert calls[0][:2] == ("item", "get")
    assert "--reveal" not in calls[0]
    assert "list" not in calls[0]
    assert _AGE not in json.dumps(metadata)

def test_vault_secret_approval_shows_item_and_field_metadata_to_reviewer(unlocked, monkeypatch):
    from tools import approval

    vst, _ = unlocked
    backend = vst._backend()
    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: [backend])
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="default": "vault-review")
    monkeypatch.setattr("tools.approval_context.get_current_session_key", lambda default="default": "vault-review")
    monkeypatch.setattr(approval, "is_approved", lambda *_args: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    seen = {}
    def smart(display, *_args):
        seen["display"] = display
        return "approve"

    def human(spec, **kwargs):
        seen.update(kwargs)
        return {"approved": True, "message": None}

    monkeypatch.setattr(approval, "_smart_verdict", smart)
    monkeypatch.setattr(approval, "_human_decision", human)

    approval.request_tool_approval("vault_secret", "review target", args={
        "env": {"K": {"item": "op:sopsid", "field": "pk"}},
    }, review=True)

    assert "SOPS" in seen["display"]
    assert "private key" in seen["display"]
    assert "op:sopsid" in seen["command"]
    assert _AGE not in seen["display"] + seen["command"]

def test_run_injects_the_field_and_redacts_it_in_every_common_form(unlocked):
    vst, tmp = unlocked
    command = ('test "$K" = "' + _AGE + '" && echo match; echo "$K"; printf %s "$K" | base64; '
               'python3 -c "import json,os; print(json.dumps(os.environ[\'K\']))"; echo "$K" > out.txt; wc -c out.txt')
    res = _call(vst, action="run", reason="test run of the injected field", command=command, workdir=str(tmp),
                env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["success"] and res["exit_code"] == 0 and res["injected"] == ["K"]
    assert res["output"].splitlines()[0] == "match"
    assert _AGE not in res["output"] and "[REDACTED]" in res["output"]
    import base64
    assert base64.b64encode(_AGE.encode()).decode() not in res["output"]

def _fake_sops(unlocked, monkeypatch):
    vst, tmp = unlocked
    binary_dir = tmp / "bin"
    binary_dir.mkdir()
    binary = binary_dir / "sops"
    binary.write_text("#!/bin/sh\ncat > \"$SOPS_STDIN_CAPTURE\"\n", encoding="utf-8")
    binary.chmod(0o700)
    capture = tmp / "sops-stdin"

    from tools.environments.local import build_subprocess_env

    def build_env(*, extra=None):
        env = build_subprocess_env(extra=extra)
        env["PATH"] = str(binary_dir) + os.pathsep + env["PATH"]
        env["SOPS_STDIN_CAPTURE"] = str(capture)
        return env

    monkeypatch.setattr("tools.environments.local.build_subprocess_env", build_env)
    return capture


def _sops_command(path):
    return f"sops set --value-stdin \"$K\" 'secrets.yaml' < {shlex.quote(str(path))}"


def _sops_run(vst, tmp, path, command=None):
    return _call(vst, action="run", reason="test sops secret file input",
                 command=command or _sops_command(path),
                 workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})


def _scratch(tmp, monkeypatch, vst):
    scratch = tmp / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(vst, "_scratch_root", lambda: str(scratch))
    return scratch


def test_run_refuses_a_swapped_symlink_at_the_sops_stdin_path(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("approved source\n", encoding="utf-8")
    source.chmod(0o600)
    target = tmp / "ssh-key"
    target.write_text("swapped target contents\n", encoding="utf-8")
    target.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    source.unlink()
    source.symlink_to(target)

    result = _sops_run(vst, tmp, source)

    assert not result["success"] and "symlinks are refused" in result["error"]
    assert not capture.exists(), "the shell or sops read the swapped-in target"


def test_run_refuses_a_symlinked_directory_in_the_sops_stdin_path(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    real_dir = tmp / "real-dir"
    real_dir.mkdir()
    source = real_dir / "secret input"
    source.write_text("secret input\n", encoding="utf-8")
    source.chmod(0o600)
    (scratch / "dirlink").symlink_to(real_dir, target_is_directory=True)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _sops_run(vst, tmp, scratch / "dirlink" / "secret input")

    assert not result["success"] and "symlinks are refused" in result["error"]
    assert not capture.exists()


def test_run_rejects_hardlinked_sops_stdin_file(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("secret input\n", encoding="utf-8")
    source.chmod(0o600)
    os.link(source, tmp / "second-link")
    capture = _fake_sops(unlocked, monkeypatch)

    result = _sops_run(vst, tmp, source)

    assert not result["success"] and "link count 2" in result["error"]
    assert not capture.exists()


def test_run_rejects_group_readable_sops_stdin_file(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("secret input\n", encoding="utf-8")
    source.chmod(0o640)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _sops_run(vst, tmp, source)

    assert not result["success"] and "mode 0640" in result["error"]
    assert not capture.exists()


def test_run_passes_open_sops_stdin_file_to_command(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _sops_run(vst, tmp, source)

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"encrypted value bytes\n"


def test_run_passes_relative_sops_stdin_resolved_against_workdir(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    (scratch / "secret input").write_text("relative bytes\n", encoding="utf-8")
    (scratch / "secret input").chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    command = "sops set --value-stdin \"$K\" 'secrets.yaml' < 'secret input'"

    result = _call(vst, action="run", reason="test sops secret file input", command=command,
                   workdir=str(scratch), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"relative bytes\n"


def test_run_passes_piped_sops_stdin_without_opening_a_file(unlocked, monkeypatch):
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    capture = _fake_sops(unlocked, monkeypatch)
    command = 'printf %s "$K" | sops set --value-stdin "$K" \'secrets.yaml\''

    result = _call(vst, action="run", reason="test sops secret pipe input", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == _AGE.encode()


def test_run_passes_cat_pipe_sops_stdin_unchecked_as_documented(unlocked, monkeypatch):
    """Documented limit: a file read by ``cat f |`` is not checked or opened by this module."""
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    command = f"cat {shlex.quote(str(source))} | sops set --value-stdin \"$K\" 'secrets.yaml'"

    result = _call(vst, action="run", reason="test cat pipe into sops is unchecked",
                   command=command, workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"encrypted value bytes\n"


def test_run_passes_eval_quoted_sops_redirect_unchecked_as_documented(unlocked, monkeypatch):
    """Documented limit: a ``<`` inside an ``eval`` argument is not checked or opened by this module. The
    shell reads the file itself when eval runs it, so the bytes reach sops without the no-follow open."""
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    inner = f"sops set --value-stdin \"$K\" 'secrets.yaml' < {shlex.quote(str(source))}"
    command = f"eval {shlex.quote(inner)}"

    result = _call(vst, action="run", reason="test eval-quoted sops redirect is unchecked",
                   command=command, workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"encrypted value bytes\n"


def test_run_passes_printf_built_sops_redirect_unchecked_as_documented(unlocked, monkeypatch):
    """Documented limit: command text built at run time (here ``eval`` over ``$(printf ...)`` with a
    ``\\074`` escape) hides the ``<`` from the parser. The shell opens the file itself, so the checked
    open does not run. This pins that the bytes still reach sops."""
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret-input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    built = f"sops set --value-stdin \"$K\" secrets.yaml \\074{shlex.quote(str(source))}"
    command = f"eval \"$(printf '{built}')\""

    result = _call(vst, action="run", reason="test printf-built redirect is unchecked",
                   command=command, workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"encrypted value bytes\n"


def test_run_refuses_a_heredoc_whose_body_contains_a_less_than(unlocked, monkeypatch):
    """A raw ``<<`` heredoc with a lone ``<`` in its body is refused even with no sops command: the
    plain character rule cannot tell a body from a file input, so it refuses (a known false positive)."""
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    command = "cat > f.xml <<EOF\n<root/>\nEOF"

    result = _call(vst, action="run", reason="test heredoc body with less-than", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert not result["success"] and "heredoc with a file input" in result["error"]


def test_run_passes_here_string_sops_stdin_without_opening_a_file(unlocked, monkeypatch):
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    capture = _fake_sops(unlocked, monkeypatch)
    command = 'sops set --value-stdin "$K" \'secrets.yaml\' <<< "$K"'

    result = _call(vst, action="run", reason="test sops secret here-string input", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == _AGE.encode() + b"\n"


def test_run_passes_a_heredoc_beside_sops_value_stdin_with_no_file_input(unlocked, monkeypatch):
    """Round-4 rule: a heredoc is refused only with a lone ``<`` (a file input) in the same command. A
    heredoc with no lone ``<`` names no file, so it runs unchanged and sops receives the body."""
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    capture = _fake_sops(unlocked, monkeypatch)
    command = 'sops set --value-stdin "$K" \'secrets.yaml\' <<EOF\nheredoc body\nEOF'

    result = _call(vst, action="run", reason="test sops heredoc with no file input", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"heredoc body\n"


def test_run_runs_a_heredoc_when_no_sops_value_stdin_is_present(unlocked, monkeypatch):
    """A heredoc in a command with no sops ``--value-stdin`` is not refused: it runs as written."""
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    command = "cat <<EOF\nplain heredoc body\nEOF"

    result = _call(vst, action="run", reason="test plain heredoc still runs", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert "plain heredoc body" in result["output"]


def test_run_passes_a_home_tilde_sops_stdin_path(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    monkeypatch.setenv("HOME", str(tmp))
    source = scratch / "value"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    command = "sops set --value-stdin \"$K\" 'secrets.yaml' < ~/scratch/value"

    result = _call(vst, action="run", reason="test sops tilde stdin path", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert result["success"] and result["exit_code"] == 0
    assert capture.read_bytes() == b"encrypted value bytes\n"


@pytest.mark.parametrize("path", [
    "'~/scratch/value'",
    '"~/scratch/value"',
    r"\~/scratch/value",
    "~",
    "~+/scratch/value",
    "~-/scratch/value",
])
def test_run_refuses_nonliteral_home_tilde_sops_stdin_paths(unlocked, monkeypatch, path):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    monkeypatch.setenv("HOME", str(tmp))
    source = scratch / "value"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    command = f'''sops set --value-stdin "$K" 'secrets.yaml' < {path}'''

    result = _call(vst, action="run", reason="test refused nonliteral sops tilde stdin path", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert not result["success"] and "stdin redirect refused" in result["error"]
    assert not capture.exists()


def test_run_refuses_a_home_tilde_sops_stdin_path_outside_scratch(unlocked, monkeypatch):
    vst, tmp = unlocked
    _scratch(tmp, monkeypatch, vst)
    monkeypatch.setenv("HOME", str(tmp))
    source = tmp / "outside"
    source.write_text("outside scratch\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)
    command = "sops set --value-stdin \"$K\" 'secrets.yaml' < ~/scratch/../outside"

    result = _call(vst, action="run", reason="test sops tilde stdin outside scratch", command=command,
                   workdir=str(tmp), env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert not result["success"] and "must be under the Hermes scratch dir" in result["error"]
    assert not capture.exists()

@pytest.mark.parametrize("command, reaches_fake_sops", [
    ("sops set --value-stdin \"$K\" 'secrets.yaml' < {path}", True),
    ("env sops set --value-stdin \"$K\" 'secrets.yaml' < {path}", True),
    ("command sops set --value-stdin \"$K\" 'secrets.yaml' < {path}", True),
    ("time sops set --value-stdin \"$K\" 'secrets.yaml' < {path}", True),
    ("sops --config x set --value-stdin \"$K\" 'secrets.yaml' < {path}", True),
    ("sops set --value-stdin \"$K\" 'secrets.yaml' <{path}", True),
    # Absolute path: the fake sops is only on PATH, so this must not reach it.
    ("/usr/bin/sops set --value-stdin \"$K\" 'secrets.yaml' < {path}", False),
])
def test_run_routes_every_sops_spelling_of_a_file_stdin_through_the_checked_open(
        unlocked, monkeypatch, command, reaches_fake_sops):
    """Each spelling reaches the no-follow open, never a bare shell redirect. Where the fake sops is
    on PATH, the approved bytes must reach it; the absolute path resolves elsewhere and never does."""
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _call(vst, action="run", reason="test sops secret file input",
                   command=command.format(path=shlex.quote(str(source))), workdir=str(tmp),
                   env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert "stdin redirect refused" not in result.get("error", "")
    if reaches_fake_sops:
        assert result["success"] and result["exit_code"] == 0
        assert capture.read_bytes() == b"encrypted value bytes\n"
    else:
        assert not result["success"]
        assert not capture.exists()


@pytest.mark.parametrize("command", [
    "< {path} sops set --value-stdin \"$K\" 'secrets.yaml'",
    "sops set --value-stdin \"$K\" 'secrets.yaml' {{v}}< {path}",
    "echo $((1<<2))\nsops set --value-stdin \"$K\" 'secrets.yaml' < {path}",
    "true # <<true\ntrue\nsops set --value-stdin \"$K\" 'secrets.yaml' < {path}",
    "cat \\'x <<EOF\n'\nEOF\nsops set --value-stdin \"$K\" 'secrets.yaml' < {path}",
    "cat \\'x <<EOF\n'\nEOF\nsops set --value-std''in \"$K\" 'secrets.yaml' < {path}",
    "cat \\'x <<EOF\n'\nEOF\ns''ops set --value-stdin \"$K\" 'secrets.yaml' < {path}",
    "sops set --value-stdin \"$K\" 'secrets.yaml' <<< \"$(<{path})\"",
    "sops set --value-stdin \"$K\" 'secrets.yaml' < {path} ; cat < {path}",
    "cat < {path} | sops set --value-stdin \"$K\" 'secrets.yaml'",
    "true $(cat < {path})",
    "sops set --value-stdin \"$K\" 'secrets.yaml' <\"{path}\"  # \"unterminated",
])
def test_run_refuses_unhandled_file_stdin_redirects(unlocked, monkeypatch, command):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _call(vst, action="run", reason="test sops secret file input",
                   command=command.format(path=shlex.quote(str(source))), workdir=str(tmp),
                   env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert not result["success"] and "stdin redirect refused" in result["error"]
    assert not capture.exists()


def test_run_refuses_a_file_stdin_redirect_with_no_sops_command(unlocked, monkeypatch):
    vst, tmp = unlocked
    scratch = _scratch(tmp, monkeypatch, vst)
    source = scratch / "secret input"
    source.write_text("encrypted value bytes\n", encoding="utf-8")
    source.chmod(0o600)
    capture = _fake_sops(unlocked, monkeypatch)

    result = _call(vst, action="run", reason="test file redirect without sops",
                   command=f"cat < {shlex.quote(str(source))}", workdir=str(tmp),
                   env={"K": {"item": "op:sopsid", "field": "pk"}})

    assert not result["success"] and "stdin redirect refused" in result["error"]
    assert not capture.exists()

def test_run_refuses_a_value_too_short_to_redact(unlocked):
    """Redaction skips needles under _MIN_REDACT, so a short value would come back verbatim."""
    vst, tmp = unlocked
    res = _call(vst, action="run", reason="test run of the injected field", command='echo "$P" > ran.txt; echo "$P"', workdir=str(tmp),
                env={"K": {"item": "op:sopsid", "field": "pk"}, "P": {"item": "op:sopsid", "field": "pin"}})
    assert not res["success"] and "P" in res["error"] and "42" not in json.dumps(res)
    assert not (tmp / "ran.txt").exists()


def test_run_refuses_reserved_names_and_bad_references(unlocked):
    vst, _ = unlocked
    for env in ({"PATH": {"item": "op:sopsid", "field": "pk"}}, {"lower": {"item": "op:sopsid", "field": "pk"}},
                {"K": {"item": "op:sopsid"}}, {}):
        res = _call(vst, action="run", reason="test run of the injected field", command="true", env=env)
        assert res["success"] is False
    res = _call(vst, action="run", reason="test run of the injected field", command="true", env={"K": {"item": "op:sopsid", "field": "nope"}})
    assert res["success"] is False and "field not found" in res["error"]
    res = _call(vst, action="run", reason="test run of the injected field", command="true", env={"K": {"item": "op:--help", "field": "pk"}})
    assert res["success"] is False


def test_run_reports_failure_and_timeout(unlocked):
    vst, _ = unlocked
    res = _call(vst, action="run", reason="test run of the injected field", command='echo "$K" >&2; exit 3', env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["success"] is False and res["exit_code"] == 3 and _AGE not in res["output"]
    res = _call(vst, action="run", reason="test run of the injected field", command="sleep 5", timeout=1, env={"K": {"item": "op:sopsid", "field": "pk"}})
    assert res["exit_code"] is None and "timed out" in res["output"]


def test_locked_backend_asks_for_an_unlock(unlocked):
    from agent.vault_backends import unlock

    vst, _ = unlocked
    unlock.lock("onepassword")
    assert _call(vst, action="search", query="")["error_type"] == "locked"
    assert _call(vst, action="run", reason="test run of the injected field", command="true",
                 env={"K": {"item": "op:sopsid", "field": "pk"}})["error_type"] == "locked"


def test_redact_handles_multiline_values():
    from tools.vault_secret_tool import redact

    secret = "line-one-secret\nline-two-secret"
    assert redact("a line-two-secret b", [secret]) == "a [REDACTED] b"


def test_run_requires_a_reason(unlocked):
    vst, tmp = unlocked
    for reason in (None, "", "  why  "):
        args = {"action": "run", "command": "echo hi > ran.txt", "workdir": str(tmp),
                "env": {"K": {"item": "op:sopsid", "field": "pk"}}}
        if reason is not None:
            args["reason"] = reason
        res = _call(vst, **args)
        assert not res["success"] and "reason is required" in res["error"]
    assert not (tmp / "ran.txt").exists()

def test_describe_ref_raises_when_item_metadata_is_unavailable(monkeypatch):
    from agent.vault_backends import describe_ref

    backend = SimpleNamespace(name="local", owns=lambda _item: True, get_meta=lambda _item: None)
    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: [backend])

    with pytest.raises(RuntimeError, match="item not found"):
        describe_ref("vault_missing")

