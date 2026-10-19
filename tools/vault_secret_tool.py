"""vault_secret: use a 1Password field (any item category) without the value entering the conversation.

Actions:
- search: items whose title contains ``query`` across every configured account (metadata only).
- fields: one item's field ids, labels, sections and types; never values.
- run: read named fields server-side, inject each as an environment variable of one ``bash -c``
  child, and return its exit code and output with every injected value (and its common
  encodings) redacted. The value never reaches the model, argv or disk.

- create / edit: write one item (off unless ``vault.onepassword.allow_write``). Values come from a
  0600 file under the scratch dir or a command's stdout; an inline value is accepted only for a
  non-secret field. The item JSON reaches ``op`` on stdin, and only the handle and labels return.

The tool cannot print a value: it has no read action, and ``run`` output is scrubbed. Hosts
gate ``run``, ``create`` and ``edit`` with a pre_tool_call hook (it carries the command and the item/field names for
the approval prompt); a host without one should not enable this tool.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import subprocess
import time
import urllib.parse
from typing import Any, Optional

_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
# Names whose value would change how the child runs or resolves programs.
_RESERVED = {"PATH", "HOME", "SHELL", "BASH_ENV", "ENV", "IFS", "LD_PRELOAD", "LD_LIBRARY_PATH",
             "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "PYTHONPATH", "PYTHONSTARTUP", "PROMPT_COMMAND"}
_OUTPUT_LIMIT = 8000
_DEFAULT_TIMEOUT = 120
_MAX_TIMEOUT = 600
_MIN_REDACT = 4


def _backend():
    from agent.vault_backends import enabled_backends

    return next((b for b in enabled_backends() if b.name == "onepassword"), None)


def _check_available() -> bool:
    try:
        from agent.vault_backends.base import is_enabled

        return is_enabled("onepassword")
    except Exception:
        return False


def _err(message: str, **extra: Any) -> str:
    return json.dumps({"success": False, "error": message, **extra}, ensure_ascii=False)


def _locked(backend) -> str:
    return _err(f"{backend.display_name} is locked. Ask the user to unlock it "
                "(browser_vault action=unlock, or their host's unlock command), then retry.",
                error_type="locked")


def _variants(value: str) -> set[str]:
    """The value plus encodings a command might echo: JSON-escaped, base64, URL-encoded, per line."""
    out = {value, value.strip(), json.dumps(value)[1:-1], urllib.parse.quote(value, safe="")}
    raw = value.encode()
    out |= {base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode()}
    out |= {line.strip() for line in value.splitlines()}
    return {v for v in out if len(v) >= _MIN_REDACT}


def redact(text: str, secrets: list[str]) -> str:
    for needle in sorted({v for s in secrets for v in _variants(s)}, key=len, reverse=True):
        text = text.replace(needle, "[REDACTED]")
    return text


def safe_error(text: str, secrets: list[str], generic: str) -> str:
    """*text* redacted, or *generic* when a value has a line too short to redact (it could survive
    in an error that echoes the payload)."""
    if any(0 < len(line.strip()) < _MIN_REDACT for s in secrets for line in s.splitlines()):
        return generic
    return redact(text, secrets)


def _parse_env(raw: Any) -> list[tuple[str, str, str]]:
    """``env``: {NAME: {"item": ..., "field": ...}} → [(NAME, item, field)]."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("env must map at least one variable name to {item, field}")
    out = []
    for name, ref in raw.items():
        name = str(name)
        if not _ENV_NAME.match(name) or name in _RESERVED:
            raise ValueError(f"{name!r} is not an allowed variable name")
        if not isinstance(ref, dict) or not ref.get("item") or not ref.get("field"):
            raise ValueError(f"{name}: give both item (from search) and field (from fields)")
        out.append((name, str(ref["item"]), str(ref["field"])))
    return out


def vault_secret_run(command: str, env: Any, workdir: str = "", timeout: Any = None,
                     reason: str = "") -> str:
    backend = _backend()
    if backend is None:
        return _err("1Password is not an enabled vault backend.")
    if not isinstance(command, str) or not command.strip():
        return _err("command is required")
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        # The approver (reviewer model, then the user) sees only the command and the field names;
        # the reason is what tells them why a secret is needed and what the command does with it.
        return _err("reason is required: say what this run is for, which task it serves and what "
                    "the command does with each field, so the approver can judge it")
    try:
        refs = _parse_env(env)
    except ValueError as exc:
        return _err(str(exc))
    try:
        limit = min(max(int(timeout or _DEFAULT_TIMEOUT), 1), _MAX_TIMEOUT)
    except (TypeError, ValueError):
        return _err("timeout must be a number of seconds")
    cwd = os.path.expanduser(workdir) if workdir else os.getcwd()
    if not os.path.isdir(cwd):
        return _err(f"workdir does not exist: {workdir}")

    from agent.vault_backends.base import UnlockRequired

    values: dict[str, str] = {}
    try:
        for name, item, field in refs:
            values[name] = backend.read_field(item, field)
    except UnlockRequired:
        return _locked(backend)
    except Exception as exc:
        return _err(redact(str(exc), list(values.values()))[:300])
    short = [name for name, value in values.items() if len(value.strip()) < _MIN_REDACT]
    if short:
        # Output redaction skips needles this short (they would shred ordinary text), so a value
        # this short could be echoed back verbatim. Refuse rather than return it.
        return _err(f"{', '.join(short)}: field is shorter than {_MIN_REDACT} characters, so its "
                    "value could not be redacted from the output; not run")

    from tools.environments.local import build_subprocess_env

    child_env = build_subprocess_env(extra=values)
    secrets = list(values.values())
    try:
        proc = subprocess.run(["bash", "-c", command], cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=limit, check=False)
        output, code = (proc.stdout or "") + (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        partial = partial.decode(errors="replace") if isinstance(partial, bytes) else partial
        output, code = partial + f"\n[timed out after {limit}s]", None
    finally:
        values.clear()
        del child_env
    output = redact(output, secrets)
    secrets.clear()
    if len(output) > _OUTPUT_LIMIT:
        output = output[: _OUTPUT_LIMIT // 2] + "\n[… output truncated …]\n" + output[-_OUTPUT_LIMIT // 2:]
    return json.dumps({"success": code == 0, "exit_code": code, "output": output,
                       "injected": [name for name, _, _ in refs]}, ensure_ascii=False)


# ── create / edit: write an item whose values come from a file or a command, never the chat ──

_MAX_FIELDS = 20
_MAX_SOURCE_BYTES = 64 * 1024
_SOURCE_TIMEOUT = 60
# A label like these is a secret even when the caller gives no type, so a literal is refused.
_SECRET_LABEL = re.compile(r"(?<![a-z])(pass(word|phrase)?|secret|token|key|credential|private|otp|pin)(?![a-z])",
                           re.IGNORECASE)
_CATEGORIES = {
    "api credential", "bank account", "credit card", "database", "driver license", "email account",
    "identity", "login", "membership", "outdoor license", "passport", "password", "reward program",
    "secure note", "server", "social security number", "software license", "wireless router", "ssh key",
}


def _scratch_root() -> str:
    from hermes_constants import get_scratch_dir

    return str(get_scratch_dir(prune=False))


def _relative_to_scratch(path: str) -> tuple[str, str]:
    """(resolved scratch root, *path* relative to it), matching the root by its own spelling or
    its resolved one (macOS /var → /private/var). Only the root is resolved: components below it
    are walked without following symlinks by _open_under."""
    raw = os.path.normpath(os.path.abspath(os.path.expanduser(_scratch_root())))
    resolved = os.path.realpath(raw)
    expanded = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    for base in (raw, resolved):
        if os.path.commonpath([expanded, base]) == base and expanded != base:
            return resolved, os.path.relpath(expanded, base)
    raise ValueError(f"{path}: value files must be under the Hermes scratch dir ({raw})")


def _open_under(root: str, rel: str) -> tuple[int, int, str]:
    """Open *rel* below *root* one component at a time with O_NOFOLLOW, so no symlink anywhere
    on the path (ancestor or leaf, even one swapped in mid-call) can redirect it. Returns
    (parent dir fd, file fd, leaf name); the caller closes both."""
    flags_dir = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parts = [p for p in rel.split(os.sep) if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError("bad path")
    dir_fd = os.open(root, flags_dir)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, flags_dir, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = nxt
        return dir_fd, os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd), parts[-1]
    except BaseException:
        os.close(dir_fd)
        raise


def _read_source_file(path: str) -> tuple[str, str]:
    """A value file: a regular file under the scratch dir, owned by this user, with no group or
    other access, reached without following any symlink. Returns (value, path relative to the
    scratch dir); errors name the path, never the content."""
    root, rel = _relative_to_scratch(path)
    try:
        dir_fd, fd, _ = _open_under(root, rel)
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        raise ValueError(f"{path}: cannot open ({reason}; symlinks are refused)") from None
    os.close(dir_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path}: not a regular file")
        if info.st_uid != os.getuid():
            raise ValueError(f"{path}: not owned by this user")
        if info.st_mode & 0o077:
            raise ValueError(f"{path}: mode {stat.S_IMODE(info.st_mode):04o}; chmod 600 it first")
        if info.st_size > _MAX_SOURCE_BYTES:
            raise ValueError(f"{path}: larger than {_MAX_SOURCE_BYTES} bytes")
        raw = os.read(fd, _MAX_SOURCE_BYTES + 1)
    finally:
        os.close(fd)
    try:
        return raw.decode("utf-8").removesuffix("\n").removesuffix("\r"), path
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not UTF-8 text") from None


def _unlink_source(path: str) -> None:
    """Delete a value file via the same no-follow walk that read it."""
    root, rel = _relative_to_scratch(path)
    dir_fd, fd, leaf = _open_under(root, rel)
    try:
        os.unlink(leaf, dir_fd=dir_fd)
    finally:
        os.close(fd)
        os.close(dir_fd)


def _no_op_credentials(env: dict[str, str]) -> dict[str, str]:
    """Drop every 1Password credential variable. The gateway's own unlock never enters the
    environment, but a session or token exported into the launch env would otherwise pass on."""
    return {k: v for k, v in env.items()
            if not (k.upper().startswith(("OP_SESSION", "OP_CONNECT_")) or k.upper().startswith("OP_SERVICE_ACCOUNT"))}


def _read_source_command(command: str, cwd: str) -> str:
    """A value command's stdout. It runs with the ordinary scrubbed child env minus any 1Password
    credential, so it can never write with the gateway's session; only its exit code is reported
    on failure."""
    from tools.environments.local import build_subprocess_env

    # stderr is never reported, so it is discarded; stdout is read with a hard cap, and the child
    # is killed as soon as it exceeds the cap or the timeout.
    proc = subprocess.Popen(["bash", "-c", command], cwd=cwd, env=_no_op_credentials(build_subprocess_env()),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks: list[bytes] = []
    size = 0
    deadline = time.monotonic() + _SOURCE_TIMEOUT
    try:
        os.set_blocking(proc.stdout.fileno(), False)
        while True:
            if time.monotonic() > deadline:
                raise ValueError(f"value command timed out after {_SOURCE_TIMEOUT}s")
            try:
                chunk = os.read(proc.stdout.fileno(), 65536)
            except BlockingIOError:
                time.sleep(0.02)
                continue
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_SOURCE_BYTES:
                raise ValueError(f"value command printed more than {_MAX_SOURCE_BYTES} bytes")
            chunks.append(chunk)
        code = proc.wait(timeout=max(deadline - time.monotonic(), 0.1))
    except subprocess.TimeoutExpired:
        raise ValueError(f"value command timed out after {_SOURCE_TIMEOUT}s") from None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if code != 0:
        raise ValueError(f"value command exited {code}")
    out = b"".join(chunks)
    chunks.clear()
    try:
        return out.decode("utf-8").removesuffix("\n").removesuffix("\r")
    except UnicodeDecodeError:
        raise ValueError("value command output is not UTF-8 text") from None


def _parse_fields(raw: Any) -> list[dict]:
    """Validate the field specs (no value is read yet)."""
    from agent.vault_backends.onepassword import PLAIN_TYPES, SECRET_TYPES

    if not isinstance(raw, list) or not raw or len(raw) > _MAX_FIELDS:
        raise ValueError(f"fields must list 1 to {_MAX_FIELDS} fields")
    out = []
    for spec in raw:
        if not isinstance(spec, dict) or not str(spec.get("label") or "").strip():
            raise ValueError("each field needs a label")
        label = str(spec["label"]).strip()
        kind = str(spec.get("type") or "").upper()
        if kind and kind not in PLAIN_TYPES | SECRET_TYPES:
            raise ValueError(f"{label}: unknown type {kind!r}")
        sources = [k for k in ("file", "command", "value") if spec.get(k) not in (None, "")]
        if len(sources) != 1:
            raise ValueError(f"{label}: give exactly one of file, command or value")
        source = sources[0]
        if source == "value":
            # The literal passed through the conversation, so it may only be a non-secret. A
            # secret-looking label is refused whatever type the caller claims for it.
            if not isinstance(spec["value"], str):
                raise ValueError(f"{label}: value must be a string")
            if kind in SECRET_TYPES or _SECRET_LABEL.search(label):
                raise ValueError(f"{label}: secret fields take a file or command, never an inline value")
        out.append({"label": label, "section": str(spec.get("section") or "").strip(), "type": kind,
                    "source": source, "ref": str(spec[source])})
    return out


def vault_secret_write(action: str, args: dict[str, Any]) -> str:
    backend = _backend()
    if backend is None:
        return _err("1Password is not an enabled vault backend.")
    if not getattr(backend, "writes_enabled", lambda: False)():
        return _err("1Password writes are off on this host (vault.onepassword.allow_write).")
    reason = str(args.get("reason") or "")
    if len(reason.strip()) < 10:
        return _err(f"reason is required: say what this {action} is for and where each value comes from, "
                    "so the approver can judge it")
    try:
        specs = _parse_fields(args.get("fields"))
    except ValueError as exc:
        return _err(str(exc))
    title = str(args.get("title") or "").strip()
    if action == "create":
        vault, category = str(args.get("vault") or "").strip(), str(args.get("category") or "").strip()
        if not (title and vault and category):
            return _err("create needs account, vault, title and category")
        if category.lower() not in _CATEGORIES:
            return _err(f"unknown category {category!r}")
    elif not str(args.get("item") or "").strip() or not title:
        # The title names the target in the approval prompt, and the backend refuses the edit
        # unless it matches the item, so the approver sees what is really changed.
        return _err("edit needs item (a handle from search) and title (that item's current title)")
    cwd = os.path.expanduser(str(args.get("workdir") or "")) or os.getcwd()
    if not os.path.isdir(cwd):
        return _err(f"workdir does not exist: {args.get('workdir')}")

    from agent.vault_backends.base import UnlockRequired

    secrets: list[str] = []
    updates: list[dict] = []
    try:
        for spec in specs:
            if spec["source"] == "file":
                value, spec["resolved"] = _read_source_file(spec["ref"])
            elif spec["source"] == "command":
                value = _read_source_command(spec["ref"], cwd)
            else:
                value = spec["ref"]
            if spec["source"] != "value":
                if len(value.strip()) < _MIN_REDACT:
                    raise ValueError(f"{spec['label']}: value is shorter than {_MIN_REDACT} characters, so it "
                                     "could not be redacted from errors; not written")
                secrets.append(value)
            updates.append({"label": spec["label"], "section": spec["section"], "type": spec["type"],
                            "value": value, "literal": spec["source"] == "value"})
        if action == "create":
            result = backend.create_item(account=str(args.get("account") or ""), vault=vault, title=title,
                                         category=category, fields=updates)
        else:
            result = backend.edit_item(item=str(args["item"]), title=title, fields=updates)
    except UnlockRequired:
        return _locked(backend)
    except Exception as exc:
        return _err(safe_error(str(exc), secrets, f"1Password {action} failed (details withheld: a value "
                               "has a line too short to redact)")[:300])
    finally:
        for update in updates:
            update["value"] = ""
        secrets.clear()
    deleted = []
    if args.get("delete_sources") is True:
        for spec in specs:
            if spec["source"] == "file":
                try:
                    _unlink_source(spec["resolved"])
                    deleted.append(spec["ref"])
                except OSError:
                    pass
    return json.dumps({"success": True, "action": action, **result,
                       "fields": [{"label": s["label"], "section": s["section"], "source": s["source"]} for s in specs],
                       "deleted_sources": deleted}, ensure_ascii=False)


def _handle(args: dict[str, Any], **_kwargs: Any) -> str:
    action = str(args.get("action") or "")
    backend = _backend()
    if backend is None:
        return _err("1Password is not an enabled vault backend.")
    from agent.vault_backends.base import UnlockRequired

    try:
        if action == "search":
            return json.dumps({"success": True, "items": backend.search_items(str(args.get("query") or ""))},
                              ensure_ascii=False)
        if action == "fields":
            item = str(args.get("item") or "")
            if not item:
                return _err("fields needs item (from search)")
            return json.dumps({"success": True, "item": item, "fields": backend.item_fields(item)},
                              ensure_ascii=False)
    except UnlockRequired:
        return _locked(backend)
    except Exception as exc:
        return _err(str(exc)[:300])
    if action == "run":
        return vault_secret_run(str(args.get("command") or ""), args.get("env"),
                                str(args.get("workdir") or ""), args.get("timeout"),
                                str(args.get("reason") or ""))
    if action in {"create", "edit"}:
        return vault_secret_write(action, args)
    return _err(f"Unknown action {action!r}; use search, fields, run, create or edit.")


VAULT_SECRET_SCHEMA = {
    "name": "vault_secret",
    "description": (
        "Use a field from any 1Password item (Secure Note, API Credential, SSH key, ...) in a shell command "
        "without the value entering the conversation. search (query: title text) → items; fields (item) → "
        "field ids and labels, never values; run (command, env: {VAR: {item, field}}, reason, optional "
        "workdir, timeout) → runs `bash -c command` with each field injected as $VAR and returns the exit code and "
        "output with the values redacted. create (account, vault, title, category, fields, reason) / edit "
        "(item, title, fields, reason) write an item when the host allows writes; each field is {label, section?, "
        "type?, and one of file (a 0600 file under the Hermes scratch dir), command (its stdout, run in "
        "workdir) or value (inline, non-secret fields only)}; the result is the item handle, title, vault, "
        "account and each field's label and source, never a value. A host enables writes "
        "(vault.onepassword.allow_write) only with an approval hook that asks the user for each one. "
        "There is no way to print a value: never try to echo, encode or "
        "write one out, and never ask the user for it. For website logins use browser_vault."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "fields", "run", "create", "edit"]},
            "query": {"type": "string", "description": "search: case-insensitive title substring."},
            "item": {"type": "string", "description": "fields, edit: an item handle from search."},
            "account": {"type": "string", "description": "create: 1Password account (required when several)."},
            "vault": {"type": "string", "description": "create: vault name or id."},
            "title": {"type": "string", "description": "create: the new item's title. edit (required): the "
                      "item's current title, shown to the approver and checked against the item."},
            "category": {"type": "string", "description": "create: item category, e.g. API Credential, "
                         "Secure Note, Password, Login."},
            "fields": {
                "type": "array",
                "description": "create, edit: fields to set. Never put a secret in value.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "section": {"type": "string"},
                        "type": {"type": "string", "description": "CONCEALED (default for file/command), "
                                 "STRING, EMAIL, URL, PHONE, DATE, MONTH_YEAR, MENU, OTP, SSHKEY, "
                                 "CREDIT_CARD_NUMBER."},
                        "file": {"type": "string", "description": "path of a 0600 file under the scratch dir "
                                 "holding the value (one trailing newline is dropped)."},
                        "command": {"type": "string", "description": "bash command whose stdout is the value."},
                        "value": {"type": "string", "description": "inline value, non-secret fields only."},
                    },
                    "required": ["label"],
                },
            },
            "delete_sources": {"type": "boolean", "description": "create, edit: delete the value files "
                               "after a successful write."},
            "command": {"type": "string", "description": "run: bash command that reads the variables. "
                        "Prefer the operations inline over a script path, so the approver sees them."},
            "reason": {"type": "string", "description": (
                "run, create, edit (required): why this needs the secret, the task it serves, and what the "
                "command does with each variable (or where each written value comes from). Shown to the "
                "reviewer and the user who approve the call.")},
            "env": {
                "type": "object",
                "description": "run: {VAR_NAME: {\"item\": handle, \"field\": field id}}.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}, "field": {"type": "string"}},
                    "required": ["item", "field"],
                },
            },
            "workdir": {"type": "string", "description": "run, and create/edit value commands: working "
                        "directory (default: current)."},
            "timeout": {"type": "integer", "description": f"run: seconds, max {_MAX_TIMEOUT}."},
        },
        "required": ["action"],
    },
}


from tools.registry import no_cache_check_fn, registry

registry.register(
    name="vault_secret",
    toolset="vault_secret",
    schema=VAULT_SECRET_SCHEMA,
    handler=_handle,
    check_fn=no_cache_check_fn(_check_available),
    emoji="🗝️",
)
