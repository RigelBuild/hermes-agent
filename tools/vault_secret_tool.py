"""vault_secret: use a 1Password field (any item category) without the value entering the conversation.

Actions:
- search: items whose title contains ``query`` across every configured account (metadata only).
- fields: one item's field ids, labels, sections and types; never values.
- run: read named fields server-side, inject each as an environment variable of one ``bash -c``
  child, and return its exit code and output with every injected value (and its common
  encodings) redacted. The value never reaches the model, argv or disk.

The tool cannot print a value: it has no read action, and ``run`` output is scrubbed. Hosts
gate ``run`` with a pre_tool_call hook (it carries the command and the item/field names for
the approval prompt); a host without one should not enable this tool.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
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


def vault_secret_run(command: str, env: Any, workdir: str = "", timeout: Any = None) -> str:
    backend = _backend()
    if backend is None:
        return _err("1Password is not an enabled vault backend.")
    if not isinstance(command, str) or not command.strip():
        return _err("command is required")
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
                                str(args.get("workdir") or ""), args.get("timeout"))
    return _err(f"Unknown action {action!r}; use search, fields or run.")


VAULT_SECRET_SCHEMA = {
    "name": "vault_secret",
    "description": (
        "Use a field from any 1Password item (Secure Note, API Credential, SSH key, ...) in a shell command "
        "without the value entering the conversation. search (query: title text) → items; fields (item) → "
        "field ids and labels, never values; run (command, env: {VAR: {item, field}}, optional workdir, "
        "timeout) → runs `bash -c command` with each field injected as $VAR and returns the exit code and "
        "output with the values redacted. There is no way to print a value: never try to echo, encode or "
        "write one out, and never ask the user for it. For website logins use browser_vault."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "fields", "run"]},
            "query": {"type": "string", "description": "search: case-insensitive title substring."},
            "item": {"type": "string", "description": "fields: an item handle from search."},
            "command": {"type": "string", "description": "run: bash command that reads the variables."},
            "env": {
                "type": "object",
                "description": "run: {VAR_NAME: {\"item\": handle, \"field\": field id}}.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}, "field": {"type": "string"}},
                    "required": ["item", "field"],
                },
            },
            "workdir": {"type": "string", "description": "run: working directory (default: current)."},
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
