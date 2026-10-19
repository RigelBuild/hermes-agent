"""1Password Login items as a vault backend (``op`` CLI).

Unlock: ``op signin --raw`` with the master password on stdin (desktop-app
integration or account-level auth) mints an ``OP_SESSION_<account>`` token.
``vault.onepassword.accounts`` lists several accounts (shorthand, sign-in address or
ID) that share the master password: one unlock signs in to each, all or nothing, and
listings and reads span them all. ``account`` (one account) still works.
A configured service-account token skips the prompt entirely (headless).
Desktop-app mode (``vault.onepassword.desktop_app``) mints no token: the 1Password
app authorises every ``op`` call itself, so an unlock only proves the app answers
and opens this session's gate; a lock closes it again.
List: ``op item list --categories Login --format json`` → title, urls,
username, vault. Any item (Secure Note, API Credential, ...) can also be searched by
title and its field labels listed (``search_items``/``item_fields``, metadata only),
and one field read server-side by item and field id (``read_field``) for
``vault_secret``'s run action, which injects it into a child process. Resolve: ``op item get <id> --vault <vault-id> ...`` (service
accounts reject an item read without a vault query).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from agent.secret_sources.base import run_cli
from agent.secret_sources.onepassword import _OP_ENV_ALLOWLIST, _scrub, find_op
from agent.vault_backends.base import LoginBackend, UnlockRequired, run_with_stdin_secret
from agent.vault_backends import unlock as _unlock
from agent.vault_store import VaultItemMeta, normalize_origin

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0
# Stands in for a session token in desktop-app mode; never passed to op.
_DESKTOP_APP_MARKER = "desktop-app"


class OnePasswordLoginBackend(LoginBackend):
    name = "onepassword"
    display_name = "1Password"
    prefix = "op:"
    needs_unlock = True

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        from agent.secret_scope import get_secret
        env_name = str(self.cfg.get("service_account_token_env") or "OP_SERVICE_ACCOUNT_TOKEN")
        self._service_token = get_secret(env_name, "") or ""
        self._item_vaults: dict[str, str] = {}  # item id -> vault id from the latest ``item list``
        self._item_accounts: dict[str, str] = {}  # item id -> account it was listed from ("" = default)
        self._any_accounts: dict[str, str] = {}  # same, for search_items / item_fields / read_field
        self.desktop_app = bool(self.cfg.get("desktop_app")) and not self._service_token

    # ── auth ────────────────────────────────────────────────────────────────

    def _op(self) -> Path:
        op = find_op(str(self.cfg.get("binary_path") or ""))
        if op is None:
            raise RuntimeError("1Password CLI (op) not found — install it or set vault.onepassword.binary_path")
        return op

    def _accounts(self) -> list[str]:
        """Configured accounts in order; ``[""]`` means op's default (single) account."""
        raw = self.cfg.get("accounts")
        if isinstance(raw, str):
            raw = [raw]
        accounts = [str(a).strip() for a in raw or [] if str(a).strip()] if isinstance(raw, (list, tuple)) else []
        if not accounts and (single := str(self.cfg.get("account") or "").strip()):
            accounts = [single]
        return list(dict.fromkeys(accounts)) or [""]

    def _tokens(self, stored: Optional[str]) -> dict[str, str]:
        """Per-account session tokens from the stored session. One account stores op's bare token;
        several store a JSON object keyed by account."""
        if not stored:
            return {}
        accounts = self._accounts()
        if len(accounts) == 1:
            return {accounts[0]: stored}
        try:
            data = json.loads(stored)
        except ValueError:
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _env(self, session_token: Optional[str], account: Optional[str] = None) -> dict[str, str]:
        from agent.secret_scope import get_secret
        env = {k: os.environ[k] for k in _OP_ENV_ALLOWLIST if k in os.environ and not k.startswith("OP_CONNECT_")}
        # Connect credentials outrank OP_SERVICE_ACCOUNT_TOKEN inside op, so they must come from the
        # profile's own secret scope like the service token does — never from the launch environment.
        for k in ("OP_CONNECT_HOST", "OP_CONNECT_TOKEN"):
            if v := get_secret(k, ""):
                env[k] = v
        env["NO_COLOR"] = "1"
        if account is None:
            account = self._accounts()[0]
        if account:
            env["OP_ACCOUNT"] = account
        if self._service_token:
            env["OP_SERVICE_ACCOUNT_TOKEN"] = self._service_token
        elif session_token and not self.desktop_app:
            # op signin --raw prints the bare token; the env var name carries the account shorthand,
            # which op also accepts as plain OP_SESSION for the default account.
            env[f"OP_SESSION_{account}" if account else "OP_SESSION"] = session_token
        return env

    def is_unlocked(self) -> bool:
        return bool(self._service_token) or _unlock.is_unlocked(self.name)

    def unlock(self, master_password: str) -> None:
        """Mint a session token from the master password (consumed on stdin, never argv)."""
        generation = _unlock.begin_unlock(self.name)
        if self.desktop_app:
            self._unlock_desktop_app(generation)
            return
        accounts = self._accounts()
        tokens: dict[str, str] = {}
        for account in accounts:
            cmd = [str(self._op()), "signin", "--raw"]
            if account:
                cmd += ["--account", account]
            proc = run_with_stdin_secret(cmd, env=self._env(None, account), secret=master_password,
                                         timeout=_TIMEOUT, label="op")
            token = (proc.stdout or "").strip()
            if proc.returncode != 0 or not token:
                where = f" ({account})" if account else ""
                raise RuntimeError(f"1Password unlock failed{where}: "
                                   f"{_scrub(proc.stderr or '')[:200] or 'no session token'}")
            tokens[account] = token
        stored = tokens[accounts[0]] if len(accounts) == 1 else json.dumps(tokens)
        if not _unlock.store_session_token(self.name, stored, generation):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _unlock_desktop_app(self, generation: int) -> None:
        """Open the session gate once the 1Password app authorises ``op`` (the app shows its own
        approval prompt). No token exists in this mode, so the gate holds a fixed marker."""
        proc = run_cli([str(self._op()), "whoami"], env=self._env(None), timeout=_TIMEOUT, label="op",
                       timeout_message="op timed out waiting for the 1Password app", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError("1Password app did not authorise the CLI (unlock the app and enable "
                               f"Settings > Developer > Integrate with 1Password CLI): {_scrub(proc.stderr or '')[:200]}")
        if not _unlock.store_session_token(self.name, _DESKTOP_APP_MARKER, generation):
            raise RuntimeError("1Password was locked while unlocking; try again")

    def _run(self, *args: str, account: Optional[str] = None) -> str:
        if account is None:
            account = self._accounts()[0]
        token = None if self._service_token else _unlock.get_session_token(self.name)
        if not self._service_token and not token:
            raise UnlockRequired(self)
        if not self._service_token and not self.desktop_app:
            token = self._tokens(token).get(account)
            if not token:
                _unlock.lock(self.name)
                raise UnlockRequired(self)
        proc = run_cli([str(self._op()), *args], env=self._env(token, account), timeout=_TIMEOUT, label="op",
                       timeout_message="op timed out", stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            err = _scrub(proc.stderr or "")
            # A service account never holds a session, so an auth failure there is not a lock to undo:
            # surface op's own error instead of an unlock prompt that reports "already unlocked".
            if not self._service_token and any(w in err.lower() for w in ("session", "sign in", "not signed in")):
                _unlock.lock(self.name)
                raise UnlockRequired(self)
            raise RuntimeError(f"op failed: {err[:200]}")
        return proc.stdout or ""

    # ── backend contract ───────────────────────────────────────────────────
    def list_items(self) -> list[VaultItemMeta]:
        if not self.is_unlocked():
            return []
        listed: list[tuple[str, dict]] = []
        for account in self._accounts():
            raw = json.loads(self._run("item", "list", "--categories", "Login", "--format", "json",
                                       account=account) or "[]")
            listed += [(account, item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        out: list[VaultItemMeta] = []
        self._item_vaults = {}
        self._item_accounts = {}
        for account, item in listed:
            vault = item.get("vault") or {}
            if item.get("id"):
                self._item_vaults[str(item["id"])] = str(vault.get("id") or "") if isinstance(vault, dict) else ""
                self._item_accounts[str(item["id"])] = account
            urls = [str(u["href"]) for u in item.get("urls") or [] if isinstance(u, dict) and u.get("href")]
            origins = _all_origins(urls)
            if not origins:
                continue
            username = str(item.get("additional_information") or "").strip() or None
            out.append(VaultItemMeta(
                id=f"{self.prefix}{item.get('id')}", kind="login", label=str(item.get("title") or origins[0]),
                origin=origins[0], created_at=str(item.get("created_at") or ""),
                identifier_type="username" if username else None, identifier=username,
                allowed_origins=_web_origins(origins)))
        return out

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return next((m for m in self.list_items() if m.id == handle), None)

    def _item_selector(self, handle: str) -> list[str]:
        """``op item get`` target for a listed login. A service account must name the item's vault
        (``--vault``) and ``item list`` reports it, so the vault comes from the same listing that
        proved the handle is a Login item; a handle nothing listed is refused before any secret read."""
        item_id = handle[len(self.prefix):]
        if item_id not in self._item_vaults:
            self.list_items()
        if item_id not in self._item_vaults:
            raise RuntimeError("1Password login not found; list logins again")
        vault_id = self._item_vaults[item_id]
        return [item_id, "--vault", vault_id] if vault_id else [item_id]

    def _item_account(self, handle: str) -> str:
        return self._item_accounts.get(handle[len(self.prefix):], self._accounts()[0])

    def resolve_password(self, handle: str) -> str:
        selector = self._item_selector(handle)
        return self._run("item", "get", *selector, "--fields", "label=password", "--reveal",
                         account=self._item_account(handle)).rstrip("\r\n")

    def resolve_otp(self, handle: str) -> Optional[str]:
        # `--otp` mints the current TOTP from the item's one-time-password field; items without one error out.
        try:
            selector = self._item_selector(handle)
            code = self._run("item", "get", *selector, "--otp", account=self._item_account(handle)).strip()
        except Exception:
            return None
        return code if code.isdigit() else None

    # ── any item: metadata search, field labels, one server-side field read ─────
    def search_items(self, query: str) -> list[dict]:
        """Items of every category whose title contains *query*; metadata only."""
        if not self.is_unlocked():
            raise UnlockRequired(self)
        needle = query.strip().lower()
        out: list[dict] = []
        for account in self._accounts():
            raw = json.loads(self._run("item", "list", "--format", "json", account=account) or "[]")
            for item in raw if isinstance(raw, list) else []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                title = str(item.get("title") or "")
                if needle and needle not in title.lower():
                    continue
                vault = item.get("vault") or {}
                self._any_accounts[str(item["id"])] = account
                out.append({"item": f"{self.prefix}{item['id']}", "title": title,
                            "category": str(item.get("category") or ""),
                            "vault": str(vault.get("name") or "") if isinstance(vault, dict) else "",
                            "account": account or "default"})
        return out

    def _get_item(self, item: str) -> dict:
        item_id = item.removeprefix(self.prefix)
        if not item_id or item_id.startswith("-"):
            raise RuntimeError("1Password item not found; search again")
        accounts = [self._any_accounts[item_id]] if item_id in self._any_accounts else self._accounts()
        last = ""
        for account in accounts:
            try:
                raw = self._run("item", "get", item_id, "--reveal", "--format", "json", account=account)
            except RuntimeError as exc:
                last = str(exc)
                continue
            data = json.loads(raw or "{}")
            if isinstance(data, dict) and data.get("id"):
                self._any_accounts[item_id] = account
                return data
        raise RuntimeError("1Password item not found" + (f": {last[:120]}" if last else ""))

    def item_fields(self, item: str) -> list[dict]:
        """Field ids and labels of one item. Values are dropped here and never returned."""
        data = self._get_item(item)
        sections = {str(sec.get("id") or ""): str(sec.get("label") or "")
                    for sec in data.get("sections") or [] if isinstance(sec, dict)}
        out = []
        for field in data.get("fields") or []:
            if not isinstance(field, dict) or not field.get("id"):
                continue
            section = field.get("section") or {}
            sid = str(section.get("id") or "") if isinstance(section, dict) else ""
            out.append({"field": str(field["id"]), "label": str(field.get("label") or ""),
                        "section": sections.get(sid, ""), "type": str(field.get("type") or ""),
                        "has_value": bool(field.get("value"))})
        return out

    def read_field(self, item: str, field: str) -> str:
        """One field's value, server-side only: the caller injects it and never returns it."""
        data = self._get_item(item)
        match = next((f for f in data.get("fields") or []
                      if isinstance(f, dict) and str(f.get("id") or "") == field), None)
        if match is None:
            raise RuntimeError("field not found on that item; list its fields again")
        value = match.get("value")
        if not isinstance(value, str) or not value:
            raise RuntimeError("that field is empty")
        return value


def _web_origins(origins: list[str]) -> tuple:
    """Fill targets are browser pages, so app URIs (``androidapp://`` etc.) never
    widen the fill set; an item whose only URI is an app URI keeps its single
    (unfillable-from-a-page) origin exactly as before."""
    web = tuple(o for o in origins if o.startswith(("http://", "https://")))
    return web or (origins[0],)


def _all_origins(urls: list[str]) -> list[str]:
    """Every normalized origin saved on the item, deduped, order preserved.

    A 1Password Login item can carry several websites; each of them is a place the
    user told 1Password the credential belongs, so all of them are valid fill targets.
    """
    out: list[str] = []
    for u in urls:
        try:
            origin = normalize_origin(u)
        except Exception:
            continue
        if origin not in out:
            out.append(origin)
    return out
