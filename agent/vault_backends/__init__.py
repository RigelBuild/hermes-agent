"""Login backends for the browser credential vault.

The local Fernet store (``agent/vault_store.py``) is one backend; 1Password
(``op``) and Bitwarden Password Manager (``bw``) are the others. Every backend
hands the agent the same shape — opaque handle + login metadata — and resolves
the password server-side at fill time only. Handles are namespaced by backend
(``vault_…`` local, ``op:…``, ``bw:…``) so the browser tools need no schema
change to route to the right one.

External managers are locked until the user unlocks them for the current
session (``agent/vault_backends/unlock.py``); the master password is typed
into a masked prompt owned by the surface (CLI panel, Desktop dialog) and is
never a tool argument, never argv, never persisted.
"""

from agent.vault_backends import base as _base
from agent.vault_backends.base import LoginBackend, UnlockRequired, backend_for_handle, enabled_backends


def describe_ref(item: str, field: str | None = None) -> dict:
    """Resolve a vault handle to display metadata only; never read a secret value."""
    backend = next((candidate for candidate in _base.enabled_backends() if candidate.owns(str(item))), None)
    if backend is None:
        raise RuntimeError("vault item not found")

    details: dict = {}
    if backend.name == "onepassword":
        item_id = str(item).removeprefix(backend.prefix)
        data = backend.describe_item(str(item))
        title = data.get("title")
        if title:
            details["title"] = str(title)
        vault = data.get("vault")
        if isinstance(vault, dict) and vault.get("name"):
            details["vault"] = str(vault["name"])
        account = backend._any_accounts.get(item_id)
        if account:
            details["account"] = str(account)
        if field is not None:
            match = next((entry for entry in data.get("fields") or []
                          if isinstance(entry, dict) and str(entry.get("id") or "") == field), None)
            if match is None:
                raise RuntimeError("vault field not found")
            if match.get("label"):
                details["field_label"] = str(match["label"])

    if backend.name != "onepassword":
        meta = backend.get_meta(item)
        if meta is not None:
            if meta.label:
                details.setdefault("title", str(meta.label))
            if meta.identifier:
                details["username"] = str(meta.identifier)
            origins = list(meta.allowed_origins) or ([meta.origin] if meta.origin else [])
            if origins:
                details["allowed_origins"] = [str(origin) for origin in origins]
    if not details:
        raise RuntimeError("vault item not found")
    return details


__all__ = ["LoginBackend", "UnlockRequired", "backend_for_handle", "describe_ref", "enabled_backends"]
