"""A gateway that cannot parse a routing row (a platform whose plugin is not loaded, e.g. ``cotal``)
must not delete it on save (RIG-8538).

A full save rewrites the whole scope in state.db, so any row the loader cannot parse used to be
dropped. These tests seed such a row beside a live one, run the saves a gateway performs, and check
the raw row is still there, byte-identical.
"""
from __future__ import annotations

import json

import hermes_state
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


UNKNOWN_KEY = "agent:main:cotal:dm:peer-1"
UNKNOWN_JSON = json.dumps({
    "session_key": UNKNOWN_KEY,
    "session_id": "sess-cotal-1",
    "origin": {"platform": "cotal", "chat_id": "peer-1", "chat_type": "dm"},
    "platform": "cotal",
})


def _make_store(tmp_path, monkeypatch) -> SessionStore:
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


def _source(user_id: str = "user-1") -> SessionSource:
    return SessionSource(platform=Platform.LOCAL, chat_id="cli", chat_name="CLI",
                         chat_type="dm", user_id=user_id)


def _durable(store: SessionStore) -> dict[str, str]:
    return store._db.load_gateway_routing_entries(scope=store._routing_scope())


def test_full_save_keeps_unparseable_row_verbatim(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store._db.save_gateway_routing_entry(UNKNOWN_KEY, UNKNOWN_JSON, scope=store._routing_scope())
    entry = store.get_or_create_session(_source())

    store._save_entries()  # the whole-index rewrite that used to delete the cotal row

    durable = _durable(store)
    assert durable[UNKNOWN_KEY] == UNKNOWN_JSON
    assert entry.session_key in durable
    store._db.close()


def test_unparseable_row_survives_repeated_saves_and_a_restart(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store._db.save_gateway_routing_entry(UNKNOWN_KEY, UNKNOWN_JSON, scope=store._routing_scope())
    store.get_or_create_session(_source())
    store._save_entries()
    store._save_entries()
    store._db.close()

    restarted = _make_store(tmp_path, monkeypatch)
    restarted.get_or_create_session(_source())
    restarted._save_entries()

    assert _durable(restarted)[UNKNOWN_KEY] == UNKNOWN_JSON
    restarted._db.close()
