"""A gateway that cannot parse a routing row (a platform whose plugin is not loaded, e.g. ``cotal``)
must not delete it on save (RIG-8538).

A full save rewrites the whole scope in state.db, so any row the loader cannot parse used to be
dropped. These tests seed such a row beside a live one, run the saves a gateway performs, and check
the raw row is still there, byte-identical.
"""
from __future__ import annotations

import json
import time

import hermes_state
import pytest
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


UNKNOWN_KEY = "agent:main:cotal:dm:peer-1"
UNKNOWN_JSON = json.dumps(
    {
        "session_key": UNKNOWN_KEY,
        "session_id": "sess-cotal-1",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "origin": {"platform": "cotal", "chat_id": "peer-1", "chat_type": "dm"},
        "platform": "local",
    }
)


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


def _assert_unknown_platform_is_unparsed():
    assert SessionStore._routing_entry_from_json(UNKNOWN_KEY, UNKNOWN_JSON) is None


@pytest.mark.parametrize(
    ("key", "entry_json", "expected"),
    [
        (UNKNOWN_KEY, UNKNOWN_JSON, True),
        (UNKNOWN_KEY, "{bad json", False),
        (UNKNOWN_KEY, json.dumps([]), False),
        (
            UNKNOWN_KEY,
            json.dumps({**json.loads(UNKNOWN_JSON), "session_id": "../unsafe"}),
            False,
        ),
        (
            UNKNOWN_KEY,
            json.dumps({**json.loads(UNKNOWN_JSON), "session_key": "agent:../unsafe"}),
            False,
        ),
        ("different-key", UNKNOWN_JSON, False),
        (
            UNKNOWN_KEY,
            json.dumps(
                {
                    **json.loads(UNKNOWN_JSON),
                    "origin": {"platform": Platform.LOCAL.value},
                }
            ),
            False,
        ),
        (
            UNKNOWN_KEY,
            json.dumps({**json.loads(UNKNOWN_JSON), "origin": {"platform": ""}}),
            False,
        ),
    ],
    ids=(
        "unknown-platform",
        "malformed-json",
        "non-object",
        "unsafe-session-id",
        "unsafe-session-key",
        "key-mismatch",
        "known-platform",
        "empty-platform",
    ),
)
def test_preservable_unparsed_row_predicate(key, entry_json, expected):
    assert SessionStore._is_preservable_unparsed_row(key, entry_json) is expected


def test_full_save_keeps_unparseable_row_verbatim(tmp_path, monkeypatch):
    _assert_unknown_platform_is_unparsed()
    store = _make_store(tmp_path, monkeypatch)
    store._db.save_gateway_routing_entry(UNKNOWN_KEY, UNKNOWN_JSON, scope=store._routing_scope())
    entry = store.get_or_create_session(_source())

    store._save_entries()

    durable = _durable(store)
    assert durable[UNKNOWN_KEY] == UNKNOWN_JSON
    assert entry.session_key in durable
    store._db.close()


def test_unparseable_row_survives_repeated_saves_and_a_restart(tmp_path, monkeypatch):
    _assert_unknown_platform_is_unparsed()
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


def test_malformed_json_does_not_abort_valid_route_save(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store._db._execute_write(
        lambda conn: conn.execute(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES (?, ?, ?, ?)",
            (store._routing_scope(), "malformed", "{bad json", time.time()),
        )
    )
    entry = store.get_or_create_session(_source())

    store._save_entries()

    durable = _durable(store)
    assert "malformed" not in durable
    assert entry.session_key in durable
    store._db.close()


def test_memory_route_replaces_unparsed_durable_copy(tmp_path, monkeypatch):
    _assert_unknown_platform_is_unparsed()
    store = _make_store(tmp_path, monkeypatch)
    store._db.save_gateway_routing_entry(UNKNOWN_KEY, UNKNOWN_JSON, scope=store._routing_scope())
    entry = store.get_or_create_session(_source())
    del store._entries[entry.session_key]
    entry.session_key = UNKNOWN_KEY
    store._entries[UNKNOWN_KEY] = entry

    store._save_entries()

    durable = _durable(store)
    assert json.loads(durable[UNKNOWN_KEY])["session_id"] == entry.session_id
    assert durable[UNKNOWN_KEY] != UNKNOWN_JSON
    store._db.close()


def test_retired_unparsed_route_is_not_preserved(tmp_path, monkeypatch):
    _assert_unknown_platform_is_unparsed()
    store = _make_store(tmp_path, monkeypatch)
    store._db.retire_gateway_routes(["sess-cotal-1"], namespace="main")
    store._db._execute_write(
        lambda conn: conn.execute(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES (?, ?, ?, ?)",
            (store._routing_scope(), UNKNOWN_KEY, UNKNOWN_JSON, time.time()),
        )
    )

    store._save_entries()

    assert UNKNOWN_KEY not in _durable(store)
    store._db.close()


def test_unparsed_route_update_from_second_db_handle_survives_save(
    tmp_path, monkeypatch
):
    _assert_unknown_platform_is_unparsed()
    store = _make_store(tmp_path, monkeypatch)
    peer = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    store._db.save_gateway_routing_entry(UNKNOWN_KEY, UNKNOWN_JSON, scope=store._routing_scope())
    store.get_or_create_session(_source())
    updated = json.loads(UNKNOWN_JSON)
    updated["display_name"] = "updated by peer"
    updated_json = json.dumps(updated)
    replace = store._db.replace_gateway_routing_entries

    def update_then_replace(entries, **kwargs):
        peer.save_gateway_routing_entry(UNKNOWN_KEY, updated_json, scope=store._routing_scope())
        return replace(entries, **kwargs)

    monkeypatch.setattr(store._db, "replace_gateway_routing_entries", update_then_replace)

    store._save_entries()

    assert _durable(store)[UNKNOWN_KEY] == updated_json
    peer.close()
    store._db.close()
