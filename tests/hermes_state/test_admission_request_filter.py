"""``list_session_admissions(request_id=...)``: native admission's legacy-retry check reads one request's rows
through ``session_admissions_request``, never the session's whole history, and returns exactly what filtering the
full scan by ``request_id`` returned."""
import json
import sqlite3
from contextlib import closing, contextmanager

import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB

_ROWS = 10_000


@pytest.fixture
def db(tmp_path):
    """s1: ``_ROWS`` terminal rows (``req-<n>``) plus a queued ``req-7`` retry under another principal; s2: a row that
    reuses ``req-7``."""
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('s1', source='cli')
        db.create_session('s2', source='cli')

        def write(conn):
            rows = [(f'a-{n}', f'req-{n}', 'p', 's1', json.dumps(['s1']), json.dumps({'text': f't{n}'}), f'd{n}',
                     'queue', 'terminal', epoch) for n in range(_ROWS)]
            rows.append(('a-retry', 'req-7', 'q', 's1', json.dumps(['s1']), json.dumps({'text': 'retry'}), 'dr',
                         'queue', 'queued', epoch))
            rows.append(('a-other', 'req-7', 'p', 's2', json.dumps(['s2']), json.dumps({'text': 'other'}), 'do',
                         'queue', 'queued', epoch))
            conn.executemany("""INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,
                lineage_json,payload_json,payload_digest,intent,status,owner_epoch) VALUES(?,?,?,?,?,?,?,?,?,?)""", rows)
        db._execute_write(write)
        yield db


@pytest.mark.parametrize('request_id', ['req-7', 'req-absent'], ids=['match', 'no-match'])
def test_filtered_read_matches_the_full_scan(db, request_id):
    full = [row for row in rt.list_session_admissions(db, session_id='s1', pending_only=False)
            if row['request_id'] == request_id]
    assert rt.list_session_admissions(db, session_id='s1', pending_only=False, request_id=request_id) == full
    assert len(full) == (2 if request_id == 'req-7' else 0)


def test_filtered_read_searches_the_request_index(db, monkeypatch, tmp_path):
    statements = []
    read = db._read_ctx

    @contextmanager
    def traced(*args, **kwargs):
        with read(*args, **kwargs) as conn:
            conn.set_trace_callback(statements.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
    monkeypatch.setattr(db, '_read_ctx', traced)
    rt.list_session_admissions(db, session_id='s1', pending_only=False, request_id='req-7')
    monkeypatch.undo()
    reads = [s for s in statements if 'session_admissions' in s and s.lstrip().upper().startswith('SELECT')]
    assert reads, statements
    with closing(sqlite3.connect(f"file:{tmp_path / 'state.db'}?mode=ro", uri=True)) as plan_conn:
        plans = [row[3] for sql in reads for row in plan_conn.execute('EXPLAIN QUERY PLAN ' + sql)]
    assert any('INDEX session_admissions_request' in plan for plan in plans), plans
    assert not any(plan.startswith('SCAN') for plan in plans), plans
