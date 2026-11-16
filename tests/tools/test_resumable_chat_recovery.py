import json

import pytest

from tools.process_registry import ProcessRegistry


def test_interrupted_entries_are_reported_on_each_boot(monkeypatch, tmp_path):
    from tools import process_registry as module

    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    path.write_text(json.dumps([
        {'session_id': f'proc_{status}', 'attempt_token': status, 'attempt_status': status,
         'session_key': 'parent'} for status in ('pending', 'running', 'claimed')
    ]))
    for _ in range(2):
        registry = ProcessRegistry()
        monkeypatch.setattr(registry, '_spawn_job', lambda *a, **k: pytest.fail('must not spawn'))
        registry.recover_from_checkpoint()
        assert all(entry['attempt_status'] == 'unknown' for entry in json.loads(path.read_text()))
        assert len(registry.pending_watchers) == 3
        assert all(w['recovery_notice'] == 'outcome UNKNOWN, verify before retrying'
                   for w in registry.pending_watchers)


def test_recorded_exit_is_left_for_existing_recovery(monkeypatch, tmp_path):
    from tools import process_registry as module
    from tools import process_registry_output_log as output_log

    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    monkeypatch.setattr(output_log, 'output_dir', lambda: tmp_path)
    log = tmp_path / 'proc_finished.log'
    log.write_text('done')
    log.with_suffix('.exit').write_text('0')
    entry = {'session_id': 'proc_finished', 'attempt_token': 'old', 'attempt_status': 'running',
             'output_log': str(log), 'pid': 99999999}
    path.write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_track_started', lambda *a, **k: None)
    registry.recover_from_checkpoint()
    assert not registry.pending_watchers
    assert log.with_suffix('.exit').read_text() == '0'

