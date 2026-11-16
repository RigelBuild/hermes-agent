import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools.process_registry import ProcessRegistry


def test_live_child_is_adopted(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    child = subprocess.Popen(['sleep', '30'])
    try:
        registry = ProcessRegistry()
        entry = {'session_id': 'proc_live', 'attempt_token': 'old', 'attempt_status': 'running',
                 'pid': child.pid, 'host_start_time': registry._safe_host_start_time(child.pid)}
        path.write_text(json.dumps([entry]))
        assert registry.recover_from_checkpoint() == 1
        assert registry.get('proc_live') is not None
        assert not registry.pending_watchers
        assert json.loads(path.read_text())[0]['attempt_status'] == 'running'
    finally:
        child.terminate()
        child.wait()


@pytest.mark.parametrize('status', ['running', 'unknown'])
def test_recorded_exit_is_adopted(monkeypatch, tmp_path, status):
    from tools import process_registry as module
    from tools import process_registry_output_log as output_log
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    monkeypatch.setattr(output_log, 'output_dir', lambda: tmp_path)
    log = tmp_path / 'proc_finished.log'
    log.write_text('done')
    log.with_suffix('.exit').write_text('0')
    entry = {'session_id': 'proc_finished', 'attempt_token': 'old', 'attempt_status': status,
             'output_log': str(log), 'pid': 99999999}
    path.write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    adopted = []
    monkeypatch.setattr(registry, '_track_started', lambda session, *a: adopted.append(session))
    assert registry.recover_from_checkpoint() == 1
    assert adopted[0].id == 'proc_finished'
    assert not registry.pending_watchers


@pytest.mark.asyncio
@pytest.mark.parametrize('delivery,remaining', [(False, True), (True, False), (None, False)])
async def test_notice_settles_only_on_delivery(monkeypatch, tmp_path, delivery, remaining):
    from tools import process_registry as module
    from gateway.run_notifications import GatewayNotificationsMixin
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    registry = ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    path.write_text(json.dumps([{'session_id': 'proc_a', 'attempt_token': 'token',
                                 'attempt_status': 'pending', 'session_key': 'parent'}]))
    registry.recover_from_checkpoint()
    watcher = registry.pending_watchers[0]
    runner = SimpleNamespace(_enqueue_process_completion_notification=AsyncMock(return_value=delivery))
    await GatewayNotificationsMixin._watch_process(runner, watcher)
    assert bool(json.loads(path.read_text())) is remaining
    next_registry = ProcessRegistry()
    next_registry.recover_from_checkpoint()
    assert bool(next_registry.pending_watchers) is remaining
