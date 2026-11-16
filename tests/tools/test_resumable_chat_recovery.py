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
        assert len(registry.pending_watchers) == 1
        assert registry.pending_watchers[0]['check_interval'] == 5
        assert json.loads(path.read_text())[0]['attempt_status'] == 'running'
    finally:
        child.terminate()
        child.wait()

@pytest.mark.parametrize('systemd_unit', ['', 'hermes-worker-test.scope'])
def test_adopted_child_death_queues_unknown(monkeypatch, tmp_path, systemd_unit):
    from tools import process_registry as module
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    child = subprocess.Popen(['sleep', '30'])
    try:
        registry = ProcessRegistry()
        path.write_text(json.dumps([{'session_id': 'proc_live', 'attempt_token': 'token',
            'attempt_status': 'running', 'pid': child.pid, 'systemd_unit': systemd_unit,
            'host_start_time': registry._safe_host_start_time(child.pid)}]))
        assert registry.recover_from_checkpoint() == 1
        assert len(registry.pending_watchers) == 1
        registry.pending_watchers.clear()
        child.terminate()
        child.wait()
        result = registry.get('proc_live')
        assert (result is not None) == bool(systemd_unit)
        assert json.loads(path.read_text())[0]['attempt_status'] == 'unknown'
        assert len(registry.pending_watchers) == 1
        registry._write_checkpoint()
        assert json.loads(path.read_text())[0]['attempt_status'] == 'unknown'
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()


@pytest.mark.asyncio
async def test_scoped_adoption_watcher_only_delivers_unknown(monkeypatch, tmp_path):
    from tools import process_registry as module
    from gateway.run_notifications import GatewayNotificationsMixin
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    registry = ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    child = subprocess.Popen(['sleep', '30'])  # noqa: ASYNC220 - test owns and reaps this fixture
    try:
        path.write_text(json.dumps([{'session_id': 'proc_live', 'attempt_token': 'token',
            'attempt_status': 'running', 'pid': child.pid, 'systemd_unit': 'hermes-worker-test.scope',
            'host_start_time': registry._safe_host_start_time(child.pid), 'notify_on_complete': True,
            'watcher_interval': 1}]))
        assert registry.recover_from_checkpoint() == 1
        watcher = registry.pending_watchers.pop()
        watcher['check_interval'] = 0
        child.terminate()
        child.wait()
        runner = SimpleNamespace(_load_background_notifications_mode=lambda: 'concise',
            _enqueue_process_completion_notification=AsyncMock(return_value=True),
            _send_watcher_message=AsyncMock(), _launching_turn_active=AsyncMock(return_value=False),
            _build_process_completion_event=lambda *args: {},
            _format_process_final_message=lambda *args: 'finished')
        await GatewayNotificationsMixin._watch_process(runner, watcher)
        assert runner._enqueue_process_completion_notification.await_count == 0
        assert runner._send_watcher_message.await_count == 0
        assert len(registry.pending_watchers) == 1
        assert registry.pending_watchers[0]['recovery_notice']
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait()


@pytest.mark.asyncio
async def test_death_only_watcher_suppresses_normal_completion(monkeypatch):
    from tools import process_registry as module
    from gateway.run_notifications import GatewayNotificationsMixin
    session = SimpleNamespace(exited=True, attempt_unknown=False, owner_task_id='', task_id='')
    monkeypatch.setattr(module.process_registry, 'get', lambda _sid: session)
    runner = SimpleNamespace(_load_background_notifications_mode=lambda: 'concise',
        _enqueue_process_completion_notification=AsyncMock(), _send_watcher_message=AsyncMock())
    await GatewayNotificationsMixin._watch_process(
        runner, {'session_id': 'proc_a', 'check_interval': 0, 'death_only': True})
    assert runner._enqueue_process_completion_notification.await_count == 0
    assert runner._send_watcher_message.await_count == 0


@pytest.mark.parametrize('status,pid,fate', [
    ('pending', None, 'gone'), ('running', 123, 'gone'),
    ('claimed', None, 'gone'), ('running', 123, 'reused'),
])
def test_interrupted_attempt_never_respawns(monkeypatch, tmp_path, status, pid, fate):
    from tools import process_registry as module
    path = tmp_path / 'processes.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', path)
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_detached_host_fate', lambda *args: fate)
    monkeypatch.setattr(registry, '_spawn_job', lambda *args, **kwargs: pytest.fail('spawned'))
    path.write_text(json.dumps([{'session_id': 'proc_a', 'attempt_token': 'token',
                                 'attempt_status': status, 'pid': pid}]))
    assert registry.recover_from_checkpoint() == 0
    assert json.loads(path.read_text())[0]['attempt_status'] == 'unknown'
    assert len(registry.pending_watchers) == 1


@pytest.mark.asyncio
async def test_notice_settles_original_profile(monkeypatch, tmp_path):
    from tools import process_registry as module
    from gateway.run_notifications import GatewayNotificationsMixin
    secondary = tmp_path / 'secondary.json'
    primary = tmp_path / 'primary.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', secondary)
    registry = ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    secondary.write_text(json.dumps([{'session_id': 'proc_a', 'attempt_token': 'token',
                                      'attempt_status': 'pending'}]))
    primary.write_text('[]')
    registry.recover_from_checkpoint()
    watcher = registry.pending_watchers[0]
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', primary)
    runner = SimpleNamespace(_enqueue_process_completion_notification=AsyncMock(return_value=True))
    await GatewayNotificationsMixin._watch_process(runner, watcher)
    assert json.loads(secondary.read_text()) == []
    assert json.loads(primary.read_text()) == []


def test_adopted_child_uses_original_checkpoint(monkeypatch, tmp_path):
    from tools import process_registry as module
    secondary = tmp_path / 'secondary.json'
    primary = tmp_path / 'primary.json'
    monkeypatch.setattr(module, 'CHECKPOINT_PATH', secondary)
    child = subprocess.Popen(['sleep', '30'])
    try:
        registry = ProcessRegistry()
        secondary.write_text(json.dumps([{'session_id': 'proc_live', 'attempt_token': 'token',
            'attempt_status': 'running', 'pid': child.pid,
            'host_start_time': registry._safe_host_start_time(child.pid)}]))
        primary.write_text('[]')
        assert registry.recover_from_checkpoint() == 1
        registry.pending_watchers.clear()
        monkeypatch.setattr(module, 'CHECKPOINT_PATH', primary)
        child.terminate()
        child.wait()
        registry.get('proc_live')
        assert json.loads(secondary.read_text())[0]['attempt_status'] == 'unknown'
        assert json.loads(primary.read_text()) == []
        assert registry.pending_watchers[0]['checkpoint_path'] == str(secondary)
    finally:
        if child.poll() is None:
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
    assert len(registry.pending_watchers) == 1


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
