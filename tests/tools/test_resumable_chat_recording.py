import json
import shlex
import threading
from types import SimpleNamespace

import pytest

from tools.process_resumable import resumable_chat_argv
from tools.process_registry import ProcessRegistry


@pytest.mark.parametrize(("command", "eligible"), [
    ("hermes chat", True),
    ("/usr/bin/hermes chat --query-file 'my prompt' --model example", True),
    ("python -m hermes_cli chat --query-file=input", True),
    ("hermes chat --api-key secret", False),
    ("hermes chat --api-key=secret", False),
    ("hermes chat -qsecret", False),
    ("hermes chat --query secret", False),
    ("hermes chat --query=secret", False),
    ("hermes chat | cat", False),
    ("hermes chat; echo x", False),
    ("hermes chat $VAR", False),
    ("hermes chat `date`", False),
    ("hermes chat > output", False),
    ("hermes chat 'unterminated", False),
    ("other chat", False),
    ("hermes run", False),
])
def test_resumable_chat_argv(command, eligible):
    assert resumable_chat_argv(command) == (shlex.split(command) if eligible else None)


def test_foreign_writer_preserves_pending_attempt(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    first, second = ProcessRegistry(), ProcessRegistry()
    entry = {"session_id": "pending", "attempt_token": "token", "status": "pending",
             "argv": ["hermes", "chat"], "request_id": "request", "resume_count": 0}
    first._write_resumable_entry(entry)
    barrier = threading.Barrier(2)
    def write(registry):
        barrier.wait()
        registry._write_checkpoint()
    threads = [threading.Thread(target=write, args=(registry,)) for registry in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert json.loads(path.read_text()) == [entry]


def test_spawn_records_pending_before_launch(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    registry = ProcessRegistry()
    captured = []
    def spawn(session, argv, *args, **kwargs):
        captured.append((json.loads(path.read_text()), argv))
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(registry, "_spawn_job", spawn)
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _: None)
    monkeypatch.setattr(registry, "_reader_loop", lambda session: None)
    session = registry.spawn_local("hermes chat --query-file prompt")
    entry, spawned = captured[0]
    assert entry[0]["status"] == "pending" and entry[0]["attempt_token"]
    assert entry[0]["request_id"] == session.request_id
    assert "--request-id" in entry[0]["argv"]
    assert session.request_id in spawned[-1]
    running = json.loads(path.read_text())[0]
    assert running["status"] == "running" and running["pid"] == 12345
    assert running["argv"] == entry[0]["argv"]


def test_ineligible_spawn_is_unchanged(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    registry = ProcessRegistry()
    captured = []
    def spawn(session, argv, *args, **kwargs):
        captured.append(argv)
        assert not path.exists()
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(registry, "_spawn_job", spawn)
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _: None)
    monkeypatch.setattr(registry, "_track_started", lambda session, *args: None)
    session = registry.spawn_local("hermes chat -q secret")
    assert not session.attempt_token
    assert captured[0][-1] == "set +m; hermes chat -q secret"
