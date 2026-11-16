import json
import os
import shlex
import threading
from types import SimpleNamespace

from tools.process_resumable import resumable_chat_argv
from tools.process_registry import ProcessRegistry


@pytest.mark.parametrize(("command", "eligible"), [
    ("hermes chat", True),
    ("/usr/bin/hermes chat --query-file 'my prompt' --model example", True),
    ("python -m hermes_cli chat --query-file=input", False),
    ("hermes chat --api S", False),
    ("hermes chat --api-k=S", False),
    ("hermes chat -Qq S", False),
    ("hermes chat -vqS", False),
    ("hermes chat --base-url https://user:pw@host/v1", False),
    ("hermes chat --request-id user", False),
    ("hermes chat --resume existing", False),
    ("hermes chat -c existing", False),
    ("hermes chat --create-if-missing", False),
    ("hermes chat --tui", False),
    ("hermes chat --tui-native", False),
    *((f"hermes chat --query-file {c}prompt", False) for c in "~*?[{#!"),
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
    entry = {"session_id": "pending", "attempt_token": "token", "attempt_status": "pending",
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
    written = json.loads(path.read_text())
    assert [item["session_id"] for item in written] == ["pending"]
    assert written[0]["attempt_status"] == "pending"
    assert written[0]["attempt_owner_pid"] == os.getpid()
    assert "attempt_owner_started_at" in written[0]


def test_live_foreign_owner_pending_attempt_is_not_classified_as_interrupted(monkeypatch, tmp_path):
    import subprocess
    import sys

    from pathlib import Path
    from tools import process_registry as module

    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    root = Path(__file__).resolve().parents[2]
    owner_program = "\n".join((
        "import sys",
        "from pathlib import Path",
        "import tools.process_registry as module",
        "from tools.process_registry import ProcessRegistry",
        "module.CHECKPOINT_PATH = Path(sys.argv[1])",
        "registry = ProcessRegistry()",
        "registry._write_resumable_entry({'session_id': 'pending', 'attempt_token': 'token',",
        "    'attempt_status': 'pending', 'argv': ['hermes', 'chat'],",
        "    'request_id': 'request', 'resume_count': 0})",
        "print('ready', flush=True)",
        "sys.stdin.readline()",
    ))
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_program, str(path)], cwd=root,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert owner.stdout.readline().strip() == "ready"
        registry = ProcessRegistry()
        assert registry.recover_from_checkpoint() == 0
        assert json.loads(path.read_text())[0]["attempt_status"] == "pending"
        assert not registry.pending_watchers

        owner.stdin.write("exit\n")
        owner.stdin.flush()
        owner.wait(timeout=5)

        assert registry.recover_from_checkpoint() == 0
        assert json.loads(path.read_text())[0]["attempt_status"] == "unknown"
        assert [notice["session_id"] for notice in registry.pending_watchers] == ["pending"]
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)


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
    assert entry[0]["attempt_status"] == "pending" and entry[0]["attempt_token"]
    assert entry[0]["request_id"] == session.request_id
    assert session.request_id in spawned[-1]
    running = json.loads(path.read_text())[0]
    assert running["attempt_status"] == "running" and running["pid"] == 12345
    assert running["argv"] == entry[0]["argv"]
    assert running["attempt_owner_pid"] == os.getpid()
    assert "attempt_owner_started_at" in running


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


def test_failed_write_falls_back(monkeypatch, tmp_path):
    from tools import process_registry as module
    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    registry = ProcessRegistry()
    def fail(_):
        raise OSError("write failed")
    monkeypatch.setattr(registry, "_write_resumable_entry", fail)
    captured = []
    monkeypatch.setattr(registry, "_spawn_job", lambda session, argv, *a, **kw: captured.append(argv) or SimpleNamespace(pid=42))
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _: None)
    monkeypatch.setattr(registry, "_track_started", lambda *a: None)
    session = registry.spawn_local("hermes chat --query-file prompt")
    assert not session.attempt_token
    assert captured[0][-1] == "set +m; hermes chat --query-file prompt"


def test_failed_spawn_removes_pending(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    registry = ProcessRegistry()
    def fail(*args, **kwargs):
        assert json.loads(path.read_text())[0]["attempt_status"] == "pending"
        raise OSError("spawn failed")
    monkeypatch.setattr(registry, "_spawn_job", fail)
    with pytest.raises(OSError, match="spawn failed"):
        registry.spawn_local("hermes chat --query-file prompt")
    assert json.loads(path.read_text()) == []


def test_pty_receives_request_id(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    registry = ProcessRegistry()
    def spawn(session, command, env):
        entry = json.loads(path.read_text())[0]
        assert entry["attempt_status"] == "pending"
        assert entry["request_id"] in command
        return session
    monkeypatch.setattr(registry, "_spawn_local_pty", spawn)
    assert registry.spawn_local("hermes chat --query-file prompt", use_pty=True).attempt_token


def test_corrupt_checkpoint_disables_admission(monkeypatch, tmp_path):
    from tools import process_registry as module
    path = tmp_path / "processes.json"
    path.write_text('{broken')
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    with pytest.raises(ValueError):
        ProcessRegistry()._write_resumable_entry({"session_id": "x"})
