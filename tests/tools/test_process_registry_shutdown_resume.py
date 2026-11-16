"""Shutdown-killed resumable chat jobs survive the kill and resume on the next boot (RIG-10319)."""

import json
from types import SimpleNamespace

import pytest

from tools import process_registry as module
from tools.process_registry import ProcessRegistry, ProcessSession

REQUEST_ID = "req-10319"
ARGV = ["hermes", "chat", "-Q", "--provider", "litellm", "-m", "m", "--yolo",
        "--query-file", "prompt.md", "--request-id", REQUEST_ID]


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    return path


def _attempt(sid="proc_att", status="running", argv=None):
    return ProcessSession(
        id=sid, command="hermes chat -Q --yolo", task_id="t", owner_task_id="t",
        started_at=1.0, pid=4242, pid_scope="host", session_key="route",
        attempt_token="tok", attempt_status=status, resume_count=0,
        argv=list(ARGV if argv is None else argv), request_id=REQUEST_ID,
        output_log="")


def _stub_signal(monkeypatch, registry):
    monkeypatch.setattr(registry, "_signal_kill", lambda *_args, **_kw: None)
    monkeypatch.setattr(registry, "_post_kill_survivors", lambda _session: [])


# (a) Shutdown kill keeps the attempt's checkpoint entry as unknown.
def test_shutdown_killed_attempt_keeps_unknown_entry(monkeypatch, checkpoint):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    _stub_signal(monkeypatch, registry)
    registry._write_checkpoint()

    registry.kill_all(source="gateway_shutdown")

    entries = json.loads(checkpoint.read_text())
    assert [e["session_id"] for e in entries] == ["proc_att"]
    assert entries[0]["attempt_status"] == "unknown"
    assert entries[0]["argv"] == ARGV


# (b) A job that exits by itself is classified by its exit, not kept as resumable.
@pytest.mark.parametrize("exit_code", [0, 1])
def test_self_exit_is_not_kept_as_resumable(checkpoint, exit_code):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    registry._write_checkpoint()

    registry._finish_exited(session, exit_code)

    entries = json.loads(checkpoint.read_text())
    assert all(e.get("session_id") != "proc_att" for e in entries)


# (c) and (d) Boot relaunch. The chat session id comes from the creation receipt
# that stores this job's request id, in the profile's state.db.
@pytest.fixture
def chat_home(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    from hermes_state_keys import local_identity

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = local_identity("profile", "principal", REQUEST_ID)
    db.set_meta("gateway.local_policy.v1:" + sid, json.dumps({
        "profile_id": "profile", "principal_id": "principal", "request_id": REQUEST_ID,
        "session_id": sid, "route": "route"}))
    return sid


def _boot_entry(*, resume_count=0, argv=None):
    return {"session_id": "proc_att", "attempt_token": "tok", "attempt_status": "unknown",
            "argv": list(ARGV if argv is None else argv), "request_id": REQUEST_ID,
            "resume_count": resume_count, "command": "hermes chat -Q --yolo",
            "owner_task_id": "t", "task_id": "t", "session_key": "route", "started_at": 1.0}


def _spawn_recorder(monkeypatch, registry, spawned):
    monkeypatch.setattr(registry, "_spawn_job",
                        lambda session, argv, *a, **kw: spawned.append(argv) or SimpleNamespace(pid=5150))
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
    monkeypatch.setattr(registry, "_reader_loop", lambda session: None)


def test_boot_relaunches_unknown_entry_with_resume(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    spawned = []
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, spawned)

    registry.recover_from_checkpoint()

    assert f"--resume {chat_home}" in spawned[0][-1]
    assert json.loads(checkpoint.read_text())[0]["resume_count"] == 1
    assert [n["session_id"] for n in registry.pending_watchers] == ["proc_att"]


def test_boot_at_resume_cap_only_notices(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry(resume_count=3)]))
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, [])
    monkeypatch.setattr(registry, "_spawn_job",
                        lambda *a, **kw: pytest.fail("relaunched past the cap"))

    registry.recover_from_checkpoint()

    assert [n["session_id"] for n in registry.pending_watchers] == ["proc_att"]
    assert json.loads(checkpoint.read_text())[0]["resume_count"] == 3


def test_boot_never_relaunches_unadmitted_argv(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry(argv=["sh", "-c", "hermes chat -Q"])]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_spawn_job",
                        lambda *a, **kw: pytest.fail("relaunched an unadmitted argv"))

    registry.recover_from_checkpoint()

    assert [n["session_id"] for n in registry.pending_watchers] == ["proc_att"]
