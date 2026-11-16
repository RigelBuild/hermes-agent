"""Shutdown-killed resumable chat jobs survive the kill and resume on the next boot."""

import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import process_registry as module
from tools.process_registry import ProcessRegistry, ProcessSession
from hermes_cli.gateway_client import GATEWAY_LOST_EXIT_CODE
from tools.process_registry_checkpoint import RESUME_NOTE, SHUTDOWN_RESUME_REASON

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


def _spawn_after_ready(registry):
    """The bootstrap's post-READY step: spawn off the loop, then arm the notices."""
    spawned, notices = registry.spawn_queued_relaunches()
    for notice in notices:
        registry._arm_watcher(notice)
    return spawned


def _boot(registry):
    """Boot recovery, then the post-READY spawn of the relaunches it claimed."""
    recovered = registry.recover_from_checkpoint(resume_shutdown_killed=True)
    _spawn_after_ready(registry)
    return recovered


# (a) Shutdown kill keeps the attempt's checkpoint entry, marked as a gateway shutdown kill.
def test_shutdown_killed_attempt_keeps_marked_entry(monkeypatch, checkpoint):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    _stub_signal(monkeypatch, registry)
    registry._write_checkpoint()

    registry.kill_all(source="gateway_shutdown")

    entries = json.loads(checkpoint.read_text())
    assert [e["session_id"] for e in entries] == ["proc_att"]
    assert entries[0]["attempt_status"] == "unknown"
    assert entries[0]["resume_reason"] == SHUTDOWN_RESUME_REASON
    assert entries[0]["argv"] == ARGV


# (b) Guard (passes before this change): a job that exits by itself outside a shutdown is
# classified by its exit, not kept as resumable.
@pytest.mark.parametrize("exit_code", [0, 1])
def test_self_exit_is_not_kept_as_resumable(checkpoint, exit_code):
    """Guard for existing behaviour, not a red-green test."""
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    registry._write_checkpoint()

    registry._finish_exited(session, exit_code)

    entries = json.loads(checkpoint.read_text())
    assert all(e.get("session_id") != "proc_att" for e in entries)


# A chat whose gateway drops while it stops exits GATEWAY_LOST_EXIT_CODE: before or after kill_all, it stays resumable.
@pytest.mark.parametrize("order", ["exit_then_kill_all", "kill_all_then_exit"])
def test_failed_exit_during_shutdown_is_kept_resumable(monkeypatch, checkpoint, order):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    _stub_signal(monkeypatch, registry)
    registry._write_checkpoint()
    registry.shutting_down = True

    if order == "exit_then_kill_all":
        registry._finish_exited(session, GATEWAY_LOST_EXIT_CODE)
        registry.kill_all(source="gateway_shutdown")
    else:
        other = _attempt(sid="proc_other")
        other.attempt_token = "tok2"
        registry._running[other.id] = other
        registry.kill_all(source="gateway_shutdown", exclude_ids=frozenset({session.id}))
        registry._finish_exited(session, GATEWAY_LOST_EXIT_CODE)

    entry = next(e for e in json.loads(checkpoint.read_text()) if e["session_id"] == "proc_att")
    assert entry["resume_reason"] == SHUTDOWN_RESUME_REASON and entry["attempt_status"] == "unknown"


# A clean exit or a task failure (model, tool, bad argument) during shutdown is the job's own outcome.
@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_own_exit_during_shutdown_is_not_kept(checkpoint, exit_code):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    registry._write_checkpoint()
    registry.shutting_down = True

    registry._finish_exited(session, exit_code)

    assert all(e.get("session_id") != "proc_att" for e in json.loads(checkpoint.read_text()))


def test_lost_gateway_exit_outside_shutdown_is_not_kept(checkpoint):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    registry._write_checkpoint()

    registry._finish_exited(session, GATEWAY_LOST_EXIT_CODE)

    assert all(e.get("session_id") != "proc_att" for e in json.loads(checkpoint.read_text()))


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
    db.close()
    return sid


def _boot_entry(*, resume_count=0, argv=None, marked=True, status="unknown"):
    entry = {"session_id": "proc_att", "attempt_token": "tok", "attempt_status": status,
             "argv": list(ARGV if argv is None else argv), "request_id": REQUEST_ID,
             "resume_count": resume_count, "command": "hermes chat -Q --yolo",
             "owner_task_id": "t", "task_id": "t", "session_key": "route", "started_at": 1.0}
    if marked:
        entry["resume_reason"] = SHUTDOWN_RESUME_REASON
    return entry


def _launched(scope_argv):
    """The chat argv inside the login-shell wrapper (after the PATH prelude)."""
    return shlex.split(scope_argv[-1].splitlines()[-1].removeprefix("set +m; "))


def _spawn_recorder(monkeypatch, registry, spawned):
    monkeypatch.setattr(registry, "_spawn_job",
                        lambda session, argv, *a, **kw: spawned.append(argv) or SimpleNamespace(pid=5150))
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
    monkeypatch.setattr(registry, "_reader_loop", lambda session: None)


def _forbid_spawn(monkeypatch, registry, why):
    monkeypatch.setattr(registry, "_spawn_job", lambda *a, **kw: pytest.fail(why))


def _notices(registry):
    return [n["recovery_notice"] for n in registry.pending_watchers if n.get("recovery_notice")]


def test_boot_relaunches_marked_entry_with_resume(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    spawned = []
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, spawned)

    assert _boot(registry) == 1

    argv = _launched(spawned[0])
    assert argv[argv.index("--resume") + 1] == chat_home
    entry = json.loads(checkpoint.read_text())[0]
    assert entry["resume_count"] == 1 and entry["argv"] == ARGV and "resume_reason" not in entry
    assert entry["resume_session_id"] == chat_home
    assert _notices(registry) == ["resumed after gateway restart (1/3)"]


def test_relaunch_swaps_query_source_for_continuation_note(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    spawned = []
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, spawned)

    _boot(registry)

    argv = _launched(spawned[0])
    assert "--query-file" not in argv and "prompt.md" not in argv
    assert argv[argv.index("--query") + 1] == RESUME_NOTE
    assert "--discard-unknown" in argv


def test_second_shutdown_and_boot_resumes_again_with_one_resume_flag(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    first = ProcessRegistry()
    _spawn_recorder(monkeypatch, first, [])
    _boot(first)
    session = first.get("proc_att")
    session.process = None

    gateway = ProcessRegistry()
    gateway._running[session.id] = session
    _stub_signal(monkeypatch, gateway)
    gateway.kill_all(source="gateway_shutdown")

    spawned = []
    second = ProcessRegistry()
    _spawn_recorder(monkeypatch, second, spawned)
    monkeypatch.setattr(second, "_detached_host_fate", lambda *_a: "gone")
    assert _boot(second) == 1

    entry = json.loads(checkpoint.read_text())[0]
    assert entry["resume_count"] == 2 and entry["argv"] == ARGV
    assert _launched(spawned[0]).count("--resume") == 1


def test_boot_at_resume_cap_only_notices(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry(resume_count=3)]))
    registry = ProcessRegistry()
    _forbid_spawn(monkeypatch, registry, "relaunched past the cap")

    _boot(registry)

    assert _notices(registry) == ["outcome UNKNOWN, verify before retrying"]
    assert json.loads(checkpoint.read_text())[0]["resume_count"] == 3


def test_dead_running_entry_without_marker_is_not_relaunched(monkeypatch, checkpoint, chat_home):
    entry = _boot_entry(marked=False, status="running")
    entry["pid"] = 999999
    checkpoint.write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_detached_host_fate", lambda *_a: "gone")
    _forbid_spawn(monkeypatch, registry, "relaunched an unmarked entry")

    _boot(registry)

    assert _notices(registry) == ["outcome UNKNOWN, verify before retrying"]


def test_boot_never_relaunches_unadmitted_argv(monkeypatch, checkpoint, chat_home):
    argv = [*ARGV[:-2], "--api-key", "sk-inline", "--request-id", REQUEST_ID]
    checkpoint.write_text(json.dumps([_boot_entry(argv=argv)]))
    registry = ProcessRegistry()
    _forbid_spawn(monkeypatch, registry, "relaunched an unadmitted argv")

    _boot(registry)

    assert _notices(registry) == ["outcome UNKNOWN, verify before retrying"]


def test_no_receipt_means_no_spawn(monkeypatch, checkpoint):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    registry = ProcessRegistry()
    _forbid_spawn(monkeypatch, registry, "relaunched without a chat session")

    _boot(registry)

    entry = json.loads(checkpoint.read_text())[0]
    assert entry["resume_reason"] == SHUTDOWN_RESUME_REASON and entry["resume_count"] == 0


def test_managed_worker_recover_does_not_relaunch(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    registry = ProcessRegistry()
    _forbid_spawn(monkeypatch, registry, "a managed worker relaunched")

    registry.recover_from_checkpoint()

    assert json.loads(checkpoint.read_text())[0]["resume_reason"] == SHUTDOWN_RESUME_REASON


def test_racing_recoverers_spawn_once(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    first, rival = ProcessRegistry(), ProcessRegistry()
    _forbid_spawn(monkeypatch, rival, "the second recoverer relaunched too")
    spawns = []

    def spawn(session, argv, *a, **kw):
        _boot(rival)  # the rival reads the checkpoint after this recoverer's claim
        spawns.append(argv)
        return SimpleNamespace(pid=5150)

    monkeypatch.setattr(first, "_spawn_job", spawn)
    monkeypatch.setattr(first, "_reader_loop", lambda session: None)
    _boot(first)

    assert len(spawns) == 1


def test_spawn_failure_restores_entry_and_keeps_recovering(monkeypatch, checkpoint, chat_home):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        registry = ProcessRegistry()
        live = {"session_id": "proc_live", "pid": child.pid, "pid_scope": "host", "command": "sleep",
                "host_start_time": registry._safe_host_start_time(child.pid), "started_at": 1.0}
        checkpoint.write_text(json.dumps([_boot_entry(), live]))

        def boom(*_a, **_kw):
            raise OSError("hermes not found")

        monkeypatch.setattr(registry, "_spawn_job", boom)
        _boot(registry)

        assert registry.get("proc_live") is not None
        entry = next(e for e in json.loads(checkpoint.read_text()) if e["session_id"] == "proc_att")
        assert entry["attempt_status"] == "unknown" and entry["resume_count"] == 0
        assert entry["resume_reason"] == SHUTDOWN_RESUME_REASON
        assert "outcome UNKNOWN, verify before retrying" in _notices(registry)
    finally:
        child.kill()
        child.wait()


@pytest.mark.asyncio
async def test_resumed_notice_keeps_the_relaunched_entry(monkeypatch, checkpoint, chat_home):
    from gateway.run_notifications import GatewayNotificationsMixin

    checkpoint.write_text(json.dumps([_boot_entry()]))
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, [])
    monkeypatch.setattr(module, "process_registry", registry)
    _boot(registry)
    notice = next(n for n in registry.pending_watchers if n.get("recovery_notice"))
    sent = []

    async def enqueue(text, _event):
        sent.append(text)
        return True

    await GatewayNotificationsMixin._watch_process(
        SimpleNamespace(_enqueue_process_completion_notification=enqueue), notice)

    assert sent == ["Background job proc_att: resumed after gateway restart (1/3)"]
    assert [e["session_id"] for e in json.loads(checkpoint.read_text())] == ["proc_att"]


def test_resumed_job_completion_is_not_deduplicated_by_the_notice(monkeypatch, checkpoint, chat_home):
    from gateway.run_notifications import GatewayNotificationsMixin
    identity = GatewayNotificationsMixin._completion_delivery_identity

    checkpoint.write_text(json.dumps([{**_boot_entry(), "notify_on_complete": True}]))
    registry = ProcessRegistry()
    _spawn_recorder(monkeypatch, registry, [])
    _boot(registry)
    notice = next(n for n in registry.pending_watchers if n.get("recovery_notice"))
    delivered = {identity({"type": "completion", **notice})}

    session = registry.get("proc_att")
    session.process = None
    registry._finish_exited(session, 0)
    completion = registry.completion_queue.get_nowait()

    assert completion["started_at"] != 1.0
    assert identity(completion) is not None and identity(completion) not in delivered


def test_no_relaunch_spawns_before_ready(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    registry = ProcessRegistry()
    spawned = []
    _spawn_recorder(monkeypatch, registry, spawned)

    assert registry.recover_from_checkpoint(resume_shutdown_killed=True) == 1
    assert spawned == [] and len(registry.pending_relaunches) == 1

    assert _spawn_after_ready(registry) == 1
    assert len(spawned) == 1 and registry.pending_relaunches == []


@pytest.mark.parametrize("source", [["--query-f", "prompt.md"], ["--query-f=prompt.md"], ["--query-file=prompt.md"],
                                    ["-qtext"], ["-q", "text"], ["--image", "pic.png", "-q", "text"]])
def test_launch_argv_drops_abbreviated_query_sources(source):
    from tools.process_registry_checkpoint import resume_launch_argv
    argv = resume_launch_argv(["hermes", "chat", "-Q", *source, "--yolo"], "sid")
    assert argv == ["hermes", "chat", "-Q", "--yolo", "--resume", "sid", "--discard-unknown", "--query", RESUME_NOTE]


def test_unspawned_claims_are_given_back(monkeypatch, checkpoint, chat_home):
    checkpoint.write_text(json.dumps([_boot_entry()]))
    registry = ProcessRegistry()
    _forbid_spawn(monkeypatch, registry, "spawned on a failed startup")
    registry.recover_from_checkpoint(resume_shutdown_killed=True)

    assert registry.give_back_pending_relaunches() == 1

    entry = json.loads(checkpoint.read_text())[0]
    assert entry["resume_reason"] == SHUTDOWN_RESUME_REASON and entry["resume_count"] == 0
    assert entry["attempt_status"] == "unknown" and registry.pending_relaunches == []


# The client's exit code decides: a gateway-stop interrupt exits 76 and is kept, a user Stop exits 130 and is not.
@pytest.mark.parametrize("exit_code, kept", [(GATEWAY_LOST_EXIT_CODE, True), (130, False)])
def test_interrupt_exit_during_shutdown(checkpoint, exit_code, kept):
    registry = ProcessRegistry()
    session = _attempt()
    registry._running[session.id] = session
    registry._write_checkpoint()
    registry.shutting_down = True

    registry._finish_exited(session, exit_code)

    assert any(e.get("session_id") == "proc_att" for e in json.loads(checkpoint.read_text())) is kept
