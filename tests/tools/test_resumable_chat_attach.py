"""RIG-10342: a recorded background chat turn can be found and attached after a restart.

Red tests. Each names the contract it needs: a parent-minted ``input_id`` recorded in argv and
in the checkpoint, a hidden ``--input-id`` child option, a hidden ``--attach-input`` attach-only
path, ``prompt.receipt`` lookup by ``input_id``, and an attempt-token fence on the attach outcome.
The relaunch-argv case (T2) lives in test_process_registry_shutdown_resume.py.
"""

import asyncio
import inspect
import json
import re
import shlex
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tools import process_registry as module
from tools.process_registry import ProcessRegistry

REQUEST_ID = "req-10342"
INPUT_ID = "input-10342"
SESSION_ID = "stored-10342"
ATTEMPT_TOKEN = "tok-10342"
ATTACH_ARGS = {"resume": SESSION_ID, "attach_input": INPUT_ID, "quiet": True,
               "query": None, "oneshot": None, "oneshot_exit": False}


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    path = tmp_path / "processes.json"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    monkeypatch.setattr(module, "worker_output_log", lambda _: None)
    return path


def _launched(scope_argv):
    """The chat argv inside the login-shell wrapper (after the PATH prelude)."""
    return shlex.split(scope_argv[-1].splitlines()[-1].removeprefix("set +m; "))


def _pending_checkpoint_entry():
    return {"session_id": "proc_att", "attempt_token": ATTEMPT_TOKEN, "attempt_status": "unknown",
            "argv": ["hermes", "chat", "-Q", "--input-id", INPUT_ID], "request_id": REQUEST_ID,
            "input_id": INPUT_ID, "resume_count": 0, "command": "hermes chat -Q --yolo",
            "owner_task_id": "t", "task_id": "t", "session_key": "route", "started_at": 1.0}


class _FakeClient:
    """Records every rpc call. ``receipts`` maps input id to a receipt dict, or to an exception to raise."""

    def __init__(self, receipts=None):
        self.calls = []
        self.receipts = receipts or {}

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == "runtime.describe":
            return {}
        if method == "session.resume":
            return {"stored_session_id": SESSION_ID, "execution_generation": 3, "pending": [], "info": {}}
        if method == "prompt.receipt":
            key = params.get("input_id")
            assert key in self.receipts, f"prompt.receipt must be keyed by input_id, got {params}"
            outcome = self.receipts[key]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return {}

    def methods(self):
        return [method for method, _ in self.calls]


@asynccontextmanager
async def _connected(client):
    yield client


def _receipt(status, *, text="the answer"):
    result = {"final_response": text, "messages": [], "completed": True, "failed": False} \
        if status == "terminal" else None
    return {"status": status, "outcome": "completed" if status == "terminal" else None,
            "admission_id": "adm-10342", "result": result, "execution_generation": 3}


async def _attach(monkeypatch, client):
    from hermes_cli import gateway_chat
    from tools.process_resumable import _chat_parser
    options = {option for action in _chat_parser()._actions for option in action.option_strings}
    assert "--attach-input" in options, (
        "contract: the chat parser must define the hidden --attach-input option (not implemented)")
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: _connected(client))
    return await asyncio.wait_for(gateway_chat.run_gateway_chat(SimpleNamespace(**ATTACH_ARGS)), 5)


# T1. A resumable chat spawn records a non-empty input_id, distinct from request_id, in the
# pending checkpoint entry and in the launched argv, before the process starts.
def test_spawn_records_input_id_before_launch(monkeypatch, checkpoint):
    registry = ProcessRegistry()
    captured = []

    def spawn(session, argv, *args, **kwargs):
        captured.append((json.loads(checkpoint.read_text()), argv))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(registry, "_spawn_job", spawn)
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _: None)
    monkeypatch.setattr(registry, "_reader_loop", lambda session: None)

    session = registry.spawn_local("hermes chat --query-file prompt")

    entries, scope_argv = captured[0]
    pending = entries[0]
    assert pending["attempt_status"] == "pending"
    input_id = pending.get("input_id")
    assert isinstance(input_id, str) and input_id, (
        "T1: spawn_local must write a non-empty 'input_id' into the pending checkpoint entry")
    assert input_id != pending["request_id"], "T1: input_id must be minted apart from request_id"
    argv = _launched(scope_argv)
    assert "--input-id" in argv, "T1: spawn_local must append '--input-id <id>' to the launched argv"
    assert argv[argv.index("--input-id") + 1] == input_id, (
        "T1: the --input-id value in argv must equal the checkpoint input_id")
    assert session.input_id == input_id, (
        "T1: ProcessSession.input_id (new field, default '') must carry the minted id")


# T3. GatewayChatView sends the input id it was built with; without one it still sends a fresh
# 32-hex id. The second half passes today; the first half is red.
@pytest.mark.asyncio
async def test_submit_uses_the_input_id_it_was_given():
    from hermes_cli.gateway_chat_view import GatewayChatView

    assert "input_id" in inspect.signature(GatewayChatView.__init__).parameters, (
        "T3: GatewayChatView must accept an input_id (not implemented)")

    sent = []

    class Peer:
        async def rpc(self, method, **params):
            sent.append((method, params))
            return {"admission_id": "adm-10342"}

    def submitted():
        return [params for method, params in sent if method == "prompt.submit"][0]

    snapshot = {"stored_session_id": SESSION_ID, "execution_generation": 3, "pending": [], "info": {}}
    await GatewayChatView(Peer(), snapshot, quiet=True, input_id=INPUT_ID).submit("hello")
    assert submitted()["input_id"] == INPUT_ID, "T3: prompt.submit must carry the constructed input_id"

    sent.clear()
    await GatewayChatView(Peer(), snapshot, quiet=True).submit("hello")
    assert re.fullmatch(r"[0-9a-f]{32}", submitted()["input_id"]), (
        "T3: with no input id, submit must still send a fresh uuid4 hex")


# T4. Attach to a settled turn: print the committed reply, return the turn's exit code, and never
# submit or resolve.
@pytest.mark.asyncio
async def test_attach_returns_settled_result_without_submitting(monkeypatch, capsys):
    client = _FakeClient({INPUT_ID: _receipt("terminal")})

    code = await _attach(monkeypatch, client)

    assert code == 0, "T4: a settled, completed attach must return the turn_exit_code of 0"
    assert "the answer" in capsys.readouterr().out, "T4: the attach must print the committed reply"
    methods = client.methods()
    assert "prompt.receipt" in methods, "T4: attach must read the receipt by input_id"
    assert "prompt.submit" not in methods, f"T4: attach must never submit; saw {methods}"
    assert "prompt.resolve_unknown" not in methods, f"T4: attach must never resolve; saw {methods}"


# T5. An input id with no admission row takes the unknown-outcome path, and never submits.
@pytest.mark.asyncio
async def test_attach_unknown_input_id_takes_unknown_path(monkeypatch, capsys):
    from hermes_cli.gateway_chat_view import _UNKNOWN_OUTCOME
    from hermes_cli.gateway_client import GATEWAY_LOST_EXIT_CODE, GatewayRPCError

    client = _FakeClient({INPUT_ID: GatewayRPCError("not_found")})

    code = await _attach(monkeypatch, client)

    out = capsys.readouterr().out
    assert code == GATEWAY_LOST_EXIT_CODE or _UNKNOWN_OUTCOME in out, (
        "T5: an unknown input id must take the unknown-outcome path (not implemented)")
    assert "prompt.submit" not in client.methods(), "T5: attach must never submit"


# T6. An admission whose status is 'unknown' takes the same unknown path, and never submits.
@pytest.mark.asyncio
async def test_attach_unknown_status_takes_unknown_path(monkeypatch, capsys):
    from hermes_cli.gateway_chat_view import _UNKNOWN_OUTCOME
    from hermes_cli.gateway_client import GATEWAY_LOST_EXIT_CODE

    client = _FakeClient({INPUT_ID: _receipt("unknown")})

    code = await _attach(monkeypatch, client)

    out = capsys.readouterr().out
    assert code == GATEWAY_LOST_EXIT_CODE or _UNKNOWN_OUTCOME in out, (
        "T6: an unknown admission status must take the unknown-outcome path (not implemented)")
    assert "prompt.submit" not in client.methods(), "T6: attach must never submit"
    assert "prompt.resolve_unknown" not in client.methods(), "T6: attach must never resolve"


# T7. The attach outcome is fenced by session_id and attempt_token, the same way _update_attempt
# is. A wrong token leaves the file byte-identical; the right token updates the entry.
# ASSUMPTION: the method name ``record_attach_outcome`` is not in the brief. The fix slice must
# implement it under this name, or the driver must rename it here.
def test_attach_outcome_write_is_fenced_by_attempt_token(checkpoint):
    assert hasattr(ProcessRegistry, "record_attach_outcome"), (
        "T7: ProcessRegistry.record_attach_outcome(entry, outcome) is not implemented")
    registry = ProcessRegistry()
    checkpoint.write_text(json.dumps([_pending_checkpoint_entry()]))
    before = checkpoint.read_bytes()

    wrong = {"session_id": "proc_att", "attempt_token": "not-this-token"}
    assert registry.record_attach_outcome(wrong, {"attempt_status": "exited"}) is None
    assert checkpoint.read_bytes() == before, "T7: a wrong attempt_token must leave the entry byte-identical"

    right = {"session_id": "proc_att", "attempt_token": ATTEMPT_TOKEN}
    updated = registry.record_attach_outcome(right, {"attempt_status": "exited"})
    assert updated is not None and updated["attempt_status"] == "exited", (
        "T7: the matching session_id and attempt_token must update the entry")
    assert json.loads(checkpoint.read_text())[0]["attempt_status"] == "exited"


# T8. The authority's receipt accepts input_id as an alternative to admission_id, resolved with
# find_session_admission for the caller's principal and session. An unknown input id is not_found.
# Uses a real SessionDB and SessionAuthority, as tests/gateway/test_authority_principal.py does.
@pytest.mark.asyncio
async def test_receipt_resolves_by_input_id(tmp_path):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
                                      find_session_admission)

    assert "input_id" in inspect.signature(SessionAuthority.receipt).parameters, (
        "T8: SessionAuthority.receipt must accept input_id (not implemented)")

    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        epoch = begin_runtime_epoch(db, instance_id="current")
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned",
                                     instance_id="current", db=db, epoch=epoch)
        authority.sessions["s"] = LiveSession(None, "route")
        # submission() uses input_id as the admission's request_id when no submission_id is given.
        admit_session_input(db, epoch=epoch, principal_id="human", session_id="s",
                            request_id=INPUT_ID, payload={"text": "question"})
        row = find_session_admission(db, principal_id="human", session_id="s", request_id=INPUT_ID)
        assert row is not None

        actor = SessionActor("human", "owned", frozenset({"session:read", "session:submit"}), "viewer")
        ref = SessionRef("owned", "s")

        by_input = await authority.receipt(actor, ref, None, input_id=INPUT_ID)
        assert by_input.admission_id == row["admission_id"], (
            "T8: receipt(input_id=...) must resolve to the admission keyed by that input id")

        with pytest.raises(RuntimeStoreError, match="not_found"):
            await authority.receipt(actor, ref, None, input_id="no-such-input")


def _recorded_turn(tmp_path, *, settle=None, claim=True):
    """A state.db holding the recorded turn: queued, started (``claim``), or settled with ``settle``."""
    from hermes_state import SessionDB
    from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                      settle_session_input)
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_keys import local_identity

    sid = local_identity("profile", "principal", REQUEST_ID)
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(sid, source="cli")
        epoch = begin_runtime_epoch(db, instance_id="current")
        admit_session_input(db, epoch=epoch, principal_id="principal", session_id=sid,
                            request_id=INPUT_ID, payload={"text": "question"})
        if claim or settle:
            claimed = claim_session_input(db, epoch=epoch, session_id=sid)
        if settle:
            outcome, result = settle
            settle_session_input(db, epoch=epoch, admission_id=claimed["admission_id"],
                                 generation=claimed["generation"], outcome=outcome, result=result)
        db.set_meta(POLICY_PREFIX + sid, json.dumps({
            "profile_id": "profile", "principal_id": "principal", "request_id": REQUEST_ID,
            "session_id": sid, "route": "route"}))
    return sid


# B1 (F3). Boot recovery finds a recorded turn that settled while the job was orphaned: the user is
# told its outcome and reply, not "UNKNOWN", and the entry stays until that notice is delivered.
def test_boot_reports_settled_turn_result(monkeypatch, checkpoint, tmp_path):
    _recorded_turn(tmp_path, settle=("completed", {"final_response": "the answer", "completed": True}))
    checkpoint.write_text(json.dumps([_pending_checkpoint_entry() | {"attempt_status": "running"}]))
    registry = ProcessRegistry()
    registry._recover_interrupted_attempts(checkpoint_path=checkpoint)

    notices = [w["recovery_notice"] for w in registry.pending_watchers if w.get("recovery_notice")]
    assert len(notices) == 1, notices
    assert "UNKNOWN" not in notices[0] and "completed" in notices[0] and "the answer" in notices[0], notices
    assert [e["session_id"] for e in json.loads(checkpoint.read_text())] == ["proc_att"], (
        "B1: the entry leaves the checkpoint only when the result notice is delivered")


# F1. A job the gateway's own shutdown stopped settles its turn 'interrupted'. Boot must leave it to
# the shutdown relaunch path (RIG-10528 open), never attach to or report the stopped turn.
def test_shutdown_killed_entry_with_interrupted_turn_keeps_relaunch(checkpoint, tmp_path):
    from tools.process_registry_checkpoint import SHUTDOWN_RESUME_REASON
    _recorded_turn(tmp_path, settle=("interrupted", {"interrupted": True,
                                                     "interrupt_message": "Gateway shutting down"}))
    entry = _pending_checkpoint_entry() | {"attempt_status": "unknown", "resume_reason": SHUTDOWN_RESUME_REASON}
    checkpoint.write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    registry._recover_interrupted_attempts(checkpoint_path=checkpoint, attach=True)

    entries = json.loads(checkpoint.read_text())
    assert entries and entries[0].get("resume_reason") == SHUTDOWN_RESUME_REASON, (
        "shutdown-killed entry was dropped before _relaunch_shutdown_killed could claim it")
    assert registry.pending_relaunches == []
    assert [w["recovery_notice"] for w in registry.pending_watchers] == ["outcome UNKNOWN, verify before retrying"]


# F2. A shutdown relaunch submits a fresh continuation turn, so its claim clears the recorded input id:
# a later orphaned boot must not attach to the original turn.
def test_shutdown_relaunch_claim_clears_recorded_input_id(checkpoint, tmp_path):
    from tools.process_registry_checkpoint import SHUTDOWN_RESUME_REASON
    entry = _pending_checkpoint_entry() | {"attempt_status": "unknown", "resume_reason": SHUTDOWN_RESUME_REASON}
    checkpoint.write_text(json.dumps([entry]))
    registry = ProcessRegistry()
    claimed = registry._claim_shutdown_resume(checkpoint, entry, SESSION_ID)

    assert claimed is not None and claimed["input_id"] == ""
    assert json.loads(checkpoint.read_text())[0]["input_id"] == ""
    # The relaunched attempt, orphaned, finds no recorded turn: the unknown notice, never the old turn.
    _recorded_turn(tmp_path, settle=("completed", {"final_response": "old answer", "completed": True}))
    registry._update_attempt(checkpoint, claimed, lambda item: item.update(attempt_status="running") or True)
    registry._recover_interrupted_attempts(checkpoint_path=checkpoint, attach=True)
    assert [w["recovery_notice"] for w in registry.pending_watchers] == ["outcome UNKNOWN, verify before retrying"]


# F4. Boot recovery attaches to a recorded turn still queued or running: after READY it spawns the
# attach-only chat (never a query, never a resolve), and no unknown notice is sent.
@pytest.mark.parametrize("claim", [False, True], ids=["queued", "started"])
def test_boot_attaches_to_a_live_recorded_turn(monkeypatch, checkpoint, tmp_path, claim):
    sid = _recorded_turn(tmp_path, claim=claim)
    checkpoint.write_text(json.dumps([_pending_checkpoint_entry() | {"attempt_status": "running"}]))
    registry = ProcessRegistry()
    spawned = []
    monkeypatch.setattr(registry, "_spawn_job",
                        lambda session, argv, *a, **kw: spawned.append(argv) or SimpleNamespace(pid=5150))
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
    monkeypatch.setattr(registry, "_reader_loop", lambda session: None)

    registry.recover_from_checkpoint(resume_shutdown_killed=True)
    assert spawned == [] and len(registry.pending_relaunches) == 1, "attach spawns only after READY"
    assert not any(w.get("recovery_notice") for w in registry.pending_watchers)
    count, notices = registry.spawn_queued_relaunches()

    assert count == 1
    argv = _launched(spawned[0])
    assert argv[argv.index("--attach-input") + 1] == INPUT_ID
    assert argv[argv.index("--resume") + 1] == sid
    assert "--query" not in argv and "--discard-unknown" not in argv and "--input-id" not in argv, argv
    assert not any(n.get("recovery_notice") for n in notices)
    entry = json.loads(checkpoint.read_text())[0]
    # The spawned attempt keeps the recorded input id, so a later orphaned boot attaches to the same turn.
    assert entry["input_id"] == INPUT_ID and entry["resume_count"] == 0


# F4. An attach claim whose spawn never happens (startup failed before READY) is given back as unknown.
def test_unspawned_attach_claim_is_given_back(monkeypatch, checkpoint, tmp_path):
    _recorded_turn(tmp_path)
    checkpoint.write_text(json.dumps([_pending_checkpoint_entry() | {"attempt_status": "running"}]))
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: None)
    registry.recover_from_checkpoint(resume_shutdown_killed=True)

    assert registry.give_back_pending_relaunches() == 1
    entry = json.loads(checkpoint.read_text())[0]
    assert entry["attempt_status"] == "unknown" and "attach_input" not in entry


# B2. Boot recovery with no settled turn for the recorded input id keeps the unknown notice and never submits.
def test_boot_keeps_unknown_notice_when_turn_is_missing(monkeypatch, checkpoint, tmp_path):
    from hermes_state import SessionDB
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_keys import local_identity

    sid = local_identity("profile", "principal", REQUEST_ID)
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(sid, source="cli")
        db.set_meta(POLICY_PREFIX + sid, json.dumps({
            "profile_id": "profile", "principal_id": "principal", "request_id": REQUEST_ID,
            "session_id": sid, "route": "route"}))

    checkpoint.write_text(json.dumps([_pending_checkpoint_entry() | {"attempt_status": "running"}]))
    registry = ProcessRegistry()
    registry._recover_interrupted_attempts(checkpoint_path=checkpoint)

    notices = [w for w in registry.pending_watchers if w.get("recovery_notice")]
    assert len(notices) == 1 and "UNKNOWN" in notices[0]["recovery_notice"], (
        "B2: a missing turn keeps the unknown notice")
    assert json.loads(checkpoint.read_text())[0]["attempt_status"] == "unknown"
