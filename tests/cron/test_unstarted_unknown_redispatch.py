"""A claimed occurrence whose owner dies before the run starts must run once after restart, and
every ``unknown`` execution the restart sweep writes must open a ``cron_incidents`` row.

The gateway stalled, the tick claimed the slot (execution row + ``fire_claim``), then the shutdown
watchdog exited the process before ``started_at``. The restart sweep marked the row ``unknown`` and
nothing re-ran it, with no incident. A started run must still never be re-run: side effects may
have happened.

Drives the REAL ``tick()``: process 1 is a subprocess that dies at the chosen point, so the
restarted scan sees a provably dead owner exactly as a gateway restart does.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Dies after the fire claim and the execution row exist, before the claimed -> running CAS.
_DIE_BEFORE_START = """
import os, sys
sys.path.insert(0, sys.argv[1])
import cron.scheduler as S
S.mark_execution_running = lambda *a, **k: os._exit(75)
S.tick(verbose=False, sync=True)
"""

# Dies right after the run won claimed -> running (started_at is set).
_DIE_AFTER_START = """
import os, sys
sys.path.insert(0, sys.argv[1])
import cron.scheduler as S
S.enter_cron_execution = lambda *a, **k: os._exit(75)
S.tick(verbose=False, sync=True)
"""

# Dies right after a provider (webhook) fire claimed the job and created its row.
_DIE_AFTER_PROVIDER_CLAIM = """
import os, sys
sys.path.insert(0, sys.argv[1])
from cron.jobs import load_jobs
from cron.scheduler_provider import InProcessCronScheduler
job_id = load_jobs()[0]["id"]
assert InProcessCronScheduler().claim_fire(job_id) is not None
os._exit(75)
"""

# Dies right after the provider's job-store claim is saved, before any later ledger write.
_DIE_AFTER_PROVIDER_STORE_CLAIM = """
import os, sys
sys.path.insert(0, sys.argv[1])
import cron.jobs as J
from cron.scheduler_provider import InProcessCronScheduler
job_id = J.load_jobs()[0]["id"]
real_save = J.save_jobs
J.save_jobs = lambda jobs, *a, **k: (real_save(jobs, *a, **k), os._exit(75))
InProcessCronScheduler().claim_fire(job_id)
os._exit(76)
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    (home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)

    import cron.executions as E
    import cron.jobs as J
    import cron.scheduler as S

    monkeypatch.setattr(J, "HERMES_DIR", home)
    monkeypatch.setattr(J, "CRON_DIR", home / "cron")
    monkeypatch.setattr(J, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(J, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(E, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(S, "_hermes_home", home)
    monkeypatch.setattr(S, "_last_dead_owner_reap_at", {})
    S._running_job_ids.clear()
    S._running_since.clear()
    S._running_futures.clear()

    counter = home / "fires.txt"
    (home / "scripts" / "fire.sh").write_text(
        f"#!/bin/sh\necho fired >> {counter}\necho fired\n", encoding="utf-8")
    (home / "scripts" / "fire.sh").chmod(0o755)
    job = J.create_job(prompt=None, schedule="every 1h", name="unknown-slot", script="fire.sh",
                       no_agent=True, deliver="local")
    slot = (J._hermes_now() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    stored = J.load_jobs()
    next(r for r in stored if r["id"] == job["id"])["next_run_at"] = slot
    J.save_jobs(stored)

    def fires() -> int:
        return counter.read_text(encoding="utf-8").count("\n") if counter.exists() else 0

    def die(script: str) -> None:
        proc = subprocess.run([sys.executable, "-c", script, str(REPO)],
                              env=dict(os.environ, HERMES_HOME=str(home)), cwd=str(REPO),
                              capture_output=True, text=True, timeout=120, check=False)
        assert proc.returncode == 75, proc.stderr[-2000:]

    yield {"job_id": job["id"], "slot": slot, "fires": fires, "die": die, "S": S, "J": J, "E": E}
    S._shutdown_parallel_pool()


def _restart(S, monkeypatch):
    """What a fresh gateway process starts with: no in-flight set, reap throttle open."""
    S._running_job_ids.clear()
    monkeypatch.setattr(S, "_last_dead_owner_reap_at", {})


def _incidents_for(job_id):
    from cron.incidents import list_incidents

    return [i for i in list_incidents() if i["job_id"] == job_id]


def test_owner_dead_before_start_redispatches_once_with_incident(env, monkeypatch):
    S, E = env["S"], env["E"]
    job_id = env["job_id"]

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    assert lost["status"] == "claimed" and lost["started_at"] is None
    assert env["fires"]() == 0

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert env["fires"]() == 1, "claimed-but-never-started occurrence must run once"
    from cron.occurrences import scheduled_instant
    rerun = E.latest_execution(job_id)
    assert rerun["id"] != lost["id"]
    assert rerun["status"] == "completed"
    assert rerun["scheduled_instant"] == scheduled_instant(env["slot"])

    incidents = _incidents_for(job_id)
    assert len(incidents) == 1
    assert "never started" in incidents[0]["error"]
    assert "was re-dispatched" in incidents[0]["error"]
    assert lost["id"] not in incidents[0]["error"], "signature text must stay stable"

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1, "the re-dispatched occurrence must not run twice"


def test_owner_dead_after_start_is_not_redispatched_but_opens_incident(env, monkeypatch):
    S, E = env["S"], env["E"]
    job_id = env["job_id"]

    env["die"](_DIE_AFTER_START)
    lost = E.latest_execution(job_id)
    assert lost["status"] == "running" and lost["started_at"]

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert env["fires"]() == 0, "a started run may have had side effects: never re-run it"
    assert E.latest_execution(job_id)["id"] == lost["id"]

    incidents = _incidents_for(job_id)
    assert len(incidents) == 1
    assert "had started" in incidents[0]["error"]
    assert "not re-run" in incidents[0]["error"]
    assert lost["id"] not in incidents[0]["error"], "signature text must stay stable"


@pytest.mark.parametrize("rewrite", ["pause_resume", "schedule_edit"])
def test_rewrite_after_claim_blocks_restore_but_opens_incident(env, monkeypatch, rewrite):
    """A schedule or lifecycle rewrite between the claim and the restart supersedes the slot."""
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    assert lost["started_at"] is None
    if rewrite == "pause_resume":
        J.pause_job(job_id)
        J.resume_job(job_id)
    else:
        J.update_job(job_id, {"schedule": J.parse_schedule("every 2h")})

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert env["fires"]() == 0, "a rewritten job must not resurrect the pre-rewrite slot"
    assert "pending_slot" not in J.get_job(job_id)

    incidents = _incidents_for(job_id)
    assert len(incidents) == 1
    assert "not restorable" in incidents[0]["error"]


def _dead_pid() -> int:
    proc = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    return int(proc.stdout.strip())


def _orphan(E, job_id, *, scheduled_instant=None, claim_token=None) -> str:
    """A claimed, never-started row whose owner process is provably gone."""
    row = E.create_execution(job_id, source="builtin", scheduled_instant=scheduled_instant)
    with E._transaction() as conn:
        conn.execute("UPDATE executions SET process_id='dead-owner', pid=?, process_started_at=NULL, "
                     "fire_claim_by=? WHERE id=?", (_dead_pid(), claim_token, row["id"]))
    return row["id"]


def test_restore_failure_still_opens_incident(env, monkeypatch):
    J, E = env["J"], env["E"]
    job_id = env["job_id"]

    def boom(*_a, **_k):
        raise OSError("jobs store unwritable")

    monkeypatch.setattr(J, "restore_unstarted_occurrence", boom)
    lost = _orphan(E, job_id, scheduled_instant=env["slot"], claim_token="dead-owner-token")
    assert E.recover_interrupted_executions() == 1
    assert E.get_execution(lost)["status"] == "unknown"

    incidents = _incidents_for(job_id)
    assert len(incidents) == 1
    assert "restoring its occurrence failed" in incidents[0]["error"]


def test_same_category_unknowns_share_one_incident(env):
    E = env["E"]
    job_id = env["job_id"]

    first = _orphan(E, job_id)
    assert E.recover_interrupted_executions() == 1
    second = _orphan(E, job_id)
    assert E.recover_interrupted_executions() == 1
    assert {E.get_execution(i)["status"] for i in (first, second)} == {"unknown"}

    incidents = _incidents_for(job_id)
    assert len(incidents) == 1, "the same loss category must dedupe into one incident"
    assert "not restorable" in incidents[0]["error"]


def _rewrite_job(J, job_id, **fields):
    stored = J.load_jobs()
    next(r for r in stored if r["id"] == job_id).update(fields)
    J.save_jobs(stored)


def test_one_minute_job_restores_the_due_occurrence_not_the_next(env, monkeypatch):
    """At one-minute cadence the advanced next_run_at is within the claim skew of now."""
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]
    _rewrite_job(J, job_id, schedule=J.parse_schedule("every 1m"))

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    assert lost["started_at"] is None
    from cron.occurrences import scheduled_instant
    assert lost["scheduled_instant"] == scheduled_instant(env["slot"])

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1, "the due one-minute occurrence must be restored and run once"
    incidents = _incidents_for(job_id)
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]


@pytest.mark.parametrize("restart_after_grace", [False, True])
def test_never_started_oneshot_runs_once_after_restart(env, monkeypatch, restart_after_grace):
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]
    slot = env["slot"]
    _rewrite_job(J, job_id, schedule={"kind": "once", "run_at": slot, "display": "once"},
                 next_run_at=slot, repeat={"times": 1, "completed": 0})

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    assert lost["started_at"] is None

    if restart_after_grace:
        real_now = J._hermes_now
        monkeypatch.setattr(J, "_hermes_now", lambda: real_now() + timedelta(minutes=10))
    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert env["fires"]() == 1, "a claimed-but-never-started one-shot must run once"
    incidents = _incidents_for(job_id)
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1, "and never twice"


def test_payload_only_edit_keeps_the_restore_and_runs_the_edited_job(env, monkeypatch):
    """Only schedule/lifecycle rewrites cancel the restore; a payload edit runs on the slot."""
    S, J = env["S"], env["J"]
    job_id = env["job_id"]

    env["die"](_DIE_BEFORE_START)
    J.update_job(job_id, {"name": "renamed"})

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1
    assert J.get_job(job_id)["name"] == "renamed"


def test_restored_one_minute_retry_dying_again_restores_the_same_occurrence(env, monkeypatch):
    """The retry consumes the restored pending_slot before its claim; the claim must still bind
    the occurrence the tick dispatches, so a second pre-start death is restorable too."""
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]
    _rewrite_job(J, job_id, schedule=J.parse_schedule("every 1m"))

    env["die"](_DIE_BEFORE_START)
    first = E.latest_execution(job_id)
    env["die"](_DIE_BEFORE_START)  # recovers + restores the first, then the retry dies too
    retry = E.latest_execution(job_id)
    assert retry["id"] != first["id"] and retry["started_at"] is None
    assert retry["scheduled_instant"] == first["scheduled_instant"]

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(retry["id"])["status"] == "unknown"
    assert env["fires"]() == 1, "the twice-lost occurrence must still run exactly once"
    assert E.latest_execution(job_id)["scheduled_instant"] == first["scheduled_instant"]


def test_same_instant_replacement_claim_is_not_restored_over(env):
    """A replacement claim for the same instant, taken after the owner died but before
    recovery, belongs to another execution: recovery must leave it and its count alone."""
    J, E = env["J"], env["E"]
    job_id = env["job_id"]
    slot = env["slot"]
    _rewrite_job(J, job_id, schedule={"kind": "once", "run_at": slot, "display": "once"},
                 next_run_at=slot, repeat={"times": 1, "completed": 0})

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    assert lost["started_at"] is None
    assert J.claim_job_for_fire(job_id, claim_ttl_seconds=0) is True
    replacement = J.get_job(job_id)["fire_claim"]
    _rewrite_job(J, job_id, fire_claim=dict(replacement, dispatch_counted=True))  # passed claim_dispatch
    before = J.get_job(job_id)
    assert before["fire_claim"]["scheduled_at"] == lost["scheduled_instant"]

    assert E.recover_interrupted_executions() == 1
    after = J.get_job(job_id)
    assert after["fire_claim"] == before["fire_claim"], "the replacement claim must survive"
    assert after["repeat"] == before["repeat"], "its dispatch count must be unchanged"
    assert after.get("run_claim") == before.get("run_claim")
    incidents = _incidents_for(job_id)
    assert len(incidents) == 1 and "not restorable" in incidents[0]["error"]


def test_provider_fire_dying_before_start_is_restored(env, monkeypatch):
    """The provider claim path records the claim token on its row, so recovery can fence on it."""
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]

    env["die"](_DIE_AFTER_PROVIDER_CLAIM)
    lost = E.latest_execution(job_id)
    assert lost["started_at"] is None
    assert lost["fire_claim_by"] == J.get_job(job_id)["fire_claim"]["by"]

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert env["fires"]() == 1, "the provider's never-started occurrence must run once"
    incidents = _incidents_for(job_id)
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]


def test_provider_death_between_claim_and_ledger_bind_is_restorable(env, monkeypatch):
    """The occurrence and claim token must reach the row before the job-store claim is saved."""
    S, E = env["S"], env["E"]
    job_id = env["job_id"]

    env["die"](_DIE_AFTER_PROVIDER_STORE_CLAIM)
    lost = E.latest_execution(job_id)
    assert lost["scheduled_instant"] and lost["fire_claim_by"]

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1, "the provider's occurrence must be restored and run once"


@pytest.mark.parametrize("crash", ["before_settle", "after_restore"])
def test_sweep_crash_after_unknown_commit_is_settled_by_the_next_sweep(env, monkeypatch, crash):
    """The unknown transition persists an unsettled marker; a later sweep finishes the restore and
    incident exactly once, even when the occurrence was already restored before the crash."""
    S, J, E = env["S"], env["J"], env["E"]
    job_id = env["job_id"]
    import cron.incidents as I

    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(job_id)
    with monkeypatch.context() as m:
        if crash == "before_settle":
            m.setattr(E, "_settle_recovered_unknown", lambda _r: (_ for _ in ()).throw(
                KeyboardInterrupt()))
        else:
            m.setattr(I, "upsert_incident", lambda *_a, **_k: (_ for _ in ()).throw(
                KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            E.recover_interrupted_executions()
    assert E.get_execution(lost["id"])["status"] == "unknown"
    assert _incidents_for(job_id) == []

    assert E.recover_interrupted_executions() == 0  # no new transition, only settlement
    assert E.recover_interrupted_executions() == 0
    incidents = _incidents_for(job_id)
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]
    assert J.get_job(job_id).get("pending_slot")

    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1


def _swept_unknown(env):
    """The lost row right after the unknown transition committed, before any settlement."""
    E = env["E"]
    env["die"](_DIE_BEFORE_START)
    lost = E.latest_execution(env["job_id"])
    with pytest.MonkeyPatch.context() as m:
        m.setattr(E, "_settle_recovered_unknown", lambda _r: None)
        assert E.recover_interrupted_executions() == 1
    return E.get_execution(lost["id"])


def test_two_settlers_of_one_row_yield_one_outcome_and_one_incident(env):
    """Two sweeps that read the same unsettled row: only the lease winner restores and decides."""
    E, J = env["E"], env["J"]
    record = _swept_unknown(env)
    assert record["recovery_pending"] == "restore"

    E._settle_recovered_unknown(dict(record))
    E._settle_recovered_unknown(dict(record))  # the slower sweep's stale snapshot
    incidents = _incidents_for(env["job_id"])
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]
    assert E.get_execution(record["id"])["recovery_pending"] is None
    assert J.get_job(env["job_id"]).get("pending_slot")


def _sweep_with_lease(env, record, lease):
    """Plant a settlement lease on the row, sweep once, and return the marker left behind."""
    E = env["E"]
    with E._transaction() as conn:
        conn.execute("UPDATE executions SET recovery_pending=? WHERE id=?", (lease, record["id"]))
    assert E.recover_interrupted_executions() == 0
    return E.get_execution(record["id"])["recovery_pending"]


def _assert_taken_over(env, record):
    assert env["E"].get_execution(record["id"])["recovery_pending"] is None
    incidents = _incidents_for(env["job_id"])
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]


def test_live_settling_lease_is_left_alone_and_a_dead_one_is_taken_over(env):
    E = env["E"]
    record = _swept_unknown(env)
    import os
    import time

    parent, now = os.getppid(), int(time.time())
    live = f"settling:other-process:{parent}:{E._process_start_time(parent)}:{now}:restore"
    assert _sweep_with_lease(env, record, live) == live
    assert _incidents_for(env["job_id"]) == []

    _sweep_with_lease(env, record, f"settling:other-process:{_dead_pid()}:None:{now}:restore")
    _assert_taken_over(env, record)


def test_lease_on_our_reused_pid_from_another_process_is_taken_over(env):
    """A restarted process that got the dead holder's pid must not treat that lease as its own."""
    import os
    import time

    record = _swept_unknown(env)
    E = env["E"]
    pid = os.getpid()
    lease = (f"settling:previous-process:{pid}:{E._process_start_time(pid)}:"
             f"{int(time.time())}:restore")
    _sweep_with_lease(env, record, lease)
    _assert_taken_over(env, record)


@pytest.mark.parametrize("age, held", [(0, True), (601, False)])
def test_unfingerprinted_live_foreign_lease_is_held_until_it_ages_out(env, age, held):
    """No start time means liveness can't be proven either way: hold it, but only for a bound."""
    import os
    import time

    record = _swept_unknown(env)
    assert env["E"]._UNFINGERPRINTED_LEASE_MAX_AGE_SECONDS == 600
    lease = f"settling:other-process:{os.getppid()}:None:{int(time.time()) - age}:restore"
    left = _sweep_with_lease(env, record, lease)
    if held:
        assert left == lease
        assert _incidents_for(env["job_id"]) == []
    else:
        _assert_taken_over(env, record)


def test_crash_after_restore_save_before_outcome_write_replays_as_redispatched(env, monkeypatch):
    S, J, E = env["S"], env["J"], env["E"]
    record = _swept_unknown(env)

    real_save = J.save_jobs

    def save_then_die(*a, **k):
        real_save(*a, **k)
        raise KeyboardInterrupt  # the process dies right after the job store saved the restore

    with monkeypatch.context() as m:
        m.setattr(J, "save_jobs", save_then_die)
        with pytest.raises(KeyboardInterrupt):
            E._settle_recovered_unknown(dict(record))
    assert J.get_job(env["job_id"]).get("pending_slot"), "the restore itself was saved"

    assert E.recover_interrupted_executions() == 0
    incidents = _incidents_for(env["job_id"])
    assert len(incidents) == 1 and "was re-dispatched" in incidents[0]["error"]
    _restart(S, monkeypatch)
    S.tick(verbose=False, sync=True)
    assert env["fires"]() == 1


def test_replayed_settlement_does_not_reopen_an_incident_resolved_in_between(env, monkeypatch):
    E = env["E"]
    import cron.incidents as I
    record = _swept_unknown(env)

    real_upsert = I.upsert_incident

    def upsert_then_die(*a, **k):
        real_upsert(*a, **k)
        raise KeyboardInterrupt  # dies after the incident write, before the marker clear

    with monkeypatch.context() as m:
        m.setattr(I, "upsert_incident", upsert_then_die)
        with pytest.raises(KeyboardInterrupt):
            E._settle_recovered_unknown(dict(record))
    assert I.close_incidents_for_recovered_job(env["job_id"]) == 1  # a successful run

    E.recover_interrupted_executions()
    E.recover_interrupted_executions()
    states = [i["state"] for i in _incidents_for(env["job_id"])]
    assert states == ["resolved"], "replaying the same execution must not reopen its incident"
