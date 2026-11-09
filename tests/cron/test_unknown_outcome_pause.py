"""Unknown cron outcomes skip shutdown-killed recurring slots and alert on retained pauses."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import pytest

from cron import executions, jobs, scheduler, scheduler_authority
from hermes_cli import gateway_client


_SHUTDOWN_ERROR = (
    "Gateway shutdown ({phase}) killed the job's tool subprocess before the run finished."
)


def _peer(monkeypatch):
    class Peer:
        async def rpc(self, method, **params):
            assert method == "cron.recover"
            return {"status": "unknown", "result": None}

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)


def _journal(job, execution_id):
    params = {"job_id": job["id"], "request_id": execution_id, "extra_prompt": None}
    journal = scheduler_authority.journal_path(job["id"], execution_id)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"params": params, "receipt": None}), encoding="utf-8")
    return journal


def _make_unknown(execution_id, error="unknown_execution: owner lost"):
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET status='unknown', error=? WHERE id=?",
            (error, execution_id),
        )


@pytest.mark.parametrize("shutdown_phase", ["post-interrupt", "final-cleanup"])
def test_shutdown_unknown_recurring_slot_is_skipped_without_replay(
        tmp_path, monkeypatch, caplog, shutdown_phase):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        killed_slot = job["next_run_at"]
        now[0] = datetime.fromisoformat(killed_slot)
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=killed_slot)
        claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=killed_slot, execution_id=execution["id"])
        assert executions.mark_execution_running(execution["id"])
        shutdown_error = _SHUTDOWN_ERROR.format(phase=shutdown_phase)
        key = scheduler._inflight_key(job["id"], tmp_path / "cron")
        token = object()
        scheduler._running_fire_owners[key] = {
            token: (claim["fire_claim"]["by"], tmp_path / "cron", execution["id"]),
        }
        try:
            assert job["id"] in scheduler.mark_running_jobs_interrupted(
                shutdown_error, only_owners={(job["id"], claim["fire_claim"]["by"])})
        finally:
            scheduler._running_fire_owners.pop(key, None)
            scheduler._interrupted_job_ids.discard(token)
        assert jobs.get_job(job["id"])["pending_slot"]["replay"]
        assert jobs.get_job(job["id"])["shutdown_interrupted_execution"] == execution["id"]
        assert executions.give_up_execution(execution["id"], reason=shutdown_error)
        journal = _journal(job, execution["id"])
        scheduler_authority.pause_for_unknown(
            job["id"], execution["id"], "unknown_execution: cron admission was not replayed")
        now[0] += timedelta(hours=2, minutes=15)
        expected_next = jobs.compute_next_run(job["schedule"], now[0].isoformat())

        scheduler_authority.reconcile_pending()
        scheduler_authority.reconcile_pending()

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "scheduled" and saved["enabled"] is True
        assert saved["paused_reason"] is None
        assert datetime.fromisoformat(saved["next_run_at"]) == datetime.fromisoformat(expected_next)
        assert datetime.fromisoformat(saved["next_run_at"]) > now[0]
        assert saved["last_status"] == "skipped"
        assert "skipped" in saved["last_error"].lower()
        assert shutdown_error in saved["last_error"]
        assert saved.get("pending_slot") is None
        assert jobs.get_due_jobs() == []
        assert executions.get_execution(execution["id"])["status"] == "unknown"
        assert "CRON_UNKNOWN_OUTCOME_SKIPPED" in caplog.text
        assert "skipped" in executions.get_execution(execution["id"])["error"].lower()
        assert not journal.exists()


def test_replay_completion_before_unknown_receipt_recovery_is_not_skipped(
        tmp_path, monkeypatch):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        killed_slot = job["next_run_at"]
        now[0] = datetime.fromisoformat(killed_slot)
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=killed_slot)
        first = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=killed_slot, execution_id=execution["id"])
        assert executions.mark_execution_running(execution["id"])
        shutdown_error = _SHUTDOWN_ERROR.format(phase="post-interrupt")
        key = scheduler._inflight_key(job["id"], tmp_path / "cron")
        token = object()
        scheduler._running_fire_owners[key] = {
            token: (first["fire_claim"]["by"], tmp_path / "cron", execution["id"]),
        }
        try:
            assert job["id"] in scheduler.mark_running_jobs_interrupted(
                shutdown_error, only_owners={(job["id"], first["fire_claim"]["by"])})
        finally:
            scheduler._running_fire_owners.pop(key, None)
            scheduler._interrupted_job_ids.discard(token)
        assert jobs.get_job(job["id"])["shutdown_interrupted_execution"] == execution["id"]
        _journal(job, execution["id"])
        assert executions.give_up_execution(execution["id"], reason=shutdown_error)

        # This ordering is legal when the job replay finishes before the unknown receipt is
        # reconciled. The replay carries the same scheduled instant in its own execution row.
        now[0] += timedelta(minutes=1)
        due = jobs.get_due_jobs()
        assert any(candidate["id"] == job["id"] for candidate in due)
        replay_execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=killed_slot)
        replay = jobs.claim_job_for_fire(
            job["id"], return_job=True, execution_id=replay_execution["id"])
        assert replay["fire_claim"]["replay"]
        assert executions.mark_execution_running(replay_execution["id"])
        assert jobs.mark_job_run(
            job["id"], True, expected_fire_owner=replay["fire_claim"]["by"],
            execution_id=replay_execution["id"])
        assert executions.finish_execution(replay_execution["id"], success=True)

        scheduler_authority.reconcile_pending()

        saved = jobs.get_job(job["id"])
        assert saved["repeat"]["completed"] == 1
        assert saved["last_status"] == "ok"
        assert saved["state"] == "scheduled"
        assert saved.get("pending_slot") is None
        assert saved.get("shutdown_interrupted_execution") is None
        assert executions.get_execution(replay_execution["id"])["status"] == "completed"
        assert not scheduler_authority.journal_path(job["id"], execution["id"]).exists()


def test_unknown_after_replay_shutdown_is_already_booked_and_not_paused(
        tmp_path, monkeypatch):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)
    notices = []
    monkeypatch.setattr(
        scheduler, "_deliver_crash_failure",
        lambda job, error, *, adapters, loop: notices.append((job["id"], error)))

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        first_slot = job["next_run_at"]
        now[0] = datetime.fromisoformat(first_slot)
        first = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=first_slot)
        first_claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=first_slot, execution_id=first["id"])
        assert executions.mark_execution_running(first["id"])
        first_key = scheduler._inflight_key(job["id"], tmp_path / "cron")
        first_token = object()
        scheduler._running_fire_owners[first_key] = {
            first_token: (first_claim["fire_claim"]["by"], tmp_path / "cron", first["id"]),
        }
        try:
            assert job["id"] in scheduler.mark_running_jobs_interrupted(
                "gateway shutdown", only_owners={(job["id"], first_claim["fire_claim"]["by"])})
        finally:
            scheduler._running_fire_owners.pop(first_key, None)
            scheduler._interrupted_job_ids.discard(first_token)
        assert executions.finish_execution(first["id"], success=False, error="gateway shutdown")

        now[0] += timedelta(minutes=1)
        assert any(candidate["id"] == job["id"] for candidate in jobs.get_due_jobs())
        replay = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=first_slot)
        journal = _journal(job, replay["id"])
        replay_claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, execution_id=replay["id"])
        assert replay_claim["fire_claim"]["replay"]
        assert executions.mark_execution_running(replay["id"])
        replay_error = "Gateway shutdown (final-cleanup) killed the replay subprocess."
        replay_key = scheduler._inflight_key(job["id"], tmp_path / "cron")
        replay_token = object()
        scheduler._running_fire_owners[replay_key] = {
            replay_token: (replay_claim["fire_claim"]["by"], tmp_path / "cron", replay["id"]),
        }
        try:
            assert job["id"] in scheduler.mark_running_jobs_interrupted(
                replay_error,
                only_owners={(job["id"], replay_claim["fire_claim"]["by"])})
        finally:
            scheduler._running_fire_owners.pop(replay_key, None)
            scheduler._interrupted_job_ids.discard(replay_token)
        expected_next = jobs.get_job(job["id"])["next_run_at"]
        assert jobs.get_job(job["id"])["last_status"] == "error"
        _make_unknown(replay["id"], replay_error)
        scheduler_authority.reconcile_pending()
        saved = jobs.get_job(job["id"])
        assert saved["state"] == "scheduled" and saved["enabled"] is True
        assert saved.get("paused_reason") is None
        assert saved["last_status"] == "error"
        assert saved["next_run_at"] == expected_next
        assert saved["repeat"]["completed"] == 1
        assert notices == []
        assert not journal.exists()


def test_completed_recurring_slot_retires_non_shutdown_unknown_receipt(
        tmp_path, monkeypatch):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)
    notices = []
    monkeypatch.setattr(
        scheduler, "_deliver_crash_failure",
        lambda job, error, *, adapters, loop: notices.append((job["id"], error)))

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        completed_slot = job["next_run_at"]
        now[0] = datetime.fromisoformat(completed_slot)
        completed = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=completed_slot)
        claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=completed_slot,
            execution_id=completed["id"])
        assert executions.mark_execution_running(completed["id"])
        assert jobs.mark_job_run(
            job["id"], True, expected_fire_owner=claim["fire_claim"]["by"],
            execution_id=completed["id"])
        assert executions.finish_execution(completed["id"], success=True)
        expected_next = jobs.get_job(job["id"])["next_run_at"]

        unknown = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=completed_slot)
        assert executions.mark_execution_running(unknown["id"])
        _make_unknown(unknown["id"], "unknown_execution: non-shutdown outcome")
        journal = _journal(job, unknown["id"])

        scheduler_authority.reconcile_pending()

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "scheduled" and saved["enabled"] is True
        assert saved["next_run_at"] == expected_next
        assert saved["repeat"]["completed"] == 1
        assert notices == []
        assert not journal.exists()


def test_skip_reconciliation_finishes_after_crash_following_job_store_write(
        tmp_path, monkeypatch):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        killed_slot = job["next_run_at"]
        now[0] = datetime.fromisoformat(killed_slot)
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=killed_slot)
        claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=killed_slot, execution_id=execution["id"])
        assert executions.mark_execution_running(execution["id"])
        shutdown_error = _SHUTDOWN_ERROR.format(phase="post-interrupt")
        assert jobs.mark_job_run(
            job["id"], False, shutdown_error,
            expected_fire_owner=claim["fire_claim"]["by"], execution_id=execution["id"],
            shutdown_interrupted=True,
        )
        _make_unknown(execution["id"], shutdown_error)
        journal = _journal(job, execution["id"])
        scheduler_authority.pause_for_unknown(
            job["id"], execution["id"], "unknown_execution: cron admission was not replayed")
        now[0] += timedelta(hours=2, minutes=15)
        expected_next = jobs.compute_next_run(job["schedule"], now[0].isoformat())
        real_mark_unknown_skipped = executions.mark_unknown_skipped

        def crash_after_jobs_write(execution_id, detail):
            assert jobs.get_job(job["id"])["last_status"] == "skipped"
            raise RuntimeError("simulated crash after jobs.json skip write")

        monkeypatch.setattr(executions, "mark_unknown_skipped", crash_after_jobs_write)
        scheduler_authority.reconcile_pending()
        partial = jobs.get_job(job["id"])
        assert partial["last_status"] == "skipped"
        assert partial["state"] == "scheduled"
        assert "shutdown_interrupted_execution" not in partial
        assert execution["id"] in partial["canonical_completions"]
        assert journal.exists()
        assert not json.loads(journal.read_text(encoding="utf-8")).get("unknown_skipped")

        monkeypatch.setattr(executions, "mark_unknown_skipped", real_mark_unknown_skipped)
        scheduler_authority.reconcile_pending()
        assert journal.exists()
        assert json.loads(journal.read_text(encoding="utf-8")).get("unknown_skipped")
        assert "skipped" in executions.get_execution(execution["id"])["error"].lower()
        scheduler_authority.reconcile_pending()
        assert not journal.exists()

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "scheduled"
        assert saved["repeat"]["completed"] == 0
        assert saved["last_status"] == "skipped"
        assert saved["next_run_at"] == expected_next
        assert executions.get_execution(execution["id"])["status"] == "unknown"
        assert "skipped" in executions.get_execution(execution["id"])["error"].lower()


def test_shutdown_unknown_final_repeat_stays_active_without_spending_repeat(
        tmp_path, monkeypatch):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", repeat=2, deliver="local")
        now[0] = datetime.fromisoformat(job["next_run_at"])
        assert jobs.mark_job_run(job["id"], True)
        after_first_run = jobs.get_job(job["id"])
        assert after_first_run["repeat"]["completed"] == 1
        killed_slot = after_first_run["next_run_at"]
        now[0] = datetime.fromisoformat(killed_slot)
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=killed_slot)
        claim = jobs.claim_job_for_fire(
            job["id"], return_job=True, scheduled_at=killed_slot, execution_id=execution["id"])
        assert executions.mark_execution_running(execution["id"])
        assert jobs.mark_job_run(
            job["id"], False, _SHUTDOWN_ERROR.format(phase="post-interrupt"),
            expected_fire_owner=claim["fire_claim"]["by"], execution_id=execution["id"],
            shutdown_interrupted=True,
        )
        _make_unknown(execution["id"], _SHUTDOWN_ERROR.format(phase="post-interrupt"))
        _journal(job, execution["id"])
        scheduler_authority.pause_for_unknown(
            job["id"], execution["id"], "unknown_execution: cron admission was not replayed")
        now[0] += timedelta(hours=2, minutes=15)
        expected_next = jobs.compute_next_run(job["schedule"], now[0].isoformat())

        scheduler_authority.reconcile_pending()

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "scheduled" and saved["enabled"] is True
        assert saved["repeat"] == {"times": 2, "completed": 1}
        assert datetime.fromisoformat(saved["next_run_at"]) == datetime.fromisoformat(expected_next)
        assert datetime.fromisoformat(saved["next_run_at"]) > now[0]
        assert saved["last_status"] == "skipped"
        assert executions.get_execution(execution["id"])["status"] == "unknown"
        assert "skipped" in executions.get_execution(execution["id"])["error"].lower()


def test_recurring_unknown_without_shutdown_marker_pauses_and_notifies_once(
        tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: datetime(2026, 11, 9, 8, 0, tzinfo=UTC))
    _peer(monkeypatch)
    notices = []

    def deliver(job, error, *, adapters, loop):
        notices.append((job["id"], error))
        return None, "delivered"

    monkeypatch.setattr(scheduler, "_deliver_crash_failure", deliver)
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h", deliver="local")
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=job["next_run_at"])
        assert executions.mark_execution_running(execution["id"])
        _make_unknown(execution["id"], _SHUTDOWN_ERROR.format(phase="final-cleanup"))
        _journal(job, execution["id"])

        scheduler_authority.reconcile_pending()
        scheduler_authority.reconcile_pending()
        assert len(notices) == 1
        assert notices[0][0] == job["id"]
        assert notices[0][1] == _SHUTDOWN_ERROR.format(phase="final-cleanup")

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "paused" and saved["enabled"] is False
        assert saved["paused_reason"] == "Canonical cron unknown; no automatic re-execution"
        assert "CRON_UNKNOWN_OUTCOME_PAUSED" in caplog.text



def test_one_shot_unknown_pause_notifies_once_across_recovery_ticks(tmp_path, monkeypatch, caplog):
    now = [datetime(2026, 11, 9, 8, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now[0])
    _peer(monkeypatch)
    notices = []

    def deliver(job, error, *, adapters, loop):
        notices.append((job["id"], error, adapters, loop))
        return None, "delivered"

    monkeypatch.setattr(scheduler, "_deliver_crash_failure", deliver)

    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "in 2h", deliver="local")
        execution = executions.create_execution(
            job["id"], source="builtin", scheduled_instant=job["next_run_at"])
        assert executions.mark_execution_running(execution["id"])
        _make_unknown(execution["id"])
        _journal(job, execution["id"])

        scheduler_authority.reconcile_pending()
        scheduler_authority.reconcile_pending()

        saved = jobs.get_job(job["id"])
        assert saved["state"] == "paused" and saved["enabled"] is False
        assert saved["paused_reason"] == "Canonical cron unknown; no automatic re-execution"
        assert len(notices) == 1
        assert notices[0][0] == job["id"]
        assert "unknown" in notices[0][1].lower()
        assert "CRON_UNKNOWN_OUTCOME_PAUSED" in caplog.text
