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
        assert jobs.mark_job_run(
            job["id"], False, shutdown_error,
            expected_fire_owner=claim["fire_claim"]["by"], execution_id=execution["id"],
            shutdown_interrupted=True,
        )
        assert jobs.get_job(job["id"])["pending_slot"]["replay"]
        assert jobs.get_job(job["id"])["shutdown_interrupted_execution"] == execution["id"]
        _make_unknown(execution["id"], shutdown_error)
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
