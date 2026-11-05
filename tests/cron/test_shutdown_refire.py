"""A shutdown-interrupted recurring cron occurrence is retried at most once."""

from __future__ import annotations

from datetime import timedelta

import pytest
from cron import executions, jobs
from cron.scheduler_provider import InProcessCronScheduler
from hermes_time import now


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return tmp_path


def _interrupted_recurring_job(temp_home, *, schedule="every 1h"):
    job = jobs.create_job(prompt="run", schedule=schedule, name="recoverable")
    scheduled_dt = now() - timedelta(minutes=1)
    scheduled = scheduled_dt.isoformat()
    jobs.update_job(job["id"], {
        "last_run_at": (scheduled_dt - timedelta(hours=1)).isoformat(),
        "next_run_at": (now() + timedelta(hours=1)).isoformat(),
    })
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(row["id"])
    executions.finish_execution(
        row["id"], success=False, error="Interrupted by gateway shutdown before terminal completion.",
        shutdown_interrupted=True,
    )
    return job["id"], scheduled, row["id"]


def _recover_and_fire(monkeypatch, *, finish_success=True):
    provider = InProcessCronScheduler()
    seen = []

    def fire_claimed(claimed, **_kwargs):
        seen.append(claimed)
        executions.finish_execution(
            claimed["execution_id"], success=finish_success,
            error=None if finish_success else "Interrupted by gateway shutdown before terminal completion.",
            shutdown_interrupted=not finish_success,
        )
        return True

    monkeypatch.setattr(provider, "fire_claimed", fire_claimed)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    provider.recover_interrupted()
    return seen


def test_interrupted_recurring_occurrence_refires_exactly_once(temp_home, monkeypatch):
    job_id, scheduled, first_id = _interrupted_recurring_job(temp_home)

    first_start = _recover_and_fire(monkeypatch)
    second_start = _recover_and_fire(monkeypatch)

    rows = executions.list_executions(job_id=job_id)
    assert [item["_scheduled_instant"] for item in first_start + second_start] == [scheduled]
    assert len(rows) == 2
    assert executions.get_execution(first_id)["status"] == "failed"
    assert rows[0]["status"] == "completed"
    assert rows[0]["scheduled_instant"] == scheduled


def test_completed_interrupted_occurrence_does_not_refire(temp_home, monkeypatch):
    job = jobs.create_job(prompt="run", schedule="every 1h", name="already-completed")
    scheduled = (now() - timedelta(minutes=1)).isoformat()
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(row["id"])
    executions.finish_execution(row["id"], success=True)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)

    assert _recover_and_fire(monkeypatch) == []
    assert len(executions.list_executions(job_id=job["id"])) == 1


def test_second_shutdown_interruption_does_not_refire(temp_home, monkeypatch):
    job_id, scheduled, _first_id = _interrupted_recurring_job(temp_home)

    first_start = _recover_and_fire(monkeypatch, finish_success=False)
    second_start = _recover_and_fire(monkeypatch)

    assert [item["_scheduled_instant"] for item in first_start] == [scheduled]
    assert second_start == []
    assert len(executions.list_executions(job_id=job_id)) == 2


def test_normal_failure_does_not_refire(temp_home, monkeypatch):
    job = jobs.create_job(prompt="run", schedule="every 1h", name="ordinary-failure")
    scheduled = (now() - timedelta(minutes=5)).isoformat()
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(row["id"])
    executions.finish_execution(row["id"], success=False, error="model request failed")
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)

    assert _recover_and_fire(monkeypatch) == []
    assert len(executions.list_executions(job_id=job["id"])) == 1


def test_shutdown_interrupted_one_shot_is_unchanged(temp_home, monkeypatch):
    job = jobs.create_job(prompt="run", schedule=(now() - timedelta(minutes=1)).isoformat(), name="one-shot")
    scheduled = jobs.get_job(job["id"])["next_run_at"]
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(row["id"])
    executions.finish_execution(
        row["id"], success=False, error="Interrupted by gateway shutdown before terminal completion.",
        shutdown_interrupted=True,
    )
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)

    assert _recover_and_fire(monkeypatch) == []
    assert len(executions.list_executions(job_id=job["id"])) == 1
