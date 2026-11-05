"""A shutdown-interrupted recurring cron occurrence is retried at most once."""

from __future__ import annotations

import threading
from datetime import UTC, timedelta, timezone

import pytest
from cron import executions, jobs, scheduler
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


def test_shutdown_replay_preserves_the_next_interval_occurrence(temp_home, monkeypatch):
    job_id, _scheduled, _first_id = _interrupted_recurring_job(temp_home)
    expected_next = jobs.get_job(job_id)["next_run_at"]
    provider = InProcessCronScheduler()
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)

    def finish_replay(claimed, **_kwargs):
        jobs.mark_job_run(
            job_id,
            True,
            execution_id=claimed["execution_id"],
            expected_fire_owner=claimed["fire_claim"]["by"],
            shutdown_replay=claimed.get("_shutdown_replay", False),
        )
        executions.finish_execution(claimed["execution_id"], success=True)
        return True

    monkeypatch.setattr(provider, "fire_claimed", finish_replay)
    provider.recover_interrupted()

    assert jobs.get_job(job_id)["next_run_at"] == expected_next


@pytest.mark.parametrize("provider_kind", ["inprocess", "chronos"])
def test_startup_replay_receives_adapters_and_loop(
    temp_home, monkeypatch, provider_kind
):
    job_id, _scheduled, _first_id = _interrupted_recurring_job(temp_home)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    adapters, loop = object(), object()
    seen = []

    if provider_kind == "inprocess":
        provider = InProcessCronScheduler()
        stop = threading.Event()
        stop.set()
        start = lambda: provider.start(stop, adapters=adapters, loop=loop, interval=1)
    else:
        from plugins.cron_providers.chronos import ChronosCronScheduler

        provider = ChronosCronScheduler()
        monkeypatch.setattr(provider, "reconcile", lambda: None)
        start = lambda: provider.start(threading.Event(), adapters=adapters, loop=loop)

    def fire_claimed(claimed, **kwargs):
        seen.append((claimed["id"], kwargs))
        executions.finish_execution(claimed["execution_id"], success=True)
        return True

    monkeypatch.setattr(provider, "fire_claimed", fire_claimed)
    start()

    assert seen == [(job_id, {"adapters": adapters, "loop": loop})]


def test_gateway_shutdown_marks_active_execution_for_replay(temp_home, monkeypatch):
    job = jobs.create_job(prompt="run", schedule="every 1h", name="shutdown-marked")
    scheduled = (now() - timedelta(minutes=1)).isoformat()
    original = executions.create_execution(job["id"], source="builtin")
    claimed = jobs.claim_job_for_fire(
        job["id"], return_job=True, scheduled_at=scheduled, execution_id=original["id"]
    )
    assert isinstance(claimed, dict)
    assert executions.mark_execution_running(original["id"])

    jobs.update_job(job["id"], {"next_run_at": (now() + timedelta(hours=1)).isoformat()})
    key = scheduler._inflight_key(job["id"])
    token = object()
    scheduler._running_job_ids.add(key)
    scheduler._running_fire_owners[key] = {
        token: (claimed["fire_claim"]["by"], temp_home.resolve()),
    }
    try:
        assert scheduler.mark_running_jobs_interrupted("gateway shutdown") == [job["id"]]
    finally:
        scheduler._running_job_ids.discard(key)
        scheduler._running_fire_owners.pop(key, None)

    assert executions.get_execution(original["id"])["shutdown_interrupted"] == 1
    executions.finish_execution(
        original["id"], success=False, error="Interrupted by gateway shutdown."
    )
    replayed = _recover_and_fire(monkeypatch)
    assert [item["execution_id"] for item in replayed]


@pytest.mark.parametrize("owner_state", ["dead", "stale"])
def test_owner_lost_without_shutdown_evidence_is_not_replayed(
    temp_home, monkeypatch, owner_state
):
    job = jobs.create_job(prompt="run", schedule="every 1h", name="owner-lost")
    scheduled = (now() - timedelta(minutes=1)).isoformat()
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    assert executions.mark_execution_running(row["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    if owner_state == "dead":
        monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    else:
        monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: True)
        monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 1)
        monkeypatch.setattr(executions, "_stale_age_seconds", lambda *_args: 2)

    assert executions.recover_interrupted_executions() == 1
    assert executions.get_execution(row["id"])["shutdown_interrupted"] == 0
    assert _recover_and_fire(monkeypatch) == []


def test_shutdown_replay_cutoff_compares_instants_across_offsets(temp_home, monkeypatch):
    _job_id, _scheduled, execution_id = _interrupted_recurring_job(temp_home)
    reference = now().astimezone(UTC)
    cutoff = reference - timedelta(hours=2)
    finished = (cutoff + timedelta(minutes=30)).astimezone(
        timezone(timedelta(hours=-2))
    )
    monkeypatch.setattr(executions, "_hermes_now", lambda: reference)
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET finished_at=? WHERE id=?",
            (finished.isoformat(), execution_id),
        )

    assert [
        item["execution_id"] for item in executions.replayable_shutdown_occurrences()
    ] == [execution_id]


def test_concurrent_recovery_reserves_and_dispatches_one_replay(temp_home, monkeypatch):
    job_id, _scheduled, _first_id = _interrupted_recurring_job(temp_home)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    select = executions.replayable_shutdown_occurrences
    both_selected = threading.Barrier(2)

    def select_then_reserve():
        candidates = select()
        both_selected.wait(timeout=10)
        return candidates

    monkeypatch.setattr(executions, "replayable_shutdown_occurrences", select_then_reserve)
    providers = [InProcessCronScheduler(), InProcessCronScheduler()]
    dispatched = []
    dispatch_lock = threading.Lock()
    failures = []

    def fire_claimed(claimed, **_kwargs):
        with dispatch_lock:
            dispatched.append(claimed["execution_id"])
        executions.finish_execution(claimed["execution_id"], success=True)
        return True

    for provider in providers:
        monkeypatch.setattr(provider, "fire_claimed", fire_claimed)

    def recover(provider):
        try:
            provider.recover_interrupted()
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=recover, args=(provider,)) for provider in providers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    rows = executions.list_executions(job_id=job_id)
    replay_ids = [row["id"] for row in rows if row["replay_of"] is not None]
    assert len(replay_ids) == 1
    assert dispatched == replay_ids
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
