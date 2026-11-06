"""A shutdown-interrupted recurring cron occurrence is retried at most once."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta, timezone

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

def test_multiplex_startup_replay_receives_profile_adapters_and_loop(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [(name, tmp_path / name) for name in ("default", "secondary")]
    job_ids = {}
    for name, home in homes:
        home.mkdir()
        token = set_hermes_home_override(str(home))
        try:
            job = jobs.create_job(prompt="run", schedule="every 1h", name=name)
            scheduled = (now() - timedelta(minutes=1)).isoformat()
            jobs.update_job(job["id"], {"next_run_at": (now() + timedelta(hours=1)).isoformat()})
            row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
            executions.mark_execution_running(row["id"])
            executions.finish_execution(
                row["id"], success=False,
                error="Interrupted by gateway shutdown before terminal completion.",
                shutdown_interrupted=True,
            )
            job_ids[name] = job["id"]
        finally:
            reset_hermes_home_override(token)

    shared_adapters, default_adapters, secondary_adapters, loop = (object() for _ in range(4))
    seen = []
    provider = InProcessCronScheduler()
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)

    def fire_claimed(claimed, **kwargs):
        seen.append((claimed["id"], kwargs))
        executions.finish_execution(claimed["execution_id"], success=True)
        return True

    monkeypatch.setattr(provider, "fire_claimed", fire_claimed)
    stop = threading.Event()
    stop.set()
    provider.start(
        stop,
        adapters=shared_adapters,
        loop=loop,
        profile_homes=homes,
        profile_adapters={"default": default_adapters, "secondary": secondary_adapters},
        default_profile="default",
    )

    assert seen == [
        (job_ids["default"], {"adapters": shared_adapters, "loop": loop}),
        (job_ids["secondary"], {"adapters": secondary_adapters, "loop": loop}),
    ]


def test_final_repeat_run_interrupted_at_shutdown_replays_once(temp_home, monkeypatch):
    job = jobs.create_job(prompt="run", schedule="every 1h", name="final-repeat", repeat=2)
    jobs.update_job(job["id"], {"repeat": {"times": 2, "completed": 1}})
    scheduled = (now() - timedelta(minutes=1)).isoformat()
    row = executions.create_execution(job["id"], source="builtin", scheduled_instant=scheduled)
    claimed = jobs.claim_job_for_fire(
        job["id"], return_job=True, scheduled_at=scheduled, execution_id=row["id"]
    )
    assert isinstance(claimed, dict)
    assert executions.mark_execution_running(row["id"])

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

    executions.finish_execution(
        row["id"], success=False, error="Interrupted by gateway shutdown before terminal completion."
    )
    interrupted = jobs.get_job(job["id"])
    assert interrupted["repeat"]["completed"] == 1
    assert not jobs.is_terminal_job(interrupted)

    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    provider = InProcessCronScheduler()
    replayed = []

    def finish_replay(replay, **_kwargs):
        replayed.append(replay)
        jobs.mark_job_run(
            job["id"],
            True,
            execution_id=replay["execution_id"],
            expected_fire_owner=replay["fire_claim"]["by"],
            shutdown_replay=replay.get("_shutdown_replay", False),
        )
        executions.finish_execution(replay["execution_id"], success=True)
        return True

    monkeypatch.setattr(provider, "fire_claimed", finish_replay)
    provider.recover_interrupted()

    finished = jobs.get_job(job["id"])
    assert len(replayed) == 1
    assert finished["repeat"]["completed"] == 2
    assert finished["state"] == "completed"
    assert finished["enabled"] is False
    assert finished["next_run_at"] is None


@pytest.mark.parametrize("replay_success", [True, False], ids=["succeeds", "unreachable-again"])
def test_shutdown_replay_of_unreachable_rung_does_not_spend_repeat_slot(
    temp_home, monkeypatch, replay_success,
):
    from cron import unreachable_retry
    from cron.scheduler_bookkeeping import finish_completed_run

    clock = [now()]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(unreachable_retry, "_hermes_now", lambda: clock[0])
    job = jobs.create_job(prompt="run", schedule="every 24h", name="retry-repeat", repeat=2)
    job_id = job["id"]

    assert jobs.mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    first_failure = jobs.get_job(job_id)
    assert first_failure["repeat"]["completed"] == 1
    retry_at = first_failure[unreachable_retry.STATE_KEY]["at"]
    clock[0] = datetime.fromisoformat(retry_at) + timedelta(seconds=1)

    rung = executions.create_execution(job_id, source="builtin", scheduled_instant=retry_at)
    claimed = jobs.claim_job_for_fire(
        job_id, return_job=True, scheduled_at=retry_at, execution_id=rung["id"]
    )
    assert isinstance(claimed, dict)
    assert executions.mark_execution_running(rung["id"])
    assert jobs.mark_job_run(
        job_id,
        False,
        "Interrupted by gateway shutdown before terminal completion.",
        expected_fire_owner=claimed["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    executions.finish_execution(
        rung["id"],
        success=False,
        error="Interrupted by gateway shutdown before terminal completion.",
        shutdown_interrupted=True,
    )

    interrupted = jobs.get_job(job_id)
    second_occurrence = interrupted["next_run_at"]
    assert interrupted[unreachable_retry.STATE_KEY]["at"] == retry_at
    assert unreachable_retry.is_retry_run(claimed)

    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    provider = InProcessCronScheduler()
    replayed = []

    def finish_replay(replay, **_kwargs):
        replayed.append(replay)
        assert executions.mark_execution_running(replay["execution_id"])
        if not replay_success:
            replay["_model_unreachable"] = True
        assert finish_completed_run(
            scheduler._RunDelivery(
                job=replay, success=replay_success,
                error=None if replay_success else "ConnectError: dns",
            ),
            replay["fire_claim"]["by"],
            replay["execution_id"],
        )
        return True

    monkeypatch.setattr(provider, "fire_claimed", finish_replay)
    provider.recover_interrupted()

    finished = jobs.get_job(job_id)
    if replay_success:
        assert finished["next_run_at"] == second_occurrence
    else:
        assert finished["next_run_at"] != second_occurrence
    assert len(replayed) == 1
    assert finished["repeat"]["completed"] == 1
    assert not jobs.is_terminal_job(finished)
    assert finished["state"] == "scheduled"
    assert finished["next_run_at"] is not None

    retry_state = finished.get(unreachable_retry.STATE_KEY)
    if replay_success:
        assert retry_state is None
    else:
        assert retry_state is not None
        assert retry_state["attempt"] == 2
        assert retry_state["at"] == finished["next_run_at"]


def test_skipped_paused_retry_replay_starts_a_fresh_unreachable_ladder(
    temp_home, monkeypatch,
):
    from cron import unreachable_retry

    clock = [now()]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(unreachable_retry, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: clock[0])
    job = jobs.create_job(prompt="run", schedule="every 24h", name="paused-retry")
    job_id = job["id"]

    assert jobs.mark_job_run(job_id, False, "ConnectError: dns", model_unreachable=True)
    retry_at = jobs.get_job(job_id)[unreachable_retry.STATE_KEY]["at"]
    clock[0] = datetime.fromisoformat(retry_at) + timedelta(seconds=1)

    rung = executions.create_execution(job_id, source="builtin", scheduled_instant=retry_at)
    claimed = jobs.claim_job_for_fire(
        job_id, return_job=True, scheduled_at=retry_at, execution_id=rung["id"]
    )
    assert isinstance(claimed, dict)
    assert executions.mark_execution_running(rung["id"])
    assert jobs.mark_job_run(
        job_id,
        False,
        "Interrupted by gateway shutdown before terminal completion.",
        expected_fire_owner=claimed["fire_claim"]["by"],
        ladder_rung=True,
        shutdown_interrupted=True,
    )
    executions.finish_execution(
        rung["id"],
        success=False,
        error="Interrupted by gateway shutdown before terminal completion.",
        shutdown_interrupted=True,
    )

    assert jobs.pause_job(job_id)
    before_replay = len(executions.list_executions(job_id=job_id))
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    InProcessCronScheduler().recover_interrupted()
    assert len(executions.list_executions(job_id=job_id)) == before_replay

    assert jobs.resume_job(job_id)
    scheduled_at = jobs.get_job(job_id)["next_run_at"]
    clock[0] = datetime.fromisoformat(scheduled_at) + timedelta(seconds=1)

    next_run = jobs.claim_job_for_fire(
        job_id, return_job=True, scheduled_at=scheduled_at
    )
    assert isinstance(next_run, dict)
    assert not unreachable_retry.is_retry_run(next_run)
    assert jobs.mark_job_run(
        job_id,
        False,
        "ConnectError: dns",
        expected_fire_owner=next_run["fire_claim"]["by"],
        model_unreachable=True,
    )

    retry_state = jobs.get_job(job_id)[unreachable_retry.STATE_KEY]
    assert retry_state["attempt"] == 1
    assert retry_state["at"] == jobs.get_job(job_id)["next_run_at"]


def test_shutdown_replay_quota_failure_holds_the_next_occurrence(temp_home, monkeypatch):
    from cron import quota_hold
    from cron.scheduler_bookkeeping import finish_completed_run

    clock = [now()]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(quota_hold, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(executions, "_hermes_now", lambda: clock[0])
    job = jobs.create_job(
        prompt="run", schedule="every 1h", name="quota-replay", repeat=3
    )
    scheduled = (clock[0] - timedelta(minutes=1)).isoformat()
    natural_next = (clock[0] + timedelta(hours=1)).isoformat()
    jobs.update_job(job["id"], {
        "last_run_at": (clock[0] - timedelta(hours=1)).isoformat(),
        "next_run_at": natural_next,
    })
    interrupted = executions.create_execution(
        job["id"], source="builtin", scheduled_instant=scheduled
    )
    assert executions.mark_execution_running(interrupted["id"])
    executions.finish_execution(
        interrupted["id"],
        success=False,
        error="Interrupted by gateway shutdown before terminal completion.",
        shutdown_interrupted=True,
    )

    plan_calls = []
    original_plan_hold = quota_hold.plan_hold

    def capture_plan_hold(job_record, hold_seconds, *, recover_consumed_fire=False):
        plan_calls.append(recover_consumed_fire)
        return original_plan_hold(
            job_record, hold_seconds, recover_consumed_fire=recover_consumed_fire
        )

    monkeypatch.setattr(quota_hold, "plan_hold", capture_plan_hold)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    provider = InProcessCronScheduler()

    def finish_replay(replay, **_kwargs):
        assert executions.mark_execution_running(replay["execution_id"])
        replay["_quota_hold_seconds"] = 2 * 60 * 60
        assert finish_completed_run(
            scheduler._RunDelivery(job=replay, success=False, error="quota exhausted"),
            replay["fire_claim"]["by"],
            replay["execution_id"],
        )
        return True

    monkeypatch.setattr(provider, "fire_claimed", finish_replay)
    provider.recover_interrupted()

    held = jobs.get_job(job["id"])
    assert plan_calls == [False]
    assert held["repeat"]["completed"] == 1
    assert held[quota_hold.STATE_KEY] == held["next_run_at"]
    assert held["next_run_at"] != natural_next
    clock[0] = datetime.fromisoformat(natural_next) + timedelta(seconds=1)
    assert job["id"] not in {due["id"] for due in jobs.get_due_jobs()}


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

def test_shutdown_replay_claim_race_releases_reservation(temp_home, monkeypatch):
    job_id, scheduled, _first_id = _interrupted_recurring_job(temp_home)
    assert jobs.claim_job_for_fire(job_id, scheduled_at=scheduled)

    assert _recover_and_fire(monkeypatch) == []
    reservations = [
        row for row in executions.list_executions(job_id=job_id) if row["replay_of"] is not None
    ]
    assert reservations == []

    stored = jobs.load_jobs()
    next(job for job in stored if job["id"] == job_id)["fire_claim"] = None
    jobs.save_jobs(stored)
    replayed = _recover_and_fire(monkeypatch)
    assert [item["_scheduled_instant"] for item in replayed] == [scheduled]
    assert _recover_and_fire(monkeypatch) == []


@pytest.mark.parametrize("owner_state", ["dead", "stale"])
def test_shutdown_recovery_reclaims_unbound_reservation(temp_home, monkeypatch, owner_state):
    job_id, scheduled, first_id = _interrupted_recurring_job(temp_home)
    abandoned = executions.create_shutdown_replay(job_id, scheduled, first_id)
    assert abandoned is not None
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id=?, pid=?, process_started_at=NULL, fire_claim_by=NULL "
            "WHERE id=?",
            ("exited-provider", 99999999, abandoned["id"]),
        )
    if owner_state == "dead":
        monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    else:
        monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: True)
        monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 1)
        monkeypatch.setattr(executions, "_claim_age_seconds", lambda _claimed_at: 2)

    replayed = _recover_and_fire(monkeypatch)

    assert [item["_scheduled_instant"] for item in replayed] == [scheduled]
    reservations = [
        row for row in executions.list_executions(job_id=job_id) if row["replay_of"] is not None
    ]
    assert len(reservations) == 1
    assert reservations[0]["id"] != abandoned["id"]


def test_shutdown_replay_save_failure_releases_reservation(temp_home, monkeypatch):
    job_id, scheduled, first_id = _interrupted_recurring_job(temp_home)
    replay = executions.create_shutdown_replay(job_id, scheduled, first_id)
    assert replay is not None

    def fail_save(_jobs, **_kwargs):
        raise OSError("jobs store unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "save_jobs", fail_save)
        with pytest.raises(OSError, match="jobs store unavailable"):
            jobs.claim_job_for_fire(
                job_id,
                return_job=True,
                scheduled_at=scheduled,
                execution_id=replay["id"],
                shutdown_replay=True,
            )

    failed_claim = executions.get_execution(replay["id"])
    assert failed_claim["fire_claim_by"] is None
    assert executions.release_shutdown_replay_reservation(replay["id"])

    replayed = _recover_and_fire(monkeypatch)
    assert [item["_scheduled_instant"] for item in replayed] == [scheduled]
    assert len(replayed) == 1
    assert _recover_and_fire(monkeypatch) == []


def test_shutdown_replay_save_failure_after_durable_claim_keeps_binding(
    temp_home, monkeypatch,
):
    job_id, scheduled, first_id = _interrupted_recurring_job(temp_home)
    replay = executions.create_shutdown_replay(job_id, scheduled, first_id)
    assert replay is not None
    save = jobs.save_jobs

    def save_then_fail(stored, **kwargs):
        save(stored, **kwargs)
        raise OSError("post-save failure")

    monkeypatch.setattr(jobs, "save_jobs", save_then_fail)
    with pytest.raises(OSError, match="post-save failure"):
        jobs.claim_job_for_fire(
            job_id,
            return_job=True,
            scheduled_at=scheduled,
            execution_id=replay["id"],
            shutdown_replay=True,
        )

    execution = executions.get_execution(replay["id"])
    assert execution["fire_claim_by"] is not None
    assert jobs.get_job(job_id)["fire_claim"]["by"] == execution["fire_claim_by"]

    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    assert executions.create_shutdown_replay(job_id, scheduled, first_id) is None


def test_reclaimed_live_reserver_cannot_claim_after_replacement(
    temp_home, monkeypatch,
):
    job_id, scheduled, first_id = _interrupted_recurring_job(temp_home)
    abandoned = executions.create_shutdown_replay(job_id, scheduled, first_id)
    assert abandoned is not None
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: True)
    monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 1)
    monkeypatch.setattr(executions, "_claim_age_seconds", lambda _claimed_at: 2)
    create_replay = executions.create_shutdown_replay

    old_claim_job_states = []
    replacements = []
    old_claims = []

    def reserve_replacement(replay_job_id, instant, replay_of):
        replacement = create_replay(replay_job_id, instant, replay_of)
        if replacement is not None:
            replacements.append(replacement)
            old_claims.append(jobs.claim_job_for_fire(
                job_id,
                return_job=True,
                scheduled_at=scheduled,
                execution_id=abandoned["id"],
                shutdown_replay=True,
            ))
            old_claim_job_states.append(jobs.get_job(job_id).get("fire_claim"))
        return replacement

    monkeypatch.setattr(executions, "create_shutdown_replay", reserve_replacement)
    monkeypatch.setattr(executions, "recover_interrupted_executions", lambda: 0)
    provider = InProcessCronScheduler()
    dispatched = []

    def finish_replay(claimed, **_kwargs):
        dispatched.append(claimed["execution_id"])
        executions.finish_execution(claimed["execution_id"], success=True)
        return True

    monkeypatch.setattr(provider, "fire_claimed", finish_replay)
    provider.recover_interrupted()

    assert old_claim_job_states == [None]
    assert len(replacements) == 1
    assert old_claims == [False]
    stale = executions.get_execution(abandoned["id"])
    assert stale["replay_of"] is None
    assert stale["fire_claim_by"] is None
    assert dispatched == [replacements[0]["id"]]


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
