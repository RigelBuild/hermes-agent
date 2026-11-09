"""Interrupted recurring occurrences remain pending and spend repeat budget once."""

from datetime import UTC, datetime, timedelta

import pytest

from cron import jobs, unreachable_retry


@pytest.fixture
def clock(monkeypatch):
    now = [datetime.now(UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(unreachable_retry, "_hermes_now", lambda: now[0])
    return now


def claim_due(job_id, clock):
    clock[0] += timedelta(seconds=1)
    due = jobs.get_due_jobs()
    assert any(job["id"] == job_id for job in due)
    return jobs.claim_job_for_fire(job_id, return_job=True)


def test_final_repeat_is_counted_when_replay_finishes(clock):
    job = jobs.create_job("report", "every 24h", repeat=1)
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    owner = first["fire_claim"]["by"]
    started = first["fire_claim"]["claimed_at"]
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=owner,
        shutdown_interrupted=True,
    )
    pending = jobs.get_job(job["id"])
    assert pending["repeat"]["completed"] == 0
    assert pending["pending_slot"]["replay"]["started_at"] == started
    replay = claim_due(job["id"], clock)
    assert replay["fire_claim"]["replay"]["started_at"] == started
    assert jobs.mark_job_run(
        job["id"], True, expected_fire_owner=replay["fire_claim"]["by"]
    )
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


def test_second_shutdown_does_not_replay_again(clock):
    job = jobs.create_job("report", "every 24h", repeat=3)
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=first["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    replay = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=replay["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    result = jobs.get_job(job["id"])
    assert result["repeat"]["completed"] == 1
    assert "pending_slot" not in result


def test_interrupted_retry_rung_does_not_count_again(clock):
    job = jobs.create_job("report", "every 24h", repeat=3)
    assert jobs.mark_job_run(job["id"], False, "offline", model_unreachable=True)
    retry = jobs.get_job(job["id"])["unreachable_retry"]
    clock[0] = datetime.fromisoformat(retry["at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=first["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1
    replay = claim_due(job["id"], clock)
    assert replay["fire_claim"]["scheduled_at"] == first["fire_claim"]["scheduled_at"]
    assert "replay" in replay["fire_claim"]
    assert jobs.mark_job_run(
        job["id"], True, expected_fire_owner=replay["fire_claim"]["by"]
    )
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


def test_unreachable_replay_carries_marker_to_retry(clock):
    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=first["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    replay = claim_due(job["id"], clock)
    marker = replay["fire_claim"]["replay"]
    assert jobs.mark_job_run(
        job["id"],
        False,
        "offline",
        expected_fire_owner=replay["fire_claim"]["by"],
        model_unreachable=True,
    )
    stored = jobs.get_job(job["id"])
    assert stored["unreachable_retry"]["replay"] == marker
    clock[0] = datetime.fromisoformat(stored["unreachable_retry"]["at"])
    rung = claim_due(job["id"], clock)
    assert rung["fire_claim"]["replay"] == marker
    assert jobs.mark_job_run(
        job["id"], True, expected_fire_owner=rung["fire_claim"]["by"]
    )
    assert "unreachable_retry" not in jobs.get_job(job["id"])


@pytest.mark.parametrize("claim", [None, "malformed"])
def test_no_claim_counts_without_replay(clock, claim):
    job = jobs.create_job("report", "every 24h", repeat=3)
    records = jobs.load_jobs()
    records[0]["fire_claim"] = claim
    jobs.save_jobs(records)
    assert jobs.mark_job_run(job["id"], False, "shutdown", shutdown_interrupted=True)
    stored = jobs.get_job(job["id"])
    assert stored["repeat"]["completed"] == 1
    assert "pending_slot" not in stored


def test_restore_unstarted_preserves_marker(clock):
    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"],
        False,
        "shutdown",
        expected_fire_owner=first["fire_claim"]["by"],
        shutdown_interrupted=True,
    )
    replay = claim_due(job["id"], clock)
    claim = replay["fire_claim"]
    assert jobs.restore_unstarted_occurrence(
        job["id"], claim["scheduled_at"], claim["by"]
    )
    assert jobs.get_job(job["id"])["pending_slot"]["replay"] == claim["replay"]


def test_refused_dispatch_keeps_replay_stamp(clock, monkeypatch):
    from cron import scheduler_gateway_gate

    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    clock[0] += timedelta(seconds=1)
    assert any(j["id"] == job["id"] for j in jobs.get_due_jobs())
    scheduler_gateway_gate._note_fire_refused([job["id"]], "no gateway")
    stored = jobs.get_job(job["id"])
    assert stored["pending_slot"]["replay"] is not None
    assert stored["last_fire_error"]["detail"] == "no gateway"


def test_second_shutdown_does_not_stamp_again(clock):
    job = jobs.create_job("report", "every 24h", repeat=3)
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    replay = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=replay["fire_claim"]["by"], shutdown_interrupted=True,
    )
    stored = jobs.get_job(job["id"])
    assert stored["repeat"]["completed"] == 1
    assert "pending_slot" not in stored or "replay" not in stored["pending_slot"]


def test_one_shot_interruption_stamps_no_replay(clock):
    job = jobs.create_job("report", "in 30 minutes", repeat=1)
    clock[0] = datetime.fromisoformat(jobs.get_job(job["id"])["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    assert "replay" not in (jobs.get_job(job["id"]).get("pending_slot") or {})


def test_no_claim_stamps_nothing(clock):
    job = jobs.create_job("report", "every 24h")
    assert jobs.mark_job_run(job["id"], False, "shutdown", shutdown_interrupted=True)
    assert "replay" not in (jobs.get_job(job["id"]).get("pending_slot") or {})


def _interrupted_replay_slot(clock):
    job = jobs.create_job("report", "every 24h", repeat=3)
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    return job["id"]


def _lock_fd(tmp_path):
    def acquire(_sched):
        return open(tmp_path / "tick.lock", "a+")

    return acquire


def test_estop_tick_leaves_replay_pending(clock, monkeypatch, tmp_path):
    from agent import estop
    from cron import scheduler_tick

    job_id = _interrupted_replay_slot(clock)
    before = jobs.get_job(job_id)["pending_slot"]
    monkeypatch.setattr(estop, "check_paused", lambda *_a, **_k: True)
    monkeypatch.setattr(scheduler_tick, "_acquire_tick_lock_or_degrade", _lock_fd(tmp_path))
    assert scheduler_tick._tick_admitted(False, None, None, True) == 0
    assert jobs.get_job(job_id)["pending_slot"] == before


def test_draining_tick_leaves_replay_pending(clock, monkeypatch, tmp_path):
    from cron import scheduler_tick

    job_id = _interrupted_replay_slot(clock)
    before = jobs.get_job(job_id)["pending_slot"]
    monkeypatch.setattr(scheduler_tick, "_acquire_tick_lock_or_degrade", _lock_fd(tmp_path))
    assert scheduler_tick._tick_admitted(False, None, None, True, can_dispatch=lambda: False) == 0
    assert jobs.get_job(job_id)["pending_slot"] == before


def test_crash_between_scan_and_claim_keeps_replay(clock):
    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    marker = jobs.get_job(job["id"])["pending_slot"]["replay"]
    # The scan moves the slot (as a crash would leave it) and no claim is taken.
    clock[0] += timedelta(seconds=1)
    assert any(j["id"] == job["id"] for j in jobs.get_due_jobs())
    assert jobs.get_job(job["id"]).get("fire_claim") is None
    replay = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert replay["fire_claim"]["replay"] == marker


def test_stale_ladder_plans_first_attempt_for_next_occurrence(clock):
    from cron import unreachable_retry

    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
    )
    replay = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "offline",
        expected_fire_owner=replay["fire_claim"]["by"], model_unreachable=True,
    )
    # Drop the replay (a schedule edit / pause clears the marker), leaving an exhausted ladder.
    stored = jobs.load_jobs()
    stored[0][unreachable_retry.STATE_KEY]["attempt"] = 3
    stored[0].pop("pending_slot", None)
    stored[0].pop("restored_replay", None)
    stored[0]["next_run_at"] = (
        datetime.fromisoformat(stored[0]["next_run_at"]) + timedelta(hours=1)
    ).isoformat()
    jobs.save_jobs(stored)
    clock[0] = datetime.fromisoformat(jobs.get_job(job["id"])["next_run_at"])
    nxt = claim_due(job["id"], clock)
    assert jobs.mark_job_run(
        job["id"], False, "offline",
        expected_fire_owner=nxt["fire_claim"]["by"], model_unreachable=True,
    )
    assert jobs.get_job(job["id"])[unreachable_retry.STATE_KEY]["attempt"] == 1


def test_catch_up_disabled_drops_replay_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job("report", "every 1h")
        stored = jobs.load_jobs()
        stale = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        stored[0]["pending_slot"] = {
            "scheduled_at": stale,
            "replay": {"scheduled_at": stale, "started_at": stale},
        }
        stored[0]["next_run_at"] = stale
        jobs.save_jobs(stored)
        assert job["id"] not in [j["id"] for j in jobs.get_due_jobs()]
        after = jobs.get_job(job["id"])
        assert "replay" not in (after.get("pending_slot") or {})
        assert "restored_replay" not in after


def test_multiplex_profiles_keep_replay_in_their_own_store(tmp_path, clock):
    """Case 4: each profile's replay lives in its own store; a gated-out profile's replay stays
    pending and does not leak into another profile's claim."""
    stores = {}
    for name in ("alpha", "beta"):
        with jobs.use_cron_store(tmp_path / name):
            job = jobs.create_job(name, "every 24h")
            stores[name] = job["id"]
            clock[0] = datetime.fromisoformat(job["next_run_at"])
            first = claim_due(job["id"], clock)
            assert jobs.mark_job_run(
                job["id"], False, "shutdown",
                expected_fire_owner=first["fire_claim"]["by"], shutdown_interrupted=True,
            )
    # Only beta is admitted this tick; alpha's gate refuses it, so alpha is never claimed.
    with jobs.use_cron_store(tmp_path / "beta"):
        beta = claim_due(stores["beta"], clock)
        assert beta["fire_claim"]["replay"] is not None
    with jobs.use_cron_store(tmp_path / "alpha"):
        stored = jobs.get_job(stores["alpha"])
        assert stored["pending_slot"]["replay"] is not None
        assert stored.get("fire_claim") is None


def test_heartbeat_does_not_move_replay_started_at(clock):
    """Case 10: the heartbeat refreshes the claim's ``at`` but never ``claimed_at``; the replay's
    ``started_at`` stays at the original claim, before any message the run posted."""
    job = jobs.create_job("report", "every 24h")
    clock[0] = datetime.fromisoformat(job["next_run_at"])
    first = claim_due(job["id"], clock)
    claimed_at = first["fire_claim"]["claimed_at"]
    owner = first["fire_claim"]["by"]
    clock[0] += timedelta(minutes=10)
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
    assert jobs.get_job(job["id"])["fire_claim"]["at"] != claimed_at
    assert jobs.mark_job_run(
        job["id"], False, "shutdown",
        expected_fire_owner=owner, shutdown_interrupted=True,
    )
    replay = jobs.get_job(job["id"])["pending_slot"]["replay"]
    assert replay["started_at"] == claimed_at
