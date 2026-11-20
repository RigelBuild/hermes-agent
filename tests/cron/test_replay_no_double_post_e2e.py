"""A cut-off check-in replays exactly once and counts once (RIG-8084).

Covers the cron store and prompt construction across a shutdown mid-drain: a fire is
in flight, gateway shutdown sweeps it (``mark_running_jobs_interrupted``), the pending
replay persists to the store, the scheduler restarts, and the next fire claims the
interrupted occurrence. The replay run's prompt carries exactly one REPLAY notice.
Once that run finishes, the occurrence counts once, and a later fire carries no replay.

It does not call ``scheduler.run_job`` or the agent, and it does not reach delivery or
Discord. Delivery targets and toolsets are stubbed in ``prompt_env``.
"""

from datetime import UTC, datetime, timedelta

import pytest

from cron import jobs, scheduler, scheduler_prompt, unreachable_retry

REPLAY_NOTICE = "REPLAY: A run of this same scheduled occurrence"


@pytest.fixture
def clock(monkeypatch):
    now = [datetime(2026, 11, 10, 10, 0, tzinfo=UTC)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(unreachable_retry, "_hermes_now", lambda: now[0])
    return now


@pytest.fixture
def prompt_env(monkeypatch, tmp_path):
    from cron import notepad
    from tools import approval_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(notepad, "render_notepad_section", lambda job_id: "")
    monkeypatch.setattr(scheduler_prompt, "_run_time_line", lambda: "")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
    monkeypatch.setattr(scheduler, "load_config_readonly", dict)
    monkeypatch.setattr(
        scheduler, "_resolve_cron_enabled_toolsets", lambda job, cfg: ["hermes-discord"]
    )
    monkeypatch.setattr(scheduler, "_resolve_cron_disabled_toolsets", lambda cfg: [])
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": "700", "thread_id": None}],
    )


def _claim(job_id, clock):
    clock[0] += timedelta(seconds=1)
    assert any(j["id"] == job_id for j in jobs.get_due_jobs())
    return jobs.claim_job_for_fire(job_id, return_job=True)


def _claim_next_occurrence(job_id, clock):
    clock[0] = datetime.fromisoformat(jobs.get_job(job_id)["next_run_at"])
    return _claim(job_id, clock)


def test_shutdown_mid_drain_replays_once_and_counts_once(
    clock, prompt_env, monkeypatch, tmp_path
):
    with jobs.use_cron_store(tmp_path / "cron"):
        # 1. A check-in fire is in flight. Its job delivers to Discord.
        job = jobs.create_job("check-in", "every 24h", repeat=3, deliver="discord:700")
        clock[0] = datetime.fromisoformat(job["next_run_at"])
        first = _claim(job["id"], clock)
        owner = first["fire_claim"]["by"]
        key = scheduler._inflight_key(job["id"])
        monkeypatch.setattr(
            scheduler,
            "_running_fire_owners",
            {key: {object(): (owner, tmp_path / "cron", None)}},
        )
        monkeypatch.setattr(scheduler, "_interrupted_job_ids", set())

        # 2. Gateway shutdown sweeps the in-flight fire mid-drain.
        marked = scheduler.mark_running_jobs_interrupted(
            "gateway shutdown", only_owners={(job["id"], owner)}
        )
        assert job["id"] in marked

        # 3. State persists: the occurrence is pending as a replay, and the repeat budget is untouched.
        stored = jobs.get_job(job["id"])
        assert stored["pending_slot"]["replay"]
        assert stored["repeat"]["completed"] == 0

        # 4. The scheduler restarts. The next fire claims the interrupted occurrence.
        monkeypatch.setattr(scheduler, "_running_fire_owners", {})
        monkeypatch.setattr(scheduler, "_interrupted_job_ids", set())
        replay_fire = _claim(job["id"], clock)
        assert replay_fire["fire_claim"]["replay"] == stored["pending_slot"]["replay"]

        # 5. The replay run's prompt carries exactly one REPLAY notice, and it reads Discord first.
        prompt = scheduler_prompt._build_job_prompt(replay_fire)
        assert prompt.count(REPLAY_NOTICE) == 1
        assert "use the discord tool (list_forum_posts, fetch_messages)" in prompt

        # 6. The replay finishes. The occurrence counts once and leaves no pending replay.
        assert jobs.mark_job_run(
            job["id"], True, expected_fire_owner=replay_fire["fire_claim"]["by"]
        )
        done = jobs.get_job(job["id"])
        assert done["repeat"]["completed"] == 1
        assert "pending_slot" not in done

        # 7. The next scheduled occurrence is a fresh run: it carries no replay notice.
        later = _claim_next_occurrence(job["id"], clock)
        assert "replay" not in later["fire_claim"]
        assert scheduler_prompt._build_job_prompt(later).count(REPLAY_NOTICE) == 0
