"""Replay job setup keeps delivery authorization separate from read authorization."""

from datetime import UTC, datetime
from types import SimpleNamespace

from gateway import session_cron
from tools import discord_tool


class _SessionDB:
    def set_session_title(self, *args, **kwargs):
        return True

    def get_compression_tip(self, session_id):
        return None

    def session_lifecycle_statuses(self, session_ids):
        return {session_id: "complete" for session_id in session_ids}

    def end_session(self, session_id, reason):
        pass

    def close(self):
        pass


def test_run_job_with_dm_discord_delivery_binds_without_validating_target(
    tmp_path, monkeypatch, cron_owner
):
    from cron import scheduler

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n", encoding="utf-8"
    )
    db = _SessionDB()
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda job: db)
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": "dm-1", "thread_id": None}],
    )
    monkeypatch.setattr(
        scheduler, "_reload_dotenv_and_publish_delivery_target", lambda job: None
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_cron_agent_setup",
        lambda *args: SimpleNamespace(
            blocked=None,
            model="test-model",
            runtime={},
            prefill_messages=None,
            max_iterations=None,
            reasoning_config=None,
            fallback_model=None,
            credential_pool=None,
            fallback_notice=None,
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_construct_cron_agent",
        lambda *args, **kwargs: SimpleNamespace(
            run_conversation=lambda *args, **kwargs: {
                "completed": True,
                "failed": False,
                "final_response": "done",
                "turn_exit_reason": "",
            },
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_agent_with_watchdog",
        lambda *args, **kwargs: {
            "completed": True,
            "failed": False,
            "final_response": "done",
            "turn_exit_reason": "",
        },
    )
    monkeypatch.setattr(
        scheduler, "_finalize_cron_session", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scheduler, "_write_usage_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scheduler, "_run_cron_cleanup_with_timeout", lambda fn, **kwargs: fn()
    )
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(discord_tool, "_in_cron_session", lambda: True)
    monkeypatch.setattr(scheduler, "_cron_preflight_enabled", lambda cfg: False)

    job = {
        "id": "replay-dm",
        "name": "DM replay",
        "prompt": "Do it",
        "deliver": "origin",
        "fire_claim": {
            "replay": {
                "started_at": datetime(2026, 11, 10, tzinfo=UTC).isoformat(),
            }
        },
    }
    execution = session_cron._execution.set((None, "session", job["id"], "fire"))
    try:
        result = scheduler.run_job(job, execution_id="fire")
        assert result[0] is True
        assert result[2] == "done"
        assert result[3] is None
    finally:
        session_cron._execution.reset(execution)


def test_prompt_toolset_resolver_keeps_disabled_discord_out_of_replay_hint(
    tmp_path, monkeypatch
):
    from cron import scheduler, scheduler_prompt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: test-model\nplatform_toolsets:\n  cron: [hermes-discord]\n"
        "agent:\n  disabled_toolsets: [discord]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [
            {
                "platform": "discord",
                "chat_id": "guild-channel",
                "thread_id": None,
            }
        ],
    )
    job = {
        "id": "replay-disabled-discord",
        "prompt": "Do it",
        "fire_claim": {"replay": {"started_at": "2026-11-10T00:00:00+00:00"}},
    }
    prompt = scheduler_prompt._replay_hint(job["fire_claim"]["replay"], job)
    assert "discord tool" not in prompt
    assert "check what this job already delivered before repeating" in prompt


def test_replay_hint_falls_back_when_toolset_lookup_fails(monkeypatch):
    from cron import scheduler, scheduler_prompt

    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": "guild-channel", "thread_id": None}],
    )

    def fail_load_config():
        raise SystemExit("backend unavailable")

    monkeypatch.setattr(scheduler, "load_config_readonly", fail_load_config)
    replay = {"started_at": "2026-11-10T00:00:00+00:00"}
    prompt = scheduler_prompt._replay_hint(replay, {"id": "replay-hint"})
    assert "discord tool" not in prompt
    assert "check what this job already delivered before repeating" in prompt


def test_run_job_records_discord_replay_toolset_resolution_failure(
    tmp_path, monkeypatch, cron_owner
):
    from cron import scheduler

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\nplatform_toolsets:\n  cron: oops\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": "guild-channel", "thread_id": None}],
    )
    monkeypatch.setattr(
        scheduler, "_reload_dotenv_and_publish_delivery_target", lambda job: None
    )
    monkeypatch.setattr(scheduler, "_cron_preflight_enabled", lambda cfg: False)
    resolve_toolsets = scheduler._resolve_cron_enabled_toolsets

    def fail_on_malformed_cron_toolsets(job, cfg):
        if (cfg.get("platform_toolsets") or {}).get("cron") == "oops":
            raise RuntimeError("Cron toolset resolution failed")
        return resolve_toolsets(job, cfg)

    monkeypatch.setattr(
        scheduler, "_resolve_cron_enabled_toolsets", fail_on_malformed_cron_toolsets
    )
    monkeypatch.setattr(
        scheduler, "_resolve_job_runtime", lambda job, job_id, jc: ({"provider": "test"}, "test-model")
    )
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda job: _SessionDB())
    monkeypatch.setattr(scheduler, "_finalize_cron_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "_write_usage_audit", lambda *args, **kwargs: None)
    job = {
        "id": "replay-bad-toolsets",
        "name": "Bad toolsets replay",
        "prompt": "Do it",
        "deliver": "origin",
        "fire_claim": {"replay": {"started_at": "2026-11-10T00:00:00+00:00"}},
    }
    execution = session_cron._execution.set((None, "session", job["id"], "fire"))
    try:
        result = scheduler.run_job(job, execution_id="fire")
        assert result[0] is False
        assert "Cron toolset resolution failed" in result[3]
    finally:
        session_cron._execution.reset(execution)
