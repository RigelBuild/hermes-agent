"""Replay prompts preserve the interrupted occurrence's delivery context."""
import asyncio
import json
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cron import scheduler_prompt

START = "2026-11-07T10:00:00+00:00"
STOP = "2026-11-07T10:05:00+00:00"


def notice(start, stop):
    return (
        f"REPLAY: A run of this same scheduled occurrence started at {start} "
        f"and was stopped by a gateway restart at {stop}. It may already "
        "have posted messages or delivered part of its result. Before you post "
        "anything, read what is there: use the discord tool (list_forum_posts, "
        "fetch_messages) on the channels and threads this job posts to. Look at "
        f"every message sent at or after {start}. Do not post again anything "
        "that is already there. Do only the steps that are missing. If nothing is "
        "missing, respond with exactly [SILENT].\n\n"
    )


@pytest.fixture
def prompt_env(tmp_path, monkeypatch):
    from cron import notepad
    from tools import approval_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(notepad, "render_notepad_section", lambda job_id: "")
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")


def test_replay_notice_follows_cron_hint_and_preserves_normal_prompt(prompt_env):
    job = {"id": "job", "prompt": "Do the task."}
    normal = scheduler_prompt._build_job_prompt(job, extra_prompt="Run detail")
    assert normal == scheduler_prompt._CRON_HINT + "Do the task.\n\n## Run Context\nRun detail"
    replay = dict(job, fire_claim={"replay": {"started_at": START, "interrupted_at": STOP}})
    actual = scheduler_prompt._build_job_prompt(replay, extra_prompt="Run detail")
    assert actual == scheduler_prompt._CRON_HINT + notice(START, STOP) + normal[len(scheduler_prompt._CRON_HINT):]


@pytest.mark.parametrize("marker", [None, "broken", 42, {}])
def test_malformed_replay_marker_uses_unknown_times(prompt_env, marker):
    job = {"id": "job", "prompt": "Do the task.", "fire_claim": {"replay": marker}}
    assert scheduler_prompt._build_job_prompt(job) == scheduler_prompt._CRON_HINT + notice("unknown", "unknown") + "Do the task."


@pytest.mark.parametrize("with_skill", [False, True])
def test_replay_notice_passes_assembled_prompt_scanner(prompt_env, monkeypatch, with_skill):
    job = {"id": "job", "prompt": "Do the task.", "fire_claim": {"replay": {"started_at": START, "interrupted_at": STOP}}}
    if with_skill:
        job["skills"] = ["example"]
        monkeypatch.setattr(scheduler_prompt, "_load_cron_skill_parts", lambda job, names: ["Use this skill."])
    prompt = scheduler_prompt._build_job_prompt(job)
    assert notice(START, STOP) in prompt
    assert scheduler_prompt._scan_assembled_cron_prompt(prompt, job, has_skills=with_skill) == prompt


def test_canonical_creation_and_execution_use_stored_replay(prompt_env, monkeypatch, tmp_path):
    from cron import jobs, scheduler
    from gateway import run, session_cron, session_local_recovery, session_policy

    stored_job = {"id": "job", "prompt": "Do the task.", "fire_claim": {"replay": {"started_at": START, "interrupted_at": STOP}}}
    monkeypatch.setattr(jobs, "get_job", lambda jid: stored_job if jid == "job" else None)
    monkeypatch.setattr(run, "_load_gateway_config", dict)
    monkeypatch.setattr(run, "_profile_runtime_scope", lambda home: nullcontext())
    monkeypatch.setattr(scheduler, "_load_cron_job_config", lambda *args: SimpleNamespace(model="test"))

    @dataclass
    class Policy:
        source: str = "cli"
        platform: str = "cli"
        model: str = "test"
        request_json: str = ""
    monkeypatch.setattr(session_policy, "build_policy", lambda *args, **kwargs: Policy())
    monkeypatch.setattr(session_policy, "bind_launch_key", lambda authority, sid, policy, *args, **kwargs: policy)
    monkeypatch.setattr(session_local_recovery, "local_identity", lambda *args: "session")
    monkeypatch.setattr(session_local_recovery, "local_source", lambda *args: SimpleNamespace())
    from gateway import session
    monkeypatch.setattr(session.SessionEntry, "to_dict", lambda self: {})
    authority = SimpleNamespace(
        profile_id="profile", db=SimpleNamespace(db_path=str(tmp_path / "state.db"), get_session=lambda sid: None),
        runner=SimpleNamespace(session_store=SimpleNamespace(_generate_session_key=lambda source: "route")),
        sessions={"session": SimpleNamespace(source=SimpleNamespace(user_id="actor"))}, pending_results={},
    )
    actor = SimpleNamespace(subject="actor")
    receipt, request_id = session_cron._create(authority, actor, {"job_id": "job", "request_id": "fire", "extra_prompt": None})
    policy = Policy(**receipt["policy"])
    assert json.loads(policy.request_json)["cron_job"] == stored_job
    monkeypatch.setattr(session_cron, "_run_identified", lambda run_job, job, execution_id, **kwargs: (True, "", scheduler_prompt._build_job_prompt(job), None))
    row = {"payload": {"text": ""}, "request_id": request_id, "principal_id": "actor", "admission_id": "admission"}
    prompt = asyncio.run(session_cron.execute(authority, SimpleNamespace(session_id="session"), row, policy))
    assert prompt == scheduler_prompt._CRON_HINT + notice(START, STOP) + "Do the task."
