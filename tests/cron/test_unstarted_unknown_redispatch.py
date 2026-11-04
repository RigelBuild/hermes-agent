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
    assert lost["id"] in incidents[0]["error"]
    assert "never started" in incidents[0]["error"]
    assert "re-dispatched" in incidents[0]["error"]

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
    assert lost["id"] in incidents[0]["error"]
    assert "had started" in incidents[0]["error"]
    assert "not re-run" in incidents[0]["error"]
