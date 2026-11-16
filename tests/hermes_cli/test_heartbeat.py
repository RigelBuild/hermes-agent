"""Tests for /heartbeat (hermes_cli/heartbeat.py)."""

import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from hermes_cli.heartbeat import (
    HeartbeatManager,
    HeartbeatState,
    MIN_INTERVAL_SECONDS,
    format_interval,
    load_heartbeat,
    migrate_heartbeat_to_session,
    parse_interval,
    parse_heartbeat_set,
    save_heartbeat,
)

@pytest.mark.parametrize("command,method", [
    ("/heartbeat 1m check", "set"),
    ("/heartbeat pause", "pause"),
    ("/heartbeat resume", "resume"),
    ("/heartbeat clear", "clear"),
])
def test_cli_reports_route_lookup_failure(command, method, monkeypatch):
    from unittest.mock import MagicMock
    from hermes_cli import cli_commands_mixin as commands

    manager = MagicMock()
    getattr(manager, method).side_effect = RuntimeError("heartbeat route lookup failed")
    cli = commands.CLICommandsMixin()
    cli._session_manager = lambda *args: manager
    cli._get_heartbeat_manager = lambda: manager
    cli._start_heartbeat_watchdog = MagicMock()
    output = []
    monkeypatch.setattr(commands, "_cp", lambda *args: output.extend(args))
    cli._handle_heartbeat_command(command)
    assert len(output) == 1
    assert "heartbeat route lookup failed" in output[0]
    cli._start_heartbeat_watchdog.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# interval parsing
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10m", 600),
        ("every 10m", 600),
        ("2h", 7200),
        ("every 2 hours", 7200),
        ("1d", 86400),
        ("90 minutes", 5400),
        ("600s", 600),
    ],
)
def test_parse_interval_valid(text, expected):
    assert parse_interval(text) == expected


@pytest.mark.parametrize("text", ["", "banana", "check CI", "every", "m10"])
def test_parse_interval_not_an_interval(text):
    assert parse_interval(text) is None


def test_parse_interval_too_small_is_rejected():
    assert parse_interval("5s") == -1
    assert parse_interval("30s") == -1
    assert parse_interval(f"{MIN_INTERVAL_SECONDS}s") == MIN_INTERVAL_SECONDS


def test_format_interval():
    assert format_interval(600) == "10m"
    assert format_interval(7200) == "2h"
    assert format_interval(86400) == "1d"
    assert format_interval(90) == "90s"


@pytest.mark.parametrize("text", [
    'at "30 7 * * *" Check deployment',
    'at "30 7 * * *" tz America/New_York --idle Check deployment',
])
def test_parse_scheduled_heartbeat(text):
    from hermes_cli.heartbeat import parse_heartbeat_set

    parsed = parse_heartbeat_set(text)
    assert parsed == (None, "30 7 * * *", "America/New_York" if "tz " in text else None,
                      "Check deployment", "idle" if "--idle" in text else "steer")


# ──────────────────────────────────────────────────────────────────────
# state + due logic
# ──────────────────────────────────────────────────────────────────────


def test_state_roundtrip():
    s = HeartbeatState(prompt="check CI", interval_seconds=600, created_at=time.time())
    loaded = HeartbeatState.from_json(s.to_json())
    assert loaded.prompt == "check CI"
    assert loaded.interval_seconds == 600
    assert loaded.status == "active"


def test_scheduled_state_roundtrip():
    s = HeartbeatState(prompt="check CI", interval_seconds=0, schedule="30 7 * * *",
                       timezone="America/New_York", created_at=time.time())
    loaded = HeartbeatState.from_json(s.to_json())
    assert (loaded.schedule, loaded.timezone) == ("30 7 * * *", "America/New_York")


@pytest.mark.parametrize("interval,schedule", [(600, "30 7 * * *"), (0, None), (None, None)])
def test_manager_requires_exactly_one_schedule(interval, schedule):
    mgr = HeartbeatManager(session_id=f"hb-exclusive-{interval}-{schedule}")
    with pytest.raises(ValueError, match="exactly one"):
        mgr.set("watch CI", interval, schedule=schedule)

@pytest.mark.parametrize("schedule,zone", [("30 7 * * *", "Mars/Base"), ("* * * * * *", "UTC")])
def test_manager_rejects_invalid_cron(schedule, zone):
    with pytest.raises(ValueError):
        HeartbeatManager("invalid-cron").set("check", schedule=schedule, timezone=zone)


def test_corrupt_schedule_is_safe():
    from tui_gateway.methods_session_control import _safe_heartbeat_snapshot

    state = HeartbeatState(prompt="check", interval_seconds=0, schedule="bad cron", timezone="Mars/Base")
    assert not state.valid_cadence()
    assert state.last_error
    assert _safe_heartbeat_snapshot(state)["next_fire_at"] is None


def test_invalid_stale_manager_does_not_resume_paused_heartbeat():
    from hermes_cli.heartbeat import save_heartbeat

    state = HeartbeatState(prompt='check', interval_seconds=0, schedule='bad cron', created_at=time.time())
    save_heartbeat('invalid-stale', state)
    stale = HeartbeatManager('invalid-stale')
    HeartbeatManager('invalid-stale').pause()
    assert stale.due_prompt() is None
    assert load_heartbeat('invalid-stale').status == 'paused'

def test_old_gateway_rejects_schedule_with_clear_error():
    from hermes_cli.gateway_automation import GatewayHeartbeatManager

    class OldConnection:
        def call(self, method, **kwargs):
            assert method == "session.automation"
            assert kwargs["kwargs"]["schedule"] == "30 7 * * *"
            return {"state": None, "error": {"type": "TypeError", "message": "unexpected keyword argument 'schedule'"}}

    manager = GatewayHeartbeatManager("old", OldConnection(), load=False)
    with pytest.raises(RuntimeError, match="gateway does not support scheduled heartbeats; update the gateway"):
        manager.set("check", schedule="30 7 * * *")

def test_gateway_interval_uses_legacy_kwargs():
    from hermes_cli.gateway_automation import GatewayHeartbeatManager

    class Connection:
        def call(self, method, **params):
            assert params['kwargs'] == {'delivery_mode': 'steer'}
            return {'state': None, 'value': None}

    GatewayHeartbeatManager('old', Connection(), load=False).set('check', 60)


def test_gateway_transport_error_suggests_upgrade_for_schedule():
    from hermes_cli.gateway_automation import GatewayHeartbeatManager
    from hermes_cli.gateway_client import GatewayClientError

    class Connection:
        def call(self, method, **params):
            raise GatewayClientError('old gateway')

    with pytest.raises(RuntimeError, match='update the gateway'):
        GatewayHeartbeatManager('old', Connection(), load=False).set('check', schedule='30 7 * * *')

def test_all_locales_receive_neutral_interval():
    from agent.i18n import SUPPORTED_LANGUAGES, t

    state = HeartbeatState(prompt='check', interval_seconds=600)
    assert state.cadence_label() == '10m'
    for lang in SUPPORTED_LANGUAGES:
        for key in ('gateway.heartbeat.set', 'gateway.heartbeat.resumed',
                    'cli.commands.heartbeat.set', 'cli.commands.heartbeat.resumed'):
            rendered = t(key, lang=lang, interval=state.cadence_label(), prompt='check')
            assert 'every every' not in rendered
            if lang != 'en':
                assert 'every 10m' not in rendered

@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2026-03-06T12:00:00-05:00", "2026-03-07T07:30:00-05:00"),
        ("2026-03-07T12:00:00-05:00", "2026-03-08T07:30:00-04:00"),
        ("2026-03-08T12:00:00-04:00", "2026-03-09T07:30:00-04:00"),
        ("2026-10-30T12:00:00-04:00", "2026-10-31T07:30:00-04:00"),
        ("2026-10-31T12:00:00-04:00", "2026-11-01T07:30:00-05:00"),
        ("2026-11-01T12:00:00-05:00", "2026-11-02T07:30:00-05:00"),
    ],
)
def test_schedule_next_fire_preserves_wall_clock_across_dst(start, expected):
    from hermes_cli.heartbeat import _next_scheduled_fire

    next_fire = _next_scheduled_fire("30 7 * * *", "America/New_York", datetime.fromisoformat(start))
    assert next_fire == datetime.fromisoformat(expected).timestamp()

@pytest.mark.parametrize("schedule,start", [
    ("30 2 * * *", "2026-03-07T12:00:00-05:00"),
    ("30 1 * * *", "2026-10-31T12:00:00-04:00"),
])
def test_gap_and_fold_fire_once(schedule, start, monkeypatch):
    from hermes_cli import heartbeat
    anchor = datetime.fromisoformat(start).timestamp()
    monkeypatch.setattr(heartbeat.time, "time", lambda: anchor)
    manager = HeartbeatManager("dst-" + schedule)
    manager.set("check", schedule=schedule, timezone="America/New_York")
    fire = manager.state.next_fire()
    assert manager.due_prompt(fire - 1) is None
    assert manager.due_prompt(fire) is not None
    assert manager.due_prompt(fire + 3600) is None


def _scheduled_fire_walk(schedule, created_at, claims):
    """Claim each fire through ``due_prompt(now)`` and return the ``next_fire`` after each claim.

    ``claims`` is a list of poll instants; each claim re-anchors the next fire the way the poller does.
    """
    manager = HeartbeatManager(f"walk-{schedule}-{created_at.isoformat()}")
    manager.set("check", schedule=schedule, timezone="America/New_York")
    manager.state.created_at = created_at.timestamp()
    manager._save(manager.state)
    fires = []
    for now in claims:
        assert manager.due_prompt(now) is not None
        fires.append(manager.state.next_fire())
    return fires


def test_scheduled_heartbeat_next_fire_at_holds_wall_clock_across_dst():
    """A 07:30 heartbeat claimed on time every day for a year reads 07:30 New York, across both DST changes."""
    manager = HeartbeatManager("walk-dst-year", route_key="test:walk-dst-year")
    manager.set("check", schedule="30 7 * * *", timezone="America/New_York")
    manager.state.created_at = datetime(2026, 2, 1, 7, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    manager._save(manager.state)
    walls = []
    for _ in range(365):
        fire = manager.state.next_fire()
        assert manager.due_prompt(fire) is not None
        walls.append(datetime.fromtimestamp(fire, ZoneInfo("America/New_York")))
    assert {(w.hour, w.minute) for w in walls} == {(7, 30)}
    assert [w.strftime("%Y-%m-%d") for w in walls] == [
        (datetime(2026, 2, 2) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(365)]


@pytest.mark.parametrize("schedule,created_at,claims,expected", [
    # 02:30 does not exist on 2026-03-08; the claim at 03:30 EDT anchors the next fire on 03-09.
    ("30 2 * * *", datetime(2026, 3, 7, 7, 30, tzinfo=ZoneInfo("America/New_York")),
     ["2026-03-08T03:30:00-04:00"], ["2026-03-09T02:30:00-04:00"]),
    # On-time claim of the repeated 01:30; the next fire is Nov 2, not the second 01:30.
    ("30 1 * * *", datetime(2026, 10, 31, 7, 30, tzinfo=ZoneInfo("America/New_York")),
     ["2026-11-01T01:30:00-04:00"], ["2026-11-02T01:30:00-05:00"]),
    # Late poll at 01:10 EST, inside the second 01:xx hour, of the 01:30 EDT fire: next fire is Nov 2.
    ("30 1 * * *", datetime(2026, 10, 31, 7, 30, tzinfo=ZoneInfo("America/New_York")),
     ["2026-11-01T01:10:00-05:00"], ["2026-11-02T01:30:00-05:00"]),
    # A late poll of a per-minute cron coalesces missed ticks: next fire is the next minute after the poll.
    ("* * * * *", datetime(2026, 5, 1, 12, 0, tzinfo=ZoneInfo("America/New_York")),
     ["2026-05-01T12:10:00-04:00"], ["2026-05-01T12:11:00-04:00"]),
    # Weekly cron: the last Monday slot (05-04) is over a day before the claim, so the scan finds none and falls back.
    ("0 9 * * 1", datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("America/New_York")),
     ["2026-05-06T10:00:00-04:00"], ["2026-05-11T09:00:00-04:00"]),
    # Daily cron claimed 24.5 h after its first due slot: the window still holds the 05-03 slot, so the next fire is 05-04.
    ("30 9 * * *", datetime(2026, 5, 1, 9, 30, tzinfo=ZoneInfo("America/New_York")),
     ["2026-05-03T10:00:00-04:00"], ["2026-05-04T09:30:00-04:00"]),
])
def test_scheduled_heartbeat_gap_and_fold_next_fire_at(schedule, created_at, claims, expected):
    fires = _scheduled_fire_walk(schedule, created_at, [datetime.fromisoformat(c).timestamp() for c in claims])
    assert fires == [datetime.fromisoformat(e).timestamp() for e in expected]


def test_scheduled_route_survives_restart_and_fires_once(monkeypatch):
    from hermes_cli import heartbeat

    start = datetime(2026, 3, 7, 12, tzinfo=ZoneInfo("America/New_York")).timestamp()
    fire = datetime(2026, 3, 8, 7, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    monkeypatch.setattr(heartbeat.time, "time", lambda: start)
    manager = HeartbeatManager("old", route_key="test:scheduled-route")
    manager.set("check CI", schedule="30 7 * * *", timezone="America/New_York")
    restored = HeartbeatManager("new", route_key="test:scheduled-route")
    assert restored.state.schedule == "30 7 * * *"
    assert restored.due_prompt(fire - 1) is None
    assert "check CI" in restored.due_prompt(fire)
    assert restored.due_prompt(fire) is None


def test_resume_inside_repeated_hour_does_not_refire_a_fired_slot(monkeypatch):
    """Fired at the first EDT 01:30, paused, resumed at 01:10 EST: the next fire is Nov 2, not the second 01:30."""
    from hermes_cli import heartbeat

    first_fire = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0).timestamp()
    resumed_at = datetime(2026, 11, 1, 1, 10, tzinfo=ZoneInfo("America/New_York"), fold=1).timestamp()
    now = {"t": first_fire}
    monkeypatch.setattr(heartbeat.time, "time", lambda: now["t"])
    manager = HeartbeatManager("resume-dst", route_key="test:resume-dst")
    manager.set("check", schedule="30 1 * * *", timezone="America/New_York")
    manager.state.created_at = datetime(2026, 10, 31, 7, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    manager._save(manager.state)
    assert manager.due_prompt(first_fire) is not None
    manager.pause()
    now["t"] = resumed_at
    manager.resume()
    expected_next = datetime(2026, 11, 2, 1, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    assert manager.state.next_fire() == expected_next
    assert manager.due_prompt(resumed_at) is None
    assert manager.due_prompt(expected_next) is not None
    assert manager.state.fire_count == 2


def test_resume_after_multi_day_pause_waits_for_the_next_slot(monkeypatch):
    """Daily 09:00 paused Mon 10:00, resumed Thu 12:00: the next fire is Fri 09:00, and nothing is due at resume."""
    from hermes_cli import heartbeat

    tz = ZoneInfo("America/New_York")
    monday_fire = datetime(2026, 5, 4, 9, 0, tzinfo=tz).timestamp()
    monday_pause = datetime(2026, 5, 4, 10, 0, tzinfo=tz).timestamp()
    thursday_resume = datetime(2026, 5, 7, 12, 0, tzinfo=tz).timestamp()
    now = {"t": monday_fire}
    monkeypatch.setattr(heartbeat.time, "time", lambda: now["t"])
    manager = HeartbeatManager("multi-day-pause", route_key="test:multi-day-pause")
    manager.set("check", schedule="0 9 * * *", timezone="America/New_York")
    manager.state.created_at = datetime(2026, 5, 3, 9, 0, tzinfo=tz).timestamp()
    manager._save(manager.state)
    assert manager.due_prompt(monday_fire) is not None
    now["t"] = monday_pause
    manager.pause()
    now["t"] = thursday_resume
    manager.resume()
    friday_fire = datetime(2026, 5, 8, 9, 0, tzinfo=tz).timestamp()
    assert manager.state.next_fire() == friday_fire
    assert manager.due_prompt(thursday_resume) is None


def test_resume_of_an_invalid_cadence_does_not_raise_and_stays_active():
    """A paused row whose cron is invalid resumes without raising, and the stored row agrees with the cache."""
    from hermes_cli.heartbeat import load_route_heartbeat

    manager = HeartbeatManager("bad-cadence-resume", route_key="test:bad-cadence-resume")
    manager.set("check", schedule="30 7 * * *", timezone="America/New_York")
    manager.state.schedule = "bad cron"
    manager._save(manager.state)
    manager.pause()
    manager.resume()
    assert manager.state.status == "active"
    assert load_route_heartbeat("test:bad-cadence-resume").status == "active"


def test_abandon_of_a_scheduled_claim_restores_the_slot_so_it_fires_again():
    """A scheduled claim whose turn never starts rewinds the anchor, so the same slot is due on the next poll."""
    tz = ZoneInfo("America/New_York")
    slot = datetime(2026, 5, 4, 7, 30, tzinfo=tz).timestamp()
    manager = HeartbeatManager("abandon-scheduled", route_key="test:abandon-scheduled")
    manager.set("check", schedule="30 7 * * *", timezone="America/New_York")
    manager.state.created_at = datetime(2026, 5, 3, 7, 30, tzinfo=tz).timestamp()
    manager._save(manager.state)
    before_anchor, before_fired = manager.state.schedule_anchor, manager.state.last_fired_at
    assert manager.due_prompt(slot) is not None
    assert manager.abandon_fire() is True
    assert manager.state.schedule_anchor == before_anchor
    assert manager.state.last_fired_at == before_fired
    assert manager.state.next_fire() == slot
    assert manager.due_prompt(slot) is not None


def test_weekly_claim_anchors_to_the_scan_start_when_no_slot_is_in_the_window():
    """A weekly cron claimed 5 days late has no slot in the one-day window: the anchor is the window start, not the poll."""
    manager = HeartbeatManager("weekly-anchor", route_key="test:weekly-anchor")
    manager.set("check", schedule="0 9 * * 1", timezone="America/New_York")
    manager.state.created_at = datetime(2026, 5, 1, 9, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    manager._save(manager.state)
    claim = datetime(2026, 5, 6, 10, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    assert manager.due_prompt(claim) is not None
    assert manager.state.last_fired_at == claim
    assert manager.state.schedule_anchor == datetime(2026, 5, 5, 10, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    assert manager.state.next_fire() == datetime(2026, 5, 11, 9, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()


def test_scheduled_list_displays_next_fire(capsys):
    from hermes_cli.subcommands.heartbeat import _list_heartbeats

    manager = HeartbeatManager("list", route_key="test:scheduled-list")
    manager.set("check CI", schedule="30 7 * * *", timezone="America/New_York")
    _list_heartbeats(None)
    output = capsys.readouterr().out
    assert "test:scheduled-list" in output
    assert "Next fire:" in output
    assert "30 7 * * *" in output


def test_is_due_anchors_on_created_then_last_fired():
    now = time.time()
    s = HeartbeatState(prompt="p", interval_seconds=600, created_at=now)
    assert s.is_due(now + 1) is False
    assert s.is_due(now + 601) is True
    s.last_fired_at = now + 601
    assert s.is_due(now + 700) is False
    assert s.is_due(now + 1300) is True


def test_paused_never_due():
    now = time.time()
    s = HeartbeatState(prompt="p", interval_seconds=60, created_at=now - 3600, status="paused")
    assert s.is_due(now) is False


def test_render_prompt_contains_instruction_and_interval():
    s = HeartbeatState(prompt="check the deploy", interval_seconds=600)
    rendered = s.render_prompt()
    assert "check the deploy" in rendered
    assert "10m" in rendered


# ──────────────────────────────────────────────────────────────────────
# manager
# ──────────────────────────────────────────────────────────────────────


def test_manager_set_pause_resume_clear():
    mgr = HeartbeatManager(session_id="hb-lifecycle-sid")
    state = mgr.set("watch CI", 600)
    assert state.status == "active"
    assert mgr.is_active()

    mgr.pause()
    assert not mgr.is_active()
    assert mgr.has_heartbeat()

    mgr.resume()
    assert mgr.is_active()

    assert mgr.clear() is True
    assert not mgr.has_heartbeat()
    assert load_heartbeat("hb-lifecycle-sid") is None


def test_manager_rejects_bad_input():
    mgr = HeartbeatManager(session_id="hb-bad-sid")
    with pytest.raises(ValueError):
        mgr.set("", 600)
    with pytest.raises(ValueError):
        mgr.set("ok", 5)


def test_manager_persists_across_instances():
    mgr = HeartbeatManager(session_id="hb-persist-sid")
    mgr.set("persisted prompt", 600)
    again = HeartbeatManager(session_id="hb-persist-sid")
    assert again.has_heartbeat()
    assert again.state.prompt == "persisted prompt"


def test_due_prompt_fires_once_and_reanchors():
    mgr = HeartbeatManager(session_id="hb-due-sid")
    mgr.set("tick", 600)
    assert mgr.due_prompt() is None
    mgr.state.created_at = time.time() - 700
    prompt = mgr.due_prompt()
    assert prompt is not None and "tick" in prompt
    assert mgr.state.fire_count == 1
    assert mgr.due_prompt() is None


@pytest.mark.parametrize("stop,stored", [("clear", None), ("pause", "paused")])
def test_a_stale_manager_does_not_fire_a_heartbeat_stopped_since(stop, stored):
    mgr = HeartbeatManager(session_id=f"hb-stale-{stop}")
    mgr.set("tick", 600).created_at = time.time() - 700
    getattr(HeartbeatManager(session_id=f"hb-stale-{stop}"), stop)()
    assert mgr.due_prompt() is None
    state = load_heartbeat(f"hb-stale-{stop}")
    assert (state.status if state else None) == stored


@pytest.mark.parametrize("stop,stored", [("clear", None), ("pause", "paused")])
def test_a_heartbeat_stopped_during_a_claim_stays_stopped(stop, stored, monkeypatch):
    from hermes_cli import heartbeat
    sid = f"hb-race-{stop}"
    mgr = HeartbeatManager(session_id=sid)
    mgr.set("tick", 600).created_at = time.time() - 700
    other = HeartbeatManager(session_id=sid)
    reread, release = threading.Event(), threading.Event()

    def load(session_id):
        state = load_heartbeat(session_id)
        reread.set()
        release.wait(5)
        return state
    monkeypatch.setattr(heartbeat, "load_heartbeat", load)
    claim = threading.Thread(target=mgr.due_prompt)
    claim.start()
    assert reread.wait(5)
    stopper = threading.Thread(target=getattr(other, stop))
    stopper.start()
    stopper.join(0.2)
    release.set()
    claim.join(5)
    stopper.join(5)
    state = load_heartbeat(sid)
    assert (state.status if state else None) == stored


def test_missed_ticks_coalesce():
    mgr = HeartbeatManager(session_id="hb-coalesce-sid")
    mgr.set("tick", 600)
    mgr.state.created_at = time.time() - 600 * 5 - 10
    assert mgr.due_prompt() is not None
    assert mgr.due_prompt() is None
    assert mgr.state.fire_count == 1


def test_resume_reanchors_instead_of_instant_fire():
    mgr = HeartbeatManager(session_id="hb-resume-sid")
    mgr.set("tick", 600)
    mgr.state.created_at = time.time() - 3600
    mgr.pause()
    mgr.resume()
    assert mgr.due_prompt() is None


def test_migrate_heartbeat_to_session():
    save_heartbeat("hb-parent-sid", HeartbeatState(prompt="carry me", interval_seconds=600, created_at=time.time()))
    assert migrate_heartbeat_to_session("hb-parent-sid", "hb-child-sid") is True
    child = load_heartbeat("hb-child-sid")
    assert child is not None and child.prompt == "carry me"
    assert load_heartbeat("hb-parent-sid") is None


def test_migrate_noop_without_source():
    assert migrate_heartbeat_to_session("hb-none-a", "hb-none-b") is False
    assert migrate_heartbeat_to_session("same", "same") is False
