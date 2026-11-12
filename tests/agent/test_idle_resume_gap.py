"""Regression coverage for preserving idle time across a turn reset."""

from __future__ import annotations

import time
from types import MethodType, SimpleNamespace

import pytest

from agent.activity_tracking import ActivityTrackingMixin
from agent.turn_context_compaction import CompactionOutcome, _idle_compaction
from agent.turn_facade_lease import DurableTurnLease


class _CompressionReached(Exception):
    """Raised by the recording double after idle compaction selects its trigger."""


def _agent(labels: list[str]) -> SimpleNamespace:
    def record_and_stop(_messages, _system, **kwargs):
        labels.append(kwargs.get("trigger"))
        raise _CompressionReached

    agent = SimpleNamespace(
        context_compressor=SimpleNamespace(
            threshold_tokens=1_000, summary_target_ratio=0.2,
            last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
        ),
        compression_enabled=True, compression_idle_compact_after_seconds=60,
        model="m", session_id="s1", _session_db=None,
        _emit_status=lambda _message: None, _compress_context=record_and_stop,
        _persist_session_activity_if_due=lambda: None,
    )
    agent._touch_activity = MethodType(ActivityTrackingMixin._touch_activity, agent)
    return agent


def _outcome() -> CompactionOutcome:
    messages = [
        {"role": "user", "content": "x" * 8_000},
        {"role": "assistant", "content": "y" * 8_000},
        {"role": "user", "content": "current ask"},
    ]
    return CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None,
        current_turn_user_idx=len(messages) - 1,
    )


def _patch_idle_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.turn_context._preflight_request_tokens", lambda *_args: 5_000)


def test_idle_compaction_uses_gap_saved_before_cached_turn_reset(monkeypatch):
    from gateway.run import GatewayRunner

    _patch_idle_estimate(monkeypatch)
    labels: list[str] = []
    agent = _agent(labels)
    agent._last_activity_ts = time.time() - 3_600
    agent._last_activity_desc = "previous turn activity"
    agent._last_activity_provenance = None

    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

    with pytest.raises(_CompressionReached):
        _idle_compaction(agent, _outcome(), "sys", "current ask", "task")

    assert labels == ["idle"]
    assert agent._idle_resume_gap_s is None


def test_lease_start_saves_idle_gap_before_clock_reset(monkeypatch):
    monkeypatch.setattr("agent.periodic_scheduler.schedule", lambda *_args: None)
    _patch_idle_estimate(monkeypatch)
    labels: list[str] = []
    agent = _agent(labels)
    agent._last_activity_ts = time.time() - 3_600
    old_activity_ts = agent._last_activity_ts
    lease = DurableTurnLease(agent, object(), "s1", "holder")

    lease.start()

    assert agent._last_activity_ts > old_activity_ts
    assert agent._last_activity_ts >= time.time() - 5
    with pytest.raises(_CompressionReached):
        _idle_compaction(agent, _outcome(), "sys", "current ask", "task")

    assert labels == ["idle"]


def test_gateway_gap_survives_later_lease_start(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("agent.periodic_scheduler.schedule", lambda *_args: None)
    labels: list[str] = []
    agent = _agent(labels)
    agent._last_activity_ts = time.time() - 3_600

    GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)
    gateway_gap = agent._idle_resume_gap_s
    DurableTurnLease(agent, object(), "s1", "holder").start()

    assert gateway_gap >= 3_590
    assert agent._idle_resume_gap_s >= gateway_gap


def test_idle_compaction_clears_gap_when_compression_disabled():
    agent = _agent([])
    agent.compression_enabled = False
    agent._idle_resume_gap_s = 3_600

    _idle_compaction(agent, _outcome(), "sys", "current ask", "task")

    assert agent._idle_resume_gap_s is None
