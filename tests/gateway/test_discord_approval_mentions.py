"""Discord approval prompts can opt into owner mentions."""

import os
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _apply_yaml_config,
)


class _FakeChannel:
    def __init__(self):
        self.sent_kwargs = None

    async def send(self, **kwargs):
        self.sent_kwargs = kwargs
        return SimpleNamespace(id=12345)


class _FakeClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_id):
        return self.channel


@pytest.mark.asyncio
async def test_exec_approval_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra={"approval_mentions": True, "approval_mentions_target": "session"})

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")
    assert "make check" in channel.sent_kwargs["content"]
    assert "allowed_mentions" in channel.sent_kwargs


@pytest.mark.asyncio
async def test_exec_approval_mentions_can_move_to_index(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra={"approval_mentions": True, "approval_mentions_target": "index"})

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert "<@111>" not in channel.sent_kwargs["content"]
    assert "allowed_mentions" not in channel.sent_kwargs


@pytest.mark.asyncio
async def test_exec_approval_mentions_disabled_does_not_ping(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra={"approval_mentions": False})

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert "<@111>" not in channel.sent_kwargs["content"]
    assert "allowed_mentions" not in channel.sent_kwargs

def test_yaml_config_bridges_approval_mentions_target():
    seeded = _apply_yaml_config({}, {"approval_mentions_target": "index"})
    assert seeded == {"approval_mentions_target": "index"}

def test_yaml_config_seeds_websocket_health_with_primary_precedence(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    seeded = _apply_yaml_config(
        {},
        {
            "websocket_liveness_interval_seconds": 11,
            "liveness_interval_seconds": 99,
            "websocket_liveness_failure_threshold": 2,
            "websocket_heartbeat_ack_max_age_seconds": 75,
            "websocket_max_latency_seconds": 30,
        },
    )

    assert os.environ["HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"] == "11"
    assert os.environ["HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"] == "2"
    assert seeded == {
        "websocket_liveness_interval_seconds": 11,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 75,
        "websocket_max_latency_seconds": 30,
    }


