"""DL-1691: a replay may read its job's ``deliver`` channel inside its window, and nothing else.

An ordinary cron run keeps the ``discord.post_channels`` boundary. A replay also reads its job's
deliver channels (and threads under them), sees only this bot's messages at or after the replay's
``started_at``, and never posts there.
"""
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from gateway.session_context import _VAR_MAP
from tools import discord_tool
from tools.discord_tool import bind_cron_replay_read, discord_core, reset_cron_replay_read

BOT = "9999"
USER = "42"
POST = "500"      # discord.post_channels
DELIVER = "700"   # the job's deliver channel, not in post_channels
OTHER = "900"     # neither allowed nor the deliver channel
GUILD = "1"
EPOCH_MS = 1420070400000
T0 = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
STARTED = T0.isoformat()


def ts(delta: timedelta = timedelta()) -> str:
    return (T0 + delta).isoformat()


def snow(delta: timedelta = timedelta()) -> str:
    return str(int((T0 + delta).timestamp() * 1000 - EPOCH_MS) << 22)


def message(msg_id: str, author: str, when: str) -> dict:
    return {"id": msg_id, "content": "x", "author": {"id": author}, "timestamp": when}


class FakeDiscord:
    """Records REST calls and answers the few paths the replay read touches."""

    def __init__(self, messages=(), threads=()):
        self.messages = list(messages)
        self.threads = list(threads)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, token, params=None, body=None, timeout=15):
        self.calls.append((method, path))
        if path == "/users/@me":
            return {"id": BOT}
        if path == f"/channels/{DELIVER}":
            return {"id": DELIVER, "type": 0, "guild_id": GUILD}
        if path == f"/channels/{OTHER}":
            return {"id": OTHER, "type": 0, "guild_id": GUILD}
        if path == f"/channels/{DELIVER}/messages":
            return self.messages
        if path == f"/guilds/{GUILD}/threads/active":
            return {"threads": self.threads}
        if path == f"/channels/{DELIVER}/threads/archived/public":
            return {"threads": [], "has_more": False}
        raise AssertionError(f"unexpected Discord call {method} {path}")

    def posted(self) -> bool:
        return any(method == "POST" for method, _ in self.calls)


@pytest.fixture
def cron(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    token = _VAR_MAP["HERMES_CRON_SESSION"].set("1")
    yield
    _VAR_MAP["HERMES_CRON_SESSION"].reset(token)


@pytest.fixture
def allowed():
    with patch("tools.discord_tool._load_post_channels_config", return_value={POST}):
        yield


@pytest.fixture
def replay(cron, allowed):
    """A live replay window on ``DELIVER`` that started at ``STARTED``; closed after the test."""
    token = bind_cron_replay_read(frozenset({DELIVER}), STARTED)
    yield
    reset_cron_replay_read(token)


def test_ordinary_cron_read_outside_post_channels_is_refused(cron, allowed):
    fake = FakeDiscord(messages=[message("1", BOT, ts(timedelta(minutes=1)))])
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert "outside discord.post_channels" in result["error"]
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_fetch_on_deliver_channel_returns_only_bot_messages_after_started(replay):
    fake = FakeDiscord(messages=[
        message("after-bot", BOT, ts(timedelta(minutes=1))),
        message("after-user", USER, ts(timedelta(minutes=2))),
        message("before-bot", BOT, ts(timedelta(minutes=-1))),
    ])
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert [m["id"] for m in result["messages"]] == ["after-bot"]


def test_replay_fetch_on_channel_outside_deliver_and_post_channels_is_refused(replay):
    fake = FakeDiscord()
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=OTHER))
    assert "outside discord.post_channels" in result["error"]
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_post_to_deliver_channel_is_refused(replay):
    fake = FakeDiscord()
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="post_message", channel_id=DELIVER, content="x"))
    assert "outside discord.post_channels" in result["error"]
    assert not fake.posted()


def test_replay_forum_list_keeps_only_bot_posts_started_or_active_after_started(replay):
    threads = [
        {"id": "11", "name": "checkin", "owner_id": BOT, "parent_id": DELIVER,
         "thread_metadata": {"create_timestamp": ts(timedelta(minutes=1))},
         "last_message_id": snow(timedelta(minutes=1))},
        {"id": "12", "name": "checkin", "owner_id": USER, "parent_id": DELIVER,
         "thread_metadata": {"create_timestamp": ts(timedelta(minutes=1))},
         "last_message_id": snow(timedelta(minutes=1))},
        {"id": "13", "name": "old", "owner_id": BOT, "parent_id": DELIVER,
         "thread_metadata": {"create_timestamp": ts(timedelta(hours=-1))},
         "last_message_id": snow(timedelta(hours=-1))},
        {"id": "14", "name": "old-but-replied", "owner_id": BOT, "parent_id": DELIVER,
         "thread_metadata": {"create_timestamp": ts(timedelta(hours=-1))},
         "last_message_id": snow(timedelta(minutes=2))},
    ]
    fake = FakeDiscord(threads=threads)
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="list_forum_posts", channel_id=DELIVER))
    assert {post["id"] for post in result["posts"]} == {"11", "14"}


def test_replay_with_unreadable_started_at_refuses_every_replay_read(cron, allowed):
    fake = FakeDiscord(messages=[message("1", BOT, ts(timedelta(minutes=1)))])
    token = bind_cron_replay_read(frozenset({DELIVER}), "not-a-time")
    try:
        with patch("tools.discord_tool._discord_request", side_effect=fake):
            result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    finally:
        reset_cron_replay_read(token)
    assert "started_at is unreadable" in result["error"]


def test_scheduler_binds_replay_window_only_for_a_replay_with_a_discord_deliver(monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(
        scheduler, "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": DELIVER, "thread_id": None}])
    scope = object.__new__(scheduler._CronRunScope)
    scope._replay_read_token = None

    scope.bind_replay_read({"id": "job"})
    assert discord_tool._CRON_REPLAY_READ.get() is None

    scope.bind_replay_read({"id": "job", "fire_claim": {"replay": {"started_at": STARTED}}})
    window = discord_tool._CRON_REPLAY_READ.get()
    assert window.deliver_channel_ids == frozenset({DELIVER})
    assert window.started_at == T0
    reset_cron_replay_read(scope._replay_read_token)
    assert discord_tool._CRON_REPLAY_READ.get() is None
