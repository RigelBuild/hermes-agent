"""A replay reads only its deliver channels inside its occurrence window."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from gateway.session_context import _VAR_MAP
from tools import discord_tool
from tools.discord_tool import (
    bind_cron_replay_read,
    discord_core,
    reset_cron_replay_read,
)

BOT = "9999"
USER = "42"
POST = "500"  # discord.post_channels
DELIVER = "700"  # the job's deliver channel, not in post_channels
OTHER = "900"  # neither allowed nor the deliver channel
GUILD = "1"
T0 = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
STARTED = T0.isoformat()


def ts(delta: timedelta = timedelta()) -> str:
    return (T0 + delta).isoformat()


def snow(delta: timedelta = timedelta()) -> str:
    return str(int((T0 + delta).timestamp() * 1000 - 1420070400000) << 22)


def message(msg_id: str, author: str, when: str) -> dict:
    return {
        "id": msg_id,
        "content": "x",
        "author": {"id": author},
        "timestamp": when,
        "attachments": [],
        "reactions": [],
    }


class FakeDiscord:
    """Records REST calls and answers the few paths the replay read touches."""

    def __init__(self, messages=(), threads=(), channels=None, bot_id=BOT):
        self.messages = list(messages)
        self.threads = list(threads)
        self.bot_id = bot_id
        self.channels = {
            POST: {"id": POST, "type": 0, "guild_id": GUILD},
            DELIVER: {"id": DELIVER, "type": 0, "guild_id": GUILD},
            OTHER: {"id": OTHER, "type": 0, "guild_id": GUILD},
        }
        self.channels.update(channels or {})
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, token, params=None, body=None, timeout=15):
        self.calls.append((method, path))
        if path == "/users/@me":
            return {"id": self.bot_id} if self.bot_id else {}
        if path.startswith("/channels/") and path.count("/") == 2:
            channel_id = path.rsplit("/", 1)[-1]
            if channel_id in self.channels:
                return self.channels[channel_id]
        if path.endswith("/messages"):
            return self.messages
        if path == f"/guilds/{GUILD}/threads/active":
            return {"threads": self.threads}
        if path.endswith("/threads/archived/public"):
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


def test_replay_fetch_on_post_channels_returns_user_and_pre_start_messages_unchanged(
    replay,
):
    messages = [
        message("before-user", USER, ts(timedelta(minutes=-1))),
        message("before-bot", BOT, ts(timedelta(minutes=-1))),
        message("after-user", USER, ts(timedelta(minutes=1))),
    ]
    fake = FakeDiscord(messages=messages)
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=POST))
    assert [item["id"] for item in result["messages"]] == [
        "before-user",
        "before-bot",
        "after-user",
    ]
    assert not any(path == "/users/@me" for _, path in fake.calls)


def test_replay_fetch_on_deliver_channel_returns_only_bot_messages_after_started(
    replay,
):
    fake = FakeDiscord(
        messages=[
            message("after-bot", BOT, ts(timedelta(minutes=1))),
            message("after-user", USER, ts(timedelta(minutes=2))),
            message("before-bot", BOT, ts(timedelta(minutes=-1))),
        ]
    )
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert [item["id"] for item in result["messages"]] == ["after-bot"]


def test_replay_forum_read_on_post_channels_is_unfiltered(replay):
    threads = [
        {
            "id": "thread-bot",
            "name": "old bot post",
            "owner_id": BOT,
            "parent_id": POST,
            "thread_metadata": {"create_timestamp": ts(timedelta(minutes=-5))},
            "last_message_id": snow(timedelta(minutes=-5)),
        },
        {
            "id": "thread-user",
            "name": "user post",
            "owner_id": USER,
            "parent_id": POST,
            "thread_metadata": {"create_timestamp": ts(timedelta(minutes=1))},
            "last_message_id": snow(timedelta(minutes=1)),
        },
    ]
    fake = FakeDiscord(threads=threads)
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="list_forum_posts", channel_id=POST))
    assert {post["id"] for post in result["posts"]} == {"thread-bot", "thread-user"}
    assert not any(path == "/users/@me" for _, path in fake.calls)


@pytest.mark.parametrize("channel_type", [1, 3])
def test_replay_read_refuses_dm_deliver_channel_at_read_time(replay, channel_type):
    fake = FakeDiscord(channels={DELIVER: {"id": DELIVER, "type": channel_type}})
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch("tools.discord_tool._operator_guilds", return_value=frozenset({GUILD})),
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert "DM and group DM" in result["error"]
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_read_refuses_other_guild_deliver_channel_at_read_time(replay):
    fake = FakeDiscord(
        channels={DELIVER: {"id": DELIVER, "type": 0, "guild_id": "other"}}
    )
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch("tools.discord_tool._operator_guilds", return_value=frozenset({GUILD})),
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert "outside the operator guilds" in result["error"]
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_thread_validator_uses_fetched_thread_and_parent_channels(replay):
    thread = "thread-502"
    fake = FakeDiscord(
        channels={
            thread: {"id": thread, "type": 11, "guild_id": GUILD, "parent_id": DELIVER},
        }
    )
    validator = discord_tool.validate_cron_replay_deliver_channels
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch("tools.discord_tool._operator_guilds", return_value=frozenset({GUILD})),
        patch(
            "tools.discord_tool.validate_cron_replay_deliver_channels", wraps=validator
        ) as validate,
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=thread))
    assert "error" not in result
    assert validate.call_args.args[1] == frozenset({thread, DELIVER})
    assert sum(path == f"/channels/{thread}" for _, path in fake.calls) == 1
    assert sum(path == f"/channels/{DELIVER}" for _, path in fake.calls) == 1
    assert any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_read_validation_failure_returns_tool_error_without_message_fetch(
    replay,
):
    fake = FakeDiscord()
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch(
            "tools.discord_tool.validate_cron_replay_deliver_channels",
            side_effect=ValueError("invalid"),
        ),
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert result["error"] == "invalid"
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_post_to_deliver_channel_is_refused(replay):
    fake = FakeDiscord()
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(
            discord_core(action="post_message", channel_id=DELIVER, content="x")
        )
    assert "outside discord.post_channels" in result["error"]
    assert not fake.posted()


def test_replay_forum_list_keeps_only_bot_posts_started_or_active_after_started(replay):
    threads = [
        {
            "id": "11",
            "name": "checkin",
            "owner_id": BOT,
            "parent_id": DELIVER,
            "thread_metadata": {"create_timestamp": ts(timedelta(minutes=1))},
            "last_message_id": snow(timedelta(minutes=1)),
        },
        {
            "id": "12",
            "name": "checkin",
            "owner_id": USER,
            "parent_id": DELIVER,
            "thread_metadata": {"create_timestamp": ts(timedelta(minutes=1))},
            "last_message_id": snow(timedelta(minutes=1)),
        },
        {
            "id": "13",
            "name": "old",
            "owner_id": BOT,
            "parent_id": DELIVER,
            "thread_metadata": {"create_timestamp": ts(timedelta(hours=-1))},
            "last_message_id": snow(timedelta(hours=-1)),
        },
        {
            "id": "14",
            "name": "old-but-replied",
            "owner_id": BOT,
            "parent_id": DELIVER,
            "thread_metadata": {"create_timestamp": ts(timedelta(hours=-1))},
            "last_message_id": snow(timedelta(minutes=2)),
        },
    ]
    fake = FakeDiscord(threads=threads)
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="list_forum_posts", channel_id=DELIVER))
    assert {post["id"] for post in result["posts"]} == {"11", "14"}


def test_replay_fetch_paginates_after_the_maximum_message_id(replay):
    fake = FakeDiscord()
    pages = [
        [message(str(index), BOT, ts(timedelta(minutes=1))) for index in range(1, 101)],
        [
            message(str(index), BOT, ts(timedelta(minutes=2)))
            for index in range(101, 151)
        ],
    ]
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch(
            "tools.discord_tool._fetch_messages",
            side_effect=[
                json.dumps({"messages": pages[0]}),
                json.dumps({"messages": pages[1]}),
            ],
        ) as fetch,
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert [call.kwargs["after"] for call in fetch.call_args_list] == [
        discord_tool._snowflake_before(int(T0.timestamp() * 1000)),
        "100",
    ]
    assert [item["id"] for item in result["messages"]] == [
        str(index) for index in range(1, 151)
    ]
    assert result["has_more"] is False


def test_replay_fetch_reports_more_after_maximum_pages(replay):
    fake = FakeDiscord()
    pages = [
        [
            message(str(index), BOT, ts(timedelta(minutes=1)))
            for index in range(page * 100, (page + 1) * 100)
        ]
        for page in range(discord_tool._REPLAY_MAX_PAGES)
    ]
    with (
        patch("tools.discord_tool._discord_request", side_effect=fake),
        patch(
            "tools.discord_tool._fetch_messages",
            side_effect=[json.dumps({"messages": page}) for page in pages],
        ) as fetch,
    ):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert fetch.call_count == discord_tool._REPLAY_MAX_PAGES
    assert result["has_more"] is True


def test_replay_refuses_when_bot_id_cannot_be_resolved(replay):
    fake = FakeDiscord(bot_id="")
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=DELIVER))
    assert "Could not resolve the bot user" in result["error"]
    assert not any(path.endswith("/messages") for _, path in fake.calls)


def test_replay_with_unreadable_started_at_refuses_every_replay_read(cron, allowed):
    fake = FakeDiscord()
    token = bind_cron_replay_read(frozenset({DELIVER}), "not-a-time")
    try:
        with patch("tools.discord_tool._discord_request", side_effect=fake):
            result = json.loads(
                discord_core(action="fetch_messages", channel_id=DELIVER)
            )
    finally:
        reset_cron_replay_read(token)
    assert "started_at is unreadable" in result["error"]


def test_replay_refuses_naive_started_at(cron, allowed):
    fake = FakeDiscord()
    token = bind_cron_replay_read(frozenset({DELIVER}), "2026-11-10T10:00:00")
    try:
        with patch("tools.discord_tool._discord_request", side_effect=fake):
            result = json.loads(
                discord_core(action="fetch_messages", channel_id=DELIVER)
            )
    finally:
        reset_cron_replay_read(token)
    assert "started_at is unreadable" in result["error"]


def test_replay_read_of_thread_under_deliver_channel_is_filtered(replay):
    thread = "thread-501"
    fake = FakeDiscord(
        messages=[
            message("before-bot", BOT, ts(timedelta(minutes=-1))),
            message("after-user", USER, ts(timedelta(minutes=1))),
            message("after-bot", BOT, ts(timedelta(minutes=2))),
        ],
        channels={
            thread: {"id": thread, "type": 11, "guild_id": GUILD, "parent_id": DELIVER}
        },
    )
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(discord_core(action="fetch_messages", channel_id=thread))
    assert [item["id"] for item in result["messages"]] == ["after-bot"]


def test_replay_forum_creation_on_deliver_channel_is_refused(replay):
    fake = FakeDiscord()
    with patch("tools.discord_tool._discord_request", side_effect=fake):
        result = json.loads(
            discord_core(
                action="create_forum_post", channel_id=DELIVER, name="x", content="x"
            )
        )
    assert "outside discord.post_channels" in result["error"]
    assert not fake.posted()


def test_scheduler_binds_only_replay_with_a_discord_deliver(monkeypatch):
    from cron import scheduler

    targets = []
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda job: targets)
    scope = object.__new__(scheduler._CronRunScope)
    scope._replay_read_token = None
    replay_job = {"id": "job", "fire_claim": {"replay": {"started_at": STARTED}}}

    scope.bind_replay_read({"id": "job"})
    scope.bind_replay_read(replay_job)
    targets.append(
        {"platform": "telegram", "chat_id": "non-discord", "thread_id": None}
    )
    scope.bind_replay_read(replay_job)
    assert discord_tool._CRON_REPLAY_READ.get() is None

    targets.append(
        {"platform": "discord", "chat_id": DELIVER, "thread_id": "thread-deliver"}
    )
    scope.bind_replay_read(replay_job)
    window = discord_tool._CRON_REPLAY_READ.get()
    assert window.deliver_channel_ids == frozenset({DELIVER, "thread-deliver"})
    assert window.started_at == T0
    reset_cron_replay_read(scope._replay_read_token)


def _replay_scope(monkeypatch, target):
    from cron import scheduler

    monkeypatch.setattr(
        scheduler,
        "_resolve_delivery_targets",
        lambda job: [{"platform": "discord", "chat_id": target, "thread_id": None}],
    )
    scope = object.__new__(scheduler._CronRunScope)
    scope._replay_read_token = None
    return scope


def test_scheduler_binds_dm_delivery_for_read_time_refusal(monkeypatch):
    scope = _replay_scope(monkeypatch, "dm-1")
    scope.bind_replay_read(
        {"id": "job", "fire_claim": {"replay": {"started_at": STARTED}}}
    )
    assert discord_tool._CRON_REPLAY_READ.get().deliver_channel_ids == frozenset(
        {"dm-1"}
    )
    reset_cron_replay_read(scope._replay_read_token)


def test_scheduler_binds_other_guild_delivery_for_read_time_refusal(monkeypatch):
    scope = _replay_scope(monkeypatch, "other-guild-channel")
    scope.bind_replay_read(
        {"id": "job", "fire_claim": {"replay": {"started_at": STARTED}}}
    )
    assert discord_tool._CRON_REPLAY_READ.get().deliver_channel_ids == frozenset(
        {"other-guild-channel"}
    )
    reset_cron_replay_read(scope._replay_read_token)
