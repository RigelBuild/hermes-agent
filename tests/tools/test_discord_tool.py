"""Tests for the Discord server introspection and management tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.discord_tool import (
    DiscordAPIError,
    _ACTIONS,
    _ADMIN_ACTIONS,
    _CORE_ACTIONS,
    _available_actions,
    _detect_capabilities,
    _discord_request,
    _get_bot_token,
    _load_allowed_actions_config,
    _reset_capability_cache,
    check_discord_tool_requirements,
    discord_admin_handler,
    discord_core,
    get_dynamic_schema_admin,
    get_dynamic_schema_core,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(response_data, status=200):
    """Create a mock for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Token / check_fn
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    @pytest.mark.parametrize("token, expected", [(None, False), ("test-token-123", True)])
    def test_requirements_follow_token(self, monkeypatch, token, expected):
        if token is None:
            monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        else:
            monkeypatch.setenv("DISCORD_BOT_TOKEN", token)
        assert check_discord_tool_requirements() is expected

    def test_get_bot_token(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "  my-token  ")
        assert _get_bot_token() == "my-token"

    def test_multiplex_scope_token_wins_over_process_environment(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope(
            {"DISCORD_BOT_TOKEN": "  active-profile-token  "}
        )
        try:
            assert _get_bot_token() == "active-profile-token"
            assert check_discord_tool_requirements() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_missing_token_fails_closed(self, monkeypatch):
        from agent import secret_scope
        from tools.registry import invalidate_check_fn_cache, registry

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        invalidate_check_fn_cache()
        try:
            assert _get_bot_token() is None
            assert check_discord_tool_requirements() is False
            assert registry.get_definitions({"discord", "discord_admin"}) == []
        finally:
            invalidate_check_fn_cache()
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

    def test_non_multiplex_scope_miss_keeps_environment_compatibility(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "process-token")
        secret_scope.set_multiplex_active(False)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        try:
            assert _get_bot_token() == "process-token"
            assert check_discord_tool_requirements() is True
        finally:
            secret_scope.reset_secret_scope(scope_token)

    def test_gateway_tool_prompt_gate_uses_active_profile_token(self, monkeypatch):
        from agent import secret_scope
        from gateway.session import _discord_tools_loaded

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "another-profile-token")
        secret_scope.set_multiplex_active(True)
        scope_token = secret_scope.set_secret_scope({"UNRELATED_SECRET": "value"})
        try:
            with patch("hermes_cli.config.load_config") as load_config:
                assert _discord_tools_loaded() is False
                load_config.assert_not_called()
        finally:
            secret_scope.reset_secret_scope(scope_token)
            secret_scope.set_multiplex_active(False)

# ---------------------------------------------------------------------------
# Channel type names
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Discord API request helper
# ---------------------------------------------------------------------------

class TestDiscordRequest:
    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_get_request(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen({"ok": True})
        result = _discord_request("GET", "/test", "token123")
        assert result == {"ok": True}

        # Verify the request was constructed correctly
        call_args = mock_urlopen_fn.call_args
        req = call_args[0][0]
        assert "https://discord.com/api/v10/test" in req.full_url
        assert req.get_header("Authorization") == "Bot token123"
        assert req.get_method() == "GET"


    @patch("tools.discord_tool.urllib.request.urlopen")
    def test_response_body_size_limit(self, mock_urlopen_fn, monkeypatch):
        monkeypatch.setattr("tools.discord_tool._DISCORD_RESPONSE_BODY_MAX_BYTES", 8)
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"x" * 9
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen_fn.return_value = mock_resp

        with pytest.raises(DiscordAPIError) as exc_info:
            _discord_request("GET", "/test", "tok")

        assert exc_info.value.status == 502
        assert "response body exceeded 8 bytes" in exc_info.value.body
        mock_resp.read.assert_called_once_with(9)


# ---------------------------------------------------------------------------
# Main handler: validation
# ---------------------------------------------------------------------------

class TestDiscordServerValidation:
    def test_no_token(self, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "DISCORD_BOT_TOKEN" in result["error"]


    def test_missing_multiple_params(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = json.loads(discord_admin_handler(action="add_role"))
        assert "error" in result
        assert "guild_id" in result["error"]
        assert "user_id" in result["error"]
        assert "role_id" in result["error"]


# ---------------------------------------------------------------------------
# Action: list_channels
# ---------------------------------------------------------------------------

class TestListChannels:
    @patch("tools.discord_tool._discord_request")
    def test_list_channels_organized(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "10", "name": "General", "type": 4, "position": 0, "parent_id": None},
            {"id": "11", "name": "chat", "type": 0, "position": 0, "parent_id": "10", "topic": "Main chat", "nsfw": False},
            {"id": "12", "name": "voice", "type": 2, "position": 1, "parent_id": "10", "topic": None, "nsfw": False},
            {"id": "13", "name": "no-category", "type": 0, "position": 0, "parent_id": None, "topic": None, "nsfw": False},
        ]
        result = json.loads(discord_admin_handler(action="list_channels", guild_id="111"))
        assert result["total_channels"] == 3  # excludes the category itself
        groups = result["channel_groups"]
        # Uncategorized first
        assert groups[0]["category"] is None
        assert len(groups[0]["channels"]) == 1
        assert groups[0]["channels"][0]["name"] == "no-category"
        # Then the category
        assert groups[1]["category"]["name"] == "General"
        assert len(groups[1]["channels"]) == 2


# ---------------------------------------------------------------------------
# Action: list_roles
# ---------------------------------------------------------------------------

class TestListRoles:
    @patch("tools.discord_tool._discord_request")
    def test_list_roles_sorted(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {"id": "1", "name": "@everyone", "position": 0, "color": 0, "mentionable": False, "managed": False, "hoist": False},
            {"id": "2", "name": "Admin", "position": 2, "color": 16711680, "mentionable": True, "managed": False, "hoist": True},
            {"id": "3", "name": "Mod", "position": 1, "color": 255, "mentionable": True, "managed": False, "hoist": True},
        ]
        result = json.loads(discord_admin_handler(action="list_roles", guild_id="111"))
        assert result["count"] == 3
        # Should be sorted by position descending
        assert result["roles"][0]["name"] == "Admin"
        assert result["roles"][0]["color"] == "#ff0000"
        assert result["roles"][1]["name"] == "Mod"
        assert result["roles"][2]["name"] == "@everyone"


# ---------------------------------------------------------------------------
# Action: search_members
# ---------------------------------------------------------------------------

class TestSearchMembers:
    @patch("tools.discord_tool._discord_request")
    def test_search_members_limit_capped(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = []
        discord_core(action="search_members", guild_id="111", query="x", limit=200)
        call_params = mock_req.call_args[1]["params"]
        assert call_params["limit"] == "100"  # Capped at 100


# ---------------------------------------------------------------------------
# Action: fetch_messages
# ---------------------------------------------------------------------------

class TestFetchMessages:
    @patch("tools.discord_tool._discord_request")
    def test_fetch_messages(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = [
            {
                "id": "1001",
                "content": "Hello world",
                "author": {"id": "42", "username": "user1", "global_name": "User One", "bot": False},
                "timestamp": "2024-01-01T12:00:00Z",
                "edited_timestamp": None,
                "attachments": [],
                "pinned": False,
            },
        ]
        result = json.loads(discord_core(action="fetch_messages", channel_id="11"))
        assert result["count"] == 1
        assert result["messages"][0]["content"] == "Hello world"
        assert result["messages"][0]["author"]["username"] == "user1"


# ---------------------------------------------------------------------------
# Action: create_thread
# ---------------------------------------------------------------------------

class TestCreateThread:
    @patch("tools.discord_tool._discord_request")
    def test_create_standalone_thread(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "800", "name": "New Thread"}
        result = json.loads(discord_core(action="create_thread", channel_id="11", name="New Thread"))
        assert result["success"] is True
        assert result["thread_id"] == "800"
        # Verify the API call
        mock_req.assert_called_once_with(
            "POST", "/channels/11/threads", "test-token",
            body={"name": "New Thread", "auto_archive_duration": 1440, "type": 11},
        )

    @patch("tools.discord_tool._discord_request")
    def test_create_thread_from_message(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.return_value = {"id": "801", "name": "Discussion"}
        result = json.loads(discord_core(
            action="create_thread", channel_id="11", name="Discussion", message_id="1001",
        ))
        assert result["success"] is True
        mock_req.assert_called_once_with(
            "POST", "/channels/11/messages/1001/threads", "test-token",
            body={"name": "Discussion", "auto_archive_duration": 1440},
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("tools.discord_tool._discord_request")
    def test_api_error_handled(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = DiscordAPIError(403, '{"message": "Missing Access"}')
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "error" in result
        assert "403" in result["error"]

    @patch("tools.discord_tool._discord_request")
    def test_unexpected_error_handled_core(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock_req.side_effect = RuntimeError("something broke")
        result = json.loads(discord_core(action="fetch_messages", channel_id="11"))
        assert "error" in result
        assert "something broke" in result["error"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_all_actions_covered(self):
        """Core + admin actions should cover all known actions."""
        assert set(_CORE_ACTIONS.keys()) | set(_ADMIN_ACTIONS.keys()) == set(_ACTIONS.keys())
        assert set(_CORE_ACTIONS.keys()) & set(_ADMIN_ACTIONS.keys()) == set()


# ---------------------------------------------------------------------------
# Toolset: discord / discord_admin only in hermes-discord
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Capability detection (privileged intents)
# ---------------------------------------------------------------------------

class TestCapabilityDetection:
    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    @patch("tools.discord_tool._discord_request")
    def test_both_intents_enabled(self, mock_req):
        # flags: GUILD_MEMBERS (1<<14) + MESSAGE_CONTENT (1<<18) = 278528
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        caps = _detect_capabilities("tok")
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True
        assert caps["detected"] is True


    @patch("tools.discord_tool._discord_request")
    def test_detection_failure_is_permissive(self, mock_req):
        """If detection fails (network/401/revoked token), expose everything
        and let runtime errors surface. Silent failure should never hide
        actions the bot actually has."""
        mock_req.side_effect = DiscordAPIError(401, "unauthorized")
        caps = _detect_capabilities("tok")
        assert caps["detected"] is False
        assert caps["has_members_intent"] is True
        assert caps["has_message_content"] is True


class TestNonBlockingCapabilityDetection:
    """The schema-build path must never block on a discord.com HTTP call.

    ``_detect_capabilities_nonblocking`` resolves memory cache → disk cache →
    permissive default (+ background detect), keeping the ~2s blocking
    detection off the first-token critical path (TTFT fix, July 2026).
    """

    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    def test_memory_cache_hit_no_network(self):
        from tools.discord_tool import _capability_cache, _detect_capabilities_nonblocking
        caps_in = {"has_members_intent": False, "has_message_content": True, "detected": True}
        _capability_cache["tok"] = caps_in
        with patch("tools.discord_tool._discord_request") as mock_req:
            caps = _detect_capabilities_nonblocking("tok")
        assert caps == caps_in
        mock_req.assert_not_called()


    def test_disk_cache_expires(self, tmp_path, monkeypatch):
        import time as _time

        import tools.discord_tool as dt
        monkeypatch.setattr(
            dt, "_capability_disk_cache_path",
            lambda: tmp_path / "discord_capabilities.json",
        )
        caps_in = {"has_members_intent": True, "has_message_content": True, "detected": True}
        dt._save_caps_to_disk("tok", caps_in)
        # Rewrite timestamp to be stale
        import json as _json
        p = tmp_path / "discord_capabilities.json"
        data = _json.loads(p.read_text())
        for entry in data.values():
            entry["ts"] = _time.time() - dt._CAPABILITY_DISK_TTL_SECONDS - 10
        p.write_text(_json.dumps(data))
        assert dt._load_caps_from_disk("tok") is None

    def test_schema_build_uses_nonblocking_path(self, monkeypatch):
        """get_dynamic_schema_core must not call the blocking detection."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        with patch("tools.discord_tool._load_caps_from_disk", return_value=None), \
             patch("tools.discord_tool.threading.Thread") as mock_thread, \
             patch("tools.discord_tool._discord_request") as mock_req:
            schema = get_dynamic_schema_core()
        # No blocking HTTP call happened on the schema-build path
        mock_req.assert_not_called()
        # Background detection was scheduled exactly once
        assert mock_thread.call_count == 1
        assert schema is not None
        actions = set(schema["parameters"]["properties"]["action"]["enum"])
        assert actions == set(_CORE_ACTIONS.keys())  # permissive default

    @patch("tools.discord_tool._discord_request")
    def test_cache_is_keyed_by_token(self, mock_req):
        """Regression: token A's capabilities must not leak to token B.

        Before the fix, the cache was a single module-global dict. The first
        call populated it and every subsequent call — regardless of token —
        returned the same cached value, producing wrong schema gating for
        rotated or multi-token deployments.
        """
        def _per_token_flags(method, path, token, **_kwargs):
            # token A: both intents; token B: neither.
            if token == "tok_a":
                return {"flags": (1 << 14) | (1 << 18)}
            return {"flags": 0}

        mock_req.side_effect = _per_token_flags

        caps_a = _detect_capabilities("tok_a")
        caps_b = _detect_capabilities("tok_b")

        assert caps_a["has_members_intent"] is True
        assert caps_a["has_message_content"] is True
        assert caps_b["has_members_intent"] is False
        assert caps_b["has_message_content"] is False
        # Each token should hit the endpoint exactly once.
        assert mock_req.call_count == 2

        # Re-requesting either token serves from its own cache entry.
        _detect_capabilities("tok_a")
        _detect_capabilities("tok_b")
        assert mock_req.call_count == 2


# ---------------------------------------------------------------------------
# Config allowlist
# ---------------------------------------------------------------------------

class TestConfigAllowlist:
    @pytest.fixture(autouse=True)
    def _reset_tools_logger(self):
        """Restore the ``tools`` logger level after cross-test pollution.

        ``AIAgent(quiet_mode=True)`` globally sets ``tools`` and
        ``tools.*`` children to ``ERROR`` (see run_agent.py quiet_mode
        block).  A persistent test process keeps that setting, so a
        streaming test will silence WARNING-level logs from
        ``tools.discord_tool`` for every test that follows.  Reset here so
        ``caplog`` can capture warnings regardless of earlier tests.
        """
        import logging as _logging
        _prev_tools = _logging.getLogger("tools").level
        _prev_dt = _logging.getLogger("tools.discord_tool").level
        _logging.getLogger("tools").setLevel(_logging.NOTSET)
        _logging.getLogger("tools.discord_tool").setLevel(_logging.NOTSET)
        try:
            yield
        finally:
            _logging.getLogger("tools").setLevel(_prev_tools)
            _logging.getLogger("tools.discord_tool").setLevel(_prev_dt)

    def test_empty_string_returns_none(self, monkeypatch):
        """Empty config means no allowlist — all actions visible."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        assert _load_allowed_actions_config() is None

    def test_comma_separated_string(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,list_channels,fetch_messages"}},
        )
        result = _load_allowed_actions_config()
        assert result == ["list_guilds", "list_channels", "fetch_messages"]


    def test_config_load_failure_is_permissive(self, monkeypatch):
        """If config can't be loaded at all, fall back to None (all allowed)."""
        def bad_load():
            raise RuntimeError("disk gone")
        monkeypatch.setattr("hermes_cli.config.load_config", bad_load)
        assert _load_allowed_actions_config() is None


# ---------------------------------------------------------------------------
# Action filtering combines intents + allowlist
# ---------------------------------------------------------------------------

class TestAvailableActions:
    def test_all_available_when_unrestricted(self):
        caps = {"detected": True, "has_members_intent": True, "has_message_content": True}
        assert _available_actions(caps, None) == list(_ACTIONS.keys())

    def test_no_members_intent_hides_member_actions(self):
        caps = {"detected": True, "has_members_intent": False, "has_message_content": True}
        actions = _available_actions(caps, None)
        assert "search_members" not in actions
        assert "member_info" not in actions
        # fetch_messages stays — MESSAGE_CONTENT affects content field but action works
        assert "fetch_messages" in actions

    def test_allowlist_intersects_with_intents(self):
        """Allowlist can only narrow — not re-enable intent-gated actions."""
        caps = {"detected": True, "has_members_intent": False, "has_message_content": True}
        allowlist = ["list_guilds", "search_members", "fetch_messages"]
        actions = _available_actions(caps, allowlist)
        # search_members gated by intent → stripped even though allowlisted;
        # result stays in canonical order regardless of allowlist order
        assert actions == ["list_guilds", "fetch_messages"]


# ---------------------------------------------------------------------------
# Dynamic schema build (integration of intents + config)
# ---------------------------------------------------------------------------

class TestDynamicSchema:
    def setup_method(self):
        _reset_capability_cache()

    def teardown_method(self):
        _reset_capability_cache()

    @patch("tools.discord_tool._discord_request")
    def test_no_token_returns_none(self, mock_req, monkeypatch):
        monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
        assert get_dynamic_schema_core() is None
        assert get_dynamic_schema_admin() is None
        mock_req.assert_not_called()


    @patch("tools.discord_tool._discord_request")
    def test_no_members_intent_hides_search_members_from_core(
        self, mock_req, monkeypatch,
    ):
        """search_members is a core action gated by GUILD_MEMBERS intent."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.return_value = {"flags": 1 << 18}  # only MESSAGE_CONTENT
        # Warm the capability cache — schema builds are non-blocking and use
        # the permissive default until detection has completed (background
        # thread + disk cache); filtering applies once caps are known.
        _detect_capabilities("tok")
        schema = get_dynamic_schema_core()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "search_members" not in actions

    @patch("tools.discord_tool._discord_request")
    def test_config_allowlist_narrows_admin_schema(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,list_channels"}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        schema = get_dynamic_schema_admin()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert actions == ["list_guilds", "list_channels"]

    @patch("tools.discord_tool._discord_request")
    def test_empty_allowlist_with_valid_values_hides_tools(self, mock_req, monkeypatch):
        """If the allowlist resolves to zero valid actions (e.g. all names
        were typos), get_dynamic_schema returns None so the tool is dropped."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "typo_one,typo_two"}},
        )
        mock_req.return_value = {"flags": (1 << 14) | (1 << 18)}
        assert get_dynamic_schema_core() is None
        assert get_dynamic_schema_admin() is None


# ---------------------------------------------------------------------------
# Runtime allowlist enforcement (defense in depth — schema already filtered)
# ---------------------------------------------------------------------------

class TestRuntimeAllowlistEnforcement:
    @patch("tools.discord_tool._discord_request")
    def test_denied_action_blocked_at_runtime(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds"}},
        )
        result = json.loads(discord_admin_handler(action="add_role", guild_id="1", user_id="2", role_id="3"))
        assert "error" in result
        assert "disabled by config" in result["error"]
        mock_req.assert_not_called()

    @patch("tools.discord_tool._discord_request")
    def test_allowed_action_proceeds(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds"}},
        )
        mock_req.return_value = []
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "guilds" in result


# ---------------------------------------------------------------------------
# 403 enrichment
# ---------------------------------------------------------------------------

class Test403Enrichment:
    @patch("tools.discord_tool._discord_request")
    def test_403_in_runtime_is_enriched(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.side_effect = DiscordAPIError(403, '{"message":"Missing Permissions"}')
        result = json.loads(discord_admin_handler(
            action="add_role", guild_id="1", user_id="2", role_id="3",
        ))
        assert "error" in result
        assert "MANAGE_ROLES" in result["error"]
        assert "Missing Permissions" in result["error"]  # Raw body preserved

    @patch("tools.discord_tool._discord_request")
    def test_non_403_errors_are_not_enriched(self, mock_req, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": ""}},
        )
        mock_req.side_effect = DiscordAPIError(500, "server error")
        result = json.loads(discord_admin_handler(action="list_guilds"))
        assert "500" in result["error"]
        assert "MANAGE_ROLES" not in result["error"]


# ---------------------------------------------------------------------------
# model_tools integration — dynamic schema replaces static
# ---------------------------------------------------------------------------

class TestModelToolsIntegration:
    def setup_method(self):
        _reset_capability_cache()
        from model_tools import _clear_tool_defs_cache
        from tools.registry import invalidate_check_fn_cache
        _clear_tool_defs_cache()
        invalidate_check_fn_cache()

    def teardown_method(self):
        _reset_capability_cache()
        from model_tools import _clear_tool_defs_cache
        from tools.registry import invalidate_check_fn_cache
        _clear_tool_defs_cache()
        invalidate_check_fn_cache()

    @patch("tools.discord_tool._discord_request")
    def test_discord_admin_schema_rebuilt_by_get_tool_definitions(
        self, mock_req, monkeypatch,
    ):
        """When model_tools.get_tool_definitions runs with discord_admin
        available, it should replace the static schema with the dynamic one."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "list_guilds,server_info"}},
        )
        # Bot without GUILD_MEMBERS intent
        mock_req.return_value = {"flags": 0}

        from model_tools import get_tool_definitions
        # skip_tool_search_assembly: this test exercises the dynamic schema
        # rebuild; under tiered disclosure the discord tools defer behind the
        # bridge, but the rebuilt schema is what tool_describe serves.
        tools = get_tool_definitions(enabled_toolsets=["hermes-discord"], quiet_mode=True,
                                     skip_tool_search_assembly=True)
        discord_admin_tool = next(
            (t for t in tools if t.get("function", {}).get("name") == "discord_admin"),
            None,
        )
        assert discord_admin_tool is not None, "discord_admin should be in the schema"
        actions = discord_admin_tool["function"]["parameters"]["properties"]["action"]["enum"]
        assert actions == ["list_guilds", "server_info"]


class TestDiscordForumActions:
    @staticmethod
    def _allow_actions(monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"discord": {"server_actions": ""}},
        )

    def test_list_forum_posts_filters_deduplicates_and_sorts(self, monkeypatch):
        from tools.discord_tool import _build_schema

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        requests = []
        thread_a = {
            "id": "20", "name": "Daily Update", "parent_id": "forum",
            "thread_metadata": {"archived": True, "create_timestamp": "2025-01-01T00:00:00Z"},
            "message_count": 2,
        }
        thread_b = {
            "id": "30", "name": "daily standup", "parent_id": "forum",
            "thread_metadata": {"archived": False, "create_timestamp": "2025-01-02T00:00:00Z"},
            "message_count": 3,
        }
        thread_wrong_parent = {**thread_b, "id": "99", "parent_id": "other"}

        def request(method, path, token, **kwargs):
            requests.append((method, path, kwargs))
            if path == "/channels/forum":
                return {"guild_id": "guild"}
            if path == "/guilds/guild/threads/active":
                return {"threads": [thread_b, thread_wrong_parent]}
            if path == "/channels/forum/threads/archived/public":
                return {"threads": [thread_a, {**thread_b, "thread_metadata": {"archived": True}}], "has_more": False}
            raise AssertionError(path)

        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(
                action="list_forum_posts", channel_id="forum", query="DAILY", limit=2,
            ))

        assert result["count"] == 2
        assert result["posts"] == [
            {
                "id": "30", "name": "daily standup", "archived": False,
                "created_at": "2025-01-02T00:00:00Z", "message_count": 3,
            },
            {
                "id": "20", "name": "Daily Update", "archived": True,
                "created_at": "2025-01-01T00:00:00Z", "message_count": 2,
            },
        ]
        assert all(post["id"] != "99" for post in result["posts"])
        assert [req[1] for req in requests] == [
            "/channels/forum", "/guilds/guild/threads/active", "/channels/forum/threads/archived/public",
        ]
        schema = _build_schema(list(_CORE_ACTIONS), tool_name="discord")
        assert {"list_forum_posts", "create_forum_post", "post_message"}.issubset(
            schema["parameters"]["properties"]["action"]["enum"]
        )

    def test_list_forum_posts_paginates_archives_and_applies_limit(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        older = {
            "id": "21", "name": "Daily older", "parent_id": "forum",
            "thread_metadata": {
                "archived": True, "create_timestamp": "2025-01-01T00:00:00Z",
                "archive_timestamp": "2025-01-03T00:00:00Z",
            },
        }
        newer = {
            "id": "22", "name": "Daily newer", "parent_id": "forum",
            "thread_metadata": {
                "archived": True, "create_timestamp": "2025-01-02T00:00:00Z",
                "archive_timestamp": "2025-01-02T00:00:00Z",
            },
        }
        calls = []

        def request(method, path, token, **kwargs):
            calls.append((path, kwargs.get("params")))
            if path == "/channels/forum":
                return {"guild_id": "guild"}
            if path == "/guilds/guild/threads/active":
                return {"threads": []}
            if kwargs.get("params", {}).get("before"):
                return {"threads": [newer], "has_more": False}
            return {"threads": [older], "has_more": True}

        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(
                action="list_forum_posts", channel_id="forum", query="daily", limit=1,
            ))

        assert result["count"] == 1
        assert result["posts"][0]["id"] == "22"
        assert calls[-1][1] == {"limit": "100", "before": "2025-01-03T00:00:00Z"}
        assert result["has_more"] is True  # a second match was cut by limit=1

    def test_list_forum_posts_has_more_false_when_complete(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        thread = {"id": "22", "name": "Daily", "parent_id": "forum", "thread_metadata": {}}

        def request(method, path, token, **kwargs):
            if path == "/channels/forum":
                return {"guild_id": "guild"}
            if path == "/guilds/guild/threads/active":
                return {"threads": [thread]}
            return {"threads": [], "has_more": False}

        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(action="list_forum_posts", channel_id="forum", limit=5))
        assert result["count"] == 1
        assert result["has_more"] is False

    def test_list_forum_posts_reports_timestamp_tie_truncation(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        page_threads = [
            {
                "id": str(index), "name": f"Daily {index}", "parent_id": "forum",
                "thread_metadata": {
                    "archived": True, "create_timestamp": f"2025-01-0{index}T00:00:00Z",
                    "archive_timestamp": "2025-01-03T00:00:00Z" if index in (99, 100)
                    else f"2025-01-0{index}T00:00:00Z",
                },
            }
            for index in range(1, 101)
        ]

        def request(method, path, token, **kwargs):
            if path == "/channels/forum":
                return {"guild_id": "guild"}
            if path == "/guilds/guild/threads/active":
                return {"threads": []}
            return {"threads": page_threads, "has_more": True}

        with patch("tools.discord_tool._discord_request", side_effect=request) as mock_request:
            result = json.loads(discord_core(action="list_forum_posts", channel_id="forum"))
        assert result["has_more"] is True
        assert result["count"] == 50
        assert mock_request.call_count == 3  # A shared page-boundary timestamp cannot advance safely.

    def test_list_forum_posts_bounds_archive_scan(self, monkeypatch):
        from tools import discord_tool

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        monkeypatch.setattr(discord_tool, "_MAX_ARCHIVED_THREAD_PAGES", 2)
        archive_calls = []

        def request(method, path, token, **kwargs):
            if path == "/channels/forum":
                return {"guild_id": "guild"}
            if path == "/guilds/guild/threads/active":
                return {"threads": []}
            archive_calls.append(kwargs.get("params"))
            page_number = len(archive_calls)
            thread = {
                "id": str(page_number), "name": f"Daily {page_number}", "parent_id": "forum",
                "thread_metadata": {
                    "archived": True, "create_timestamp": f"2025-01-0{page_number}T00:00:00Z",
                    "archive_timestamp": f"2025-01-0{page_number}T00:00:00Z",
                },
            }
            return {"threads": [thread], "has_more": True}

        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(action="list_forum_posts", channel_id="forum", limit=1))
        assert len(archive_calls) == 2
        assert result["count"] == 1
        assert result["has_more"] is True

    def test_create_forum_post_splits_starter_and_remainder(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        calls = []

        def request(method, path, token, **kwargs):
            calls.append((method, path, kwargs.get("body")))
            if path == "/channels/forum/threads":
                return {"id": "thread-1", "name": "2025-01-01", "message": {"id": "starter-1"}}
            return {"id": f"message-{len(calls)}"}

        content = ("line of text\n" * 200).strip()  # ~2.6k chars, splits once at a newline
        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(
                action="create_forum_post", channel_id="forum", name="2025-01-01", content=content,
            ))

        assert result == {
            "success": True, "thread_id": "thread-1", "name": "2025-01-01",
            "starter_message_id": "starter-1", "additional_message_ids": ["message-2"],
        }
        mentions = {"parse": ["users"], "replied_user": True}
        assert [(method, path) for method, path, _ in calls] == [
            ("POST", "/channels/forum/threads"), ("POST", "/channels/thread-1/messages")]
        thread_body = calls[0][2]
        assert thread_body["name"] == "2025-01-01"
        assert thread_body["auto_archive_duration"] == 1440
        assert thread_body["message"]["allowed_mentions"] == mentions
        assert calls[1][2]["allowed_mentions"] == mentions
        starter, rest = thread_body["message"]["content"], calls[1][2]["content"]
        assert len(starter) <= 2000 and len(rest) <= 2000
        assert starter.endswith("line of text\n")  # split after a newline
        assert starter + rest == content  # nothing added or stripped

    def test_message_chunks_keep_code_fences_balanced(self):
        from tools.discord_tool import _message_chunks

        content = "```python\n" + "print('x')\n" * 400 + "```"
        chunks = _message_chunks(content)
        assert len(chunks) > 1
        assert all(len(chunk) <= 2000 for chunk in chunks)
        assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
        assert all(chunk.startswith("```python\n") for chunk in chunks)
        assert _message_chunks("short") == ["short"]

    def test_message_chunks_preserve_indentation_across_splits(self):
        from tools.discord_tool import _message_chunks

        content = "".join(f"- item {i}\n    nested detail {i}\n" for i in range(150))
        chunks = _message_chunks(content)
        assert len(chunks) > 1
        assert "".join(chunks) == content
        assert all(len(chunk) <= 2000 for chunk in chunks)

    def test_post_channels_scope_limits_post_targets(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"server_actions": "", "post_channels": ["forum"]}},
        )
        channels = {
            "/channels/post-in-forum": {
                "id": "post-in-forum", "type": 11, "parent_id": "forum", "thread_metadata": {}},
            "/channels/elsewhere": {"id": "elsewhere", "type": 11, "parent_id": "other"},
            # A text channel under an allowlisted category must not inherit the category's scope.
            "/channels/text-under-forum": {"id": "text-under-forum", "type": 0, "parent_id": "forum"},
        }

        def request(method, path, token, **kwargs):
            if method == "GET":
                return channels[path]
            if path == "/channels/forum/threads":
                return {"id": "t", "name": "n", "message": {"id": "s"}}
            return {"id": "m"}

        with patch("tools.discord_tool._discord_request", side_effect=request) as mock_request:
            ok = json.loads(discord_core(action="post_message", channel_id="post-in-forum", content="hi"))
            created = json.loads(discord_core(
                action="create_forum_post", channel_id="forum", name="n", content="hi"))
            denied = json.loads(discord_core(action="post_message", channel_id="elsewhere", content="hi"))
            denied_forum = json.loads(discord_core(
                action="create_forum_post", channel_id="elsewhere", name="n", content="hi"))
            denied_text = json.loads(discord_core(
                action="post_message", channel_id="text-under-forum", content="hi"))
            denied_thread_as_forum = json.loads(discord_core(
                action="create_forum_post", channel_id="post-in-forum", name="n", content="hi"))
        assert ok["success"] is True
        assert created["success"] is True
        assert "post_channels" in denied["error"]
        assert "post_channels" in denied_forum["error"]
        assert "post_channels" in denied_text["error"]
        assert "post_channels" in denied_thread_as_forum["error"]
        posted_paths = [c.args[1] for c in mock_request.call_args_list if c.args[0] == "POST"]
        assert all("elsewhere" not in path for path in posted_paths)

    @pytest.mark.parametrize("value", [None, "", []])
    def test_post_channels_empty_means_unrestricted(self, monkeypatch, value):
        from tools.discord_tool import _load_post_channels_config

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"discord": {"post_channels": value}})
        assert _load_post_channels_config() is None

    def test_post_channels_fails_closed(self, monkeypatch):
        from tools.discord_tool import _load_post_channels_config

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"discord": {"post_channels": {"a": 1}}})
        with pytest.raises(ValueError, match="refusing to post"):
            _load_post_channels_config()

        def broken():
            raise OSError("unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", broken)
        with pytest.raises(ValueError, match="refusing to post"):
            _load_post_channels_config()
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"discord": {"post_channels": 123}})
        assert _load_post_channels_config() == {"123"}

    def test_rest_message_mentions_follow_profile_config(self, monkeypatch):
        from tools.discord_tool import _allowed_mentions

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"discord": {"allow_mentions": {
                "everyone": True, "roles": True, "users": False, "replied_user": False,
            }}},
        )
        with patch("tools.discord_tool.get_secret", return_value=None):
            assert _allowed_mentions() == {"parse": ["everyone", "roles"], "replied_user": False}

    def test_create_forum_post_supports_required_tags(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        with patch("tools.discord_tool._discord_request", return_value={
            "id": "thread-1", "name": "daily", "message": {"id": "starter-1"},
        }) as request:
            discord_core(
                action="create_forum_post", channel_id="forum", name="daily", content="text",
                applied_tags=["123456789012345678"],
            )
        assert request.call_args.kwargs["body"]["applied_tags"] == ["123456789012345678"]

    def test_chunk_failures_report_partial_delivery(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)

        def fail_second_chunk(method, path, token, **kwargs):
            if path == "/channels/forum/threads":
                return {"id": "thread-1", "name": "daily", "message": {"id": "starter-1"}}
            raise DiscordAPIError(500, "failed")

        with patch("tools.discord_tool._discord_request", side_effect=fail_second_chunk):
            created = json.loads(discord_core(
                action="create_forum_post", channel_id="forum", name="daily", content="x" * 2001,
            ))
        assert "thread-1" in created["error"]
        assert "starter-1" in created["error"]
        assert "partial" in created["error"].lower()

        def fail_after_one_message(method, path, token, **kwargs):
            if method == "GET":
                return {"id": "thread", "thread_metadata": {"archived": False}}
            if len([call for call in calls if call[0] == "POST"]) == 1:
                raise DiscordAPIError(500, "failed")
            calls.append((method, path))
            return {"id": "sent-1"}

        calls = []
        with patch("tools.discord_tool._discord_request", side_effect=fail_after_one_message):
            posted = json.loads(discord_core(
                action="post_message", channel_id="thread", content="y" * 2001,
            ))
        assert "sent-1" in posted["error"]
        assert "partial" in posted["error"].lower()

    def test_post_message_unarchives_and_splits_long_content(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        calls = []

        def request(method, path, token, **kwargs):
            calls.append((method, path, kwargs.get("body")))
            if method == "GET":
                return {"id": "thread", "type": 11, "thread_metadata": {"archived": True}}
            if method == "PATCH":
                return {"id": "thread", "thread_metadata": {"archived": False}}
            return {"id": f"message-{len(calls)}"}

        with patch("tools.discord_tool._discord_request", side_effect=request):
            result = json.loads(discord_core(
                action="post_message", channel_id="thread", content="b" * 2001,
            ))

        assert result == {"success": True, "message_ids": ["message-3", "message-4"]}
        assert calls[:2] == [
            ("GET", "/channels/thread", None),
            ("PATCH", "/channels/thread", {"archived": False}),
        ]
        sent = [body for method, _path, body in calls[2:]]
        assert [method for method, _p, _b in calls[2:]] == ["POST", "POST"]
        assert all(len(body["content"]) <= 2000 for body in sent)
        assert all(body["allowed_mentions"] == {"parse": ["users"], "replied_user": True} for body in sent)
        assert "".join(body["content"].rsplit(" (", 1)[0] for body in sent) == "b" * 2001

    def test_discord_request_retries_short_rate_limits(self, monkeypatch):
        from tools import discord_tool

        sleeps = []
        monkeypatch.setattr(discord_tool.time, "sleep", sleeps.append)
        responses = [DiscordAPIError(429, '{"retry_after": 0.5}'), {"id": "ok"}]

        def once(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("tools.discord_tool._discord_request_once", side_effect=once):
            assert discord_tool._discord_request("POST", "/x", "tok") == {"id": "ok"}
        assert sleeps == [0.5]

    def test_global_rate_limit_from_headers_delays_later_requests(self, monkeypatch):
        from tools import discord_tool

        clock = [100.0]
        sleeps = []
        monkeypatch.setattr(discord_tool.time, "monotonic", lambda: clock[0])

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        monkeypatch.setattr(discord_tool.time, "sleep", sleep)
        monkeypatch.setattr(discord_tool, "_global_rate_limit_until", 0.0)
        error = DiscordAPIError(429, "", {"Retry-After": "2", "X-RateLimit-Global": "true"})
        assert discord_tool._rate_limit_wait(error) == 2.0
        responses = [error, {"id": "ok"}]

        def once(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch("tools.discord_tool._discord_request_once", side_effect=once):
            assert discord_tool._discord_request("POST", "/x", "tok") == {"id": "ok"}
        assert discord_tool._global_rate_limit_until == 102.0
        assert sleeps == [2.0]  # the retry waits once; the global gate is already past

    def test_discord_request_gives_up_on_long_or_repeated_rate_limits(self, monkeypatch):
        from tools import discord_tool

        monkeypatch.setattr(discord_tool.time, "sleep", lambda _s: None)
        with patch("tools.discord_tool._discord_request_once",
                   side_effect=DiscordAPIError(429, '{"retry_after": 60}')) as once:
            with pytest.raises(DiscordAPIError):
                discord_tool._discord_request("POST", "/x", "tok")
        assert once.call_count == 1
        with patch("tools.discord_tool._discord_request_once",
                   side_effect=DiscordAPIError(429, '{"retry_after": 0.1}')) as once:
            with pytest.raises(DiscordAPIError):
                discord_tool._discord_request("POST", "/x", "tok")
        assert once.call_count == discord_tool._RATE_LIMIT_RETRIES + 1

    def test_new_actions_validate_required_params_and_name_limit(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
        self._allow_actions(monkeypatch)
        with patch("tools.discord_tool._discord_request") as request:
            missing = json.loads(discord_core(
                action="create_forum_post", channel_id="forum", name="today",
            ))
            too_long = json.loads(discord_core(
                action="create_forum_post", channel_id="forum", name="x" * 101, content="text",
            ))
        assert "content" in missing["error"]
        assert "100 characters" in too_long["error"]
        empty = json.loads(discord_core(
            action="create_forum_post", channel_id="forum", name="today", content="",
        ))
        invalid_tags = json.loads(discord_core(
            action="create_forum_post", channel_id="forum", name="today", content="text",
            applied_tags=["not-a-snowflake"],
        ))
        too_many_tags = json.loads(discord_core(
            action="create_forum_post", channel_id="forum", name="today", content="text",
            applied_tags=["123456789012345678"] * 6,
        ))
        assert "content" in empty["error"]
        assert "positive 64-bit" in invalid_tags["error"]
        assert "five" in too_many_tags["error"]
        request.assert_not_called()
