"""Per-call delegation model tiers."""

import json
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool
from tools.registry import registry


TIER_MODELS = {
    "easy": "fixture/easy",
    "medium": "fixture/medium",
    "hard": "fixture/hard",
}


def _parent():
    from tests.tools.test_delegate import _make_mock_parent

    return _make_mock_parent(depth=0)


def _config(*, tiers=True):
    config = {
        "model": "fixture/default",
        "base_url": "https://tiers.example/v1",
        "api_key": "fixture-key",
    }
    if tiers:
        config["tiers"] = {
            tier: {
                "model": model,
                "base_url": "https://tiers.example/v1",
                "api_key": "fixture-key",
            }
            for tier, model in TIER_MODELS.items()
        }
    return config


def _dispatch(args, config, monkeypatch, tmp_path, *, resolve_credentials=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    child = MagicMock()
    resolver_patch = (
        patch.object(delegate_tool, "_resolve_delegation_credentials", side_effect=resolve_credentials)
        if resolve_credentials is not None else nullcontext()
    )
    with (
        patch.object(delegate_tool, "_load_config", return_value=config),
        patch.object(delegate_tool, "_resolve_workspace_hint", return_value=str(tmp_path)),
        patch.object(delegate_tool, "_model_background_value", return_value=False),
        patch.object(delegate_tool, "_run_batch", return_value='{"ok": true}'),
        patch("run_agent.AIAgent", return_value=child) as agent_class,
        resolver_patch,
    ):
        result = json.loads(registry.dispatch("delegate_task", args, parent_agent=_parent()))
    return result, agent_class


def test_non_mapping_tiers_config_is_rejected(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {"goal": "Inspect fixture", "tier": "easy"},
        {**_config(), "tiers": ["easy"]}, monkeypatch, tmp_path,
    )

    assert "error" in result
    assert "expected a mapping" in result["error"]
    agent_class.assert_not_called()


def test_tier_provider_does_not_inherit_base_endpoint(monkeypatch, tmp_path):
    config = _config()
    config["tiers"]["hard"] = {"model": "openrouter/model", "provider": "openrouter"}
    resolved = {}

    def resolve_credentials(routing_cfg, _parent):
        resolved.update(routing_cfg)
        return {"model": "openrouter/model", "provider": "openrouter", "base_url": None,
                "api_key": "fixture-key", "api_mode": None, "request_overrides": None,
                "command": None, "args": None}

    _dispatch(
        {"goal": "Inspect fixture", "tier": "hard"}, config, monkeypatch, tmp_path,
        resolve_credentials=resolve_credentials,
    )

    assert resolved["model"] == "openrouter/model"
    assert resolved["provider"] == "openrouter"
    assert not {"base_url", "api_key", "api_mode"} & resolved.keys()
    assert "fixture/default" not in resolved.values()


def test_selected_tier_drops_base_route_options_and_inherits_parent_provider(monkeypatch, tmp_path):
    config = _config()
    config.update({
        "provider": "nous",
        "request_overrides": {"temperature": 0.7},
        "fallback_providers": [{"provider": "nous", "model": "fallback-model"}],
        "max_iterations": 123,
    })
    config["tiers"]["hard"] = {"model": TIER_MODELS["hard"]}
    resolved = {}

    def resolve_credentials(routing_cfg, _parent):
        resolved.update(routing_cfg)
        return {
            "model": TIER_MODELS["hard"], "provider": None, "base_url": None,
            "api_key": None, "api_mode": None, "request_overrides": None,
            "command": None, "args": None,
        }

    result, agent_class = _dispatch(
        {"goal": "Inspect fixture", "tier": "hard"}, config, monkeypatch, tmp_path,
        resolve_credentials=resolve_credentials,
    )

    assert result == {"ok": True}
    assert resolved == {"max_iterations": 123, "model": TIER_MODELS["hard"]}
    assert not {"provider", "base_url", "api_key", "api_mode", "request_overrides", "fallback_providers"} & resolved.keys()
    child_kwargs = agent_class.call_args.kwargs
    assert child_kwargs["provider"] == "openrouter"
    assert child_kwargs["base_url"] == "https://openrouter.ai/api/v1"


def test_tier_is_ignored_when_tiers_are_unconfigured(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {"goal": "Inspect fixture", "tier": "hard"},
        _config(tiers=False), monkeypatch, tmp_path,
    )

    assert result == {"ok": True}
    assert agent_class.call_args.kwargs["model"] == "fixture/default"
    assert agent_class.call_args.kwargs["provider"] == "custom"


@pytest.mark.parametrize("tier", [None, "unknown"])
def test_configured_tiers_refuse_missing_or_unknown_tier_without_spawning(
    tier, monkeypatch, tmp_path,
):
    args = {"goal": "Inspect fixture"}
    if tier is not None:
        args["tier"] = tier

    result, agent_class = _dispatch(args, _config(), monkeypatch, tmp_path)

    assert "error" in result
    assert all(name in result["error"] for name in ("easy", "medium", "hard"))
    agent_class.assert_not_called()


def test_control_calls_need_no_tier(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {"action": "list"}, _config(), monkeypatch, tmp_path,
    )

    assert result["action"] == "list"
    assert "error" not in result
    agent_class.assert_not_called()


@pytest.mark.parametrize("tier,model", TIER_MODELS.items())
def test_each_tier_resolves_its_model_and_provider_for_child(
    tier, model, monkeypatch, tmp_path,
):
    result, agent_class = _dispatch(
        {"goal": "Inspect fixture", "tier": tier}, _config(), monkeypatch, tmp_path,
    )

    assert result == {"ok": True}
    child_kwargs = agent_class.call_args.kwargs
    assert child_kwargs["model"] == model
    assert child_kwargs["provider"] == "custom"
    assert child_kwargs["base_url"] == "https://tiers.example/v1"
    assert child_kwargs["api_key"] == "fixture-key"


def test_batch_uses_selected_tier_for_every_task(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {
            "tasks": [{"goal": "Inspect fixture A"}, {"goal": "Inspect fixture B"}],
            "tier": "hard",
        },
        _config(),
        monkeypatch,
        tmp_path,
    )

    assert result == {"ok": True}
    assert agent_class.call_count == 2
    assert [call.kwargs["model"] for call in agent_class.call_args_list] == [
        TIER_MODELS["hard"], TIER_MODELS["hard"],
    ]


def test_batch_without_tier_is_refused_before_spawning(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {"tasks": [{"goal": "Inspect fixture A"}, {"goal": "Inspect fixture B"}]},
        _config(),
        monkeypatch,
        tmp_path,
    )

    assert "error" in result
    assert all(name in result["error"] for name in ("easy", "medium", "hard"))
    agent_class.assert_not_called()


def test_missing_requested_tier_is_a_config_error(monkeypatch, tmp_path):
    config = _config()
    del config["tiers"]["hard"]

    result, agent_class = _dispatch(
        {"goal": "Inspect fixture", "tier": "hard"}, config, monkeypatch, tmp_path,
    )

    assert "error" in result
    assert "hard" in result["error"]
    assert "delegation.tiers" in result["error"]
    agent_class.assert_not_called()


def test_without_tiers_legacy_call_uses_delegation_model(monkeypatch, tmp_path):
    result, agent_class = _dispatch(
        {"goal": "Inspect fixture"}, _config(tiers=False), monkeypatch, tmp_path,
    )

    assert result == {"ok": True}
    child_kwargs = agent_class.call_args.kwargs
    assert child_kwargs["model"] == "fixture/default"
    assert child_kwargs["provider"] == "custom"


def test_schema_requires_tier_only_when_tiers_are_configured(monkeypatch):
    for enabled in (True, False):
        config = _config(tiers=enabled)
        monkeypatch.setattr(delegate_tool, "_load_config", lambda config=config: config)
        definition = registry.get_definitions({"delegate_task"})[0]["function"]
        parameters = definition["parameters"]
        if enabled:
            assert "tier" in parameters["required"]
            assert parameters["properties"]["tier"]["enum"] == ["easy", "medium", "hard"]
            assert all(
                label in parameters["properties"]["tier"]["description"]
                for label in ("easy", "medium", "hard")
            )
        else:
            assert "tier" not in parameters["required"]


_UNSET = object()


@pytest.mark.parametrize(
    ("tier_effort", "base_effort", "expected"),
    [("high", "low", "high"), (_UNSET, "low", "low"), (None, "low", "low"), ("", "low", "low"),
     (False, "high", False)],
)
def test_tier_reasoning_effort_overrides_delegation_effort(
    tier_effort, base_effort, expected, monkeypatch, tmp_path,
):
    from hermes_constants import parse_reasoning_effort

    config = {**_config(), "reasoning_effort": base_effort}
    if tier_effort is not _UNSET:
        config["tiers"]["easy"]["reasoning_effort"] = tier_effort
    _, agent_class = _dispatch({"goal": "Inspect fixture", "tier": "easy"}, config, monkeypatch, tmp_path)

    assert agent_class.call_args.kwargs["reasoning_config"] == parse_reasoning_effort(expected)


def test_delegation_effort_applies_without_tiers(monkeypatch, tmp_path):
    from hermes_constants import parse_reasoning_effort

    config = {**_config(tiers=False), "reasoning_effort": "low"}
    _, agent_class = _dispatch({"goal": "Inspect fixture"}, config, monkeypatch, tmp_path)

    assert agent_class.call_args.kwargs["reasoning_config"] == parse_reasoning_effort("low")


def test_tier_effort_typo_is_refused(monkeypatch, tmp_path):
    config = _config()
    config["tiers"]["easy"]["reasoning_effort"] = "hgih"
    result, agent_class = _dispatch({"goal": "Inspect fixture", "tier": "easy"}, config, monkeypatch, tmp_path)

    assert "unknown reasoning_effort 'hgih'" in result["error"]
    agent_class.assert_not_called()


_BACKEND_PARAMS = ({"option": {"type": "string", "enum": ["fast"]}}, "Backend guidance.")


def test_tiers_hide_backend_params_from_schema(monkeypatch):
    for enabled in (True, False):
        config = _config(tiers=enabled)
        monkeypatch.setattr(delegate_tool, "_load_config", lambda config=config: config)
        with patch("tools.delegate_tool_routes.schema_extension", return_value=_BACKEND_PARAMS):
            definition = registry.get_definitions({"delegate_task"})[0]["function"]
        props = definition["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert ("option" in props) is not enabled


def test_tiered_spawn_refuses_backend_params(monkeypatch, tmp_path):
    with patch("tools.delegate_tool_routes.schema_extension", return_value=_BACKEND_PARAMS):
        result, agent_class = _dispatch(
            {"tasks": [{"goal": "Inspect fixture", "option": "fast"}], "tier": "easy"},
            _config(), monkeypatch, tmp_path,
        )

    assert "take precedence over delegation.backend" in result["error"]
    assert "option" in result["error"]
    agent_class.assert_not_called()
