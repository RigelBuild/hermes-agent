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
