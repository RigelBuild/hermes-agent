"""Terminal tool schemas advertise the effective configured timeout."""

from unittest.mock import patch

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import terminal_tool
from tools.registry import registry
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope


@pytest.fixture
def terminal_config_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    with patch.dict("os.environ"):
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)
        monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
        yield home


def _registered_timeout_description() -> str:
    definitions = registry.get_definitions({"terminal"}, quiet=True)
    terminal = next(definition["function"] for definition in definitions
                    if definition["function"]["name"] == "terminal")
    return terminal["parameters"]["properties"]["timeout"]["description"]


def test_registered_terminal_schema_advertises_configured_timeout(
    terminal_config_home,
):
    (terminal_config_home / "config.yaml").write_text(
        "terminal:\n  timeout: 60\n", encoding="utf-8"
    )

    assert "default 60" in _registered_timeout_description()
    assert "default 180" in terminal_tool.TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]


def test_registered_terminal_schema_keeps_default_timeout(terminal_config_home):
    assert DEFAULT_CONFIG["terminal"]["timeout"] == 180

    assert "default 180" in _registered_timeout_description()


def test_registered_terminal_schema_uses_bound_timeout_over_ambient(
    terminal_config_home, monkeypatch,
):
    monkeypatch.setenv("TERMINAL_TIMEOUT", "90")
    home_token = set_hermes_home_override(str(terminal_config_home.parent / "routed"))
    scope = set_terminal_scope({"TERMINAL_ENV": "local", "TERMINAL_TIMEOUT": "60"})
    try:
        description = _registered_timeout_description()
    finally:
        reset_terminal_scope(scope)
        reset_hermes_home_override(home_token)

    assert "default 60" in description
    assert "default 90" not in description


def test_registered_terminal_schema_skips_launch_timeout_for_routed_home_without_scope(
    terminal_config_home, monkeypatch,
):
    monkeypatch.setenv("TERMINAL_TIMEOUT", "90")
    routed_home = terminal_config_home.parent / "routed"
    routed_home.mkdir()
    (routed_home / "config.yaml").write_text(
        "terminal:\n  timeout: 60\n", encoding="utf-8"
    )
    home_token = set_hermes_home_override(str(routed_home))
    try:
        description = _registered_timeout_description()
    finally:
        reset_hermes_home_override(home_token)

    assert "default 180" in description
    assert "default 90" not in description


