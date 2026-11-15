"""Terminal tool schemas advertise the effective configured timeout."""

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools import terminal_tool
from tools.registry import registry


@pytest.fixture
def terminal_config_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    return home


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
