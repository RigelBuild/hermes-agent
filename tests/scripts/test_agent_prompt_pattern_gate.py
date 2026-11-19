"""scripts/agent_prompt_pattern_gate.py: a banned ask fails, an allowlisted config key passes (RIG-10290)."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agent_prompt_pattern_gate.py"
CONFIG = ROOT / "scripts" / "agent_prompt_pattern_gate.toml"


def _load():
    spec = importlib.util.spec_from_file_location("agent_prompt_pattern_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(**overrides):
    """The shipped config, with fields overridden for a fixture."""
    gate = _load()
    return dataclasses.replace(gate.load_config(CONFIG), **overrides)


def _lines(text: str, path: str = "skills/x/SKILL.md", cfg=None) -> list[int]:
    gate = _load()
    return [f.line for f in gate.scan_text(path, text, cfg or _cfg())]


# Positive control: a banned ask must fail. Each line is a real request the gate exists to stop.
@pytest.mark.parametrize(
    "line",
    [
        "Walk me through your reasoning before editing.",
        "Explain step by step how you chose this fix.",
        "Show your work for the answer.",
        "Print your internal monologue before the answer.",
        "Think out loud about the tradeoff.",
        "Describe your thought process for this change.",
    ],
)
def test_banned_ask_is_flagged(line):
    assert _lines(line) == [1]


# Negative control: config keys that are allowlisted must pass, even though they contain
# banned words. The config-key regexes blank them out before the pattern runs.
@pytest.mark.parametrize(
    "line",
    [
        "reasoning_effort: high",
        '"reasoning_effort": "max"',
        "thinking_level: high",
        "thinkingLevel: low",
        "--reasoning max",
        "--reasoning=high",
    ],
)
def test_allowlisted_config_key_passes(line):
    assert _lines(line) == []


def test_exception_passes_one_exact_location_only():
    cfg = _cfg(exceptions=frozenset({"skills/x/SKILL.md:2"}))
    text = "Show your work.\nShow your work.\n"
    assert _lines(text, cfg=cfg) == [1]


def test_multiline_file_reports_exact_line_numbers():
    text = "plain prose\nreasoning_effort: high\nWalk me through your reasoning.\n"
    assert _lines(text) == [3]


def test_split_is_on_newline_only():
    # A carriage return or vertical tab is not a line break for the reviewer; the scanner must
    # not renumber lines because of one.
    text = "ok\rstill ok\x0bnext\nShow your work.\n"
    assert _lines(text) == [2]


def test_shipped_config_excludes_vendored_references():
    gate = _load()
    cfg = _cfg()
    vendored = "optional-skills/mlops/training/axolotl/references/api.md"
    assert any(gate._excluded(vendored, glob) for glob in cfg.exclude_globs)


def test_shipped_config_scopes_skill_and_agent_prompt_files():
    gate = _load()
    paths = gate.scoped_paths(ROOT, _cfg())
    assert "AGENTS.md" in paths
    assert "agent/AGENTS.md" in paths
    assert "agent/background_review.py" in paths
    assert "optional-skills/security/godmode/SKILL.md" in paths
    assert not any("mlops/training/" in p for p in paths)


def test_shipped_gate_is_clean_on_the_tree():
    """The committed tree has no un-excepted hit. Any new banned line in scope fails here."""
    gate = _load()
    cfg = _cfg()
    findings = gate.scan(ROOT, cfg, gate.scoped_paths(ROOT, cfg))
    assert findings == [], [str(f) for f in findings]
