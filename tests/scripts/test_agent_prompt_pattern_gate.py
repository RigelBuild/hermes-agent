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
        "Fill the 'reasoning' field with why.",
        "if reasoning == x:",
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


def test_stale_exception_fails_the_gate(tmp_path):
    """An exception that matches no raw hit is reported stale, so the gate fails."""
    gate = _load()
    skill = tmp_path / "skills" / "x" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Plain prose.\n", encoding="utf-8")
    cfg = _cfg(exceptions=frozenset({"skills/x/SKILL.md:1"}), symbol_exceptions=frozenset())
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["skills/x/SKILL.md:1: stale exception"]


@pytest.mark.parametrize(
    "config_text",
    [
        'globs = ["skills/**/SKILL.md"]\npattern = "(unclosed"\n',
        'globs = [""]\npattern = "show your work"\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'exceptions = ["agent/prompt_builder.py:1"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = ["agent/prompt_builder.md:keep"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = ["agent/prompt_builder.py"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = ["agent/prompt_builder.py:not a symbol"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = ["agent/prompt_builder.py:Foo..bar"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = ["agent/prompt_builder.py:Foo.bar.baz"]\n',
        'globs = ["skills/**/SKILL.md"]\n'
        'pattern = "show your work"\n'
        'symbol_exceptions = "agent/prompt_builder.py:keep"\n',
    ],
    ids=[
        "bad-regex",
        "bad-glob",
        "py-path-line-exception",
        "non-py-symbol-exception",
        "symbol-without-colon",
        "symbol-not-identifier",
        "symbol-empty-part",
        "symbol-nested-too-deep",
        "symbol-not-list",
    ],
)
def test_invalid_config_exits_2(tmp_path, monkeypatch, config_text):
    gate = _load()
    bad = tmp_path / "bad.toml"
    bad.write_text(config_text, encoding="utf-8")
    load = gate.load_config
    monkeypatch.setattr(gate, "load_config", lambda _path: load(bad))
    assert gate.main(["--root", str(tmp_path)]) == 2


@pytest.fixture
def fixture_gate(monkeypatch):
    """The shipped patterns with no exceptions, so a fixture tree is judged on its own lines."""
    gate = _load()
    shipped = gate.load_config
    empty = dataclasses.replace(
        shipped(CONFIG), exceptions=frozenset(), symbol_exceptions=frozenset()
    )
    monkeypatch.setattr(gate, "load_config", lambda _path: empty)
    return gate


def test_main_exit_code_follows_banned_line(tmp_path, fixture_gate):
    skill = tmp_path / "skills" / "x" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Show your work.\n", encoding="utf-8")
    assert fixture_gate.main(["--root", str(tmp_path)]) == 1
    skill.write_text("Plain prose.\n", encoding="utf-8")
    assert fixture_gate.main(["--root", str(tmp_path)]) == 0


def test_python_assignment_passes_but_prose_in_same_file_is_flagged():
    text = 'reasoning_effort = "high"\nShow your work.\n'
    assert _lines(text, path="agent/prompt_builder.py") == [2]


@pytest.mark.parametrize(
    "line",
    [
        'reasoning_effort = "high"',
        "model.reasoning = x",
        'kwargs["reasoning"] = x',
    ],
)
def test_python_config_assignment_is_not_flagged(line):
    assert _lines(line, path="agent/prompt_builder.py") == []


def test_json_reasoning_key_in_prose_is_flagged():
    line = 'Return JSON: {"reasoning": "<why>", "answer": "<final>"}'
    assert _lines(line) == [1]


def test_symbol_exception_exempts_only_its_span(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'def keep():\n    return "Show your work."\n\n\nNOTE = "Show your work."\n',
        encoding="utf-8",
    )
    flagged = _cfg(exceptions=frozenset(), symbol_exceptions=frozenset())
    findings = gate.scan(tmp_path, flagged, gate.scoped_paths(tmp_path, flagged))
    assert [str(f) for f in findings] == [
        "agent/prompt_builder.py:2: disallowed prompt pattern",
        "agent/prompt_builder.py:5: disallowed prompt pattern",
    ]
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["agent/prompt_builder.py:5: disallowed prompt pattern"]


def test_symbol_exception_covers_decorator_and_comment_run_above(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "# Show your work in this note.\n"
        "@decorator  # Show your work.\n"
        "def keep():\n"
        "    return 1\n"
        "\n"
        "# Show your work.\n"
        "\n"
        "def other():\n"
        "    return 2\n",
        encoding="utf-8",
    )
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["agent/prompt_builder.py:6: disallowed prompt pattern"]


def test_symbol_exception_with_zero_raw_hits_is_stale(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text("def keep():\n    return 1\n", encoding="utf-8")
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["agent/prompt_builder.py:0: stale symbol exception: keep"]


def test_symbol_exception_naming_no_symbol_is_stale(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text("def keep():\n    return 1\n", encoding="utf-8")
    gone = frozenset({("agent/prompt_builder.py", "gone")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=gone)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["agent/prompt_builder.py:0: stale symbol exception: gone"]


def test_symbol_exception_missing_from_tree_is_stale(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text("Plain prose.\n", encoding="utf-8")
    gone = frozenset({("agent/gone.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=gone)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["agent/gone.py:0: stale symbol exception: keep"]


def test_ambiguous_symbol_exception_exempts_nothing_and_fails(tmp_path, monkeypatch):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'def keep():\n    return "Show your work."\n\nkeep = "Walk me through your reasoning."\n',
        encoding="utf-8",
    )
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == [
        "agent/prompt_builder.py:0: ambiguous symbol exception: keep",
        "agent/prompt_builder.py:2: disallowed prompt pattern",
        "agent/prompt_builder.py:4: disallowed prompt pattern",
    ]
    monkeypatch.setattr(gate, "load_config", lambda _path: cfg)
    assert gate.main(["--root", str(tmp_path)]) == 1


def test_unparseable_symbol_file_exits_2(tmp_path, monkeypatch):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text("def keep(:\n    return 1\n", encoding="utf-8")
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    monkeypatch.setattr(gate, "load_config", lambda _path: cfg)
    assert gate.main(["--root", str(tmp_path)]) == 2


def test_markdown_exception_and_symbol_exception_in_one_tree(tmp_path):
    gate = _load()
    skill = tmp_path / "skills" / "x" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Show your work.\n", encoding="utf-8")
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text("def keep():\n    return 'Show your work.'\n", encoding="utf-8")
    exempt = frozenset({("agent/prompt_builder.py", "keep")})
    cfg = _cfg(exceptions=frozenset({"skills/x/SKILL.md:1"}), symbol_exceptions=exempt)
    assert gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg)) == []
    cfg = _cfg(
        exceptions=frozenset({"skills/x/SKILL.md:1", "skills/x/SKILL.md:9"}),
        symbol_exceptions=exempt,
    )
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == ["skills/x/SKILL.md:9: stale exception"]


def test_dotted_symbol_exempts_only_that_method(tmp_path):
    gate = _load()
    module = tmp_path / "agent" / "prompt_builder.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "class Box:\n"
        "    NOTE = 'Show your work.'\n"
        "    def keep(self):\n"
        "        return 'Show your work.'\n"
        "    def other(self):\n"
        "        return 'Show your work.'\n",
        encoding="utf-8",
    )
    exempt = frozenset({("agent/prompt_builder.py", "Box.keep")})
    cfg = _cfg(exceptions=frozenset(), symbol_exceptions=exempt)
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    assert [str(f) for f in findings] == [
        "agent/prompt_builder.py:2: disallowed prompt pattern",
        "agent/prompt_builder.py:6: disallowed prompt pattern",
    ]


def _copied_findings(tmp_path, rel, text):
    """Findings for ``rel`` after ``text`` is written there and scanned with the shipped config.

    Other exempt files are absent from this tree, so their entries report stale; only ``rel`` counts.
    """
    gate = _load()
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    cfg = _cfg(exceptions=frozenset())
    findings = gate.scan(tmp_path, cfg, gate.scoped_paths(tmp_path, cfg))
    return [str(f) for f in findings if f.path == rel]


def test_banned_line_in_skill_review_prompt_is_flagged(tmp_path):
    rel = "agent/background_review.py"
    lines = (ROOT / rel).read_text(encoding="utf-8").split("\n")
    opener = lines.index("_SKILL_REVIEW_PROMPT = (")
    lines.insert(opener + 1, '    "Walk me through your reasoning. "')
    assert _copied_findings(tmp_path, rel, "\n".join(lines)) == [
        f"{rel}:{opener + 2}: disallowed prompt pattern"
    ]


def test_banned_line_in_build_summary_prompt_is_flagged(tmp_path):
    rel = "agent/context_compressor.py"
    lines = (ROOT / rel).read_text(encoding="utf-8").split("\n")
    method = lines.index("    def _build_summary_prompt(")
    opener = next(
        i for i in range(method, len(lines)) if lines[i].strip() == "_summarizer_preamble = ("
    )
    lines.insert(opener + 1, '            "Walk me through your reasoning. "')
    assert _copied_findings(tmp_path, rel, "\n".join(lines)) == [
        f"{rel}:{opener + 2}: disallowed prompt pattern"
    ]


def test_banned_module_constant_is_flagged(tmp_path):
    rel = "agent/background_review.py"
    text = (ROOT / rel).read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    line_no = text.count("\n") + 1
    text += '_NEW_REVIEW_PROMPT = "Show your work."\n'
    assert _copied_findings(tmp_path, rel, text) == [f"{rel}:{line_no}: disallowed prompt pattern"]
