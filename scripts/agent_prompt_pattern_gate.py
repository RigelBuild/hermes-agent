#!/usr/bin/env python3
"""Fail when agent prompts ask a model to show, explain or walk through its reasoning.

RIG-10290. Prompts that an agent reads or sends (skills, their references, the
prompt-builder modules, the compressor's summary prompts, the scheduler prompt,
and the repo AGENTS.md files) must not ask the model for its reasoning, thinking,
chain of thought, or step-by-step narration. Those requests pull hidden reasoning
into visible output, and the orion ``agent-prompt-pattern-gate`` blocks them for
the same reason. This scanner is the hermes port of that gate.

A line is flagged when the ``pattern`` matches it after every ``configKeys``
match is blanked out. Config keys such as ``reasoning_effort: high`` or
``--reasoning`` are therefore never flagged. A flagged line is allowed only when
its exact ``path:line`` appears in ``exceptions``.

Config: ``agent_prompt_pattern_gate.toml`` beside this file. It is read with
``tomllib`` and has no third-party dependency.

Usage:
    python3 scripts/agent_prompt_pattern_gate.py            # scan the repo
    python3 scripts/agent_prompt_pattern_gate.py --root DIR # scan another tree

Exit codes: 0 clean, 1 at least one finding, 2 configuration or I/O error.
Each finding prints as ``<path>:<line>: disallowed prompt pattern``.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_NAME = "agent_prompt_pattern_gate.toml"
FINDING_TEXT = "disallowed prompt pattern"


@dataclass(frozen=True)
class Config:
    globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    pattern: re.Pattern[str]
    config_keys: tuple[re.Pattern[str], ...]
    exceptions: frozenset[str]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {FINDING_TEXT}"


def load_config(path: Path) -> Config:
    """Read and validate the TOML config. Raises ``ValueError`` on a bad shape."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"read config {path}: {exc}") from exc

    globs = raw.get("globs")
    if not isinstance(globs, list) or not globs or not all(isinstance(g, str) for g in globs):
        raise ValueError("config: globs must be a non-empty list of strings")
    exclude = raw.get("exclude_globs", [])
    if not isinstance(exclude, list) or not all(isinstance(g, str) for g in exclude):
        raise ValueError("config: exclude_globs must be a list of strings")
    pattern_src = raw.get("pattern")
    if not isinstance(pattern_src, str) or not pattern_src:
        raise ValueError("config: pattern must be a non-empty string")
    keys_src = raw.get("config_keys", [])
    if not isinstance(keys_src, list) or not all(isinstance(k, str) for k in keys_src):
        raise ValueError("config: config_keys must be a list of strings")
    exceptions = raw.get("exceptions", [])
    if not isinstance(exceptions, list) or not all(isinstance(e, str) for e in exceptions):
        raise ValueError("config: exceptions must be a list of strings")

    try:
        pattern = re.compile(pattern_src)
        keys = tuple(re.compile(k) for k in keys_src)
    except re.error as exc:
        raise ValueError(f"config: bad regex: {exc}") from exc

    return Config(
        globs=tuple(globs),
        exclude_globs=tuple(exclude),
        pattern=pattern,
        config_keys=keys,
        exceptions=frozenset(exceptions),
    )


def scoped_paths(root: Path, cfg: Config) -> list[str]:
    """Repo-relative POSIX paths matched by ``globs`` and not by ``exclude_globs``."""
    seen: dict[str, None] = {}
    for glob in cfg.globs:
        for match in sorted(root.glob(glob)):
            if not match.is_file():
                continue
            rel = match.relative_to(root).as_posix()
            if any(_excluded(rel, ex) for ex in cfg.exclude_globs):
                continue
            seen.setdefault(rel, None)
    return list(seen)


def _excluded(rel: str, glob: str) -> bool:
    """True when ``rel`` matches ``glob``. ``**`` matches any number of path segments."""
    return _segments_match(rel.split("/"), glob.split("/"))


def _segments_match(path: list[str], pat: list[str]) -> bool:
    if not pat:
        return not path
    head, rest = pat[0], pat[1:]
    if head == "**":
        return any(_segments_match(path[i:], rest) for i in range(len(path) + 1))
    if not path:
        return False
    return fnmatch.fnmatchcase(path[0], head) and _segments_match(path[1:], rest)


def scan_text(path: str, text: str, cfg: Config) -> list[Finding]:
    """Findings for one file's text. ``path`` is the repo-relative name used for exceptions."""
    findings: list[Finding] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for key in cfg.config_keys:
            line = key.sub(lambda m: " " * len(m.group(0)), line)
        if cfg.pattern.search(line) and f"{path}:{number}" not in cfg.exceptions:
            findings.append(Finding(path, number))
    return findings


def scan(root: Path, cfg: Config, paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        findings.extend(scan_text(rel, text, cfg))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=None, help="repo root (default: this checkout)")
    args = parser.parse_args(argv)

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    try:
        cfg = load_config(Path(__file__).resolve().parent / CONFIG_NAME)
        findings = scan(root, cfg, scoped_paths(root, cfg))
    except (OSError, ValueError) as exc:
        print(f"agent-prompt-pattern-gate: {exc}", file=sys.stderr)
        return 2

    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
