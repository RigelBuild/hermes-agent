#!/usr/bin/env python3
"""Fail when agent prompts ask a model to show, explain or walk through its reasoning.

RIG-10290. Prompts that an agent reads or sends (skills, their references, the
prompt-builder modules, the compressor's summary prompts, the scheduler prompt,
and the repo AGENTS.md files) must not ask the model for its reasoning, thinking,
chain of thought, or step-by-step narration. Those requests pull hidden reasoning
into visible output, and the source repo's agent-prompt-pattern gate blocks them for
the same reason. This scanner is the hermes port of that gate.

A line is flagged when the ``pattern`` matches it after every ``configKeys``
match is blanked out. Config keys such as ``reasoning_effort: high`` or
``--reasoning`` are therefore never flagged. A flagged line is allowed only when
its exact ``path:line`` appears in ``exceptions``; those entries are markdown only.
A ``.py`` file listed in ``file_exceptions`` is skipped whole.

Config: ``agent_prompt_pattern_gate.toml`` beside this file. It is read with
``tomllib`` and has no third-party dependency.

Usage:
    python3 scripts/agent_prompt_pattern_gate.py            # scan the repo
    python3 scripts/agent_prompt_pattern_gate.py --root DIR # scan another tree

Exit codes: 0 clean, 1 at least one finding, 2 configuration or I/O error.
Each finding prints as ``<path>:<line>: disallowed prompt pattern``.
A stale exception, an entry in ``exceptions`` that matches no raw hit,
prints as ``<path>:<line>: stale exception``.
A ``file_exceptions`` entry with no raw hit, or a path not in the tree,
prints as ``<path>:0: stale file exception``.
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
STALE_TEXT = "stale exception"
STALE_FILE_TEXT = "stale file exception"


@dataclass(frozen=True)
class Config:
    globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    pattern: re.Pattern[str]
    config_keys: tuple[re.Pattern[str], ...]
    exceptions: frozenset[str]
    file_exceptions: frozenset[str]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    text: str = FINDING_TEXT

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.text}"


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
    if not all(e.rpartition(":")[0].endswith(".md") for e in exceptions):
        raise ValueError(
            "config: exceptions must be markdown; use file_exceptions for .py"
        )
    file_exceptions = raw.get("file_exceptions", [])
    if not isinstance(file_exceptions, list) or not all(
        isinstance(f, str) for f in file_exceptions
    ):
        raise ValueError("config: file_exceptions must be a list of strings")
    if not all(f.endswith(".py") for f in file_exceptions):
        raise ValueError("config: file_exceptions must be .py paths")

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
        file_exceptions=frozenset(file_exceptions),
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


def raw_hit_lines(text: str, cfg: Config) -> list[int]:
    """1-based line numbers matched by ``pattern`` after every config key is blanked out."""
    hits: list[int] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for key in cfg.config_keys:
            line = key.sub(lambda m: " " * len(m.group(0)), line)
        if cfg.pattern.search(line):
            hits.append(number)
    return hits


def scan_text(path: str, text: str, cfg: Config) -> list[Finding]:
    """Findings for one file's text. ``path`` is the repo-relative name used for exceptions."""
    if path in cfg.file_exceptions:
        return []
    return [
        Finding(path, number)
        for number in raw_hit_lines(text, cfg)
        if f"{path}:{number}" not in cfg.exceptions
    ]


def scan(root: Path, cfg: Config, paths: list[str]) -> list[Finding]:
    """Findings for ``paths``; stale exceptions and stale file exceptions are reported too."""
    findings: list[Finding] = []
    hit_locations: set[str] = set()
    hit_files: set[str] = set()
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        hits = raw_hit_lines(text, cfg)
        if rel in cfg.file_exceptions:
            if hits:
                hit_files.add(rel)
            continue
        for number in hits:
            location = f"{rel}:{number}"
            hit_locations.add(location)
            if location not in cfg.exceptions:
                findings.append(Finding(rel, number))
    for location in sorted(cfg.exceptions - hit_locations):
        rel, _, number = location.rpartition(":")
        findings.append(Finding(rel, int(number), STALE_TEXT))
    for rel in sorted(cfg.file_exceptions - hit_files):
        findings.append(Finding(rel, 0, STALE_FILE_TEXT))
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
