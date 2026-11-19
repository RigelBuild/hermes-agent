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
A raw hit inside the source span of a ``symbol_exceptions`` entry is allowed too. A span
is a top-level or class-nested def, class, or assignment, with its decorators and the
comment lines directly above it.

Config: ``agent_prompt_pattern_gate.toml`` beside this file. It is read with
``tomllib`` and has no third-party dependency.

Usage:
    python3 scripts/agent_prompt_pattern_gate.py            # scan the repo
    python3 scripts/agent_prompt_pattern_gate.py --root DIR # scan another tree

Exit codes: 0 clean, 1 at least one finding, 2 configuration, I/O, or parse error.
Each finding prints as ``<path>:<line>: disallowed prompt pattern``.
A stale exception, an entry in ``exceptions`` that matches no raw hit,
prints as ``<path>:<line>: stale exception``.
A ``symbol_exceptions`` entry that names no symbol, whose span has no raw hit, or whose
file is not scanned, prints as ``<path>:0: stale symbol exception``.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_NAME = "agent_prompt_pattern_gate.toml"
FINDING_TEXT = "disallowed prompt pattern"
STALE_TEXT = "stale exception"
STALE_SYMBOL_TEXT = "stale symbol exception"


@dataclass(frozen=True)
class Config:
    globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    pattern: re.Pattern[str]
    config_keys: tuple[re.Pattern[str], ...]
    exceptions: frozenset[str]
    symbol_exceptions: frozenset[tuple[str, str]]


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
            "config: exceptions must be markdown; use symbol_exceptions for .py"
        )
    entries = raw.get("symbol_exceptions", [])
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError("config: symbol_exceptions must be a list of strings")
    symbols: list[tuple[str, str]] = []
    for entry in entries:
        src, _, symbol = entry.rpartition(":")
        if not src.endswith(".py"):
            raise ValueError(f"config: symbol_exceptions must be <.py path>:<symbol>: {entry}")
        if not symbol or not all(part.isidentifier() for part in symbol.split(".")):
            raise ValueError(f"config: symbol_exceptions has a bad symbol: {entry}")
        symbols.append((src, symbol))

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
        symbol_exceptions=frozenset(symbols),
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


def _bound_names(node: ast.stmt) -> list[str]:
    """Names a top-level or class-body statement binds: a def, a class, or assignment targets."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _node_span(node: ast.stmt, lines: list[str]) -> tuple[int, int]:
    """First and last 1-based line of ``node``, with its decorators and the comment run above."""
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    while first > 1 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    return first, node.end_lineno or node.lineno


def _symbol_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    """Dotted symbol -> spans of its top-level or class-nested definitions. Raises ``SyntaxError``."""
    lines = text.split("\n")
    spans: dict[str, list[tuple[int, int]]] = {}
    for node in ast.parse(text).body:
        for name in _bound_names(node):
            spans.setdefault(name, []).append(_node_span(node, lines))
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                for name in _bound_names(child):
                    spans.setdefault(f"{node.name}.{name}", []).append(_node_span(child, lines))
    return spans


def _exempt_spans(path: str, text: str, cfg: Config) -> dict[str, list[tuple[int, int]]]:
    """Spans of each ``symbol_exceptions`` symbol in ``path``. An undefined symbol maps to ``[]``."""
    names = sorted(sym for rel, sym in cfg.symbol_exceptions if rel == path)
    if not names:
        return {}
    try:
        defined = _symbol_spans(text)
    except SyntaxError as exc:
        raise ValueError(f"parse {path}: {exc}") from exc
    return {name: defined.get(name, []) for name in names}


def _covering_symbols(exempt: dict[str, list[tuple[int, int]]], number: int) -> list[str]:
    """Exempt symbols whose span contains 1-based line ``number``."""
    return [
        sym
        for sym, spans in exempt.items()
        if any(first <= number <= last for first, last in spans)
    ]


def scan_text(path: str, text: str, cfg: Config) -> list[Finding]:
    """Findings for one file's text. ``path`` is the repo-relative name used for exceptions."""
    exempt = _exempt_spans(path, text, cfg)
    return [
        Finding(path, number)
        for number in raw_hit_lines(text, cfg)
        if f"{path}:{number}" not in cfg.exceptions and not _covering_symbols(exempt, number)
    ]


def scan(root: Path, cfg: Config, paths: list[str]) -> list[Finding]:
    """Findings for ``paths``; stale exceptions and stale symbol exceptions are reported too."""
    findings: list[Finding] = []
    hit_locations: set[str] = set()
    hit_symbols: set[tuple[str, str]] = set()
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        exempt = _exempt_spans(rel, text, cfg)
        for number in raw_hit_lines(text, cfg):
            location = f"{rel}:{number}"
            hit_locations.add(location)
            covering = _covering_symbols(exempt, number)
            hit_symbols.update((rel, sym) for sym in covering)
            if not covering and location not in cfg.exceptions:
                findings.append(Finding(rel, number))
    for location in sorted(cfg.exceptions - hit_locations):
        rel, _, number = location.rpartition(":")
        findings.append(Finding(rel, int(number), STALE_TEXT))
    for rel, sym in sorted(cfg.symbol_exceptions - hit_symbols):
        findings.append(Finding(rel, 0, STALE_SYMBOL_TEXT))
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
