"""Seal the lock packages drift() requires into one read-only store.

A Nix artifact cannot run pm's installer, so it ships the packages itself.
``seal`` writes them into a directory with its own ``facts.json``. Each fact
records the lock identity (version, target, artifact sha256s) and the tree
digest, the same shape ``pm install`` writes. A Nix install stamp points
``sealedStore`` at that directory. ``drift()`` searches it after the user's
own facts, so a stale ``~/.hermes/tools/facts.json`` cannot hide it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pm import paths
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import Store, tree_digest


def sealed_roots() -> tuple[Path, ...]:
    """The sealed store a Nix install stamp declares, if it exists as a directory.

    Read-only and never created here: a missing store contributes nothing,
    so drift() reports the packages it cannot find instead of guessing.
    """
    stamp = paths.install_stamp_path(paths.repo_root())
    try:
        declared = json.loads(stamp.read_text(encoding="utf-8-sig")).get("sealedStore")
    except (OSError, ValueError, AttributeError):
        return ()
    if not isinstance(declared, str) or not Path(declared).is_absolute():
        return ()
    root = Path(declared)
    return (root,) if (root / "facts.json").is_file() else ()



def required_packages(names: list[str], target: str) -> list[str]:
    """The lock packages drift() requires on ``target``, in lock order."""
    required = []
    for name in names:
        try:
            package = get_package(name)
        except KeyError:
            continue
        if package.optional or package.internal or package.missing_reason(target) is not None:
            continue
        required.append(name)
    return required


def seal(lockfile_path: Path, target: str, trees: dict[str, Path], out: Path) -> Path:
    """Copy each required package's unpacked tree into ``out`` and record its lock identity.

    Refuses a missing package, and a tree whose executable is absent, so a
    broken Nix build fails at build time instead of at the first drift check.
    """
    lock = Lockfile(lockfile_path)
    required = required_packages(lock.names(), target)
    for name in required:
        if name not in trees:
            raise ValueError(f"sealed store is missing required package {name} for {target}")
        package = get_package(name)
        binary = package.binary(trees[name], target)
        if binary is not None and not binary.is_file():
            raise ValueError(f"{name} tree has no executable at {binary}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    store = Store(out)
    facts = Facts(out / "facts.json")
    for name in required:
        version = lock.version(name)
        entry = f"{name}-{version}-{target}"
        shutil.copytree(trees[name], store.entry(entry), symlinks=True)
        facts.record(
            name, version, entry, get_package(name).env(store.entry(entry), target), store.root,
            target=target,
            artifacts=[artifact["sha256"] for artifact in lock.artifacts(name, target)],
            digest=tree_digest(store.entry(entry)),
        )
    return out
