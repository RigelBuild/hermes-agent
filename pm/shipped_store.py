"""Build the read-only PM store shipped by Nix."""

from __future__ import annotations

import json
import shutil
from functools import cache
from pathlib import Path

from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import tree_digest


@cache
def shipped_roots(stamp: Path) -> tuple[Path, ...]:
    """Read the declared shipped store from an install stamp."""
    try:
        data = json.loads(stamp.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ()
    if not isinstance(data, dict) or data.get("distribution") != "nix":
        return ()
    declared = data.get("shippedStore")
    if not isinstance(declared, str) or not Path(declared).is_absolute():
        return ()
    root = Path(declared)
    return (root,) if (root / "facts.json").is_file() else ()


def required_packages(names: list[str], target: str) -> list[str]:
    """Select lock packages eligible for drift checks on the target."""
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
    """Link each required tree into the store and record its lock identity."""
    lock = Lockfile(lockfile_path)
    required = required_packages(lock.names(), target)
    if set(trees) != set(required):
        missing = sorted(set(required) - set(trees))
        extra = sorted(set(trees) - set(required))
        raise ValueError(f"shipped store package set mismatch: missing={missing}, extra={extra}")
    for name in required:
        package = get_package(name)
        binary = package.binary(trees[name], target)
        if binary is not None and not binary.is_file():
            raise ValueError(f"{name} tree has no executable at {binary}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    facts = Facts(out / "facts.json")
    for name in required:
        version = lock.version(name)
        entry = f"{name}-{version}-{target}"
        tree = trees[name].resolve()
        (out / entry).symlink_to(tree, target_is_directory=True)
        facts.record(
            name, version, entry, get_package(name).env(out / entry, target), out,
            target=target,
            artifacts=[artifact["sha256"] for artifact in lock.artifacts(name, target)],
            digest=tree_digest(tree),
        )
    return out
