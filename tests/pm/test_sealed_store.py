"""A Nix artifact seals the lock packages drift() requires into a read-only store.

Regression for RIG-9869: on a host with a stale ~/.hermes/tools/facts.json, a
Nix-installed Hermes reported "install out of sync (node, pnpm)".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm import paths
from pm.sealed_store import required_packages, seal
from pm.lock import Facts, Lockfile
from pm.registry import get_package
from pm.store import current_target


def _binary_tree(tree: Path, package: str, target: str) -> Path:
    """An unpacked package tree with its executable where pm expects it."""
    binary = get_package(package).binary(tree, target)
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    return tree


def _required(target: str) -> list[str]:
    return required_packages(Lockfile(paths.lockfile_path()).names(), target)


def _seal_required(tmp_path: Path, target: str) -> Path:
    trees = tmp_path / "trees"
    out = tmp_path / "sealed"
    seal(paths.lockfile_path(), target,
         {name: _binary_tree(trees / name, name, target) for name in _required(target)}, out)
    return out


def _stale_user_facts(target: str) -> None:
    """What an old `pm install` left behind: node 26.7.0 and no pnpm."""
    store = paths.store_root()
    entry = f"node-26.7.0-{target}"
    _binary_tree(store / entry, "node", target)
    Facts(store / "facts.json").record("node", "26.7.0", entry, {}, store)


@pytest.fixture
def install_root(tmp_path, monkeypatch):
    """A checkout whose install stamp lives at its own root, under a private HERMES_HOME."""
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: package)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(package))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    return package


def test_stale_user_facts_report_no_drift_under_a_nix_runtime(install_root, tmp_path):
    from pm.install import drift

    target = current_target()
    _stale_user_facts(target)
    sealed = _seal_required(tmp_path, target)
    stamp = {"distribution": "nix", "sealedStore": str(sealed)}
    (install_root / "install-stamp.json").write_text(json.dumps(stamp), encoding="utf-8")

    assert drift() == {}


def test_stale_store_without_a_sealed_runtime_still_reports_drift(install_root):
    from pm.install import drift

    _stale_user_facts(current_target())

    problems = drift()
    assert "node" in problems and "pnpm" in problems


def test_sealed_entries_record_the_lock_identity(tmp_path):
    target = current_target()
    sealed = _seal_required(tmp_path, target)
    lock = Lockfile(paths.lockfile_path())
    facts = Facts(sealed / "facts.json")

    for name in _required(target):
        fact = facts.get(name)
        assert fact["version"] == lock.version(name)
        assert fact["target"] == target
        assert fact["artifacts"] == [artifact["sha256"] for artifact in lock.artifacts(name, target)]
        assert get_package(name).binary(sealed / fact["entry"], target).is_file()


def test_seal_refuses_a_store_missing_a_required_package(tmp_path):
    target = current_target()
    names = _required(target)
    trees = {name: _binary_tree(tmp_path / "trees" / name, name, target) for name in names[1:]}

    with pytest.raises(ValueError, match=names[0]):
        seal(paths.lockfile_path(), target, trees, tmp_path / "sealed")


def test_seal_refuses_a_tree_without_its_executable(tmp_path):
    target = current_target()
    names = _required(target)
    trees = {name: _binary_tree(tmp_path / "trees" / name, name, target) for name in names}
    trees[names[0]] = tmp_path / "empty"
    trees[names[0]].mkdir()

    with pytest.raises(ValueError, match=names[0]):
        seal(paths.lockfile_path(), target, trees, tmp_path / "sealed")
