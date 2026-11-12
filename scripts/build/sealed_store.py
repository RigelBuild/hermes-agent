"""Seal a Nix build's pm packages into the store its install stamp names.

Invoked from nix/hermes-agent.nix as ``python3 -m scripts.build.sealed_store``.
Each ``name=tree`` argument maps a lock package to its unpacked derivation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pm.sealed_store import seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("trees", nargs="+", metavar="NAME=TREE")
    args = parser.parse_args()

    trees: dict[str, Path] = {}
    for pair in args.trees:
        name, sep, tree = pair.partition("=")
        if not sep or not name or not tree:
            parser.error(f"expected NAME=TREE, got {pair!r}")
        trees[name] = Path(tree)
    seal(args.lockfile, args.target, trees, args.out)


if __name__ == "__main__":
    main()
