"""Validate release version changes and mint deterministic release tags."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence

VERSION_FILE = ".github/rigel-version"
VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rigel\.([1-9][0-9]*)")
TAG_PATTERN = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rigel\.([1-9][0-9]*)")


class ReleaseTagError(Exception):
    """An expected release-tag operation failed."""


def _run(argv: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", check=False)
    if check and result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or f"{argv[0]} exited {result.returncode}"
        raise ReleaseTagError(message)
    return result


def version_at(commit: str, *, required: bool = True) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{VERSION_FILE}"], capture_output=True, check=False
    )
    if result.returncode:
        if required:
            raise ReleaseTagError(f"::error::expected {VERSION_FILE} at {commit} to hold one non-empty line")
        return None

    try:
        contents = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        contents = ""
    lines = contents.split("\n")
    if contents.endswith("\n"):
        lines.pop()
    if len(lines) != 1 or not lines[0]:
        if required:
            raise ReleaseTagError(f"::error::expected {VERSION_FILE} at {commit} to hold one non-empty line")
        return None
    return lines[0]


def validate_next_version(version: str) -> None:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ReleaseTagError(f"::error::version '{version}' does not match the required X.Y.Z-rigel.N shape")

    base = ".".join(match.groups()[:3])
    tags = _run(["git", "tag", "--list", f"v{base}-rigel.*"], check=True).stdout.splitlines()
    sequences = [
        int(tag_match.group(4))
        for tag in tags
        if (tag_match := TAG_PATTERN.fullmatch(tag)) is not None
        and ".".join(tag_match.groups()[:3]) == base
    ]
    expected_sequence = max(sequences, default=0) + 1
    expected = f"{base}-rigel.{expected_sequence}"
    if version != expected:
        raise ReleaseTagError(f"::error::expected {expected}, got {version}")


def check_pr(base_sha: str, head_sha: str) -> None:
    merge_base = _run(["git", "merge-base", base_sha, head_sha], check=True).stdout.strip()
    diff = _run(["git", "diff", "--quiet", merge_base, head_sha, "--", VERSION_FILE])
    if diff.returncode == 0:
        print(f"{VERSION_FILE} is unchanged; no release version check is needed")
        return
    if diff.returncode != 1:
        message = diff.stderr.strip() or f"git diff exited {diff.returncode}"
        raise ReleaseTagError(message)

    base_version = version_at(merge_base, required=False)
    head_version = version_at(head_sha)
    base_tip_version = version_at(base_sha, required=False)
    if base_version == head_version or head_version == base_tip_version:
        print(f"{VERSION_FILE} version is unchanged; no release version check is needed")
        return
    validate_next_version(head_version)
    print(f"{head_version} is the next release version")


def _tag_commit(tag: str) -> str | None:
    ref = f"refs/tags/{tag}"
    exists = _run(["git", "show-ref", "--verify", "--quiet", ref])
    if exists.returncode:
        return None
    commit = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"])
    if commit.returncode:
        return "<not a commit>"
    return commit.stdout.strip()


def _peel_remote_tag(repo: str, tag_object: dict[str, str]) -> str:
    if tag_object["type"] != "tag":
        return tag_object["sha"]
    result = _run(["gh", "api", f"repos/{repo}/git/tags/{tag_object['sha']}"])
    if result.returncode:
        raise ReleaseTagError(result.stderr.strip() or result.stdout.strip() or f"gh exited {result.returncode}")
    return json.loads(result.stdout)["object"]["sha"]


def _remote_tag_commit(repo: str, tag: str) -> str | None:
    result = _run(["gh", "api", f"repos/{repo}/git/ref/tags/{tag}"])
    if result.returncode:
        return None
    tag_ref = json.loads(result.stdout)
    return _peel_remote_tag(repo, tag_ref["object"])


def _check_existing_tag(repo: str, tag: str, target: str) -> bool:
    found = False
    for actual in (_tag_commit(tag), _remote_tag_commit(repo, tag)):
        if actual is None:
            continue
        found = True
        if actual != target:
            raise ReleaseTagError(f"::error::{tag} already exists at {actual}, expected {target}")
    if found:
        print(f"{tag} already points at {target}")
    return found


def mint(sha: str, repo: str) -> None:
    version = version_at(sha)
    assert version is not None
    tag = f"v{version}"

    target = sha
    while True:
        parent = _run(["git", "rev-parse", "--verify", f"{target}^1"])
        if parent.returncode:
            break
        parent_sha = parent.stdout.strip()
        if version_at(parent_sha, required=False) != version:
            break
        target = parent_sha

    if _check_existing_tag(repo, tag, target):
        return
    validate_next_version(version)

    response = _run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/git/refs",
            "-f",
            f"ref=refs/tags/{tag}",
            "-f",
            f"sha={target}",
        ]
    )
    if response.returncode == 0:
        print(f"Created {tag} at {target}")
        return

    output = response.stdout + response.stderr
    if "HTTP 422" in output and "Reference already exists" in output:
        actual_result = _run(["gh", "api", f"repos/{repo}/git/ref/tags/{tag}"])
        if actual_result.returncode:
            raise ReleaseTagError(actual_result.stderr.strip() or actual_result.stdout.strip())
        tag_ref = json.loads(actual_result.stdout)
        actual = _peel_remote_tag(repo, tag_ref["object"])
        if actual == target:
            print(f"{tag} already points at {target}")
            return
        raise ReleaseTagError(f"::error::{tag} already exists at {actual}, expected {target}")

    raise ReleaseTagError(output.rstrip() or f"gh exited {response.returncode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check-pr", help="validate a changed release version")
    check_parser.add_argument("base_sha")
    check_parser.add_argument("head_sha")
    mint_parser = commands.add_parser("mint", help="mint the deterministic release tag")
    mint_parser.add_argument("sha")
    mint_parser.add_argument("repo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check-pr":
            check_pr(args.base_sha, args.head_sha)
        else:
            mint(args.sha, args.repo)
        return 0
    except (OSError, ReleaseTagError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
