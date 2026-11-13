"""Regression tests for release version validation and tag minting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/releases/release_tag.py"
GH_STUB = '''#!{python}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with Path(os.environ["GH_STUB_LOG"]).open("a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\\n")
mode = os.environ.get("GH_STUB_MODE", "success")
if "--method" in args and args[args.index("--method") + 1] == "POST":
    if mode in {"422-mismatch", "422-same", "422-annotated"}:
        print(json.dumps({"message": "Reference already exists", "status": "422"}))
        print("gh: Reference already exists (HTTP 422)", file=sys.stderr)
        sys.exit(1)
    if mode == "500-failure":
        print("gh: Internal Server Error (HTTP 500)", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
if any("/git/tags/" in arg for arg in args):
    tag_endpoint = next(arg for arg in args if "/git/tags/" in arg)
    if tag_endpoint.endswith("/git/tags/tag-object-sha"):
        if mode == "annotated-nested":
            print(json.dumps({"object": {"sha": "nested-tag-object-sha", "type": "tag"}}))
        elif mode in {"annotated", "422-annotated"}:
            print(json.dumps({"object": {"sha": os.environ["GH_STUB_REMOTE_SHA"], "type": "commit"}}))
        else:
            sys.exit(1)
        sys.exit(0)
    if tag_endpoint.endswith("/git/tags/nested-tag-object-sha") and mode == "annotated-nested":
        print(json.dumps({"object": {"sha": os.environ["GH_STUB_REMOTE_SHA"], "type": "commit"}}))
        sys.exit(0)
    sys.exit(1)
if any("/git/ref/tags/" in arg for arg in args):
    count_file = Path(os.environ["GH_STUB_QUERY_COUNT"])
    count = int(count_file.read_text(encoding="utf-8")) if count_file.exists() else 0
    count += 1
    count_file.write_text(str(count), encoding="utf-8")
    if mode in {"422-mismatch", "422-same", "422-annotated"} and count == 1:
        sys.exit(1)
    sha = os.environ.get("GH_STUB_REMOTE_SHA")
    if sha:
        if mode in {"annotated", "422-annotated", "annotated-nested"}:
            print(json.dumps({"object": {"sha": "tag-object-sha", "type": "tag"}}))
        else:
            print(json.dumps({"object": {"sha": sha, "type": "commit"}}))
        sys.exit(0)
    sys.exit(1)
sys.exit(2)
'''


@dataclass(frozen=True)
class GitCase:
    repo: Path
    env: dict[str, str]


@pytest.fixture
def git_case(tmp_path: Path) -> GitCase:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Release Tag Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release-tag-test@example.invalid"], cwd=repo, check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH_STUB.replace("#!{python}", f"#!{sys.executable}", 1), encoding="utf-8")
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GH_STUB_LOG": str(tmp_path / "gh-calls.jsonl"),
        "GH_STUB_QUERY_COUNT": str(tmp_path / "gh-query-count"),
    }
    return GitCase(repo, env)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str, *, version: str | None = None, note: str | None = None) -> str:
    if version is not None:
        version_file = repo / ".github/rigel-version"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(f"{version}\n", encoding="utf-8")
    if note is not None:
        (repo / "README.md").write_text(note, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", message], cwd=repo, check=True)
    return _git(repo, "rev-parse", "HEAD")


def _run(case: GitCase, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=case.repo,
        env={**case.env, **env},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _post_sha(case: GitCase) -> str:
    calls = [
        json.loads(line)
        for line in Path(case.env["GH_STUB_LOG"]).read_text(encoding="utf-8-sig").splitlines()
    ]
    post = next(call for call in calls if "--method" in call)
    return next(arg.removeprefix("sha=") for arg in post if arg.startswith("sha="))


def test_check_pr_accepts_first_release_version_one(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "base")
    head = _commit(git_case.repo, "set first release version", version="2026.9.24-rigel.1")

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == 0, result.stderr
    assert "2026.9.24-rigel.1 is the next release version" in result.stdout


def test_check_pr_accepts_next_release_number(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", base)
    head = _commit(git_case.repo, "bump release version", version="2026.9.24-rigel.2")

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == 0, result.stderr
    assert "2026.9.24-rigel.2 is the next release version" in result.stdout


def test_check_pr_accepts_version_already_at_base_tip(git_case: GitCase) -> None:
    ancestor = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "switch", "-c", "pull-request")
    head = _commit(git_case.repo, "bump version", version="2026.9.24-rigel.2")
    _git(git_case.repo, "switch", "--detach", ancestor)
    base_tip = _commit(git_case.repo, "base independently reaches version", version="2026.9.24-rigel.2")

    result = _run(git_case, "check-pr", base_tip, head)

    assert result.returncode == 0, result.stderr
    assert ".github/rigel-version version is unchanged" in result.stdout


@pytest.mark.parametrize(("version", "returncode"), [("2026.9.24-rigel.11", 0), ("2026.9.24-rigel.10", 1)])
def test_check_pr_uses_numeric_maximum_release_tag(
    git_case: GitCase, version: str, returncode: int
) -> None:
    base = _commit(git_case.repo, "version nine", version="2026.9.24-rigel.9")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.9", base)
    _git(git_case.repo, "tag", "v2026.9.24-rigel.10", base)
    head = _commit(git_case.repo, "set next version", version=version)

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == returncode, result.stderr
    if returncode == 0:
        assert f"{version} is the next release version" in result.stdout
    else:
        assert "expected 2026.9.24-rigel.11" in result.stderr

def test_check_pr_rejects_skipped_release_number(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", base)
    head = _commit(git_case.repo, "skip release number", version="2026.9.24-rigel.3")

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == 1
    assert "expected 2026.9.24-rigel.2, got 2026.9.24-rigel.3" in result.stderr


def test_check_pr_rejects_malformed_version(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    head = _commit(git_case.repo, "malformed release version", version="not-a-version")

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == 1
    assert "does not match the required X.Y.Z-rigel.N shape" in result.stderr


def test_check_pr_skips_unchanged_version_file(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    head = _commit(git_case.repo, "change unrelated file", note="release tooling\n")

    result = _run(git_case, "check-pr", base, head)

    assert result.returncode == 0, result.stderr
    assert ".github/rigel-version is unchanged" in result.stdout


def test_check_pr_uses_merge_base_for_stale_pr_after_minted_bump(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", base)
    _git(git_case.repo, "switch", "-c", "pull-request")
    pr_head = _commit(git_case.repo, "change unrelated PR file", note="stale pull request\n")
    _git(git_case.repo, "switch", "--detach", base)
    bump = _commit(git_case.repo, "bump release version", version="2026.9.24-rigel.2")
    minted = _run(git_case, "mint", bump, "RigelBuild/hermes-agent")
    assert minted.returncode == 0, minted.stderr
    _git(git_case.repo, "tag", "v2026.9.24-rigel.2", bump)

    result = _run(git_case, "check-pr", bump, pr_head)

    assert result.returncode == 0, result.stderr
    assert ".github/rigel-version is unchanged" in result.stdout


def test_mint_walks_back_to_the_version_bump_commit(git_case: GitCase) -> None:
    first = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", first)
    bump = _commit(git_case.repo, "set second release version", version="2026.9.24-rigel.2")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")

    result = _run(git_case, "mint", release_commit, "RigelBuild/hermes-agent")

    assert result.returncode == 0, result.stderr
    assert _post_sha(git_case) == bump
    assert f"Created v2026.9.24-rigel.2 at {bump}" in result.stdout


@pytest.mark.parametrize("existing", ["local", "remote"])
def test_mint_accepts_existing_tag_at_same_commit(git_case: GitCase, existing: str) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")
    if existing == "local":
        _git(git_case.repo, "tag", "v2026.9.24-rigel.1", bump)
        result = _run(git_case, "mint", release_commit, "RigelBuild/hermes-agent")
    else:
        result = _run(
            git_case,
            "mint",
            release_commit,
            "RigelBuild/hermes-agent",
            GH_STUB_REMOTE_SHA=bump,
        )

    assert result.returncode == 0, result.stderr
    assert f"v2026.9.24-rigel.1 already points at {bump}" in result.stdout


def test_mint_rejects_existing_tag_at_different_commit(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", release_commit)

    result = _run(git_case, "mint", release_commit, "RigelBuild/hermes-agent")

    assert result.returncode == 1
    assert f"already exists at {release_commit}, expected {bump}" in result.stderr


def test_mint_rejects_422_tag_race_at_different_commit(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")
    result = _run(
        git_case,
        "mint",
        release_commit,
        "RigelBuild/hermes-agent",
        GH_STUB_MODE="422-mismatch",
        GH_STUB_REMOTE_SHA=release_commit,
    )

    assert result.returncode == 1
    assert f"already exists at {release_commit}, expected {bump}" in result.stderr


def test_mint_accepts_nested_annotated_remote_tag(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")

    result = _run(
        git_case,
        "mint",
        release_commit,
        "RigelBuild/hermes-agent",
        GH_STUB_MODE="annotated-nested",
        GH_STUB_REMOTE_SHA=bump,
    )

    assert result.returncode == 0, result.stderr
    assert f"v2026.9.24-rigel.1 already points at {bump}" in result.stdout


def test_mint_walks_first_parent_when_merge_version_is_on_second_parent(git_case: GitCase) -> None:
    base = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    _git(git_case.repo, "tag", "v2026.9.24-rigel.1", base)
    _git(git_case.repo, "switch", "-c", "version-branch")
    second_parent = _commit(git_case.repo, "set second release version", version="2026.9.24-rigel.2")
    _git(git_case.repo, "switch", "--detach", base)
    _commit(git_case.repo, "first-parent-only change", note="first parent\n")
    _git(git_case.repo, "merge", "--no-ff", "version-branch", "-m", "merge version branch")
    merge_commit = _git(git_case.repo, "rev-parse", "HEAD")

    result = _run(git_case, "mint", merge_commit, "RigelBuild/hermes-agent")

    assert result.returncode == 0, result.stderr
    assert _post_sha(git_case) == merge_commit
    assert _post_sha(git_case) != second_parent


def test_mint_accepts_annotated_remote_tag_at_target(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")

    result = _run(
        git_case,
        "mint",
        release_commit,
        "RigelBuild/hermes-agent",
        GH_STUB_MODE="annotated",
        GH_STUB_REMOTE_SHA=bump,
    )

    assert result.returncode == 0, result.stderr
    assert f"v2026.9.24-rigel.1 already points at {bump}" in result.stdout


def test_mint_accepts_annotated_tag_won_by_422_race(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")

    result = _run(
        git_case,
        "mint",
        release_commit,
        "RigelBuild/hermes-agent",
        GH_STUB_MODE="422-annotated",
        GH_STUB_REMOTE_SHA=bump,
    )

    assert result.returncode == 0, result.stderr
    assert f"v2026.9.24-rigel.1 already points at {bump}" in result.stdout


def test_mint_rejects_remote_tag_at_different_commit(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")
    release_commit = _commit(git_case.repo, "follow-up release commit", note="follow-up\n")

    result = _run(
        git_case,
        "mint",
        release_commit,
        "RigelBuild/hermes-agent",
        GH_STUB_REMOTE_SHA=release_commit,
    )

    assert result.returncode == 1
    assert f"already exists at {release_commit}, expected {bump}" in result.stderr


def test_mint_reports_non_422_post_failure(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")

    result = _run(git_case, "mint", bump, "RigelBuild/hermes-agent", GH_STUB_MODE="500-failure")

    assert result.returncode == 1
    assert "gh: Internal Server Error (HTTP 500)" in result.stderr


def test_mint_accepts_422_race_when_remote_tag_already_points_at_target(git_case: GitCase) -> None:
    bump = _commit(git_case.repo, "first release version", version="2026.9.24-rigel.1")

    result = _run(
        git_case,
        "mint",
        bump,
        "RigelBuild/hermes-agent",
        GH_STUB_MODE="422-same",
        GH_STUB_REMOTE_SHA=bump,
    )

    assert result.returncode == 0, result.stderr
    assert "already points at" in result.stdout
    assert f"v2026.9.24-rigel.1 already points at {bump}" in result.stdout
