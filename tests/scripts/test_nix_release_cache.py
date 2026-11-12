"""Release assets form a flat cache, with upstream NARs kept externally referenced."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts.releases.nix_release_cache import assemble, fetch_upstream_narinfo, write_pin


SHARED_HASH = "1b9p3x1z5r7c2m8k4w6v0q9s3d5f7h1j"
LOCAL_HASH = "6v4d2c8b0ssfmzqk3p9lhrx5j7wn1dyg"


def _narinfo(store_hash: str, nar_name: str, *, nar_hash: str, nar_size: int = 100) -> str:
    return (
        f"StorePath: /nix/store/{store_hash}-hermes-agent\n"
        f"URL: nar/{nar_name}\n"
        "Compression: xz\n"
        f"FileHash: sha256:{nar_name}\n"
        "FileSize: 42\n"
        f"NarHash: {nar_hash}\n"
        f"NarSize: {nar_size}\n"
        "References: \n"
        f"Deriver: {store_hash}-hermes-agent.drv\n"
        "Sig: cache-release-1:OURS\n"
    )


def _platform(
    root: Path,
    name: str,
    paths: list[tuple[str, str, str]],
    *,
    public_key: str = "cache-release-1:PUBLIC\n",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "platform").write_text(name + "\n", encoding="utf-8")
    (directory / "nix-release-cache.pub").write_text(public_key, encoding="utf-8")
    (directory / f"{name}.path-info.json").write_text(
        f'{{"/nix/store/{paths[0][0]}-hermes-agent": {{}}}}\n', encoding="utf-8"
    )
    cache = directory / "cache"
    nar_dir = cache / "nar"
    nar_dir.mkdir(parents=True)
    (cache / "nix-cache-info").write_text(
        "StoreDir: /nix/store\nWantMassQuery: 1\nPriority: 30\n", encoding="utf-8"
    )
    for store_hash, nar_name, nar_hash in paths:
        (cache / f"{store_hash}.narinfo").write_text(
            _narinfo(store_hash, nar_name, nar_hash=nar_hash), encoding="utf-8"
        )
        (nar_dir / nar_name).write_bytes(f"nar bytes for {store_hash}".encode())
    return directory


class _Response(BytesIO):
    status = 200


def _upstream_record(store_hash: str = SHARED_HASH, *, nar_hash: str = "sha256:shared") -> bytes:
    return (
        f"StorePath: /nix/store/{store_hash}-hermes-agent\n"
        "URL: nar/upstream-real.nar.zst\n"
        "Compression: zstd\n"
        "FileHash: sha256:upstream-file\n"
        "FileSize: 777\n"
        f"NarHash: {nar_hash}\n"
        "NarSize: 100\n"
        "References: \n"
        "Sig: cache-nixos-1:UPSTREAM\n"
    ).encode()


def test_assemble_keeps_upstream_path_narinfo_and_uses_absolute_nar_url(tmp_path: Path) -> None:
    upstream_hash = SHARED_HASH
    platform = _platform(
        tmp_path / "platforms",
        "hermes-agent-v2026.11.1-rigel.1-linux-x64",
        [
            (upstream_hash, "local-upstream.nar.xz", "sha256:shared"),
            (LOCAL_HASH, "local-only.nar.xz", "sha256:local"),
        ],
    )
    requests: list[str] = []

    def open_url(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith(f"/{upstream_hash}.narinfo"):
            return _Response(_upstream_record())
        raise HTTPError(request.full_url, 404, "not found", {}, None)

    output = tmp_path / "release-assets"
    count = assemble(
        [platform], output, upstream_url="https://cache.nixos.org", opener=open_url, sleeper=lambda _: None
    )

    upstream_narinfo = (output / f"{upstream_hash}.narinfo").read_text(encoding="utf-8")
    assert "URL: https://cache.nixos.org/nar/upstream-real.nar.zst\n" in upstream_narinfo
    assert "Compression: zstd\n" in upstream_narinfo
    assert "FileHash: sha256:upstream-file\n" in upstream_narinfo
    assert "FileSize: 777\n" in upstream_narinfo
    assert "NarHash: sha256:shared\n" in upstream_narinfo
    assert "NarSize: 100\n" in upstream_narinfo
    assert "References: \n" in upstream_narinfo
    assert "Sig: cache-release-1:OURS\n" in upstream_narinfo
    assert "UPSTREAM" not in upstream_narinfo
    assert not (output / "local-upstream.nar.xz").exists()
    assert (output / "local-only.nar.xz").is_file()
    assert (output / f"{LOCAL_HASH}.narinfo").is_file()
    assert count == 6
    assert any(url.endswith(f"/{upstream_hash}.narinfo") for url in requests)


def test_upstream_get_retries_transient_response_with_bounded_attempts() -> None:
    attempts = 0
    waits: list[float] = []

    def open_url(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HTTPError(request.full_url, 503, "busy", {}, None)
        return _Response(b"StorePath: /nix/store/x\n")

    result = fetch_upstream_narinfo(
        SHARED_HASH,
        "https://cache.nixos.org",
        opener=open_url,
        sleeper=waits.append,
    )
    assert result == b"StorePath: /nix/store/x\n"
    assert attempts == 3
    assert len(waits) == 2


def test_upstream_get_fails_after_bounded_retries() -> None:
    attempts = 0

    def open_url(request, timeout):
        nonlocal attempts
        attempts += 1
        raise HTTPError(request.full_url, 503, "busy", {}, None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fetch_upstream_narinfo(
            SHARED_HASH,
            "https://cache.nixos.org",
            opener=open_url,
            sleeper=lambda _: None,
        )
    assert attempts == 3


def test_upstream_narinfo_identity_must_match_local_path() -> None:
    from scripts.releases.nix_release_cache import use_upstream_narinfo

    local = _narinfo(SHARED_HASH, "local.nar.xz", nar_hash="sha256:local").encode()
    upstream = _upstream_record(nar_hash="sha256:other")
    with pytest.raises(ValueError, match="different StorePath, NarHash"):
        use_upstream_narinfo(local, upstream, "https://cache.nixos.org", source="fixture")


def test_reconcile_assets_only_removes_stale_cache_assets_and_counts_existing(tmp_path: Path) -> None:
    from scripts.releases.nix_release_cache import reconcile_assets, validate_asset_count

    output = tmp_path / "release-assets"
    output.mkdir()
    (output / "nix-cache-info").write_text("StoreDir: /nix/store\n", encoding="utf-8")
    (output / "nix-release-cache.pub").write_text("key", encoding="utf-8")

    existing = [
        f"{LOCAL_HASH}.narinfo",
        "obsolete.nar.xz",
        "release-notes.txt",
        "nix-release-cache.pub",
    ]
    assert reconcile_assets(output, existing) == [f"{LOCAL_HASH}.narinfo", "obsolete.nar.xz"]
    with pytest.raises(ValueError, match=r"peak is 3 assets.*final count is 3"):
        validate_asset_count(output, ["release-notes.txt"], max_assets=2)


def test_asset_count_rejects_peak_before_stale_cleanup(tmp_path: Path) -> None:
    from scripts.releases.nix_release_cache import validate_asset_count

    output = tmp_path / "release-assets"
    output.mkdir()
    (output / "new.narinfo").write_text("new", encoding="utf-8")
    existing = ["stale.nar.xz", "retained-a", "retained-b"]

    with pytest.raises(ValueError, match=r"peak is 4 assets.*final count is 3"):
        validate_asset_count(output, existing, max_assets=3)

    assert validate_asset_count(output, existing, max_assets=4) == 3
def test_reconcile_cli_prints_stale_asset_names_one_per_line(tmp_path: Path, capsys) -> None:
    from scripts.releases.nix_release_cache import main

    output = tmp_path / "release-assets"
    output.mkdir()
    (output / "nix-cache-info").write_text("StoreDir: /nix/store\n", encoding="utf-8")
    existing = tmp_path / "existing-assets.json"
    existing.write_text(json.dumps(["obsolete.nar.xz", "release-notes.txt"]), encoding="utf-8")

    assert main(["reconcile", "-out", str(output), "-existing-assets", str(existing)]) == 0
    assert capsys.readouterr().out == "obsolete.nar.xz\n"


def test_assemble_dedupes_shared_paths_but_rejects_identity_conflicts(tmp_path: Path) -> None:
    first = _platform(
        tmp_path / "same",
        "hermes-agent-v2026.11.1-rigel.1-linux-x64",
        [(SHARED_HASH, "shared.nar.xz", "sha256:shared")],
    )
    second = _platform(
        tmp_path / "same",
        "hermes-agent-v2026.11.1-rigel.1-linux-arm64",
        [(SHARED_HASH, "shared.nar.xz", "sha256:shared")],
    )
    output = tmp_path / "deduped"
    assert assemble([first, second], output) == 6
    assert (output / f"{SHARED_HASH}.narinfo").is_file()
    assert (output / "shared.nar.xz").is_file()

    conflict_root = tmp_path / "conflict"
    conflict_a = _platform(
        conflict_root,
        "hermes-agent-v2026.11.1-rigel.1-linux-x64",
        [(SHARED_HASH, "shared.nar.xz", "sha256:shared")],
    )
    conflict_b = _platform(
        conflict_root,
        "hermes-agent-v2026.11.1-rigel.1-linux-arm64",
        [(SHARED_HASH, "shared.nar.xz", "sha256:changed")],
    )
    with pytest.raises(ValueError, match="different NarHash"):
        assemble([conflict_a, conflict_b], tmp_path / "conflict-out")


def test_assemble_rejects_missing_nar_and_asset_limit(tmp_path: Path) -> None:
    missing = _platform(
        tmp_path / "missing",
        "hermes-agent-v2026.11.1-rigel.1-linux-x64",
        [
            (LOCAL_HASH, "local.nar.xz", "sha256:local"),
            (SHARED_HASH, "shared.nar.xz", "sha256:shared"),
        ],
    )
    (missing / "cache" / "nar" / "local.nar.xz").unlink()
    with pytest.raises(ValueError, match="missing NAR"):
        assemble([missing], tmp_path / "missing-out")

    limited = _platform(
        tmp_path / "limited",
        "hermes-agent-v2026.11.1-rigel.1-linux-x64",
        [(LOCAL_HASH, "local.nar.xz", "sha256:local")],
    )
    output = tmp_path / "limited-out"
    with pytest.raises(ValueError, match="allows 2"):
        assemble([limited], output, max_assets=2)
    assert not output.exists()


def test_write_pin_uses_single_path_info_entry(tmp_path: Path) -> None:
    path_info = tmp_path / "path-info.json"
    store_path = f"/nix/store/{SHARED_HASH}-hermes-agent"
    path_info.write_text(f'{{"{store_path}": {{"narSize": 100}}}}', encoding="utf-8")
    output = tmp_path / "nix" / "release-closure" / "x86_64-linux.json"

    write_pin("x86_64-linux", "v2026.11.1-rigel.1", path_info, output)

    assert output.read_text(encoding="utf-8") == (
        '{\n  "tag": "v2026.11.1-rigel.1",\n'
        f'  "storePath": "{store_path}"\n}}\n'
    )
    path_info.write_text(
        f'{{"{store_path}": {{}}, "/nix/store/{LOCAL_HASH}-other": {{}}}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly one store path"):
        write_pin("x86_64-linux", "v2026.11.1-rigel.1", path_info, output)


def test_write_pin_rejects_invalid_system_tag_and_path(tmp_path: Path) -> None:
    path_info = tmp_path / "path-info.json"
    store_path = f"/nix/store/{SHARED_HASH}-hermes-agent"
    path_info.write_text(f'{{"{store_path}": {{}}}}', encoding="utf-8")
    output = tmp_path / "pin.json"

    with pytest.raises(ValueError, match="unknown system"):
        write_pin("riscv64-linux", "v2026.11.1-rigel.1", path_info, output)
    with pytest.raises(ValueError, match="release pattern"):
        write_pin("x86_64-linux", "not-a-release", path_info, output)

    path_info.write_text('{"/tmp/not-a-store-path": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Nix store path"):
        write_pin("x86_64-linux", "v2026.11.1-rigel.1", path_info, output)


def test_write_pin_accepts_nix_path_info_array(tmp_path: Path) -> None:
    store_path = f"/nix/store/{SHARED_HASH}-hermes-agent"
    path_info = tmp_path / "path-info.json"
    path_info.write_text(json.dumps([{"path": store_path, "narSize": 100}]), encoding="utf-8")
    output = tmp_path / "pin.json"

    write_pin("x86_64-linux", "v2026.11.1-rigel.1", path_info, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "tag": "v2026.11.1-rigel.1",
        "storePath": store_path,
    }
