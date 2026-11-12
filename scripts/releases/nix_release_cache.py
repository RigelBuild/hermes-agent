"""Assemble the flat signed Nix cache published as release assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

MAX_RELEASE_ASSETS = 1000
MAX_UPSTREAM_ATTEMPTS = 3
UPSTREAM_TIMEOUT_SECONDS = 30
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
RELEASE_SYSTEMS = {"x86_64-linux", "aarch64-linux", "aarch64-darwin"}
TAG_PATTERN = re.compile(r"^v[A-Za-z0-9._+-]*-rigel\.[A-Za-z0-9._+-]+$")
STORE_PATH_PATTERN = re.compile(r"^/nix/store/([0-9a-df-np-sv-z]{32})-.+$")

Opener = Callable[..., object]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class Narinfo:
    store_path: str
    nar_hash: str
    nar_size: str
    references: tuple[str, ...]

    @property
    def identity(self) -> tuple[str, str, tuple[str, ...]]:
        return self.nar_hash, self.nar_size, self.references


def parse_narinfo(data: bytes, *, source: str = "narinfo") -> Narinfo:
    """Parse closure identity fields from a binary-cache narinfo."""
    fields: dict[str, list[str]] = {}
    for raw_line in data.decode("utf-8").splitlines():
        key, separator, value = raw_line.partition(":")
        if separator:
            fields.setdefault(key, []).append(value.lstrip(" "))

    def one(name: str) -> str:
        values = fields.get(name, [])
        if len(values) != 1:
            raise ValueError(f"{source}: expected one {name} field, found {len(values)}")
        return values[0]

    store_path = one("StorePath")
    if not STORE_PATH_PATTERN.fullmatch(store_path):
        raise ValueError(f"{source}: invalid Nix store path {store_path!r}")
    nar_hash = one("NarHash")
    nar_size = one("NarSize")
    references = tuple(sorted(one("References").split()))
    return Narinfo(
        store_path=store_path,
        nar_hash=nar_hash,
        nar_size=nar_size,
        references=references,
    )


def _field(data: bytes, name: str, *, source: str) -> str:
    values = parse_fields(data).get(name, ())
    if len(values) != 1:
        raise ValueError(f"{source}: expected one {name} field, found {len(values)}")
    return values[0]


def parse_fields(data: bytes) -> dict[str, tuple[str, ...]]:
    fields: dict[str, list[str]] = {}
    for raw_line in data.decode("utf-8").splitlines():
        key, separator, value = raw_line.partition(":")
        if separator:
            fields.setdefault(key, []).append(value.lstrip(" "))
    return {key: tuple(values) for key, values in fields.items()}


def _replace_fields(data: bytes, replacements: dict[str, str], *, source: str) -> bytes:
    found = {key: 0 for key in replacements}
    lines: list[bytes] = []
    for line in data.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        key, separator, _value = body.partition(b":")
        key_text = key.decode("utf-8")
        if separator and key_text in replacements:
            found[key_text] += 1
            replacement = f"{key_text}: {replacements[key_text]}".encode("utf-8")
            lines.append(replacement + ending)
        else:
            lines.append(line)
    for key, count in found.items():
        if count != 1:
            raise ValueError(f"{source}: expected one {key} field, found {count}")
    return b"".join(lines)


def flatten_narinfo(data: bytes, *, source: str) -> tuple[bytes, str]:
    url = _field(data, "URL", source=source)
    if not url.startswith("nar/"):
        raise ValueError(f"{source}: URL {url!r} is not under nar/")
    nar_name = url.removeprefix("nar/")
    if not nar_name or "/" in nar_name or "\\" in nar_name or nar_name in {".", ".."}:
        raise ValueError(f"{source}: URL {url!r} is not a single NAR file name")
    return _replace_fields(data, {"URL": nar_name}, source=source), nar_name


def fetch_upstream_narinfo(
    store_hash: str,
    base_url: str,
    *,
    opener: Opener | None = None,
    sleeper: Sleeper = time.sleep,
) -> bytes | None:
    """GET an upstream narinfo; return None on 404, retry transient failures."""
    open_url = opener or urlopen
    url = f"{base_url.rstrip('/')}/{store_hash}.narinfo"
    request = Request(url, method="GET")
    last_error: BaseException | None = None

    for attempt in range(1, MAX_UPSTREAM_ATTEMPTS + 1):
        try:
            response = open_url(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
            try:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                if status == 404:
                    return None
                if status != 200:
                    error = RuntimeError(f"GET {url}: HTTP {status}")
                    if status not in RETRYABLE_HTTP_STATUSES:
                        raise error
                    last_error = error
                else:
                    return response.read()
            finally:
                response.close()
        except HTTPError as error:
            if error.code == 404:
                error.close()
                return None
            if error.code not in RETRYABLE_HTTP_STATUSES:
                raise RuntimeError(f"GET {url}: HTTP {error.code}") from error
            last_error = error
            error.close()
        except (URLError, TimeoutError, OSError) as error:
            last_error = error

        if attempt < MAX_UPSTREAM_ATTEMPTS:
            sleeper(float(attempt))

    raise RuntimeError(
        f"GET {url} failed after {MAX_UPSTREAM_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _absolute_upstream_url(base_url: str, url: str, *, source: str) -> str:
    base = base_url.rstrip("/") + "/"
    absolute = urljoin(base, url)
    parsed_base = urlsplit(base)
    parsed_absolute = urlsplit(absolute)
    if parsed_absolute.scheme != parsed_base.scheme or parsed_absolute.netloc != parsed_base.netloc:
        raise ValueError(f"{source}: upstream narinfo URL has a different origin: {url!r}")
    if not parsed_absolute.path.startswith("/nar/"):
        raise ValueError(f"{source}: upstream narinfo URL is not under /nar/: {url!r}")
    return absolute


def use_upstream_narinfo(local: bytes, upstream: bytes, base_url: str, *, source: str) -> bytes:
    """Use upstream fetch metadata but preserve local closure identity and Sig."""
    local_narinfo = parse_narinfo(local, source=source)
    upstream_url = _field(upstream, "URL", source=f"upstream {source}")
    replacements = {
        "URL": _absolute_upstream_url(base_url, upstream_url, source=source),
        "Compression": _field(upstream, "Compression", source=f"upstream {source}"),
        "FileHash": _field(upstream, "FileHash", source=f"upstream {source}"),
        "FileSize": _field(upstream, "FileSize", source=f"upstream {source}"),
    }
    # Validate the upstream record's complete closure identity. These fields
    # are not copied: the release keeps its own identity and signature.
    upstream_narinfo = parse_narinfo(upstream, source=f"upstream {source}")
    if upstream_narinfo.store_path != local_narinfo.store_path or upstream_narinfo.identity != local_narinfo.identity:
        raise ValueError(f"upstream {source}: different StorePath, NarHash, NarSize or References")
    return _replace_fields(local, replacements, source=source)


@dataclass(frozen=True)
class Asset:
    data: bytes | None = None
    source: Path | None = None

    def digest(self) -> bytes:
        digest = hashlib.sha256()
        if self.source is not None:
            with self.source.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(self.data or b"")
        return digest.digest()


def _add_asset(assets: dict[str, Asset], name: str, asset: Asset) -> None:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid release asset name {name!r}")
    previous = assets.get(name)
    if previous is None:
        assets[name] = asset
        return
    if previous.digest() != asset.digest():
        raise ValueError(f"conflicting contents for release asset {name}")


def _read_platform(directory: Path) -> tuple[str, Path, bytes, bytes]:
    name = (directory / "platform").read_text(encoding="utf-8").strip()
    if not name:
        raise ValueError(f"{directory}: empty platform name")
    path_infos = sorted(directory.glob("*.path-info.json"))
    if len(path_infos) != 1:
        raise ValueError(f"{directory}: expected one path-info.json, found {len(path_infos)}")
    if path_infos[0].name != name + ".path-info.json":
        raise ValueError(f"{directory}: path-info file does not match platform {name}")
    public_key = (directory / "nix-release-cache.pub").read_bytes()
    cache_info = (directory / "cache" / "nix-cache-info").read_bytes()
    return name, path_infos[0], public_key, cache_info


def _store_hash(store_path: str) -> str:
    match = STORE_PATH_PATTERN.fullmatch(store_path)
    if match is None:
        raise ValueError(f"invalid Nix store path {store_path!r}")
    return match.group(1)


def assemble(
    platform_dirs: list[Path] | list[str],
    output_dir: Path | str,
    *,
    upstream_url: str | None = None,
    max_assets: int = MAX_RELEASE_ASSETS,
    opener: Opener | None = None,
    sleeper: Sleeper = time.sleep,
) -> int:
    """Merge platform caches into a flat asset directory without partial output."""
    if not platform_dirs:
        raise ValueError("no platform directories given")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty")

    assets: dict[str, Asset] = {}
    identities: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    seen_platforms: set[str] = set()
    for raw_directory in platform_dirs:
        directory = Path(raw_directory)
        name, path_info, public_key, cache_info = _read_platform(directory)
        if name in seen_platforms:
            raise ValueError(f"platform {name} given twice")
        seen_platforms.add(name)
        _add_asset(assets, "nix-cache-info", Asset(data=cache_info))
        _add_asset(assets, "nix-release-cache.pub", Asset(data=public_key))
        _add_asset(assets, path_info.name, Asset(source=path_info))

        cache = directory / "cache"
        nar_files = tuple((cache / "nar").glob("*"))
        if not nar_files:
            raise ValueError(f"platform {name} has no NAR files")
        narinfos = sorted(cache.glob("*.narinfo"))
        if not narinfos:
            raise ValueError(f"platform {name} has no narinfo files")

        for narinfo_path in narinfos:
            local = narinfo_path.read_bytes()
            parsed = parse_narinfo(local, source=str(narinfo_path))
            store_hash = _store_hash(parsed.store_path)
            previous_identity = identities.get(parsed.store_path)
            if previous_identity is not None:
                if previous_identity != parsed.identity:
                    raise ValueError(
                        f"{narinfo_path}: {parsed.store_path} has a different NarHash, NarSize or References on another platform"
                    )
                continue
            identities[parsed.store_path] = parsed.identity

            upstream_record = None
            if upstream_url is not None:
                upstream_record = fetch_upstream_narinfo(
                    store_hash, upstream_url, opener=opener, sleeper=sleeper
                )
            if upstream_record is not None:
                narinfo_data = use_upstream_narinfo(
                    local, upstream_record, upstream_url, source=str(narinfo_path)
                )
            else:
                narinfo_data, nar_name = flatten_narinfo(local, source=str(narinfo_path))
                nar_source = cache / "nar" / nar_name
                if not nar_source.is_file():
                    raise ValueError(f"{narinfo_path}: missing NAR {nar_source}")
                _add_asset(assets, nar_name, Asset(source=nar_source))

            _add_asset(assets, store_hash + ".narinfo", Asset(data=narinfo_data))

    if len(assets) > max_assets:
        raise ValueError(f"release has {len(assets)} assets; GitHub allows {max_assets} per release")

    output.mkdir(parents=True, exist_ok=True)
    for name, asset in sorted(assets.items()):
        target = output / name
        if asset.source is not None:
            shutil.copyfile(asset.source, target)
        else:
            target.write_bytes(asset.data or b"")
    return len(assets)


def write_pin(system: str, tag: str, path_info_file: Path | str, output_file: Path | str) -> None:
    if system not in RELEASE_SYSTEMS:
        raise ValueError(f"unknown system {system!r}; want one of {', '.join(sorted(RELEASE_SYSTEMS))}")
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError(f"tag {tag!r} does not match the v*-rigel.* release pattern")
    data = json.loads(Path(path_info_file).read_text(encoding="utf-8"))
    if isinstance(data, dict) and len(data) == 1:
        store_path = next(iter(data))
    elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        store_path = data[0].get("path")
        if not isinstance(store_path, str):
            raise ValueError(f"{path_info_file}: path-info array entry must contain a string path")
    else:
        count = len(data) if isinstance(data, (dict, list)) else "not an object or array"
        raise ValueError(f"{path_info_file}: expected exactly one store path, found {count}")
    _store_hash(store_path)
    payload = {"tag": tag, "storePath": store_path}
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_release_cache_asset(name: str) -> bool:
    if name in {"nix-cache-info", "nix-release-cache.pub"}:
        return True
    if re.fullmatch(r"[0-9a-df-np-sv-z]{32}\.narinfo", name):
        return True
    if re.fullmatch(r".+\.nar(?:\.[A-Za-z0-9_-]+)?", name):
        return True
    return name.startswith("hermes-agent-v") and "-rigel." in name and name.endswith(".path-info.json")


def reconcile_assets(output_dir: Path | str, existing_names: list[str]) -> list[str]:
    """Return stale cache assets, leaving unrelated release assets untouched."""
    output = Path(output_dir)
    new_names = {path.name for path in output.iterdir() if path.is_file()}
    return sorted(name for name in set(existing_names) - new_names if _is_release_cache_asset(name))


def validate_asset_count(
    output_dir: Path | str,
    existing_names: list[str],
    *,
    max_assets: int = MAX_RELEASE_ASSETS,
) -> int:
    """Fail if retained existing assets plus the upload set exceed GitHub's cap."""
    output = Path(output_dir)
    stale_names = set(reconcile_assets(output, existing_names))
    retained_names = set(existing_names) - stale_names
    new_names = {path.name for path in output.iterdir() if path.is_file()}
    total = len(retained_names | new_names)
    if total > max_assets:
        raise ValueError(f"release would have {total} assets; GitHub allows {max_assets}")
    return total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    assemble_parser = commands.add_parser("assemble", help="merge platform caches into flat release assets")
    assemble_parser.add_argument("-out", required=True, type=Path, help="empty directory for release assets")
    assemble_parser.add_argument("-upstream", dest="upstream_url", help="upstream binary cache URL")
    assemble_parser.add_argument("platform_dirs", nargs="+", type=Path)

    pin_parser = commands.add_parser("write-pin", help="write one system's release closure pin")
    pin_parser.add_argument("-system", required=True)
    pin_parser.add_argument("-tag", required=True)
    pin_parser.add_argument("-path-info", required=True, type=Path)
    pin_parser.add_argument("-out", required=True, type=Path)

    reconcile_parser = commands.add_parser("reconcile", help="print existing cache assets absent from the new asset set")
    reconcile_parser.add_argument("-out", required=True, type=Path, help="assembled release asset directory")
    reconcile_parser.add_argument("-existing-assets", required=True, type=Path, help="JSON array of current release asset names")

    count_parser = commands.add_parser("count", help="check the release asset cap after stale assets are removed")
    count_parser.add_argument("-out", required=True, type=Path, help="assembled release asset directory")
    count_parser.add_argument("-existing-assets", required=True, type=Path, help="JSON array of remaining release asset names")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "assemble":
            count = assemble(args.platform_dirs, args.out, upstream_url=args.upstream_url)
            print(f"wrote {count} release assets to {args.out}")
        elif args.command == "write-pin":
            write_pin(args.system, args.tag, args.path_info, args.out)
        elif args.command == "reconcile":
            existing = json.loads(args.existing_assets.read_text(encoding="utf-8"))
            if not isinstance(existing, list) or not all(isinstance(name, str) for name in existing):
                raise ValueError(f"{args.existing_assets}: expected a JSON array of asset names")
            for asset in reconcile_assets(args.out, existing):
                print(asset)
        elif args.command == "count":
            existing = json.loads(args.existing_assets.read_text(encoding="utf-8"))
            if not isinstance(existing, list) or not all(isinstance(name, str) for name in existing):
                raise ValueError(f"{args.existing_assets}: expected a JSON array of asset names")
            count = validate_asset_count(args.out, existing)
            print(f"release will have {count} assets")
        return 0
    except (OSError, ValueError, RuntimeError, HTTPError, URLError) as error:
        print(f"nix-release-cache: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
