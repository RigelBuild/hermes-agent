#!/usr/bin/env bash
# Validate a changed release version or mint its deterministic release tag.
# The version source is .github/rigel-version: one line, the bare version, no `v`.
set -euo pipefail

readonly VERSION_FILE=.github/rigel-version
readonly VERSION_RE='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rigel\.([1-9][0-9]*)$'
readonly TAG_RE='^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rigel\.([1-9][0-9]*)$'
export LC_ALL=C

usage() {
  echo "Usage: release-tag.sh check-pr <base-sha> <head-sha> | mint <sha> <owner/repo>" >&2
  exit 2
}

# Print the version at COMMIT. An optional commit with no usable file prints
# nothing: a PR base that predates the file, or a parent in mint's walk.
version_at() {
  local commit="$1" required="${2:-yes}" lines=()
  mapfile -t lines < <(git show "${commit}:${VERSION_FILE}" 2>/dev/null || true)
  if [ "${#lines[@]}" -ne 1 ] || [ -z "${lines[0]}" ]; then
    if [ "$required" = yes ]; then
      echo "::error::expected ${VERSION_FILE} at ${commit} to hold one non-empty line" >&2
      return 1
    fi
    return 0
  fi
  printf '%s\n' "${lines[0]}"
}

# True when decimal string LEFT is greater than RIGHT (no leading zeros).
greater_decimal() {
  local left="$1" right="$2"
  [ "${#left}" -gt "${#right}" ] || { [ "${#left}" -eq "${#right}" ] && [[ "$left" > "$right" ]]; }
}

validate_next_version() {
  local version="$1" base tag max="" expected
  if [[ ! "$version" =~ $VERSION_RE ]]; then
    echo "::error::version '${version}' does not match the required X.Y.Z-rigel.N shape" >&2
    return 1
  fi
  base="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
  while IFS= read -r tag; do
    [[ "$tag" =~ $TAG_RE ]] || continue
    [ "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}" = "$base" ] || continue
    if [ -z "$max" ] || greater_decimal "${BASH_REMATCH[4]}" "$max"; then
      max="${BASH_REMATCH[4]}"
    fi
  done < <(git tag --list "v${base}-rigel.*")
  if [ -z "$max" ]; then
    expected="${base}-rigel.1"
  else
    expected="${base}-rigel.$((10#$max + 1))"
  fi
  if [ "$version" != "$expected" ]; then
    echo "::error::expected ${expected}, got ${version}" >&2
    return 1
  fi
}

check_pr() {
  local base_sha="$1" head_sha="$2" base_version head_version
  if git diff --quiet "$base_sha" "$head_sha" -- "$VERSION_FILE"; then
    echo "${VERSION_FILE} is unchanged; no release version check is needed"
    return 0
  fi
  base_version="$(version_at "$base_sha" no)"
  head_version="$(version_at "$head_sha")"
  if [ "$base_version" = "$head_version" ]; then
    echo "${VERSION_FILE} version is unchanged; no release version check is needed"
    return 0
  fi
  validate_next_version "$head_version"
  echo "${head_version} is the next release version"
}

mint() {
  local sha="$1" repo="$2" version tag target parent parent_version response rc actual
  version="$(version_at "$sha")"
  tag="v${version}"
  if git show-ref --verify --quiet "refs/tags/${tag}" || gh api "repos/${repo}/git/ref/tags/${tag}" >/dev/null 2>&1; then
    echo "${tag} already exists; nothing to mint"
    return 0
  fi

  # Walk back to the oldest commit that set this version.
  target="$sha"
  while parent="$(git rev-parse --verify "${target}^1" 2>/dev/null)"; do
    parent_version="$(version_at "$parent" no)"
    [ "$parent_version" = "$version" ] || break
    target="$parent"
  done
  validate_next_version "$version"

  set +e
  response="$(gh api --method POST "repos/${repo}/git/refs" -f "ref=refs/tags/${tag}" -f "sha=${target}" 2>&1)"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "Created ${tag} at ${target}"
    return 0
  fi
  if [[ "$response" == *"HTTP 422"* && "$response" == *"Reference already exists"* ]]; then
    actual="$(gh api "repos/${repo}/git/ref/tags/${tag}" --jq '.object.sha')"
    if [ "$actual" = "$target" ]; then
      echo "${tag} already points at ${target}"
      return 0
    fi
    echo "::error::${tag} already exists at ${actual}, expected ${target}" >&2
    return 1
  fi
  printf '%s\n' "$response" >&2
  return "$rc"
}

[ "$#" -ge 1 ] || usage
command="$1"
shift
case "$command" in
  check-pr)
    [ "$#" -eq 2 ] || usage
    check_pr "$1" "$2"
    ;;
  mint)
    [ "$#" -eq 2 ] || usage
    mint "$1" "$2"
    ;;
  *) usage ;;
esac
