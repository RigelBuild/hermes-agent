"""Discord server introspection and management tool (REST API + bot token).

The model-visible schema is filtered by two gates: privileged intents from GET /applications/@me
(search_members / member_info need GUILD_MEMBERS; fetch_messages / list_pins are annotated when
MESSAGE_CONTENT is missing) and the ``discord.server_actions`` config allowlist. Per-guild
permissions are NOT pre-checked — a call-time 403 is mapped to guidance by :func:`_enrich_403`.
"""

import functools
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
_DISCORD_RESPONSE_BODY_MAX_BYTES = 4 * 1024 * 1024
_DISCORD_ERROR_BODY_MAX_BYTES = 64 * 1024
_MAX_ARCHIVED_THREAD_PAGES = 10

# Application flag bits (GET /applications/@me → "flags"); the *_LIMITED bit is the
# <100-guild variant of the same intent.
_FLAGS_GUILD_MEMBERS = (1 << 14) | (1 << 15)
_FLAGS_MESSAGE_CONTENT = (1 << 18) | (1 << 19)


class DiscordAPIError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Discord API error {status}: {body}")


def _read_limited_response_body(source: Any, limit: int, *, label: str) -> bytes:
    body = source.read(limit + 1)
    if len(body) > limit:
        raise DiscordAPIError(502, f"Discord API {label} exceeded {limit} bytes.")
    return body


def _get_bot_token() -> Optional[str]:
    """Resolve the Discord bot token under the active profile secret scope."""
    return (get_secret("DISCORD_BOT_TOKEN", "") or "").strip() or None


_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_MAX_WAIT_SECONDS = 10.0


def _rate_limit_wait(error: "DiscordAPIError") -> Optional[float]:
    """Seconds to wait before retrying a 429, or ``None`` when the wait is unknown or too long."""
    try:
        retry_after = float(json.loads(error.body).get("retry_after"))
    except (TypeError, ValueError, AttributeError):
        return None
    if retry_after < 0 or retry_after > _RATE_LIMIT_MAX_WAIT_SECONDS:
        return None
    return retry_after


def _discord_request(
    method: str, path: str, token: str, params: Optional[dict[str, str]] = None,
    body: Optional[dict[str, Any]] = None, timeout: int = 15) -> Any:
    """Make a request to the Discord REST API, waiting out short 429 rate limits."""
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return _discord_request_once(method, path, token, params=params, body=body, timeout=timeout)
        except DiscordAPIError as error:
            wait = _rate_limit_wait(error) if error.status == 429 else None
            if wait is None or attempt == _RATE_LIMIT_RETRIES:
                raise
            logger.info("Discord rate limited %s %s; retrying in %.2fs", method, path, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _discord_request_once(
    method: str, path: str, token: str, params: Optional[dict[str, str]] = None,
    body: Optional[dict[str, Any]] = None, timeout: int = 15) -> Any:
    """Make one request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode("utf-8"), method=method,
        headers={
            "Authorization": f"Bot {token}", "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            body = _read_limited_response_body(resp, _DISCORD_RESPONSE_BODY_MAX_BYTES, label="response body")
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            error_body = _read_limited_response_body(
                e, _DISCORD_ERROR_BODY_MAX_BYTES, label="error body").decode("utf-8", errors="replace")
        except DiscordAPIError as too_large:
            error_body = too_large.body
        except Exception:
            error_body = ""
        raise DiscordAPIError(e.code, error_body) from e


_CHANNEL_TYPE_NAMES = {
    0: "text", 2: "voice", 4: "category", 5: "announcement", 10: "announcement_thread",
    11: "public_thread", 12: "private_thread", 13: "stage", 15: "forum", 16: "media"}


def _channel_type_name(type_id: int) -> str:
    return _CHANNEL_TYPE_NAMES.get(type_id, f"unknown({type_id})")


# ── capability detection (application intents) ──────────────────────────────
# Per-token in-process cache: the app/me endpoint is hit at most once per process.
_capability_cache: dict[str, dict[str, Any]] = {}

# Privileged intents change only when the user flips them in the Developer Portal, so
# 24h disk staleness is harmless: a hidden action re-appears on the next refresh; an
# exposed action the bot lost fails at call time with an enriched 403.
_CAPABILITY_DISK_TTL_SECONDS = 24 * 3600

# One background detection per (process, token) at most.
_capability_bg_started: set = set()
_capability_bg_lock = threading.Lock()

# Permissive default (``detected`` False = detection failed/pending): all actions
# exposed, call-time 403s mapped to guidance by ``_enrich_403``.
_PERMISSIVE_CAPS = {"has_members_intent": True, "has_message_content": True, "detected": False}


def _capability_disk_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "discord_capabilities.json"


def _token_cache_key(token: str) -> str:
    """Stable non-reversible cache key for a bot token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _read_caps_file(path: Path) -> dict[str, Any]:
    """Disk cache contents ({token_key: {"caps", "ts"}}); {} when missing/corrupt."""
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_caps_from_disk(token: str) -> Optional[dict[str, Any]]:
    """Return fresh disk-cached capabilities for *token*, or None."""
    try:
        entry = _read_caps_file(_capability_disk_cache_path()).get(_token_cache_key(token))
        if not isinstance(entry, dict) or time.time() - float(entry.get("ts", 0)) > _CAPABILITY_DISK_TTL_SECONDS:
            return None
        caps = entry.get("caps")
        return caps if isinstance(caps, dict) and "has_members_intent" in caps else None
    except Exception:
        return None


def _save_caps_to_disk(token: str, caps: dict[str, Any]) -> None:
    try:
        path = _capability_disk_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read_caps_file(path)
        data[_token_cache_key(token)] = {"caps": caps, "ts": time.time()}
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)
    except Exception:
        logger.debug("discord capability disk-cache write failed", exc_info=True)


def _detect_capabilities_nonblocking(token: str) -> dict[str, Any]:
    """Schema-build lookup: in-process cache → fresh disk cache → permissive default plus a
    fire-and-forget background detection that fills the disk cache for the NEXT process
    (the ~2-5s blocking HTTPS call must stay off the cold-start critical path)."""
    cached = _capability_cache.get(token)
    if cached is not None:
        return cached
    disk = _load_caps_from_disk(token)
    if disk is not None:
        _capability_cache[token] = disk
        return disk

    # Cold start — pin the permissive default for THIS process: schemas must not change
    # between agent inits within a live process or the per-conversation prompt cache breaks.
    caps_default = dict(_PERMISSIVE_CAPS)
    _capability_cache[token] = caps_default
    with _capability_bg_lock:
        if token not in _capability_bg_started:
            _capability_bg_started.add(token)

            def _bg_detect() -> None:
                try:
                    caps = _fetch_capabilities(token)
                    if caps.get("detected"):
                        _save_caps_to_disk(token, caps)
                except Exception:
                    logger.debug("background discord capability detection failed", exc_info=True)

            threading.Thread(target=_bg_detect, name="discord-caps-detect", daemon=True).start()
    return caps_default


def _fetch_capabilities(token: str) -> dict[str, Any]:
    """Fetch capabilities from GET /applications/@me. Pure network fetch — never touches
    the in-process cache (background detection must not mutate schemas mid-process).
    Detection failure is permissive."""
    caps: dict[str, Any] = dict(_PERMISSIVE_CAPS)
    try:
        app = _discord_request("GET", "/applications/@me", token, timeout=5)
        flags = int(app.get("flags", 0) or 0)
        caps["has_members_intent"] = bool(flags & _FLAGS_GUILD_MEMBERS)
        caps["has_message_content"] = bool(flags & _FLAGS_MESSAGE_CONTENT)
        caps["detected"] = True
    except Exception as exc:  # nosec — detection is best-effort
        logger.info("Discord capability detection failed (%s); exposing all actions.", exc)
    return caps


def _detect_capabilities(token: str, *, force: bool = False) -> dict[str, Any]:
    """Blocking detection via GET /applications/@me, cached per token (the warm-up path;
    schema builds use the non-blocking variant). ``force`` re-fetches."""
    if token in _capability_cache and not force:
        return _capability_cache[token]
    caps = _fetch_capabilities(token)
    _capability_cache[token] = caps
    return caps


def _reset_capability_cache() -> None:
    """Test hook: clear the detection cache."""
    global _capability_cache, _capability_bg_started
    _capability_cache = {}
    with _capability_bg_lock:
        _capability_bg_started = set()


# ── action implementations ───────────────────────────────────────────────────
def _listing(key: str, items: list[dict[str, Any]]) -> str:
    return json.dumps({key: items, "count": len(items)})


def _member_summary(m: dict[str, Any], *, full: bool) -> dict[str, Any]:
    """Member row; ``full`` adds the avatar/join fields member_info exposes
    (key order is part of the result text, so the two shapes stay explicit)."""
    user = m.get("user", {})
    row = {
        "user_id": user.get("id"), "username": user.get("username"), "display_name": user.get("global_name"),
        "nickname": m.get("nick"), "avatar": user.get("avatar"), "bot": user.get("bot", False),
        "roles": m.get("roles", []), "joined_at": m.get("joined_at"), "premium_since": m.get("premium_since")}
    return row if full else {k: v for k, v in row.items() if k not in ("avatar", "joined_at", "premium_since")}


def _message_summary(msg: dict[str, Any]) -> dict[str, Any]:
    author = msg.get("author", {})
    return {
        "id": msg["id"], "content": msg.get("content", ""),
        "author": {
            "id": author.get("id"), "username": author.get("username"),
            "display_name": author.get("global_name"), "bot": author.get("bot", False)},
        "timestamp": msg.get("timestamp"), "edited_timestamp": msg.get("edited_timestamp"),
        "attachments": [
            {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
            for a in msg.get("attachments", [])],
        "reactions": [
            {"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)}
            for r in msg.get("reactions", [])] if msg.get("reactions") else [],
        "pinned": msg.get("pinned", False)}


def _limit_param(limit: Any, default: int) -> str:
    """Discord caps list endpoints at 100 per page."""
    try:
        return str(min(int(limit), 100))
    except (TypeError, ValueError):
        return str(min(default, 100))


def _list_guilds(token: str, **_kwargs: Any) -> str:
    guilds = _discord_request("GET", "/users/@me/guilds", token)
    return _listing("guilds", [
        {
            "id": g["id"], "name": g["name"], "icon": g.get("icon"),
            "owner": g.get("owner", False), "permissions": g.get("permissions")}
        for g in guilds])


def _server_info(token: str, guild_id: str, **_kwargs: Any) -> str:
    g = _discord_request("GET", f"/guilds/{guild_id}", token, params={"with_counts": "true"})
    return json.dumps({
        "id": g["id"], "name": g["name"], "description": g.get("description"), "icon": g.get("icon"),
        "owner_id": g.get("owner_id"), "member_count": g.get("approximate_member_count"),
        "online_count": g.get("approximate_presence_count"), "features": g.get("features", []),
        "premium_tier": g.get("premium_tier"), "premium_subscription_count": g.get("premium_subscription_count"),
        "verification_level": g.get("verification_level")})


def _list_channels(token: str, guild_id: str, **_kwargs: Any) -> str:
    """All channels grouped by category (uncategorized first), each sorted by position."""
    channels = _discord_request("GET", f"/guilds/{guild_id}/channels", token)
    cats = sorted((ch for ch in channels if ch["type"] == 4), key=lambda c: c.get("position", 0))
    groups: dict[Optional[str], list[dict[str, Any]]] = {None: [], **{c["id"]: [] for c in cats}}
    for ch in channels:
        if ch["type"] == 4:  # category
            continue
        parent = ch.get("parent_id")
        groups[parent if parent in groups else None].append({
            "id": ch["id"], "name": ch.get("name", ""), "type": _channel_type_name(ch["type"]),
            "position": ch.get("position", 0), "topic": ch.get("topic"), "nsfw": ch.get("nsfw", False)})
    for group in groups.values():
        group.sort(key=lambda c: c["position"])
    result = [{"category": None, "channels": groups[None]}] if groups[None] else []
    result += [{"category": {"id": c["id"], "name": c["name"]}, "channels": groups[c["id"]]} for c in cats]
    return json.dumps({"channel_groups": result, "total_channels": sum(len(g["channels"]) for g in result)})


def _channel_info(token: str, channel_id: str, **_kwargs: Any) -> str:
    ch = _discord_request("GET", f"/channels/{channel_id}", token)
    return json.dumps({
        "id": ch["id"], "name": ch.get("name"), "type": _channel_type_name(ch["type"]),
        "guild_id": ch.get("guild_id"), "topic": ch.get("topic"), "nsfw": ch.get("nsfw", False),
        "position": ch.get("position"), "parent_id": ch.get("parent_id"),
        "rate_limit_per_user": ch.get("rate_limit_per_user", 0), "last_message_id": ch.get("last_message_id")})


def _list_roles(token: str, guild_id: str, **_kwargs: Any) -> str:
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    return _listing("roles", [
        {
            "id": r["id"], "name": r["name"],
            "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
            "position": r.get("position", 0), "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False), "member_count": r.get("member_count"),
            "hoist": r.get("hoist", False)}
        for r in sorted(roles, key=lambda r: r.get("position", 0), reverse=True)])


def _member_info(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    m = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    return json.dumps(_member_summary(m, full=True))


def _search_members(token: str, guild_id: str, query: str, limit: int = 20, **_kwargs: Any) -> str:
    """Name-prefix member search (requires the GUILD_MEMBERS intent)."""
    params = {"query": query, "limit": _limit_param(limit, 20)}
    members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params=params)
    return _listing("members", [_member_summary(m, full=False) for m in members])


def _fetch_messages(
    token: str, channel_id: str, limit: int = 50,
    before: Optional[str] = None, after: Optional[str] = None, **_kwargs: Any) -> str:
    """``before``/``after`` are message snowflakes for reverse/forward pagination."""
    params: dict[str, str] = {"limit": _limit_param(limit, 50)}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    messages = _discord_request("GET", f"/channels/{channel_id}/messages", token, params=params)
    return _listing("messages", [_message_summary(msg) for msg in messages])


def _list_pins(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Pinned messages (content truncated for overview)."""
    messages = _discord_request("GET", f"/channels/{channel_id}/pins", token)
    return _listing("pinned_messages", [
        {
            "id": msg["id"], "content": msg.get("content", "")[:200],
            "author": msg.get("author", {}).get("username"), "timestamp": msg.get("timestamp")}
        for msg in messages])


def _create_thread(
    token: str, channel_id: str, name: str, message_id: Optional[str] = None,
    auto_archive_duration: int = 1440, **_kwargs: Any) -> str:
    """Create a thread — anchored to ``message_id`` when given, else standalone public."""
    body: dict[str, Any] = {"name": name, "auto_archive_duration": auto_archive_duration}
    path = f"/channels/{channel_id}/threads"
    if message_id:
        path = f"/channels/{channel_id}/messages/{message_id}/threads"
    else:
        body["type"] = 11  # PUBLIC_THREAD
    thread = _discord_request("POST", path, token, body=body)
    return json.dumps({"success": True, "thread_id": thread["id"], "name": thread.get("name")})


def _thread_created_at(thread: dict[str, Any]) -> tuple[int, str]:
    """Creation time in epoch milliseconds and ISO form, using the snowflake when needed."""
    metadata = thread.get("thread_metadata") or {}
    created = metadata.get("create_timestamp")
    if created:
        try:
            parsed = datetime.fromisoformat(created)
            millis = int(parsed.timestamp() * 1000)
            return millis, created
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        millis = (int(thread["id"]) >> 22) + 1420070400000
        created = datetime.fromtimestamp(millis / 1000, tz=UTC).isoformat().replace("+00:00", "Z")
        return millis, created
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return 0, ""


def _list_forum_posts(
    token: str, channel_id: str, query: str = "", limit: int = 50, **_kwargs: Any,
) -> str:
    """List active and archived forum threads, newest first."""
    channel = _discord_request("GET", f"/channels/{channel_id}", token)
    guild_id = channel.get("guild_id")
    active = _discord_request("GET", f"/guilds/{guild_id}/threads/active", token)
    threads = list(active.get("threads", []))
    archived: list[dict[str, Any]] = []
    before = None
    has_more = False
    for _ in range(_MAX_ARCHIVED_THREAD_PAGES):
        params = {"limit": "100"}
        if before:
            params["before"] = before
        page = _discord_request(
            "GET", f"/channels/{channel_id}/threads/archived/public", token, params=params,
        )
        page_threads = page.get("threads", [])
        archived.extend(page_threads)
        if not page.get("has_more") or not page_threads:
            has_more = False
            break
        next_before = (page_threads[-1].get("thread_metadata") or {}).get("archive_timestamp")
        if not next_before or next_before == before:
            has_more = True
            break
        # Discord paginates archives by timestamp. If the page boundary timestamp
        # is shared by another thread, the cursor cannot reliably include all ties.
        # Report possible truncation instead of silently returning an incomplete list.
        if any(
            (thread.get("thread_metadata") or {}).get("archive_timestamp") == next_before
            for thread in page_threads[:-1]
        ):
            has_more = True
            break
        before = next_before
        has_more = True
    else:
        has_more = True

    by_id: dict[str, dict[str, Any]] = {}
    for thread in [*threads, *archived]:
        if thread.get("parent_id") == channel_id:
            by_id.setdefault(str(thread.get("id", "")), thread)
    filtered = [
        thread for thread in by_id.values()
        if query.casefold() in str(thread.get("name", "")).casefold()
    ]
    filtered.sort(key=lambda thread: _thread_created_at(thread)[0], reverse=True)
    posts = []
    result_limit = max(1, int(_limit_param(limit, 50)))
    has_more = has_more or len(filtered) > result_limit
    for thread in filtered[:result_limit]:
        _, created_at = _thread_created_at(thread)
        posts.append({
            "id": thread.get("id"), "name": thread.get("name"),
            "archived": bool((thread.get("thread_metadata") or {}).get("archived", False)),
            "created_at": created_at or None, "message_count": thread.get("message_count", 0),
        })
    return json.dumps({"posts": posts, "count": len(posts), "has_more": has_more})


_DISCORD_MESSAGE_MAX_CHARS = 2000


def _message_chunks(content: str) -> list[str]:
    """Split at newlines/spaces, keeping code fences balanced in every chunk (gateway splitter)."""
    from gateway.platforms.base import BasePlatformAdapter
    return BasePlatformAdapter.truncate_message(content, _DISCORD_MESSAGE_MAX_CHARS)


def _load_post_channels_config() -> Optional[set[str]]:
    """``discord.post_channels`` (comma string or list of channel IDs), or ``None`` when unset."""
    try:
        from hermes_cli.config import load_config
        raw = (load_config().get("discord") or {}).get("post_channels")
    except Exception as exc:
        logger.debug("discord: could not load post_channels (%s); not restricting.", exc)
        return None
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, (str, int)):
        raw = str(raw).split(",")
    if not isinstance(raw, (list, tuple)):
        logger.warning("discord.post_channels: unexpected type %s; ignoring.", type(raw).__name__)
        return None
    return {str(item).strip() for item in raw if str(item).strip()}


def _require_post_scope(token: str, channel_id: str, channel: Optional[dict[str, Any]] = None) -> None:
    """Refuse posts outside ``discord.post_channels``; a thread counts via its parent channel."""
    allowed = _load_post_channels_config()
    if allowed is None or str(channel_id) in allowed:
        return
    if channel is None:
        channel = _discord_request("GET", f"/channels/{channel_id}", token) or {}
    if str(channel.get("parent_id") or "") in allowed:
        return
    raise ValueError(
        f"Channel {channel_id} is outside discord.post_channels; allowed: {', '.join(sorted(allowed)) or '<none>'}")


def _allowed_mentions() -> dict[str, Any]:
    """Resolve the current profile's Discord mention policy, defaulting to safe mentions."""
    try:
        from hermes_cli.config import load_config
        discord_cfg = load_config().get("discord") or {}
    except Exception as exc:
        logger.debug("discord: could not load mention settings (%s); using safe defaults.", exc, exc_info=True)
        discord_cfg = {}
    configured = discord_cfg.get("allow_mentions")
    configured = configured if isinstance(configured, dict) else {}
    defaults = {"everyone": False, "roles": False, "users": True, "replied_user": True}
    settings = {}
    for key, default in defaults.items():
        env_value = get_secret(f"DISCORD_ALLOW_MENTION_{key.upper()}", None)
        value = env_value if env_value is not None else configured.get(key, default)
        settings[key] = value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes", "on"}
    return {
        "parse": [kind for kind in ("everyone", "roles", "users") if settings[kind]],
        "replied_user": settings["replied_user"],
    }


def _send_message_chunks(
    token: str, channel_id: str, content: str, allowed_mentions: Optional[dict[str, Any]] = None,
) -> list[str]:
    return _send_chunks(token, channel_id, _message_chunks(content), allowed_mentions)


def _send_chunks(
    token: str, channel_id: str, chunks: list[str], allowed_mentions: Optional[dict[str, Any]] = None,
) -> list[str]:
    mentions = allowed_mentions if allowed_mentions is not None else _allowed_mentions()
    message_ids = []
    for index, chunk in enumerate(chunks, start=1):
        try:
            message = _discord_request(
                "POST", f"/channels/{channel_id}/messages", token,
                body={"content": chunk, "allowed_mentions": mentions},
            )
        except Exception as exc:
            sent = ", ".join(message_ids) if message_ids else "none"
            status = exc.status if isinstance(exc, DiscordAPIError) else 502
            raw_error = exc.body if isinstance(exc, DiscordAPIError) else str(exc)
            raise DiscordAPIError(
                status,
                f"Partial delivery: sent message IDs {sent}; failed chunk {index}/{len(chunks)}. "
                f"Retrying may duplicate sent messages. Raw error: {raw_error}",
            ) from exc
        message_ids.append(message["id"])
    return message_ids


def _require_forum_post_text(name: str, content: str) -> None:
    if not isinstance(name, str) or not name or len(name) > 100:
        raise ValueError("Forum post name must be between 1 and 100 characters.")
    if not isinstance(content, str) or not content:
        raise ValueError("Forum post content must not be empty.")


def _validate_applied_tags(applied_tags: Optional[list[str]]) -> None:
    if applied_tags is None or applied_tags == []:
        return
    if not isinstance(applied_tags, list) or any(
        not isinstance(tag, str)
        or not tag.isascii()
        or not tag.isdigit()
        or len(tag) > 20
        or int(tag) == 0
        or int(tag) >= 2**64
        for tag in applied_tags
    ):
        raise ValueError("applied_tags must be a list of positive 64-bit Discord forum tag snowflake IDs.")
    if len(applied_tags) > 5:
        raise ValueError("A forum post can have at most five applied_tags.")


def _forum_post_body(name: str, starter: str, applied_tags: Optional[list[str]]) -> dict[str, Any]:
    _validate_applied_tags(applied_tags)
    body: dict[str, Any] = {
        "name": name,
        "message": {"content": starter, "allowed_mentions": _allowed_mentions()},
        "auto_archive_duration": 1440,
    }
    if applied_tags:
        body["applied_tags"] = applied_tags
    return body


def _create_forum_post(
    token: str, channel_id: str, name: str, content: str,
    applied_tags: Optional[list[str]] = None, **_kwargs: Any,
) -> str:
    """Create a tagged forum thread with a starter message, then append long content."""
    _require_forum_post_text(name, content)
    _require_post_scope(token, channel_id)
    chunks = _message_chunks(content)
    body = _forum_post_body(name, chunks[0], applied_tags)
    thread = _discord_request("POST", f"/channels/{channel_id}/threads", token, body=body)
    starter = thread.get("message") or {}
    try:
        message_ids = (
            _send_chunks(token, thread["id"], chunks[1:], allowed_mentions=body["message"]["allowed_mentions"])
            if len(chunks) > 1 else []
        )
    except Exception as exc:
        status = exc.status if isinstance(exc, DiscordAPIError) else 502
        detail = exc.body if isinstance(exc, DiscordAPIError) else str(exc)
        raise DiscordAPIError(
            status,
            f"Partial forum post created: thread_id={thread['id']}, "
            f"starter_message_id={starter.get('id')}. {detail}",
        ) from exc
    return json.dumps({
        "success": True, "thread_id": thread["id"], "name": thread.get("name"),
        "starter_message_id": starter.get("id"), "additional_message_ids": message_ids,
    })




def _post_message(token: str, channel_id: str, content: str, **_kwargs: Any) -> str:
    """Post message chunks to a channel/thread, reopening an archived thread first."""
    channel = _discord_request("GET", f"/channels/{channel_id}", token)
    _require_post_scope(token, channel_id, channel)
    if (channel.get("thread_metadata") or {}).get("archived"):
        try:
            _discord_request("PATCH", f"/channels/{channel_id}", token, body={"archived": False})
        except DiscordAPIError as exc:
            raise DiscordAPIError(
                exc.status, f"Could not unarchive thread before posting: {exc.body}",
            ) from exc
    message_ids = _send_message_chunks(token, channel_id, content)
    return json.dumps({"success": True, "message_ids": message_ids})


def _mutation(method: str, path: str, message: str):
    """Body-less write action: ``path``/``message`` are format templates over the action kwargs."""
    def _action(token: str, **kw: Any) -> str:
        _discord_request(method, path.format(**kw), token)
        return json.dumps({"success": True, "message": message.format(**kw)})
    return _action


_pin_message = _mutation("PUT", "/channels/{channel_id}/pins/{message_id}", "Message {message_id} pinned.")
_unpin_message = _mutation("DELETE", "/channels/{channel_id}/pins/{message_id}", "Message {message_id} unpinned.")
_delete_message = _mutation(
    "DELETE", "/channels/{channel_id}/messages/{message_id}", "Message {message_id} deleted.")
_add_role = _mutation(
    "PUT", "/guilds/{guild_id}/members/{user_id}/roles/{role_id}", "Role {role_id} added to user {user_id}.")
_remove_role = _mutation(
    "DELETE", "/guilds/{guild_id}/members/{user_id}/roles/{role_id}",
    "Role {role_id} removed from user {user_id}.")


# ── action dispatch + metadata ───────────────────────────────────────────────
# Single source of truth: (action, handler, required-param signature, description). Order is
# the schema/enum order; the signature drives runtime required-param validation.
_ACTION_MANIFEST = [
    ("list_guilds", _list_guilds, "()", "list servers the bot is in"),
    ("server_info", _server_info, "(guild_id)", "server details + member counts"),
    ("list_channels", _list_channels, "(guild_id)", "all channels grouped by category"),
    ("channel_info", _channel_info, "(channel_id)", "single channel details"),
    ("list_roles", _list_roles, "(guild_id)", "roles sorted by position"),
    ("member_info", _member_info, "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", _search_members, "(guild_id, query)", "find members by name prefix"),
    ("fetch_messages", _fetch_messages, "(channel_id)", "recent messages; optional before/after snowflakes"),
    ("list_pins", _list_pins, "(channel_id)", "pinned messages in a channel"),
    ("pin_message", _pin_message, "(channel_id, message_id)", "pin a message"),
    ("unpin_message", _unpin_message, "(channel_id, message_id)", "unpin a message"),
    ("delete_message", _delete_message, "(channel_id, message_id)", "delete a message"),
    ("create_thread", _create_thread, "(channel_id, name)", "create a public thread; optional message_id anchor"),
    ("list_forum_posts", _list_forum_posts, "(channel_id)", "find forum posts by title; use for daily find-or-create workflows; response has_more flags archive scan truncation"),
    ("create_forum_post", _create_forum_post, "(channel_id, name, content)", "create a forum post with starter message; optional applied_tags for required forum tags; use for daily forum workflows; limited by discord.post_channels"),
    ("post_message", _post_message, "(channel_id, content)", "post into a channel or forum post; use for daily forum workflows; limited by discord.post_channels"),
    ("add_role", _add_role, "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", _remove_role, "(guild_id, user_id, role_id)", "remove a role"),
]
_ACTIONS = {name: fn for name, fn, _sig, _desc in _ACTION_MANIFEST}
_REQUIRED_PARAMS: dict[str, list[str]] = {
    name: [p.strip() for p in sig.strip("()").split(",") if p.strip()]
    for name, _fn, sig, _desc in _ACTION_MANIFEST}

# Two tools share one action table: ``discord`` (core participation actions every bot
# user wants) and ``discord_admin`` (everything else).
_CORE_ACTION_NAMES = frozenset({
    "fetch_messages", "search_members", "create_thread", "list_forum_posts", "create_forum_post", "post_message",
})
_CORE_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _CORE_ACTION_NAMES}
_ADMIN_ACTIONS = {k: v for k, v in _ACTIONS.items() if k not in _CORE_ACTION_NAMES}

# Actions that require the GUILD_MEMBERS privileged intent.
_INTENT_GATED_MEMBERS = frozenset({"member_info", "search_members"})


def _load_allowed_actions_config() -> Optional[list[str]]:
    """``discord.server_actions`` allowlist (comma string or YAML list), or ``None`` when
    unrestricted. Unknown names are dropped with a warning."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); allowing all actions.", exc)
        return None
    raw = (cfg.get("discord") or {}).get("server_actions")
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        raw = raw.split(",")
    elif not isinstance(raw, (list, tuple)):
        logger.warning("discord.server_actions: unexpected type %s; ignoring.", type(raw).__name__)
        return None
    names = [str(n).strip() for n in raw if str(n).strip()]
    invalid = [n for n in names if n not in _ACTIONS]
    if invalid:
        logger.warning(
            "discord.server_actions: unknown action(s) ignored: %s. Known: %s",
            ", ".join(invalid), ", ".join(_ACTIONS.keys()))
    return [n for n in names if n in _ACTIONS]


def _available_actions(caps: dict[str, Any], allowlist: Optional[list[str]]) -> list[str]:
    """Visible actions from intents + config allowlist, in :data:`_ACTIONS` order."""
    members_ok = caps.get("has_members_intent", True)
    return [
        name for name in _ACTIONS
        if (members_ok or name not in _INTENT_GATED_MEMBERS) and (allowlist is None or name in allowlist)]


# ── schema construction ──────────────────────────────────────────────────────
_TOOL_DESCRIPTIONS = {
    "discord_admin": (
        "Manage a Discord server via the REST API.",
        "Call list_guilds first to discover guild_ids, then list_channels for "
        "channel_ids. Runtime errors will tell you if the bot lacks a specific "
        "per-guild permission (e.g. MANAGE_ROLES for add_role).",
    ),
    "discord": (
        "Read and participate in a Discord server.",
        "Use the channel_id from the current conversation context. "
        "Use search_members to look up user IDs by name prefix.",
    ),
}

_SCHEMA_PROPERTIES: dict[str, Any] = {
    "guild_id": {"type": "string", "description": "Discord server (guild) ID."},
    "channel_id": {"type": "string", "description": "Discord channel ID."},
    "user_id": {"type": "string", "description": "Discord user ID."},
    "role_id": {"type": "string", "description": "Discord role ID."},
    "message_id": {"type": "string", "description": "Discord message ID."},
    "query": {"type": "string", "description": "Member prefix (search_members) or case-insensitive forum-post title substring (list_forum_posts)."},
    "name": {"type": "string", "description": "New thread/forum post name (maximum 100 characters for create_forum_post)."},
    "content": {"type": "string", "description": "Message text (create_forum_post or post_message); longer text is split into 2000-character messages."},
    "applied_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5, "description": "Optional forum tag IDs (create_forum_post); required when the forum channel enforces REQUIRE_TAG."},
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Max results (default 50). Applies to fetch_messages, search_members, list_forum_posts.",
    },
    "before": {"type": "string", "description": "Snowflake ID for reverse pagination (fetch_messages)."},
    "after": {"type": "string", "description": "Snowflake ID for forward pagination (fetch_messages)."},
    "auto_archive_duration": {
        "type": "integer",
        "enum": [60, 1440, 4320, 10080],
        "description": "Thread archive duration in minutes (create_thread, default 1440).",
    },
}

_CONTENT_NOTE = (
    "\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. "
    "{names} will return message metadata (author, "
    "timestamps, attachments, reactions, pin state) but `content` will be "
    "empty for messages not sent as a direct mention to the bot or in DMs. "
    "Enable the intent in the Discord Developer Portal to see all content."
)


def _build_schema(
    actions: list[str], caps: Optional[dict[str, Any]] = None, tool_name: str = "discord",
) -> Optional[dict[str, Any]]:
    """Tool schema for the filtered action list; ``None`` when empty (drop the tool)."""
    caps = caps or {}
    if not actions:
        return None
    manifest_block = "\n".join(
        f"  {name}{sig}  — {desc}" for name, _fn, sig, desc in _ACTION_MANIFEST if name in actions)
    content_note = ""
    affected_actions = {"fetch_messages", "list_pins"} & set(actions)
    if affected_actions and caps.get("detected") and caps.get("has_message_content") is False:
        content_note = _CONTENT_NOTE.format(names=" and ".join(sorted(affected_actions)))
    lead, guidance = _TOOL_DESCRIPTIONS.get(tool_name, _TOOL_DESCRIPTIONS["discord"])
    return {
        "name": tool_name,
        "description": f"{lead}\n\nAvailable actions:\n{manifest_block}\n\n{guidance}{content_note}",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": actions}, **_SCHEMA_PROPERTIES},
            "required": ["action"]}}


def _get_dynamic_schema(action_subset: dict[str, Any], tool_name: str) -> Optional[dict[str, Any]]:
    """Build a dynamic schema for *action_subset* filtered by intents + config."""
    token = _get_bot_token()
    if not token:
        return None
    caps = _detect_capabilities_nonblocking(token)
    actions = [a for a in _available_actions(caps, _load_allowed_actions_config()) if a in action_subset]
    return _build_schema(actions, caps, tool_name=tool_name) if actions else None


get_dynamic_schema_core = functools.partial(_get_dynamic_schema, _CORE_ACTIONS, "discord")
get_dynamic_schema_admin = functools.partial(_get_dynamic_schema, _ADMIN_ACTIONS, "discord_admin")


# ── 403 error enrichment ─────────────────────────────────────────────────────
_NO_MANAGE_MESSAGES = "Bot lacks MANAGE_MESSAGES permission in this channel"
_VIEW_HISTORY = "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
_ROLE_HIERARCHY = "Either the bot lacks MANAGE_ROLES, or the target role sits higher than the bot's highest role."

# Per-action guidance for a call-time 403 (per-guild permissions are never pre-checked).
_ACTION_403_HINT = {
    "pin_message": (
        f"{_NO_MANAGE_MESSAGES}. "
        "Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, or a per-channel overwrite."),
    "unpin_message": f"{_NO_MANAGE_MESSAGES}.",
    "delete_message": f"{_NO_MANAGE_MESSAGES}, or cannot view the channel/message.",
    "create_thread": "Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it.",
    "add_role": (
        f"{_ROLE_HIERARCHY} Roles can only be assigned below the bot's own position in the role hierarchy."),
    "remove_role": _ROLE_HIERARCHY,
    "fetch_messages": _VIEW_HISTORY,
    "list_pins": _VIEW_HISTORY,
    "channel_info": "Bot cannot view this channel (missing VIEW_CHANNEL).",
    "search_members": (
        "Likely missing the Server Members privileged intent — enable it in the Discord Developer Portal "
        "under your bot's settings."),
    "member_info": "Bot cannot see this guild member (missing Server Members intent or insufficient permissions)."}


def _enrich_403(action: str, body: str) -> str:
    """Return a user-friendly guidance string for a 403 on ``action``."""
    hint = _ACTION_403_HINT.get(action)
    base = f"Discord API 403 (forbidden) on '{action}'."
    return f"{base} {hint} (Raw: {body})" if hint else f"{base} (Raw: {body})"


def check_discord_tool_requirements() -> bool:
    """Tool is available only when a Discord bot token is configured."""
    return bool(_get_bot_token())


# ── handlers ─────────────────────────────────────────────────────────────────
_HANDLER_DEFAULTS = {
    "guild_id": "", "channel_id": "", "user_id": "", "role_id": "", "message_id": "", "query": "",
    "name": "", "content": "", "applied_tags": None, "limit": 50, "before": "", "after": "", "auto_archive_duration": 1440}


def _run_discord_action(action: str, valid_actions: dict[str, Any], tool_label: str, **params: Any) -> str:
    """Shared handler logic for both discord tools (``params`` default per :data:`_HANDLER_DEFAULTS`)."""
    token = _get_bot_token()
    if not token:
        return tool_error("DISCORD_BOT_TOKEN not configured.")
    action_fn = valid_actions.get(action)
    if not action_fn:
        return tool_error(f"Unknown action: {action}", available_actions=list(valid_actions.keys()))
    # Config-level allowlist gate (defense in depth): a stale cached schema from a prior
    # config must not let denied actions through.
    allowlist = _load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return tool_error(
            f"Action '{action}' is disabled by config (discord.server_actions). "
            f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}")
    kwargs = {k: params.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}
    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not kwargs.get(p)]
    if missing:
        return tool_error(f"Missing required parameters for '{action}': {', '.join(missing)}")
    try:
        return action_fn(token=token, **kwargs)
    except ValueError as e:
        return tool_error(str(e))
    except DiscordAPIError as e:
        logger.warning("Discord API error in %s action '%s': %s", tool_label, action, e)
        return tool_error(_enrich_403(action, e.body) if e.status == 403 else str(e))
    except Exception as e:
        logger.exception("Unexpected error in %s action '%s'", tool_label, action)
        return tool_error(f"Unexpected error: {e}")


# ``discord`` = core participation trio; ``discord_admin`` = server management.
discord_core = functools.partial(_run_discord_action, valid_actions=_CORE_ACTIONS, tool_label="discord")
discord_admin_handler = functools.partial(
    _run_discord_action, valid_actions=_ADMIN_ACTIONS, tool_label="discord_admin")


# Static (un-detected) schemas at import; the intent/config-filtered ones come from
# get_dynamic_schema_core/admin via model_tools' dynamic schema overrides.
for _name, _actions, _handler in (
    ("discord", _CORE_ACTIONS, discord_core), ("discord_admin", _ADMIN_ACTIONS, discord_admin_handler),
):
    registry.register(
        name=_name,
        toolset=_name,
        schema=_build_schema(list(_actions), caps={"detected": False}, tool_name=_name),
        handler=lambda args, _h=_handler, **kw: _h(**{"action": "", **args}),
        check_fn=check_discord_tool_requirements,
        requires_env=["DISCORD_BOT_TOKEN"])
