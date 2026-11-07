"""Platform scope rules for configured toolsets."""

from typing import Set


# Toolsets without a restriction entry are available on every platform. Core discord permits cron and cotal;
# cotal calls are mesh-scoped, and discord_admin stays Discord-only.
_TOOLSET_PLATFORM_RESTRICTIONS = {"discord": {"discord", "cron", "cotal"}, "discord_admin": {"discord"}}


def toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return whether ``ts_key`` is available on ``platform``."""
    allowed: set[str] | None = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed
