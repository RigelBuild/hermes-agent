"""Platform scope rules for configured toolsets."""

from typing import Set


# Toolsets without a restriction entry are available on every platform. ``discord`` also runs in
# cron so scheduled jobs can post into a forum; its posting actions stay limited to
# ``discord.post_channels``. ``discord_admin`` (pins, roles, server listing) stays Discord-only.
_TOOLSET_PLATFORM_RESTRICTIONS = {"discord": {"discord", "cron"}, "discord_admin": {"discord"}}


def toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return whether ``ts_key`` is available on ``platform``."""
    allowed: set[str] | None = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed
