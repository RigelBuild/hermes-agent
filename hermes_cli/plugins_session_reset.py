"""Host routing for ``PluginContext.reset_session``."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("hermes_cli.plugins")


def reset_plugin_session(ctx: Any, session_id: str, reason: str) -> bool:
    """Ask the live gateway to schedule a reset of the identified session."""
    manager = ctx._manager
    if not ctx._gateway_injection_allowed():
        logger.warning("reset_session: gateway reset denied for plugin %s; set "
                       "plugins.entries.%s.allow_gateway_injection: true to allow it",
                       ctx.plugin_id, ctx.plugin_id)
        return False
    if not manager.has_gateway_session_resetter:
        logger.warning("reset_session: no live gateway is available")
        return False
    try:
        return bool(manager.reset_gateway_session(
            session_id=session_id, reason=reason, plugin_id=ctx.plugin_id, plugin_home=manager.home_path,
        ))
    except Exception:
        logger.warning("reset_session: gateway scheduling failed for plugin %s", ctx.plugin_id, exc_info=True)
        return False
