"""The Discord exec-approval card's button view.

Built by ``build_exec_approval_view`` from ``adapter._define_discord_view_classes``, because discord.py can
be installed lazily after the adapter module is imported.
"""
import asyncio
import logging
from typing import Optional

from agent.i18n import t
from plugins.platforms.discord.adapter_config_readers import (
    _read_discord_prompt_timeout, _resolve_exec_approval_admin_gate,
)

logger = logging.getLogger(__name__)


def build_exec_approval_view(discord, base, *, unauthorized):
    """Return the ``ExecApprovalView`` class on ``base`` (the adapter's shared ``_HermesView``)."""
    class ExecApprovalView(base):
        """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
        Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

        def __init__(
            self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
            require_admin: bool = False, admin_user_ids: Optional[set] = None,
            allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
            request_id: Optional[str] = None, allow_once: bool = True, approval_extra: Optional[dict] = None,
        ):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_read_discord_prompt_timeout())
            self.session_key = session_key
            self.request_id = request_id
            self.pending_retire: Optional[str] = None  # settle reason that arrived before the card was sent
            self._backend_settled = False
            self._retire_lock = asyncio.Lock()
            self.require_admin = require_admin
            self.admin_user_ids = {str(a).strip() for a in (admin_user_ids or set()) if str(a).strip()}
            # The platform block: its admins answer any approval even with require_admin off (approval_actor).
            self.approval_extra = approval_extra
            self._localize_buttons(
                allow_once="gateway.exec_approval.action_once", allow_session="gateway.exec_approval.action_session",
                allow_always="gateway.exec_approval.action_always", deny="gateway.exec_approval.action_deny")
            if smart_denied or not allow_session:
                self.remove_item(self.allow_session)
                self.remove_item(self.allow_always)
            elif not allow_permanent:
                self.remove_item(self.allow_always)
            if not allow_once:  # Deny-only card: the command could not be shown in full
                self.remove_item(self.allow_once)
            if request_id:
                # Stable ids let a press after a reconnect find its request (see the adapter's listener).
                for attr, choice in (("allow_once", "once"), ("allow_session", "session"),
                                     ("allow_always", "always"), ("deny", "deny")):
                    item = getattr(self, attr, None)
                    if item is not None and hasattr(item, "custom_id"):
                        item.custom_id = f"hermes:exec:{request_id}:{choice}"

        def _admin_gate(self) -> tuple[bool, set, Optional[dict]]:
            """``(require_admin, admin_user_ids, platform extra)`` read at press time: from the adopting adapter for an
            adopted card, else this view's own copy."""
            source = self.policy_source
            if source is None:
                return self.require_admin, self.admin_user_ids, self.approval_extra
            require_admin, admin_ids = _resolve_exec_approval_admin_gate(getattr(source.config, "extra", None))
            return require_admin, admin_ids, getattr(source.config, "extra", None)

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Base admission always required; with ``require_admin`` the clicker must
            also be an admin. Fails closed (logged once) when no admins are configured."""
            if not super()._check_auth(interaction):
                return False
            require_admin, admin_user_ids, _extra = self._admin_gate()
            if not require_admin:
                return True
            user = getattr(interaction, "user", None)
            try:
                uid = str(getattr(user, "id", "") or "")
            except Exception:
                uid = ""
            if uid and uid in admin_user_ids:
                return True
            if not admin_user_ids:
                logger.warning(
                    "[Discord] require_admin_for_exec_approval is enabled but "
                    "no admins are configured (allow_admin_from is empty) — "
                    "exec approval buttons are disabled for everyone. Add "
                    "admin user IDs under the discord platform's "
                    "allow_admin_from, or disable the toggle."
                )
            return False

        async def on_timeout(self):
            # The card can no longer be answered, so settle the request now as withdrawn: the agent fails
            # closed at once instead of waiting for /stop. A withdrawal, not a user deny.
            await self.retire("timeout", backend_settled=False)
            if self.request_id and self.session_key:
                from tools.approval import withdraw_gateway_approval
                withdraw_gateway_approval(self.session_key, self.request_id,
                                          "the approval card timed out before anyone answered it")

        async def retire(self, reason="resolved", *, backend_settled=True):
            """Disable a card when its particular backend request is no longer pending."""
            self.stop()
            # A UI timeout can precede a text answer. Serialize their edits, and
            # never let a queued UI timeout overwrite a backend settlement.
            async with self._retire_lock:
                if self._backend_settled:
                    return
                self._backend_settled = backend_settled
                self.resolved = True
                self._disable_all()
                footer = t({
                    "resolved": "platform.discord.approval.already_resolved",
                    "timeout": "platform.discord.prompt.expired_footer",
                    "gone": "platform.discord.approval.no_longer_pending",
                }.get(reason, "platform.discord.slash_confirm.resolved_cancel"))
                # The timeout notice edits only the content, so it never restores the controls.
                await self._expire_embed(footer, keep_controls=False)

        async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label_key: str):
            """Resolve the approval via the gateway approval queue and update the embed."""
            if not await self._gate(
                interaction, resolved_msg=t("platform.discord.approval.already_resolved"),
                unauth_msg=unauthorized(),
            ):
                return
            from gateway.approval_actor import approval_refusal, refusal_text
            # Admins of the approval's own chat scope (``extra``), as ``/approve``: never the DM-only
            # ``admin_user_ids``, which stays this view's own require_admin click gate.
            extra = self._admin_gate()[2] or {}
            if requester := approval_refusal(self.session_key, getattr(interaction.user, "id", ""),
                                             extra=extra, request_id=self.request_id):
                return await interaction.response.send_message(refusal_text(requester), ephemeral=True)
            label = t(label_key)
            self.resolved = True
            self._backend_settled = True
            # Unblock the waiting agent thread FIRST. A click after the approval
            # wait timed out (count == 0) must not claim "Approved".
            try:
                from tools.approval import resolve_gateway_approval_card
                count = resolve_gateway_approval_card(
                    self.session_key, choice, self.request_id, actor_id=getattr(interaction.user, "id", ""),
                    extra=extra)
                logger.info(
                    "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                logger.error("Failed to resolve gateway approval from button: %s", exc, exc_info=True)
                # Nothing was recorded and the request may still be pending: tell only the clicker
                # and keep the card live for a retry. A later click, the request's observer, or
                # (with no request id) the view timeout does the final edit.
                self.resolved = False
                self._backend_settled = False
                await interaction.response.send_message(
                    t("platform.discord.approval.click_failed"), ephemeral=True)
                return
            if not count:
                self._backend_settled = False
                if self.request_id:
                    # The request already settled through another path, whose observer owns this
                    # card; editing it here too could land last and hide the real outcome.
                    await interaction.response.send_message(
                        t("platform.discord.approval.no_longer_pending"), ephemeral=True)
                    return
                color = discord.Color.dark_grey()
                label = t("platform.discord.approval.expired")
            try:
                await self._finalize_embed(
                    interaction, color,
                    t("platform.discord.approval.by_user", label=label, user=interaction.user.display_name) if count else label)
            except Exception:
                if count:
                    # The answer is recorded but this click's edit failed, and the observer skips cards
                    # whose click owns the edit: stamp the card through its message instead.
                    self._backend_settled = False
                    await self.retire("resolved")
                raise

        async def press(self, interaction: discord.Interaction, custom_id: str) -> bool:
            """Run the button whose ``custom_id`` is ``custom_id``. A press that reached a dead client's view
            routes here. Returns False, acting on nothing, when this card has no such live button: it was
            removed for this card, it is disabled, or the id is unknown."""
            button = next((child for child in self.children if getattr(child, "custom_id", None) == custom_id), None)
            if button is None or getattr(button, "disabled", False):
                return False
            choice = custom_id.rpartition(":")[2]
            labels = {
                "once": (discord.Color.green(), "platform.discord.approval.resolved_once"),
                "session": (discord.Color.blue(), "platform.discord.approval.resolved_session"),
                "always": (discord.Color.purple(), "platform.discord.approval.resolved_always"),
                "deny": (discord.Color.red(), "platform.discord.approval.resolved_deny"),
            }
            color, label_key = labels[choice]
            await self._resolve(interaction, choice, color, label_key)
            return True

        # Decorator labels are placeholders; ``_localize_buttons`` in __init__ sets the real text.
        @discord.ui.button(label="Allow Once", style=discord.ButtonStyle.green)
        async def allow_once(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "once", discord.Color.green(), "platform.discord.approval.resolved_once")

        @discord.ui.button(label="Allow Session", style=discord.ButtonStyle.grey)
        async def allow_session(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "session", discord.Color.blue(), "platform.discord.approval.resolved_session")

        @discord.ui.button(label="Always Allow", style=discord.ButtonStyle.blurple)
        async def allow_always(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "always", discord.Color.purple(), "platform.discord.approval.resolved_always")

        async def resolve_unmatched_press(self, interaction: discord.Interaction, choice: str) -> bool:
            """Resolve this card's still-pending request for a press no live view routes. Runs the
            live view's own gates: base admission, ``require_admin``, the approval's authority, then the Deny-only
            check inside ``resolve_gateway_approval_card``. True when the press was answered or settled the request."""
            from tools.approval import resolve_gateway_approval_card
            from gateway.approval_actor import approval_refusal, refusal_text
            # The card's own request is the only one resolved: resolve_gateway_approval_card finds nothing elsewhere.
            if not self._check_auth(interaction):
                await interaction.response.send_message(unauthorized(), ephemeral=True)
                return True
            user_id = getattr(interaction.user, "id", "")
            extra = self._admin_gate()[2] or {}
            if requester := approval_refusal(self.session_key, user_id, extra=extra, request_id=self.request_id):
                await interaction.response.send_message(refusal_text(requester), ephemeral=True)
                return True
            offered = {getattr(item, "custom_id", "").rpartition(":")[2] for item in self.children}
            if choice not in offered:
                logger.info("[Discord] unmatched press dropped, choice not offered: request_id=%s choice=%s",
                            self.request_id, choice)
                return False
            count = resolve_gateway_approval_card(
                self.session_key, choice, self.request_id, actor_id=user_id, extra=extra)
            if not count:
                return False
            logger.info("[Discord] unmatched press resolved pending approval: request_id=%s choice=%s",
                        self.request_id, choice)
            # The card is already finished, so this press only answers the clicker; the settle observer edits the card.
            label_key = {"once": "platform.discord.approval.resolved_once",
                         "session": "platform.discord.approval.resolved_session",
                         "always": "platform.discord.approval.resolved_always"}.get(choice, "platform.discord.approval.resolved_deny")
            await interaction.response.send_message(t(label_key), ephemeral=True)
            return True

        @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
        async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "deny", discord.Color.red(), "platform.discord.approval.resolved_deny")

    return ExecApprovalView
