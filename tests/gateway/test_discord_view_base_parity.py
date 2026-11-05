"""Parity tests for the shared ``_HermesView`` base behind the Discord component views.

Every view must keep its own user-visible rejection strings and the shared
timeout behaviour (buttons disabled, embed greyed with the expiry footer).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.adapter import (
    ChoicePickerView,
    ClarifyChoiceView,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
)


def _interaction(user_id=1):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="alice", roles=[]),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock(), defer=AsyncMock()),
        message=SimpleNamespace(embeds=[]),
        data={"values": ["x"]},
        channel_id=5,
    )


async def _noop(*_a, **_k):
    return ""


def _views():
    return {
        "exec": ExecApprovalView(session_key="s", allowed_user_ids=set()),
        "slash": SlashConfirmView(session_key="s", confirm_id="c", allowed_user_ids=set()),
        "update": UpdatePromptView(session_key="s", allowed_user_ids=set()),
        "clarify": ClarifyChoiceView(choices=["a"], clarify_id="c", allowed_user_ids=set()),
        "model": ModelPickerView(
            providers=[], current_model="m", current_provider="p", session_key="s",
            on_model_selected=_noop, allowed_user_ids=set(),
        ),
        "choice": ChoicePickerView(choices=[{"value": "v"}], on_choice_selected=_noop, allowed_user_ids=set()),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,call",
    [
        ("exec", lambda v, i: v._resolve(i, "once", None, "x")),
        ("slash", lambda v, i: v._resolve(i, "once", None, "x")),
        ("update", lambda v, i: v._respond(i, "y", None, "x")),
        ("clarify", lambda v, i: v._resolve_choice(i, 0, "a")),
        ("clarify", lambda v, i: v._on_other(i)),
        ("model", lambda v, i: v._on_provider_selected(i)),
        ("model", lambda v, i: v._on_back(i)),
        ("choice", lambda v, i: v._on_select(i)),
    ],
)
async def test_unauthorized_click_uses_the_shared_notice(monkeypatch, name, call):
    """Every view refuses a stranger with the ONE gateway-wide sentence that names the fix command."""
    from gateway.platforms.base import unauthorized_action_notice

    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    view = _views()[name]
    interaction = _interaction()
    await call(view, interaction)
    expected = unauthorized_action_notice("discord")
    interaction.response.send_message.assert_awaited_once_with(expected, ephemeral=True)
    interaction.response.edit_message.assert_not_called()
    assert view.resolved is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["exec", "slash", "update", "clarify"])
async def test_on_timeout_disables_and_greys_embed(name):
    view = _views()[name]
    embed = SimpleNamespace(color=None, set_footer=lambda *, text: setattr(embed, "footer", text))
    msg = SimpleNamespace(embeds=[embed], edit=AsyncMock())
    view._message = msg
    await view.on_timeout()
    assert view.resolved is True
    assert all(child.disabled for child in view.children)
    assert embed.footer
    msg.edit.assert_awaited_once_with(embed=embed, view=view)


@pytest.mark.asyncio
async def test_on_timeout_without_message_is_safe():
    for view in _views().values():
        await view.on_timeout()


def test_one_day_prompt_timeout_never_outlives_backing_waits(monkeypatch):
    """Only exec approvals get the full day; other cards expire with the request they answer."""
    import hermes_cli.config
    cfg = {"approvals": {"discord_prompt_timeout": 86400}, "agent": {"clarify_timeout": 3600}}
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: cfg)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    views = _views()
    assert views["exec"].timeout == 86400
    assert views["slash"].timeout == 300
    assert views["update"].timeout == 300
    assert views["clarify"].timeout == 3600


@pytest.mark.asyncio
async def test_resolved_exec_view_is_stopped():
    view = ExecApprovalView(session_key="s", allowed_user_ids=set())
    await view._finalize_embed(_interaction(), color=None, footer="done")
    assert view.resolved is True
    assert getattr(view, "stopped", False) is True


def test_short_backing_wait_beats_the_view_floor(monkeypatch):
    import hermes_cli.config
    cfg = {"approvals": {"discord_prompt_timeout": 86400}, "clarify": {"timeout": 5}}
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: cfg)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    assert _views()["clarify"].timeout == 5


@pytest.mark.asyncio
async def test_old_update_card_cannot_answer_a_newer_prompt(tmp_path, monkeypatch):
    import json
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    prompt = home / ".update_prompt.json"
    prompt.write_text(json.dumps({"prompt": "old?", "id": "old"}))
    old_card = UpdatePromptView(session_key="s", allowed_user_ids={"1"})
    prompt.write_text(json.dumps({"prompt": "new?", "id": "new"}))
    new_card = UpdatePromptView(session_key="s", allowed_user_ids={"1"})

    stale = _interaction()
    await old_card._respond(stale, "y", None, "platform.discord.prompt.affirm")
    assert not (home / ".update_response").exists()
    assert old_card.resolved is True
    stale.response.send_message.assert_awaited_once()

    await new_card._respond(_interaction(), "n", None, "platform.discord.prompt.negate")
    assert (home / ".update_response").read_text() == "prompt-id:new\nn"


@pytest.mark.asyncio
async def test_clarify_view_stays_live_when_its_edit_fails():
    view = ClarifyChoiceView(choices=["a"], clarify_id="c", allowed_user_ids=set())
    interaction = _interaction()
    interaction.response.edit_message = AsyncMock(side_effect=RuntimeError("gone"))
    await view._finish(interaction, None, "done", log_edit_failure=False)
    assert view.resolved is True
    assert getattr(view, "stopped", False) is False


@pytest.mark.asyncio
async def test_update_answer_names_the_prompt_it_answers(tmp_path, monkeypatch):
    import json
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".update_prompt.json").write_text(json.dumps({"prompt": "p?", "id": "abc"}))
    card = UpdatePromptView(session_key="s", allowed_user_ids={"1"})
    await card._respond(_interaction(), "y", None, "platform.discord.prompt.affirm")
    assert (home / ".update_response").read_text() == "prompt-id:abc\ny"
