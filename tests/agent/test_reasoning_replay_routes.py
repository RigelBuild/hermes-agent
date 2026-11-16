"""Stored reasoning is replayed to every chat-completions route that can read it, on the
carriers that route reads; strict schemas never receive a reasoning key; a route that still
rejects one is learned per (provider, host, model) and stripped on a single retry."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.message_sanitization import (
    apply_reasoning_content_policy, reapply_reasoning_echo, reasoning_replay_route,
    rejected_reasoning_carriers,
)
from agent.transports.chat_completions import ChatCompletionsTransport

_DETAILS = [{"type": "reasoning.text", "text": "thought"}]
_TURN = {"role": "assistant", "content": "", "reasoning": "thought", "reasoning_content": "thought",
         "reasoning_details": _DETAILS,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"},
                         "extra_content": {"google": {"thought_signature": "sig"}}}]}
_BARE = {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
                                                             "function": {"name": "t", "arguments": "{}"}}]}


def _wire(provider, base_url, model="m", rejected=()):
    route = reasoning_replay_route("chat_completions", provider, model, base_url, rejected=rejected)
    out = []
    for source in (_TURN, _BARE):
        api_msg = deepcopy(source)
        apply_reasoning_content_policy(source, api_msg, route.pad, route.carriers)
        out.append(api_msg)
    return ChatCompletionsTransport().convert_messages(out, model=model, base_url=base_url, provider=provider)


@pytest.mark.parametrize("provider,base_url,model,expected", [
    # strict schemas: nothing, not even the must-echo pad (#45655, #70233)
    ("mistral", "https://api.mistral.ai/v1", "deepseek-r1", set()),
    ("custom", "https://api.groq.com/openai/v1", "m", set()),
    ("custom", "https://api.cerebras.ai/v1", "m", set()),
    ("fireworks", "https://api.fireworks.ai/inference/v1", "m", {"reasoning_content"}),
    # OpenRouter-format gateways, incl. the Portal (#118182 does not reproduce)
    ("nous", "https://inference-api.nousresearch.com/v1", "z-ai/glm-5.3",
     {"reasoning_content", "reasoning", "reasoning_details"}),
    ("kilocode", "https://api.kilo.ai/api/gateway", "m", {"reasoning_content", "reasoning", "reasoning_details"}),
    ("novita", "https://api.novita.ai/openai/v1", "m", {"reasoning_content", "reasoning_details"}),
    ("ollama-cloud", "https://ollama.com/v1", "m", {"reasoning_content", "reasoning"}),
    ("custom", "http://127.0.0.1:8080/v1", "m", {"reasoning_content", "reasoning"}),
    # everyone else: the de-facto standard alias
    ("zai", "https://api.z.ai/api/paas/v4", "glm-5.3", {"reasoning_content"}),
    ("openai-api", "https://api.openai.com/v1", "gpt-5", {"reasoning_content"}),
    # Anthropic refuses replayed reasoning text sent through this gateway, so only Claude models on
    # the gateway host are stripped. Other models served there keep the carriers and pad they had.
    ("litellm", "https://llm-gateway.internal.rigel.build/v1", "claude-opus-5-5", set()),
    ("custom", "https://llm-gateway.internal.rigel.build/v1", "claude-opus-5-5", set()),
    ("custom", "https://llm-gateway.internal.rigel.build/v1", "openrouter/deepseek/deepseek-v4", {"reasoning_content", "reasoning"}),
    ("custom", "https://llm-gateway.internal.rigel.build/v1", "gpt-6-luna", {"reasoning_content", "reasoning"}),
    (None, "https://llm-gateway.internal.rigel.build/v1", "claude-opus-5-5", set()),
])
def test_each_route_gets_exactly_the_carriers_it_reads(provider, base_url, model, expected):
    turn, bare = _wire(provider, base_url, model)
    assert {k for k in ("reasoning_content", "reasoning", "reasoning_details") if k in turn} == expected
    if "reasoning_content" in expected:
        assert turn["reasoning_content"] == "thought"
    assert "extra_content" not in turn["tool_calls"][0]  # Gemini routes only
    # A reasoning-less turn is padded only when the route reads reasoning_content and the model is
    # must-echo (DeepSeek/Kimi/MiMo). A strict route never gets the pad (#45655).
    route = reasoning_replay_route("chat_completions", provider, model, base_url)
    assert (bare.get("reasoning_content") == " ") is (route.pad and "reasoning_content" in route.carriers)
    assert _TURN["reasoning_details"] is _DETAILS and "reasoning" in _TURN  # history untouched


def test_must_echo_pads_and_aliased_gemini_keeps_its_signature():
    _, bare = _wire("deepseek", "https://api.deepseek.com/v1", "deepseek-v4-flash")
    assert bare["reasoning_content"] == " "  # DeepSeek 400s without it; "" is rejected too
    gemini_alias, _ = _wire("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "my-alias")
    assert gemini_alias["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}


@pytest.mark.parametrize("provider", ["custom", "litellm"])
def test_litellm_gateway_sends_no_replayed_reasoning_on_any_assistant_turn(provider):
    """The final chat_completions body for the Rigel LiteLLM gateway carries no reasoning key on
    any Claude assistant turn. The runtime names a configured LiteLLM entry ``custom``, so that
    provider runs first. The check drives the copy helper and the route the agent would use."""
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api
    from agent.reasoning_carriers import shape_wire_carriers

    url = "https://llm-gateway.internal.rigel.build/v1"
    model = "claude-opus-5-5"
    agent = SimpleNamespace(provider=provider, model=model, base_url=url, api_mode="chat_completions")
    agent._reasoning_replay_route = lambda: reasoning_replay_route(
        agent.api_mode, agent.provider, agent.model, agent.base_url)
    assert agent._reasoning_replay_route().carriers == frozenset()
    stored = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {**_TURN, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    api = []
    for source in stored:
        msg = deepcopy(source)
        copy_reasoning_content_for_api(agent, source, msg)
        api.append(msg)
    wire = shape_wire_carriers(api, model=model, base_url=url)
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=model, messages=wire, tools=None, base_url=url, provider_name=provider,
        supports_reasoning=True, reasoning_config={"enabled": True, "effort": "high"},
    )
    assistant = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
    assert assistant, "fixture must contain an assistant turn"
    for msg in assistant:
        assert not {"reasoning", "reasoning_content", "reasoning_details"} & set(msg)
    assert _TURN["reasoning"] == "thought"  # stored history untouched


def test_litellm_gateway_profile_cannot_reintroduce_native_details():
    """Step 4: a profile that declares native reasoning_details must not add the carrier back on this route."""
    from unittest.mock import patch
    from types import SimpleNamespace

    url = "https://llm-gateway.internal.rigel.build/v1"
    declared = SimpleNamespace(native_reasoning_details_type="litellm.native_assistant")
    with patch("providers.get_provider_profile", return_value=declared):
        route = reasoning_replay_route("chat_completions", "litellm", "claude-opus-5-5", url)
    assert "reasoning_details" not in route.carriers
    assert route.carriers == frozenset()


class _Rejection(Exception):
    status_code = 400

    def __init__(self, body):
        super().__init__(body)
        self.body = body


def _agent(model, db=None):
    return SimpleNamespace(
        provider="nous", model=model, base_url="https://inference-api.nousresearch.com/v1",
        api_mode="chat_completions", _force_ascii_payload=False, _image_rejecting_models=set(),
        log_prefix="", _vprint=lambda *a, **k: None, session_id="s1" if db else None, _session_db=db,
    )


def _recover(agent, body, sent):
    from agent.turn_recovery import recover_before_classification

    return recover_before_classification(
        agent, _Rejection(body), messages=[], api_messages=sent,
        api_kwargs={"messages": sent}, active_system_prompt="",
    )[0]


def test_named_field_rejection_strips_once_per_model_and_never_loops(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="z-ai/glm-5.3-flash:US")
    lane = _agent("z-ai/glm-5.3-flash:US", db)
    sent = [{"role": "user", "content": "hi"}, deepcopy(_TURN)]
    sent[1].pop("tool_calls")
    body = "Extra inputs are not permitted, field: 'messages[1].reasoning_details'"
    # A generic upstream error (content filter behind "Provider returned error") is not a rejection.
    assert _recover(lane, "Provider returned error", sent) is False
    assert not rejected_reasoning_carriers(lane)

    assert _recover(lane, body, sent) is True
    rejected = lane._reasoning_rejecting_routes[("nous", "inference-api.nousresearch.com", "z-ai/glm-5.3-flash:US")]
    assert rejected == {"reasoning", "reasoning_details"}
    # The retry re-shapes the same api_messages for the narrowed route: reasoning_content survives.
    route = reasoning_replay_route("chat_completions", "nous", lane.model, lane.base_url, rejected=rejected)
    reapply_reasoning_echo(sent, route.pad, route.carriers)
    assert sent[1]["reasoning_content"] == "thought"
    assert "reasoning" not in sent[1] and "reasoning_details" not in sent[1]
    # Same rejection again: nothing new to strip -> normal error path, no loop.
    assert _recover(lane, body, [{"role": "assistant", "reasoning_details": _DETAILS}]) is False
    # A resumed process (fresh agent, same session) never re-learns it with another 400.
    assert rejected_reasoning_carriers(_agent(lane.model, db)) == {"reasoning", "reasoning_details"}
    # A sibling model on the same gateway keeps every carrier.
    sibling = reasoning_replay_route("chat_completions", "nous", "moonshotai/kimi-k3", lane.base_url)
    assert "reasoning_details" in sibling.carriers


def test_route_shaping_keeps_private_native_carrier_for_the_transport():
    """A resumed Gemini/Copilot row carries its signature only as a private reasoning_details record;
    route shaping must leave it for ``shape_wire_carriers`` even on routes that never read the array."""
    from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route
    from agent.reasoning_carriers import CARRIER_TYPE, shape_wire_carriers

    base = "https://generativelanguage.googleapis.com/v1beta/openai"
    record = {"type": CARRIER_TYPE, "extra_content": {"google": {"thought_signature": "SIG"}}}
    row = {"role": "assistant", "content": "391", "reasoning_details": [{"type": "reasoning.text", "text": "t"}, record]}
    route = reasoning_replay_route("chat_completions", "gemini", "gemini-3.5-pro", base)
    api = dict(row)
    apply_reasoning_content_policy(row, api, route.pad, route.carriers)
    wire = shape_wire_carriers([api], model="gemini-3.5-pro", base_url=base)[0]
    assert wire["extra_content"] == record["extra_content"]
    assert "reasoning_details" not in wire


def test_value_complaint_never_strips_reasoning_content_from_a_must_echo_route():
    from types import SimpleNamespace
    from agent.message_sanitization import record_reasoning_field_rejection

    agent = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
                            api_mode="chat_completions", _reasoning_rejecting_routes={}, session_id=None)
    sent = [{"role": "assistant", "content": "", "reasoning_content": " ", "tool_calls": [{"id": "c"}]}]
    for body in ("reasoning_content is not allowed to be empty",
                 "Extra inputs are not permitted, field: 'messages[1].reasoning_content'"):
        assert record_reasoning_field_rejection(agent, body, sent) == frozenset()
    assert agent._reasoning_rejecting_routes == {}
