"""RIG-10585: a fallback entry's own ``reasoning_effort`` is sent on the call it serves.

The main route asks for ``medium``. The primary is usage-limited, so the call falls back to a
configured entry. An entry that declares ``reasoning_effort: low`` must reach the wire as ``low``.
An entry without the field keeps the main value (today's behaviour).
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import call_llm

MAIN_URL = "https://main-gateway.example.test/v1"
FALLBACK_URL = "https://fallback-gateway.example.test/v1"
MAIN_EFFORT = {"enabled": True, "effort": "medium"}


@pytest.fixture(autouse=True)
def _fresh_state():
    ac._reset_aux_unhealthy_cache()
    ac._client_cache.clear()
    yield
    ac._reset_aux_unhealthy_cache()
    ac._client_cache.clear()


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.usage = None
    return resp


def _effort_level(reasoning) -> str | None:
    """The effort level on the wire, whether sent as a dict or a bare level string."""
    if isinstance(reasoning, dict):
        return reasoning.get("effort")
    return reasoning


def _run_fallback_call(*, chain: list) -> list:
    """Run one call whose main route is usage-limited. Return (base_url, effort level) per wire send."""
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(**kwargs):
            extra = kwargs.get("extra_body") or {}
            reasoning = extra.get("reasoning", kwargs.get("reasoning_effort"))
            sent.append((base_url, _effort_level(reasoning)))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    runtime = {
        "provider": "custom",
        "model": "gpt-6-luna",
        "base_url": MAIN_URL,
        "api_key": "sk-main",
    }
    ac._mark_provider_unhealthy(
        "custom",
        base_url=MAIN_URL,
        model="gpt-6-luna",
        reason=aux_errors.USAGE_LIMIT_REASON,
    )
    with (
        patch("agent.auxiliary_client._create_openai_client", side_effect=construct),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "gpt-6-luna", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": chain},
        ),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
    ):
        call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "hi"}],
            main_runtime=runtime,
            reasoning_config=MAIN_EFFORT,
        )
    return sent


def test_fallback_entry_reasoning_effort_low_is_sent(monkeypatch):
    """An entry with ``reasoning_effort: low`` serves the call: the wire carries ``low``, not the main ``medium``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chain = [
        {
            "provider": "custom",
            "model": "claude-haiku-5-5",
            "base_url": FALLBACK_URL,
            "api_key": "sk-fallback",
            "reasoning_effort": "low",
        }
    ]
    sent = _run_fallback_call(chain=chain)
    assert sent == [(FALLBACK_URL, "low")]


def test_fallback_entry_without_reasoning_effort_keeps_main_value(monkeypatch):
    """An entry without ``reasoning_effort`` keeps today's behaviour: the main ``medium`` is sent."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chain = [
        {
            "provider": "custom",
            "model": "claude-haiku-5-5",
            "base_url": FALLBACK_URL,
            "api_key": "sk-fallback",
        }
    ]
    sent = _run_fallback_call(chain=chain)
    assert sent == [(FALLBACK_URL, "medium")]
