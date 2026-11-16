"""A usage limit on one model quarantines that model, not the whole gateway (RIG-10475).

A Codex ``usage_limit_reached`` 429 on ``gpt-6-luna`` served through a LiteLLM gateway must not
hide ``claude-haiku-5-5`` on that same gateway: the gateway serves several models, and the limit
belongs to one of them. The configured ``fallback_chain`` must reach the Haiku entry.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.auxiliary_client import call_llm

LITELLM_URL = "https://llm-gateway.internal.rigel.build/v1"


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def _codex_usage_limit_429() -> Exception:
    """The 429 body Codex returns once a ChatGPT-account usage window is spent."""
    exc = Exception(
        "Error code: 429 - {'error': {'type': 'usage_limit_reached', "
        "'message': 'The usage limit has been reached', 'code': 'usage_limit_reached'}}"
    )
    exc.status_code = 429
    return exc


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp


def _client(base_url: str, create):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = "sk-litellm"
    client.chat.completions.create = create
    return client


def test_usage_limit_on_one_model_still_runs_haiku_fallback_on_same_gateway():
    """Primary gpt-6-luna 429 usage_limit_reached; the claude-haiku-5-5 fallback on the same
    LiteLLM base URL must be called and succeed."""
    primary = _client(LITELLM_URL, MagicMock(side_effect=_codex_usage_limit_429()))
    haiku = _client(LITELLM_URL, MagicMock(return_value=_ok_response("HAIKU")))
    chain = [{"provider": "litellm", "model": "claude-haiku-5-5", "base_url": LITELLM_URL}]

    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("litellm", "gpt-6-luna", LITELLM_URL, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "gpt-6-luna")), \
         patch("agent.auxiliary_client._get_auxiliary_task_config",
               return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(haiku, "claude-haiku-5-5")), \
         patch("agent.auxiliary_client._recover_provider_pool", return_value=False), \
         patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")):
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "Reply OK"}])

    assert result.choices[0].message.content == "HAIKU"
    assert haiku.chat.completions.create.call_count == 1
    # The limit is scoped to gpt-6-luna; the gateway endpoint itself stays eligible.
    assert not ac._is_provider_unhealthy("litellm", LITELLM_URL, "claude-haiku-5-5")
    assert ac._is_provider_unhealthy("litellm", LITELLM_URL, "gpt-6-luna")


def test_usage_limit_is_logged_as_usage_limit_not_payment_or_credit(caplog):
    """The log names a usage limit; it must not send people chasing billing."""
    caplog.set_level("WARNING", logger="agent.auxiliary_client")
    primary = _client(LITELLM_URL, MagicMock(side_effect=_codex_usage_limit_429()))
    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("litellm", "gpt-6-luna", LITELLM_URL, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "gpt-6-luna")), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._recover_provider_pool", return_value=False), \
         patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0), \
         pytest.raises(Exception):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
    mark = [r.getMessage() for r in caplog.records if "marking" in r.getMessage()]
    assert mark, "expected an unhealthy-mark log line"
    assert all("payment / credit" not in m and "payment error" not in m for m in mark)
    assert all("usage limit" in m for m in mark)
