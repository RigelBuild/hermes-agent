"""Review r94-1 (RIG-10475): a usage limit is scoped to its model at every reader that can see it.

M1: a later ``auto`` call must not send to the limited main model.
M2: ``is_usage_limit`` reads the structured code first; a billing code stays endpoint-wide.
M3: the scope rules hold on a ``custom`` base_url primary, on sync and async, and for a billing 402.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import async_call_llm, call_llm
from agent.error_classifier import ClassifiedError, FailoverReason

LITELLM_URL = "https://llm-gateway.internal.rigel.build/v1"
CUSTOM_URL = "https://custom-gateway.internal.rigel.build/v1"


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    ac._reset_aux_unhealthy_cache()
    yield
    ac._reset_aux_unhealthy_cache()


def _usage_limit_429() -> Exception:
    """The Codex 429 as the SDK raises it: the parsed body rides on ``.body``, as the classifier reads it."""
    exc = Exception(
        "Error code: 429 - {'error': {'type': 'usage_limit_reached', "
        "'message': 'The usage limit has been reached', 'code': 'usage_limit_reached'}}"
    )
    exc.status_code = 429
    exc.body = {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached",
                          "code": "usage_limit_reached"}}
    return exc


def _billing_code_429() -> Exception:
    """A hard budget wall that also carries the usage-limit phrase: the structured code is billing."""
    exc = Exception(
        "Error code: 429 - {'error': {'code': 'terminal_quota_exhausted', "
        "'message': 'usage limit reached: the key budget is exhausted'}}"
    )
    exc.status_code = 429
    exc.body = {"error": {"code": "terminal_quota_exhausted",
                          "message": "usage limit reached: the key budget is exhausted"}}
    return exc


def _billing_402() -> Exception:
    exc = Exception("Error code: 402 - {'error': {'message': 'Insufficient credits'}}")
    exc.status_code = 402
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
    client.api_key = "sk-gateway"
    client.chat.completions.create = create
    return client


def _route_builds(primary, sibling):
    """The limited model builds the primary; any other model (the fallback candidate) builds the sibling."""
    def build(provider, model, *_args, **_kwargs):
        return (primary, model) if model == "gpt-6-luna" else (sibling, "claude-haiku-5-5")
    return build


def _verdict(reason=FailoverReason.billing, **context) -> ClassifiedError:
    return ClassifiedError(reason, status_code=429, error_context=context)


# ── M2: is_usage_limit reads the structured code first ─────────────────────────────


def test_billing_code_with_usage_limit_phrase_is_not_a_usage_limit():
    failure = aux_errors.Failure(
        _billing_code_429(),
        _verdict(billing_code="terminal_quota_exhausted"),
    )
    assert aux_errors.is_usage_limit(failure) is False


def test_structured_usage_limit_code_is_a_usage_limit_without_text():
    failure = aux_errors.Failure(
        Exception("opaque gateway body"),
        _verdict(usage_limit_code="usage_limit_reached"),
    )
    assert aux_errors.is_usage_limit(failure) is True


def test_insufficient_credits_is_not_a_usage_limit():
    failure = aux_errors.Failure(_billing_402(), _verdict())
    assert aux_errors.is_usage_limit(failure) is False


def test_text_fallback_still_catches_a_body_with_no_structure():
    exc = Exception("Error code: 429 - the usage limit has been reached")
    exc.status_code = 429
    failure = aux_errors.Failure(exc, _verdict(billing_unverified=True))
    assert aux_errors.is_usage_limit(failure) is True


def test_usage_limit_tokens_are_one_shared_tuple():
    """agent_runtime_helpers and the aux policy must read the same token list."""
    from agent import agent_runtime_helpers as arh
    from agent import error_classifier as ec

    assert aux_errors.USAGE_LIMIT_TEXT is ec.USAGE_LIMIT_TEXT
    assert arh.USAGE_LIMIT_TEXT is ec.USAGE_LIMIT_TEXT


# ── M1: a later auto call must not send to the limited main model ─────────────────


def test_second_auto_call_skips_the_limited_main_model(monkeypatch):
    """Two auto calls in a row: the first marks gpt-6-luna; the second must not send to it.

    Only the OpenAI constructor is stubbed. The dispatcher, the main route and its mark check run for real.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    limited_sends = []

    def create(*, model, **_kwargs):
        if model == "gpt-6-luna":
            limited_sends.append(model)
            raise _usage_limit_429()
        return _ok_response("HAIKU")

    def construct(*, api_key, base_url, **_kwargs):
        client = _client(base_url, MagicMock(side_effect=create))
        client.api_key = api_key
        return client

    chain = [{"provider": "custom", "model": "claude-haiku-5-5", "base_url": LITELLM_URL}]
    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": LITELLM_URL, "api_key": "sk-test"}

    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "gpt-6-luna", LITELLM_URL, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        sends_after_first = len(limited_sends)
        assert sends_after_first == 1, "the first auto call must reach the limited model to mark it"
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert len(limited_sends) == sends_after_first, (
        "the second auto call re-sent to the limited main model"
    )


# ── M3: parametrised over a custom primary and sync/async ─────────────────────────


@pytest.mark.parametrize("base_url", [LITELLM_URL, CUSTOM_URL], ids=["litellm", "custom"])
def test_usage_limit_keeps_sibling_model_sync(base_url):
    primary = _client(base_url, MagicMock(side_effect=_usage_limit_429()))
    haiku = _client(base_url, MagicMock(return_value=_ok_response("HAIKU")))
    chain = [{"provider": "custom", "model": "claude-haiku-5-5", "base_url": base_url}]

    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("custom", "gpt-6-luna", base_url, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "gpt-6-luna")), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(haiku, "claude-haiku-5-5")), \
         patch("agent.auxiliary_client._recover_provider_pool", return_value=False), \
         patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "HAIKU"
    assert haiku.chat.completions.create.call_count == 1


@pytest.mark.parametrize("base_url", [LITELLM_URL, CUSTOM_URL], ids=["litellm", "custom"])
def test_usage_limit_keeps_sibling_model_async(base_url):
    primary = _client(base_url, AsyncMock(side_effect=_usage_limit_429()))
    haiku = _client(base_url, AsyncMock(return_value=_ok_response("HAIKU")))
    chain = [{"provider": "custom", "model": "claude-haiku-5-5", "base_url": base_url}]

    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("custom", "gpt-6-luna", base_url, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "gpt-6-luna")), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(haiku, "claude-haiku-5-5")), \
         patch("agent.auxiliary_client._recover_provider_pool", return_value=False), \
         patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        result = asyncio.run(async_call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}]))

    assert result.choices[0].message.content == "HAIKU"
    assert haiku.chat.completions.create.call_count == 1


def test_billing_402_hides_the_whole_endpoint_and_sets_the_endpoint_key():
    """A confirmed 402 is account-wide: the Haiku entry on the same gateway is skipped and the
    endpoint key (no model) is set."""
    primary = _client(LITELLM_URL, MagicMock(side_effect=_billing_402()))
    haiku = _client(LITELLM_URL, MagicMock(return_value=_ok_response("HAIKU")))
    chain = [{"provider": "litellm", "model": "claude-haiku-5-5", "base_url": LITELLM_URL}]

    with patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("litellm", "gpt-6-luna", LITELLM_URL, None, None)), \
         patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "gpt-6-luna")), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(haiku, "claude-haiku-5-5")), \
         patch("agent.auxiliary_client._recover_provider_pool", return_value=False), \
         patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0), \
         pytest.raises(Exception):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])

    assert haiku.chat.completions.create.call_count == 0
    assert ac._is_provider_unhealthy("litellm", LITELLM_URL)
