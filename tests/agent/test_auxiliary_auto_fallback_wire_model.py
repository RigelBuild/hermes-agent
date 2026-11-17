"""RIG-10494: an auto route sends the requested model when the main client or discovery serves the route.

Main model M is usage-limited, so the route falls back to the configured entry F. The request asks for X.
F's client must receive F's model, not X: F may not serve X. A fallback entry sends its own model; a
main-served or discovery route keeps X (#94).
"""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import call_llm

MAIN_URL = "https://main-gateway.example.test/v1"
FALLBACK_URL = "https://fallback-gateway.example.test/v1"
FALLBACK_CHAIN = [{"provider": "custom", "model": "claude-haiku-5-5", "base_url": FALLBACK_URL,
                   "api_key": "sk-fallback"}]


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
    return resp


def _run_auto_call(*, request_model: str, main_marked: bool, chain: list, calls: int = 1,
                    auto_runtime: bool = False) -> list:
    """Run ``calls`` auto calls requesting ``request_model``; return (base_url, model) per wire send."""
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    runtime = {"provider": "auto", "model": "", "base_url": "", "api_key": ""} if auto_runtime else \
        {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    if main_marked:
        ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="gpt-6-luna",
                                    reason=aux_errors.USAGE_LIMIT_REASON)
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", request_model, None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": chain}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        for _ in range(calls):
            call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}], main_runtime=runtime)
    return sent


def test_main_marked_request_x_falls_back_and_sends_fallback_model(monkeypatch):
    """M marked, request X: the route serves fallback F, so F receives F's model, not X."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run_auto_call(request_model="cheap-model-x", main_marked=True, chain=FALLBACK_CHAIN)
    assert sent == [(FALLBACK_URL, "claude-haiku-5-5")]


def test_main_marked_request_x_fallback_model_on_cached_second_call(monkeypatch):
    """A second call with the same X must still send F's model, whether it builds or reuses the client."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run_auto_call(request_model="cheap-model-x", main_marked=True, chain=FALLBACK_CHAIN, calls=2)
    assert sent == [(FALLBACK_URL, "claude-haiku-5-5"), (FALLBACK_URL, "claude-haiku-5-5")]


def test_main_served_request_x_is_sent(monkeypatch):
    """M not marked: the main client serves the route, and the explicit X is sent (#94)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run_auto_call(request_model="cheap-model-x", main_marked=False, chain=FALLBACK_CHAIN)
    assert sent == [(MAIN_URL, "cheap-model-x")]


def test_main_served_x_cached_then_limited_sends_main_model(monkeypatch):
    """Main serves X and caches its client. X is then marked limited; the second call must send the main model, not X."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": FALLBACK_CHAIN}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="cheap-model-x",
                                    reason=aux_errors.USAGE_LIMIT_REASON)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent == [(MAIN_URL, "cheap-model-x"), (MAIN_URL, "gpt-6-luna")]


def test_main_marked_request_x_main_fallback_providers_sends_entry_model(monkeypatch):
    """M marked, no task chain, configured fallback_providers: the entry's model is sent, not X."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    main_chain = [{"provider": "custom", "model": "claude-haiku-5-5", "base_url": FALLBACK_URL,
                   "api_key": "sk-fallback"}]
    with patch("hermes_cli.config.load_config_readonly", return_value={}), \
         patch("hermes_cli.fallback_config.get_fallback_chain", return_value=main_chain):
        sent = _run_auto_call(request_model="cheap-model-x", main_marked=True, chain=[], auto_runtime=True)
    assert sent == [(FALLBACK_URL, "claude-haiku-5-5")]


def test_main_marked_request_x_async_sends_fallback_model(monkeypatch):
    """Async path: M marked, request X, task fallback_chain F: F receives F's model, not X."""
    import asyncio
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        async def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        client.chat.completions.acreate = MagicMock(side_effect=create)
        return client

    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="gpt-6-luna",
                                reason=aux_errors.USAGE_LIMIT_REASON)
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": FALLBACK_CHAIN}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        asyncio.run(ac.async_call_llm(task="title_generation",
                                      messages=[{"role": "user", "content": "hi"}], main_runtime=runtime))
    assert sent == [(FALLBACK_URL, "claude-haiku-5-5")]


def test_discovery_route_request_x_is_sent(monkeypatch):
    """No main provider: discovery picks OpenRouter; the requested model is sent, as before RIG-10494.

    A configured model (``model.provider: auto`` with ``auxiliary.<task>.model``) must not be replaced by
    OpenRouter's default. Only a fallback entry's model replaces X.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "google/gemini-3-flash-preview", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._discovery_chain_allowed", return_value=True), \
         patch("agent.auxiliary_client._get_provider_chain",
               return_value=[("openrouter", lambda model=None: (construct(api_key="sk-or", base_url="https://openrouter.ai/api/v1"),
                                                                 model or "nvidia/nemotron-3-ultra-550b-a55b:free"))]):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}],
                 main_runtime={"provider": "auto", "model": "", "base_url": "", "api_key": ""})
    assert sent == [("https://openrouter.ai/api/v1", "google/gemini-3-flash-preview")]


def test_main_marked_request_x_sends_x_again_after_limit_clears(monkeypatch):
    """X is usage-limited and the route falls back to the main model. The limit then clears: the next
    call on the same cached route must send X, not the substitute M built while X was limited."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="cheap-model-x",
                                    reason=aux_errors.USAGE_LIMIT_REASON)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        ac._reset_aux_unhealthy_cache()
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent == [(MAIN_URL, "gpt-6-luna"), (MAIN_URL, "cheap-model-x")]


def test_limit_mark_during_build_does_not_cache_substitute(monkeypatch):
    """A concurrent call marks X limited while this call builds the client. The route then sends M; that
    substitute must not be cached under X's key, so X is sent again once the limit clears."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client

    real_resolve = ac.resolve_provider_client
    marked = []

    def resolve_with_race(*args, **kwargs):
        if not marked:
            marked.append(True)
            ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="cheap-model-x",
                                        reason=aux_errors.USAGE_LIMIT_REASON)
            ac._evict_cached_clients("auto", "cheap-model-x")
        return real_resolve(*args, **kwargs)

    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve_with_race), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        ac._reset_aux_unhealthy_cache()
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "three"}], main_runtime=runtime)

    assert sent == [(MAIN_URL, "gpt-6-luna"), (MAIN_URL, "cheap-model-x"), (MAIN_URL, "cheap-model-x")]


def _capture_construct(sent: list, built: list | None = None):
    def construct(*, api_key, base_url, **_kwargs):
        if built is not None:
            built.append(base_url)
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url

        def create(*, model, **_kwargs):
            sent.append((base_url, model))
            return _ok_response("OK")

        client.chat.completions.create = MagicMock(side_effect=create)
        return client
    return construct


def test_limit_expiring_during_build_does_not_cache_substitute(monkeypatch):
    """M1 repro (deterministic, keyed on the requested model X). Step 1: no mark on X when the substitute flag is
    read, so the flag is False. Step 2: the mark on X lands before the route checks it, so the route substitutes
    and builds the fallback client. Step 3: the mark expires before the store-time re-check. The fallback client
    must not be cached under X's key: the next call sends X through the main route."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []
    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    real_unhealthy = ac._is_provider_unhealthy
    x_checks = [False, True]  # flag read: no mark; route check: mark live; later checks use the real store

    def unhealthy_on_x(label, base_url=None, model=None):
        if model == "cheap-model-x" and x_checks:
            return x_checks.pop(0)
        return real_unhealthy(label, base_url, model)

    with patch("agent.auxiliary_client._create_openai_client", side_effect=_capture_construct(sent)), \
         patch("agent.auxiliary_client._is_provider_unhealthy", side_effect=unhealthy_on_x), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": FALLBACK_CHAIN}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        ac._reset_aux_unhealthy_cache()  # the mark on X has expired before the next call
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent[-1] == (MAIN_URL, "cheap-model-x")


def test_provider_wide_mark_second_call_reuses_cached_client(monkeypatch):
    """A provider-wide mark on the main endpoint: the first call sends the fallback entry's own model. The
    second call reuses the cached fallback client and builds no new client while the mark holds."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent, built = [], []
    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    ac._mark_provider_unhealthy("custom", base_url=MAIN_URL)  # provider-wide: no model
    with patch("agent.auxiliary_client._create_openai_client", side_effect=_capture_construct(sent, built)), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": FALLBACK_CHAIN}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent == [(FALLBACK_URL, "claude-haiku-5-5"), (FALLBACK_URL, "claude-haiku-5-5")]
    assert built == [FALLBACK_URL]


def test_mark_on_cached_main_client_rebuilds_as_substitute(monkeypatch):
    """X is served by the main client and cached with the gate off. X is then marked limited: the gate flips, so
    the lookup must not reuse the cached entry. The call builds a fresh client and still sends X on the main route."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent, built = [], []
    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    with patch("agent.auxiliary_client._create_openai_client", side_effect=_capture_construct(sent, built)), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": FALLBACK_CHAIN}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        ac._mark_provider_unhealthy("custom", base_url=MAIN_URL, model="cheap-model-x",
                                    reason=aux_errors.USAGE_LIMIT_REASON)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent == [(MAIN_URL, "cheap-model-x"), (MAIN_URL, "gpt-6-luna")]
    assert built == [MAIN_URL, MAIN_URL]


def test_provider_wide_mark_keeps_auto_client_cached(monkeypatch):
    """A provider-wide mark on the main endpoint with discovery serving X: the second call reuses the cached
    discovery client and builds no new client."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent, built = [], []
    openrouter_url = "https://openrouter.ai/api/v1"
    runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    construct = _capture_construct(sent, built)
    ac._mark_provider_unhealthy("custom", base_url=MAIN_URL)  # provider-wide: no model
    with patch("agent.auxiliary_client._create_openai_client", side_effect=construct), \
         patch("agent.auxiliary_client._resolve_task_provider_model",
               return_value=("auto", "cheap-model-x", None, None, None)), \
         patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": []}), \
         patch("agent.auxiliary_client._transient_retry_count", return_value=0), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._discovery_chain_allowed", return_value=True), \
         patch("agent.auxiliary_client._get_provider_chain", return_value=[
            ("openrouter", lambda model=None: (
                construct(api_key="sk-or", base_url=openrouter_url),
                 model or "nvidia/nemotron-3-ultra-550b-a55b:free"))]):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "one"}], main_runtime=runtime)
        call_llm(task="title_generation", messages=[{"role": "user", "content": "two"}], main_runtime=runtime)

    assert sent == [(openrouter_url, "cheap-model-x"), (openrouter_url, "cheap-model-x")]
    assert built == [openrouter_url]
