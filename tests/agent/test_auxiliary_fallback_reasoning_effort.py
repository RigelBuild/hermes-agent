"""RIG-10585: a served fallback entry's own ``reasoning_effort`` replaces the main level for that call.

The main route asks for ``medium``. A configured entry that declares ``low`` must reach the wire as
``low``, from ``fallback_chain`` and from ``fallback_providers`` alike. An entry without the field keeps
the main value with the same wire shape as before. An explicit caller disable wins over the entry.
"""

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI

from agent import auxiliary_client as ac
from agent import auxiliary_error_policy as aux_errors
from agent.auxiliary_client import async_call_llm, call_llm

MAIN_URL = "https://main-gateway.example.test/v1"
FALLBACK_URL = "https://fallback-gateway.example.test/v1"
MAIN_EFFORT = {"enabled": True, "effort": "medium"}
CALLER_DISABLED = {"enabled": False}


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


def _quota_429() -> Exception:
    exc = Exception(
        "Error code: 429 - {'error': {'message': 'Weekly usage limit reached', "
        "'type': 'usage_limit_reached', 'code': 'rate_limit_exceeded'}}"
    )
    exc.status_code = 429
    return exc


def _entry(provider: str, model: str, *, base_url: str = FALLBACK_URL, reasoning_effort=None) -> dict:
    entry = {"provider": provider, "model": model, "base_url": base_url, "api_key": "sk-fallback"}
    if reasoning_effort is not None:
        entry["reasoning_effort"] = reasoning_effort
    return entry


def _run(*, chain=(), main_chain=(), reasoning_config=MAIN_EFFORT, main_marked=True,
         primary_error=None, auto_runtime=False, async_mode=False, task_provider="auto") -> list:
    """Run one auxiliary call. Return ``(base_url, wire kwargs)`` for each send, in order. ``async_mode`` drives
    ``async_call_llm`` through the real ``_to_async_client`` with ``openai.AsyncOpenAI`` patched."""
    sent = []

    def record(base_url, kwargs):
        sent.append((base_url, kwargs))
        if base_url == MAIN_URL and primary_error is not None:
            raise primary_error
        return _ok_response("OK")

    def construct(*, api_key, base_url, **_kwargs):
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url
        client.HERMES_SKIP_ASYNC_WRAP = False  # a bare MagicMock would opt out of the async rebuild
        client.chat.completions.create = MagicMock(side_effect=lambda **kw: record(base_url, kw))
        return client

    class RecordingAsyncOpenAI(AsyncOpenAI):
        """A real ``AsyncOpenAI`` (so wire ``isinstance`` checks hold) whose sends are recorded."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            url = str(kwargs["base_url"])

            async def acreate(**kw):
                return record(url, kw)

            self.chat.completions.create = acreate

    if auto_runtime:
        runtime = {"provider": "auto", "model": "", "base_url": "", "api_key": ""}
    else:
        runtime = {"provider": "custom", "model": "gpt-6-luna", "base_url": MAIN_URL, "api_key": "sk-main"}
    if main_marked:
        ac._mark_provider_unhealthy(
            "custom", base_url=MAIN_URL, model="gpt-6-luna", reason=aux_errors.USAGE_LIMIT_REASON)
    request_model = "cheap-model-x" if main_marked or auto_runtime or task_provider != "auto" else "gpt-6-luna"
    with (
        patch("agent.auxiliary_client._create_openai_client", side_effect=construct),
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=(task_provider, request_model, None, None, None)),
        patch("agent.auxiliary_client._get_auxiliary_task_config",
              return_value={"fallback_chain": list(chain)}),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("hermes_cli.fallback_config.get_fallback_chain", return_value=list(main_chain)),
        patch("openai.AsyncOpenAI", RecordingAsyncOpenAI),
    ):
        call_kwargs = dict(
            task="title_generation",
            messages=[{"role": "user", "content": "hi"}],
            main_runtime=runtime,
            reasoning_config=reasoning_config,
        )
        if async_mode:
            asyncio.run(async_call_llm(**call_kwargs))
        else:
            call_llm(**call_kwargs)
    return sent


def test_entry_without_level_keeps_main_wire_shape_for_custom(monkeypatch):
    """No entry level: the custom wire is the base shape: top-level ``reasoning_effort``, no nested reasoning."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(chain=[_entry("custom", "claude-haiku-5-5")])
    assert [url for url, _ in sent] == [FALLBACK_URL]
    kwargs = sent[0][1]
    assert kwargs["reasoning_effort"] == "medium"
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_entry_without_level_keeps_main_wire_shape_for_openrouter(monkeypatch):
    """No entry level on an OpenRouter fallback: the base shape is top-level ``verbosity``, no nested reasoning."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(chain=[_entry("openrouter", "anthropic/claude-haiku-5-5")])
    assert [url for url, _ in sent] == [FALLBACK_URL]
    kwargs = sent[0][1]
    assert kwargs["verbosity"] == "medium"
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_fallback_entry_reasoning_effort_low_is_sent(monkeypatch):
    """A ``fallback_chain`` entry with ``reasoning_effort: low`` serves the call: the wire carries ``low``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")])
    assert [url for url, _ in sent] == [FALLBACK_URL]
    kwargs = sent[0][1]
    assert kwargs["reasoning_effort"] == "low"
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_async_fallback_entry_reasoning_effort_low_is_sent(monkeypatch):
    """Async auto route served by a ``fallback_chain`` entry: the async rebuild keeps the entry, so ``low``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")], async_mode=True)
    assert [(url, kw["reasoning_effort"]) for url, kw in sent] == [(FALLBACK_URL, "low")]


def test_async_explicit_provider_unavailable_sends_entry_level(monkeypatch):
    """Async call, explicit task provider with no credentials: the chain entry serves it and its ``low`` is sent."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    sent = _run(chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")],
                main_marked=False, task_provider="deepseek", async_mode=True)
    # The served label ``fallback_chain[0](custom)`` has no profile, so the generic nested shape is sent.
    assert [(url, kw["extra_body"]["reasoning"]["effort"]) for url, kw in sent] == [(FALLBACK_URL, "low")]


def test_carry_fallback_entry_ignores_untagged_source():
    """A source with no entry: the routed client stays untagged and nothing raises."""
    class Plain:
        pass

    source, routed = Plain(), Plain()
    assert ac._carry_fallback_entry(source, routed) is routed
    assert not hasattr(routed, "_hermes_fallback_entry")



def test_fallback_providers_entry_low_is_sent(monkeypatch):
    """A top-level ``fallback_providers`` entry with ``reasoning_effort: low`` sends its own ``low``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(
        main_chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")],
        auto_runtime=True, main_marked=False,
    )
    assert [url for url, _ in sent] == [FALLBACK_URL]
    kwargs = sent[0][1]
    assert kwargs["reasoning_effort"] == "low"
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_caller_disabled_wins_over_entry_level(monkeypatch):
    """The caller turned reasoning off (``enabled: False``): the entry's ``low`` must not re-enable it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(
        chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")],
        reasoning_config=CALLER_DISABLED,
    )
    assert [url for url, _ in sent] == [FALLBACK_URL]
    kwargs = sent[0][1]
    assert kwargs["reasoning_effort"] == "none"
    assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_recovery_ladder_fallback_sends_entry_level(monkeypatch):
    """The primary fails with a usage limit, so the recovery ladder builds the fallback request.
    That request must carry the entry's ``low``, while the primary keeps the main ``medium``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(
        chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")],
        main_marked=False, primary_error=_quota_429(),
    )
    assert [url for url, _ in sent] == [MAIN_URL, FALLBACK_URL]
    assert sent[0][1]["reasoning_effort"] == "medium"
    assert sent[1][1]["reasoning_effort"] == "low"


def test_invalid_fallback_providers_level_warning_names_fallback_providers(monkeypatch, caplog):
    """An invalid level on a ``fallback_providers`` entry is ignored, and the warning names that list."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        sent = _run(
            main_chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="bogus")],
            auto_runtime=True, main_marked=False,
        )
    assert sent[0][1]["reasoning_effort"] == "medium"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("fallback_providers" in msg and "bogus" in msg for msg in warnings), warnings


def test_async_recovery_ladder_fallback_sends_entry_level(monkeypatch):
    """Async primary fails with a quota 429: the async rebuild of the fallback client keeps its entry, so ``low``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _run(
        chain=[_entry("custom", "claude-haiku-5-5", reasoning_effort="low")],
        main_marked=False, primary_error=_quota_429(), async_mode=True,
    )
    assert [(url, kw["reasoning_effort"]) for url, kw in sent] == [(MAIN_URL, "medium"), (FALLBACK_URL, "low")]


def test_entry_level_does_not_leak_into_a_later_primary_call(monkeypatch):
    """A fallback-served call, then a healthy primary call in the same process: the second sends the main level."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chain = [_entry("custom", "claude-haiku-5-5", reasoning_effort="low")]
    first = _run(chain=chain)
    assert [(url, kw["reasoning_effort"]) for url, kw in first] == [(FALLBACK_URL, "low")]
    ac._reset_aux_unhealthy_cache()
    second = _run(chain=chain, main_marked=False)
    assert [(url, kw["reasoning_effort"]) for url, kw in second] == [(MAIN_URL, "medium")]


def test_same_provider_retry_on_entry_client_keeps_entry_level():
    """An auth-refresh retry rebuilds the request for an entry-tagged client: it keeps the entry's ``low``."""
    retry_client = MagicMock()
    retry_client.base_url = FALLBACK_URL
    ac._tag_fallback_entry(retry_client, _entry("custom", "claude-haiku-5-5", reasoning_effort="low"), "fallback_chain")
    with patch("agent.auxiliary_client._get_cached_client", return_value=(retry_client, "claude-haiku-5-5")):
        client, kwargs = ac._prepare_same_provider_retry(
            task="title_generation", resolved_provider="custom", resolved_model="claude-haiku-5-5",
            resolved_base_url=FALLBACK_URL, resolved_api_key="sk-fallback", resolved_api_mode=None,
            main_runtime=None, final_model="claude-haiku-5-5", messages=[{"role": "user", "content": "hi"}],
            temperature=None, max_tokens=None, tools=None, effective_timeout=30.0, effective_extra_body={},
            reasoning_config=MAIN_EFFORT, async_mode=False,
        )
    assert client is retry_client
    assert kwargs["reasoning_effort"] == "low"
