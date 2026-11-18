"""Summary text is message content only; reasoning is never stored as the summary.

Model thinking (``reasoning`` / ``reasoning_content`` / ``reasoning_details``) must never become the
stored compaction summary, because every later request would replay it as plain text (RIG-10290 §3.1).
A reply whose visible ``content`` is empty or think-only takes the existing empty-content failure path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX


def _compressor(**overrides):
    kwargs = dict(model="test/model", quiet_mode=True, tail_mode="legacy")
    kwargs.update(overrides)
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(**kwargs)


def _turns():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


def test_content_summary_is_stored_and_reasoning_is_ignored():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="visible summary body",
            reasoning_content="PRIVATE_THINKING_STEP",
        ))]
    )
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert out.startswith(SUMMARY_PREFIX)
    assert "visible summary body" in out
    assert "PRIVATE_THINKING_STEP" not in out


def test_reasoning_only_reply_is_never_stored_as_summary():
    reasoning = "PRIVATE_THINKING_STEP " * 40
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            reasoning_content=reasoning,
        ))]
    )
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is None


def test_whitespace_content_with_reasoning_is_not_stored_for_dict_and_object():
    class _Msg:
        content = " "
        reasoning_content = "PRIVATE_OBJECT_REASONING"

    for message in (
        {"content": " ", "reasoning_content": "PRIVATE_DICT_REASONING"},
        _Msg(),
    ):
        response = {"choices": [{"message": message}]}
        with patch("agent.context_compressor.call_llm", return_value=response):
            out = _compressor()._generate_summary(_turns())
        assert out is None


def test_think_only_content_is_treated_as_empty():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="<think>PRIVATE_INLINE_THINKING</think>",
            reasoning_content=None,
        ))]
    )
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is None


def test_empty_content_without_reasoning_still_fails():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    with patch("agent.context_compressor.call_llm", return_value=mock_response):
        out = _compressor()._generate_summary(_turns())
    assert out is None
