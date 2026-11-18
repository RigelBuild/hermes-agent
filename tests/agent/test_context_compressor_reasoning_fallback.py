"""Summary text is message content only; reasoning is never stored as the summary.

Model thinking (``reasoning`` / ``reasoning_content`` / ``reasoning_details``) must never become the
stored compaction summary, because every later request would replay it as plain text (RIG-10290 §3.1).
A reply whose visible ``content`` is empty or think-only takes the existing empty-content failure path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

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


@pytest.mark.parametrize("field,value", [
    ("reasoning", "PRIVATE_THINKING_STEP " * 40),
    ("reasoning_content", "PRIVATE_THINKING_STEP " * 40),
    ("reasoning_details", [{"type": "reasoning.text", "text": "PRIVATE_THINKING_STEP"}]),
])
def test_reasoning_only_reply_is_never_stored_as_summary(field, value):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", **{field: value}))]
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


def test_list_content_keeps_the_text_part_and_drops_thinking():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=[
        {"type": "thinking", "thinking": "PRIVATE_LIST_THINKING"},
        {"type": "text", "text": "visible list summary"},
    ]))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out == f"{SUMMARY_PREFIX}\nvisible list summary"


THINK_PREFIX_SHAPES = [
    # Prefilled <think>: reasoning, then a bare closing tag.
    "PRIVATE_THINK step one</think>\n\n## Goal\nREAL_SUMMARY",
    # Mid-line opening tag that never closes.
    "REAL_SUMMARY <think>PRIVATE_THINK unterminated",
]


@pytest.mark.parametrize("content", THINK_PREFIX_SHAPES)
def test_think_prefix_or_unclosed_tail_is_never_stored(content):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert "PRIVATE_THINK" not in out
    assert "REAL_SUMMARY" in out


def _micro_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_micro_list_content_keeps_the_text_part_and_drops_thinking():
    response = _micro_response([
        {"type": "thinking", "thinking": "PRIVATE_MICRO_THINKING"},
        {"type": "text", "text": "visible micro summary"},
    ])
    with patch("agent.auxiliary_client.call_llm", return_value=response):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == "visible micro summary"


@pytest.mark.parametrize("content", THINK_PREFIX_SHAPES)
def test_micro_think_prefix_or_unclosed_tail_is_never_stored(content):
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(content)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out is not None
    assert "PRIVATE_THINK" not in out
    assert "REAL_SUMMARY" in out
