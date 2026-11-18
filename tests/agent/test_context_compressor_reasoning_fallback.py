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
    assert out.startswith(SUMMARY_PREFIX)
    assert "visible list summary" in out
    assert "PRIVATE_LIST_THINKING" not in out


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


# A summary that names a think tag in prose or code must survive whole (N1). Only the bare-tag
# cuts are at risk; the reasoning blocks are closed pairs and stay stripped.
TAG_MENTION_SUMMARIES = [
    "## Goal\nFix the `<think>` leak in the stream filter.\n## Progress\nPatched scrubber.\n## Next Steps\nShip PR #103.",
    "## Goal\nStrip the `</think>` closer before storing.\n## Next Steps\nShip PR #103.",
    "## Goal\nFix the leak.\n```\n<think>\n```\n## Next Steps\nShip PR #103.",
]


@pytest.mark.parametrize("summary", TAG_MENTION_SUMMARIES)
def test_tag_mention_in_summary_survives_on_compressor_path(summary):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=summary))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert out.endswith(summary)


@pytest.mark.parametrize("summary", TAG_MENTION_SUMMARIES)
def test_tag_mention_in_summary_survives_on_micro_path(summary):
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(summary)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == summary


def test_micro_bare_string_message_is_still_stored():
    response = SimpleNamespace(choices=[SimpleNamespace(message="bare micro summary")])
    with patch("agent.auxiliary_client.call_llm", return_value=response):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == "bare micro summary"


# F1: a stray backtick or fence in the reasoning must not hide the real bare tag that follows it.
# Each entry is (reply, PRIVATE text that must not be stored, visible text that must be stored).
STRAY_CODE_SHAPES = [
    (
        "Let me draft the fix:\n```python\ndef scrub(x):\n    PRIVATE_THINK\n</think>\n## Goal\nFix leak.\n"
        "## Commands\n```bash\nmake test\n```\n",
        "PRIVATE_THINK",
        "## Goal\nFix leak.",
    ),
    ("PRIVATE_THINK the ` key</think>REAL_SUMMARY uses `x`.", "PRIVATE_THINK", "REAL_SUMMARY uses `x`."),
    ("## Goal\nREAL_SUMMARY ``` note\n<think>PRIVATE_THINK ```", "PRIVATE_THINK", "REAL_SUMMARY"),
]


@pytest.mark.parametrize("reply,private,visible", STRAY_CODE_SHAPES)
def test_stray_code_delimiter_in_reasoning_does_not_leak_on_compressor_path(reply, private, visible):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert private not in out
    assert visible in out


@pytest.mark.parametrize("reply,private,visible", STRAY_CODE_SHAPES)
def test_stray_code_delimiter_in_reasoning_does_not_leak_on_micro_path(reply, private, visible):
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(reply)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out is not None
    assert private not in out
    assert visible in out


def test_tilde_fenced_tag_mention_survives_on_micro_path():
    summary = "## Goal\n~~~\n<think>\n~~~\n## Next\nShip PR #103."
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(summary)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == summary


# F2: tool-call XML and an attribute-carrying reasoning opener must never be stored.
# Each entry is (reply, PRIVATE text that must not be stored, visible text that must be stored).
TOOL_XML_SHAPES = [
    ("## Goal\nREAL_SUMMARY\n<tool_call>{\"name\": \"terminal\", \"arguments\": {\"cmd\": \"PRIVATE_CALL\"}}"
     "</tool_call>\n## Next\nShip.", "PRIVATE_CALL", "## Next\nShip."),
    ("## Goal\nREAL_SUMMARY\n<tool_result>PRIVATE_OUTPUT</tool_result>\nShip.", "PRIVATE_OUTPUT", "Ship."),
    ("## Goal\nREAL_SUMMARY\n<think type=\"x\">PRIVATE_THINK", "PRIVATE_THINK", "REAL_SUMMARY"),
]


@pytest.mark.parametrize("reply,private,visible", TOOL_XML_SHAPES)
def test_tool_xml_and_attribute_opener_are_not_stored_on_compressor_path(reply, private, visible):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert private not in out and "<tool_" not in out and "<think" not in out
    assert visible in out


@pytest.mark.parametrize("reply,private,visible", TOOL_XML_SHAPES)
def test_tool_xml_and_attribute_opener_are_not_stored_on_micro_path(reply, private, visible):
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(reply)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out is not None
    assert private not in out and "<tool_" not in out and "<think" not in out
    assert visible in out


def test_inline_tool_call_mention_survives_on_micro_path():
    summary = "## Goal\nFix `<tool_call>x</tool_call>` parse.\n## Next\nShip PR #103."
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(summary)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == summary


# F3: a closed reasoning pair inside one inline span must not cut the summary text before it.
PAIR_IN_SPAN_SHAPES = [
    ("## Goal\nFix the `<think>...</think>` stripping in the parser.\n## Next\nShip.",
     "## Goal\nFix the `` stripping in the parser.\n## Next\nShip."),
    ("## Goal\nShip.\n## Notes\nThe `<thinking>a</thinking>` case is covered.",
     "## Goal\nShip.\n## Notes\nThe `` case is covered."),
    ("## Goal\nFix `<think>x</think>` parse.\n<think>PRIVATE_THINK", "## Goal\nFix `` parse."),
]


@pytest.mark.parametrize("reply,visible", PAIR_IN_SPAN_SHAPES)
def test_pair_in_span_keeps_prefix_on_compressor_path(reply, visible):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = _compressor()._generate_summary(_turns())
    assert out is not None
    assert out.endswith(visible)


@pytest.mark.parametrize("reply,visible", PAIR_IN_SPAN_SHAPES)
def test_pair_in_span_keeps_prefix_on_micro_path(reply, visible):
    with patch("agent.auxiliary_client.call_llm", return_value=_micro_response(reply)):
        out = _compressor()._micro_summarize_one("user: hi\nassistant: hello")
    assert out == visible
