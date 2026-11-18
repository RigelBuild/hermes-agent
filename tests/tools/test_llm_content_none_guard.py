"""Tests for None guard on response.choices[0].message.content.strip().

OpenAI-compatible APIs return ``message.content = None`` when the model
responds with tool calls only or reasoning-only output (e.g. DeepSeek-R1,
Qwen-QwQ via OpenRouter with ``reasoning.enabled = True``).  Calling
``.strip()`` on ``None`` raises ``AttributeError``.

These tests verify that ``extract_content()`` handles ``content is None``
safely and never returns reasoning fields as the visible answer.
"""

import types

from agent.auxiliary_client import extract_content, has_reasoning


# ── helpers ────────────────────────────────────────────────────────────────

def _make_response(content, **msg_attrs):
    """Build a minimal OpenAI-compatible ChatCompletion response stub.

    Extra keyword args are set as attributes on the message object
    (e.g. reasoning="...", reasoning_content="...", reasoning_details=[...]).
    """
    message = types.SimpleNamespace(content=content, tool_calls=None, **msg_attrs)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


# ── extract_content() ──────────────────────────────────────────────────────

class TestExtractContent:
    """agent/auxiliary_client.py — extract_content()"""

    def test_normal_content_returned(self):
        response = _make_response("  Hello world  ")
        assert extract_content(response) == "Hello world"

    def test_none_content_returns_empty(self):
        response = _make_response(None)
        assert extract_content(response) == ""

    def test_think_blocks_stripped_with_remaining_content(self):
        response = _make_response("<think>internal reasoning</think>The answer is 42.")
        assert extract_content(response) == "The answer is 42."

    def test_think_only_content_returns_empty_not_reasoning(self):
        """Content that is only think blocks is empty; reasoning fields are never the answer."""
        response = _make_response(
            "<think>some reasoning</think>",
            reasoning="The actual reasoning output",
        )
        assert extract_content(response) == ""

    def test_reasoning_only_reply_returns_empty(self):
        response = _make_response(None, reasoning="private chain of thought")
        assert extract_content(response) == ""

    def test_content_preferred_over_reasoning(self):
        """When both content and reasoning exist, content wins."""
        response = _make_response("Actual answer", reasoning="Internal reasoning")
        assert extract_content(response) == "Actual answer"

    def test_dict_message_and_whitespace_return_empty_for_reasoning(self):
        assert extract_content({"content": " ", "reasoning_content": "dict reasoning"}) == ""
        assert extract_content(
            {"choices": [{"message": {"content": "", "reasoning": "from response"}}]}
        ) == ""

    def test_string_message_passthrough(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message="plain summary text")]
        )
        assert extract_content(response) == "plain summary text"

    def test_list_content_drops_thinking_parts(self):
        response = _make_response([
            {"type": "thinking", "thinking": "internal chain of thought"},
            {"type": "text", "text": "visible answer"},
        ])
        assert extract_content(response) == "visible answer"

    def test_list_content_with_only_thinking_returns_empty(self):
        response = _make_response([{"type": "thinking", "thinking": "internal chain of thought"}])
        assert extract_content(response) == ""

    def test_unterminated_think_block_returns_empty(self):
        response = _make_response("<think>unterminated cot answer")
        assert extract_content(response) == ""

    def test_orphan_think_tag_is_stripped(self):
        response = _make_response("</think>The answer is 42.")
        assert extract_content(response) == "The answer is 42."


# ── has_reasoning() ────────────────────────────────────────────────────────

class TestHasReasoning:
    """agent/auxiliary_client.py — has_reasoning(), used only to pick the retry log line."""

    def test_reasoning_field_detected(self):
        assert has_reasoning(_make_response(None, reasoning="thinking")) is True

    def test_plain_content_is_not_reasoning(self):
        assert has_reasoning(_make_response("Hello")) is False

    def test_blank_reasoning_is_not_reasoning(self):
        assert has_reasoning(_make_response(None, reasoning="   ")) is False

    def test_reasoning_details_list_detected(self):
        details = [{"type": "reasoning.text", "text": "thinking"}]
        assert has_reasoning(_make_response(None, reasoning_details=details)) is True

    def test_empty_reasoning_details_is_not_reasoning(self):
        assert has_reasoning(_make_response(None, reasoning_details=[])) is False
