"""Tests for StreamingThinkScrubber.

These tests lock in the contract the scrubber must satisfy so downstream
consumers (ACP, api_server, TTS, CLI, gateway) never see reasoning
blocks leaking through the stream_delta_callback.  The scenarios map
directly to the MiniMax-M2.7 / DeepSeek / Qwen3 streaming patterns that
break the older per-delta regex strip.
"""

from __future__ import annotations

import pytest

from agent.think_scrubber import StreamingThinkScrubber


def _drive(scrubber: StreamingThinkScrubber, deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    out = [scrubber.feed(d) for d in deltas]
    out.append(scrubber.flush())
    return "".join(out)


class TestClosedPairs:
    """Closed <tag>...</tag> pairs are always stripped, regardless of boundary."""

    def test_closed_pair_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning</think>Hello world"]) == "Hello world"


    @pytest.mark.parametrize(
        "tag",
        ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"],
    )
    def test_all_tag_variants(self, tag: str) -> None:
        s = StreamingThinkScrubber()
        delta = f"<{tag}>x</{tag}>Hello"
        assert _drive(s, [delta]) == "Hello"



class TestUnterminatedOpen:
    """Unterminated open tag discards all subsequent content to end of stream."""

    def test_open_at_stream_start(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning text with no close"]) == ""



    def test_prose_mentioning_tag_not_stripped(self) -> None:
        """Mid-line '<think>' in prose is preserved (no boundary)."""
        s = StreamingThinkScrubber()
        text = "Use the <think> element for reasoning"
        assert _drive(s, [text]) == text


class TestOrphanClose:
    """Orphan close tags (no prior open) are stripped without boundary check."""

    def test_orphan_close_alone(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello</think>world"]) == "Helloworld"




class TestPartialTagsAcrossDeltas:
    """Partial tags at delta boundaries must be held back, not emitted raw."""

    def test_split_open_tag_held_back(self) -> None:
        """'<' arrives alone, 'think>' completes it on next delta."""
        s = StreamingThinkScrubber()
        # At stream start, last_emitted_ended_newline=True, so <think> at 0 is boundary
        assert (
            _drive(s, ["<", "think>reasoning</think>done"])
            == "done"
        )

    def test_split_open_tag_not_at_boundary(self) -> None:
        """Mid-line split '<' + 'think>X</think>' is a closed pair.

        Closed pairs are always stripped (matching
        ``_strip_think_blocks`` case 1), even without a block
        boundary — a closed pair is an intentional bounded construct.
        """
        s = StreamingThinkScrubber()
        out = _drive(s, ["word<", "think>prose</think>more"])
        assert out == "wordmore"




class TestTheMiniMaxScenario:
    """The exact pattern run_agent per-delta regex strip breaks."""

    def test_minimax_split_open(self) -> None:
        """delta1='<think>', delta2='Let me check', delta3='</think>done'."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "Let me check their config", "</think>", "done"])
        assert out == "done"


    def test_minimax_unterminated_reasoning_at_end(self) -> None:
        """Unclosed reasoning at stream end is dropped entirely."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "The user wants", " to know something"])
        assert out == ""


class TestResetAndReentry:
    def test_reset_clears_in_block_state(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("<think>hanging")
        s.reset()
        # After reset, a new turn works cleanly
        assert _drive(s, ["Hello world"]) == "Hello world"

    def test_reset_clears_buffered_partial_tag(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("word<")
        s.reset()
        assert _drive(s, ["fresh content"]) == "fresh content"


class TestFlushBehaviour:



    def test_flush_restores_stream_start_boundary(self) -> None:
        """End-of-stream flush must re-arm block-boundary gating.

        Thinking-only / empty-response retries flush then stream again
        without ``reset()``.  If flush left ``_last_emitted_ended_newline``
        False (e.g. after emitting a held-back ``<``), the next stream's
        opening ``<think>`` looked mid-line and leaked into the UI.
        """
        s = StreamingThinkScrubber()
        assert s.feed("word") == "word"
        assert s.flush() == ""
        assert (
            _drive(s, ["<think>", "secret reasoning", "</think>", "Visible answer"])
            == "Visible answer"
        )

    def test_flush_partial_tag_tail_does_not_poison_next_stream(self) -> None:
        """Flushing a held-back ``<`` must not make the next open tag leak."""
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s.flush() == "<"
        assert _drive(s, ["<think>hidden</think>Hello"]) == "Hello"


class TestRealisticStreaming:
    """Character-by-character streaming must work as well as larger chunks."""

    def test_char_by_char_closed_pair(self) -> None:
        s = StreamingThinkScrubber()
        deltas = list("<think>x</think>Hello world")
        assert _drive(s, deltas) == "Hello world"


    def test_reasoning_then_real_response_first_word_preserved(self) -> None:
        """Regression: the first word of the final response must NOT be eaten.

        Stefan's screenshot bug — 'Let me check' was being rendered as
        ' me check'.  The scrubber must not consume any character of
        post-close content.
        """
        s = StreamingThinkScrubber()
        deltas = [
            "<think>",
            "User wants to know things",
            "</think>",
            "Let me check their config.",
        ]
        assert _drive(s, deltas) == "Let me check their config."

    def test_no_tag_passthrough_is_identical(self) -> None:
        """Streams without any reasoning tags pass through byte-for-byte."""
        s = StreamingThinkScrubber()
        deltas = ["Hello ", "world ", "how ", "are ", "you?"]
        assert _drive(s, deltas) == "Hello world how are you?"


class TestChineseReasoningTags:
    """MiniMax-M3 emits Chinese reasoning tags (#43827); both surfaces must hide them."""

    def test_split_chinese_tag_scrubbed_in_stream(self) -> None:
        s = StreamingThinkScrubber()
        deltas = ["<思", "考>让我想想", "……</思考>", "\n答案是 42"]
        assert _drive(s, deltas) == "\n答案是 42"

    def test_final_response_strip_hides_chinese_tags(self) -> None:
        from agent.agent_runtime_helpers import strip_think_blocks

        out = strip_think_blocks(None, "<反思>内部推理</反思>最终答案\n<推理>未闭合的推理")
        assert "内部推理" not in out and "未闭合" not in out
        assert "最终答案" in out

    def test_cli_replay_strip_hides_chinese_tags(self) -> None:
        from cli import _strip_reasoning_tags

        assert _strip_reasoning_tags("<思考>secret</思考>答案") == "答案"


class TestNamespacedReasoningTags:
    """MiniMax-M3 also emits the namespaced ``<mm:think>``; neither surface may deliver it."""

    def test_split_namespaced_tag_scrubbed_in_stream(self) -> None:
        s = StreamingThinkScrubber()
        deltas = ["<mm:th", "ink>secret plan", "</mm:think>", "Answer."]
        assert _drive(s, deltas) == "Answer."

    def test_final_response_strip_hides_block_orphan_and_unterminated(self) -> None:
        from agent.agent_runtime_helpers import strip_think_blocks

        out = strip_think_blocks(None, "a <mm:think>hidden</mm:think> b. Report.</mm:think>Fixed.\n<mm:think>open")
        assert out == "a  b. Report.Fixed."


class TestSummaryVisibleText:
    """Summary-only scrub (``summary_visible_text``): stored summary text carries no reasoning."""

    def test_block_list_drops_thinking_and_keeps_text(self) -> None:
        from agent.think_scrubber import summary_visible_text

        parts = [{"type": "thinking", "thinking": "PRIVATE"}, {"type": "text", "text": "visible"}]
        assert summary_visible_text(parts) == "visible"

    def test_orphan_close_takes_the_reasoning_before_it(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("PRIVATE step</think>\n\n## Goal") == "## Goal"

    def test_unclosed_open_takes_the_tail(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("visible <think>PRIVATE tail") == "visible"

    def test_closed_pair_keeps_its_neighbours(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("A <think>r</think> B") == "A  B"

    def test_non_text_content_is_empty(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text(None) == ""

    def test_backticked_tag_mention_survives(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\nFix the `<think>` leak.\n## Next\nShip."
        assert summary_visible_text(text) == text

    def test_backticked_closer_mention_survives(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\nStrip the `</think>` closer.\n## Next\nShip."
        assert summary_visible_text(text) == text

    def test_fenced_tag_mention_survives(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\n```\n<think>\n```\n## Next\nShip."
        assert summary_visible_text(text) == text

    def test_reasoning_before_a_real_closer_is_still_cut_around_code(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("PRIVATE `x`</think>\n## Goal") == "## Goal"

    def test_odd_fence_in_reasoning_does_not_hide_real_closer(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = (
            "Let me draft the fix:\n```python\ndef scrub(x):\n    PRIVATE_THINK\n</think>\n## Goal\nFix leak.\n"
            "## Commands\n```bash\nmake test\n```\n"
        )
        assert summary_visible_text(text) == (
            "## Goal\nFix leak.\n## Commands\n```bash\nmake test\n```"
        )

    def test_stray_backtick_in_reasoning_does_not_hide_real_closer(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("PRIVATE_THINK the ` key</think>REAL_SUMMARY uses `x`.") == (
            "REAL_SUMMARY uses `x`."
        )

    def test_stray_fence_in_summary_does_not_hide_real_opener(self) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text("## Goal\nREAL_SUMMARY ``` note\n<think>PRIVATE_THINK ```") == (
            "## Goal\nREAL_SUMMARY ``` note"
        )

    def test_tilde_fenced_tag_mention_survives(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\n~~~\n<think>\n~~~\n## Next\nShip."
        assert summary_visible_text(text) == text

    def test_fenced_closer_mention_is_cut_known_limit(self) -> None:
        """Known limit (RIG-10913): a fence that contains a bare closer protects nothing, so a closer
        quoted inside a summary's code block still cuts. Storing reasoning is the worse failure."""
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\n```\n</think>\n```\n## Next\nShip."
        assert summary_visible_text(text) == "```\n## Next\nShip."

    @pytest.mark.parametrize("text,private,visible", [
        ("## Goal\nREAL\n<tool_call>{\"name\": \"terminal\", \"arguments\": {\"cmd\": \"rm -rf\"}}</tool_call>\n"
         "## Next\nShip.", "rm -rf", "## Goal\nREAL\n\n## Next\nShip."),
        ("## Goal\nREAL\n<tool_result>PRIVATE_OUTPUT</tool_result>\nShip.", "PRIVATE_OUTPUT", "## Goal\nREAL\n\nShip."),
        ("## Goal\nREAL\n<function_calls><invoke name=\"x\"></invoke></function_calls>\nShip.", "invoke",
         "## Goal\nREAL\n\nShip."),
        ("## Goal\nREAL.\n<function name=\"terminal\">{\"cmd\":\"ls\"}</function>\nShip.", "terminal",
         "## Goal\nREAL.\n\nShip."),
        ("## Goal\nREAL\n<tool_call>{\"name\": \"terminal\", \"arguments\": {", "tool_call", "## Goal\nREAL"),
        ("## Goal\nREAL</tool_call>\nShip.", "tool_call", "## Goal\nREALShip."),
        ("## Goal\nREAL\nterminal<arg_key>cmd</arg_key>\n<arg_value>ls</arg_value>\nShip.", "arg_",
         "## Goal\nREAL\nShip."),
        ("## Goal\nREAL\n<think type=\"x\">PRIVATE_THINK", "PRIVATE_THINK", "## Goal\nREAL"),
    ])
    def test_tool_call_xml_and_attribute_opener_are_cut(self, text: str, private: str, visible: str) -> None:
        from agent.think_scrubber import summary_visible_text

        out = summary_visible_text(text)
        assert private not in out
        assert out == visible

    @pytest.mark.parametrize("text", [
        "## Goal\nFix `<tool_call>x</tool_call>` parse.",
        "## Goal\nDrop the stray `</tool_call>` closer.\n## Next\nShip.",
        "## Goal\nFix the `<think type=\"x\">` opener.\n## Next\nShip.",
    ])
    def test_inline_tool_or_attribute_tag_mention_survives(self, text: str) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text(text) == text

    def test_long_backtick_run_is_linear(self) -> None:
        import time
        from agent.think_scrubber import summary_visible_text

        start = time.perf_counter()
        summary_visible_text("`" * 40000)
        assert time.perf_counter() - start < 0.5

    @pytest.mark.parametrize("text,visible", [
        ("## Goal\nFix the `<think>...</think>` stripping in the parser.\n## Progress\nDone A.\n## Next\nShip.",
         "## Goal\nFix the `` stripping in the parser.\n## Progress\nDone A.\n## Next\nShip."),
        ("## Goal\nShip.\n## Notes\nThe `<thinking>a</thinking>` case is covered.",
         "## Goal\nShip.\n## Notes\nThe `` case is covered."),
        ("## Goal\nFix `<think>x</think>` parse.\n<think>PRIVATE_THINK", "## Goal\nFix `` parse."),
    ])
    def test_closed_pair_inside_inline_span_keeps_the_prefix(self, text: str, visible: str) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text(text) == visible

    def test_reasoning_between_two_tag_spans_is_kept_known_limit(self) -> None:
        """Known limit (L8): two single-tag spans are both mentions, so reasoning the model wraps in
        backticked tags is kept. It has the same shape as a summary naming both tags (F4)."""
        from agent.think_scrubber import summary_visible_text

        text = "`<think>`PRIVATE_THINK\nmore PRIVATE_THINK\n`</think>`\n## Goal\nREAL"
        assert summary_visible_text(text) == text

    @pytest.mark.parametrize("text", [
        "## Goal\nStrip `<think>` openers and `</think>` closers from summaries.\n## Next\nShip.",
        "## Goal\nThe `<think>` opener and\nthe `</think>` closer are handled.\n## Next\nShip.",
        "## Goal\nFix `<think>` handling.\n## Progress\n- Done A.\n## Decisions\n- Keep `</think>` closers.\n## Next\nShip.",
    ])
    def test_two_single_tag_spans_survive(self, text: str) -> None:
        from agent.think_scrubber import summary_visible_text

        assert summary_visible_text(text) == text

    @pytest.mark.parametrize("text", [
        "PRIVATE_THINK\n```\n<think>\n```\n</think>REAL",
        "PRIVATE_THINK about the `<think>` tag.</think>\n## Goal\nREAL",
        "PRIVATE_THINK: the parser drops `<think>` openers.\nMore PRIVATE_THINK.\n</think>\n## Goal\nREAL",
        "PRIVATE_THINK: handle `<thinking>` too\n</thinking>\n## Goal\nREAL",
        "PRIVATE_THINK\n```xml\n<think>\n```\nmore PRIVATE_THINK\n</think>\n## Goal\nREAL",
    ])
    def test_mention_in_reasoning_does_not_pair_with_real_closer(self, text: str) -> None:
        from agent.think_scrubber import summary_visible_text

        out = summary_visible_text(text)
        assert "PRIVATE_THINK" not in out
        assert out.endswith("REAL")

    def test_span_before_line_end_arg_closer_survives(self) -> None:
        from agent.think_scrubber import summary_visible_text

        text = "## Goal\n- Parser: `<arg_key>` fragments end with </arg_value>\n## Next\nShip."
        assert summary_visible_text(text) == text
