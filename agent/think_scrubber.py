"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``_strip_think_blocks`` is correct for a complete string but, run per-delta, erases an
opening ``<think>`` that arrives alone, so downstream state machines leak reasoning. This class
holds partial tags at delta boundaries until resolved; ``flush()`` releases held-back prose that
was not a tag; ``reset()`` at the top of each turn. An open tag only starts a block at a block
boundary (stream start / after a newline / whitespace-only line so far), so prose that *mentions*
``<think>`` is not suppressed; closed pairs are always suppressed (intentional).
"""

from __future__ import annotations

import re
from typing import Any, Tuple

__all__ = ["THINK_CLOSE_TAGS", "THINK_OPEN_TAGS", "THINK_TAG_NAMES", "StreamingThinkScrubber"]

# The one list of model reasoning tag names. Every surface that hides reasoning (this scrubber,
# the CLI stream filter, the gateway stream filter, the final-response regex stripper) binds to
# these; a tag added here is covered everywhere. Consumers match case-insensitively, so the
# literal tags are lowercase. The CJK names cover models (MiniMax-M3) that emit Chinese reasoning
# tags: 思考 (think), 反思 (reflect), 推理 (reason), 推敲 (deliberate). MiniMax-M3 also emits the
# namespaced ``<mm:think>``; streaming surfaces match literal tags, so a namespaced name is listed whole.
THINK_TAG_NAMES: tuple[str, ...] = (
    "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
    "思考", "反思", "推理", "推敲", "mm:think",
)
THINK_OPEN_TAGS: tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf`` (held-back partial-tag
    tail), ``_last_emitted_ended_newline`` (True iff the last emission ended with ``\\n`` or nothing
    was emitted yet — decides whether an open tag at buffer position 0 sits at a block boundary).
    """

    # Literal tags so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: tuple[str, ...] = THINK_OPEN_TAGS
    _CLOSE_TAGS: tuple[str, ...] = THINK_CLOSE_TAGS
    _ALL_TAGS: tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        # Reasoning text the most recent feed() stripped from inside think blocks (tags excluded).
        self.last_hidden: str = ""

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion ("" when it is all reasoning or held back)."""
        self.last_hidden = ""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        hidden: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix; the rest is reasoning.
                    hidden.append(self._hold_partial(buf, self._CLOSE_TAGS))
                    break
                hidden.append(buf[:close_idx])
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                # Pair tags are exact ``<name>``/``</name>``: inner text sits between them.
                hidden.append(buf[buf.index(">", pair[0]) + 1:buf.rindex("<", pair[0], pair[1])])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        self.last_hidden = "".join(hidden)
        return "".join(out)

    def _hold_partial(self, buf: str, tags: tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """End-of-stream flush: inside an unterminated block the held-back content is discarded (leaking
        partial reasoning is worse than a truncated answer), otherwise the tail is emitted verbatim.
        Always resets the boundary flag — intra-turn retries flush then stream again without ``reset()``,
        and a stale False flag made the new stream's opening ``<think>`` look mid-line."""
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(buf: str, tags: tuple[str, ...]) -> tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = []
        for tag in self._OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, already_emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        """True iff *idx* is a block boundary: position 0 after a newline-terminated (or no) prior emission,
        or any position whose preceding text on the current line is whitespace-only (when no newline
        precedes it in *buf*, the prior emission must also have ended with a newline)."""
        prior_newline = already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag (full matches are real tags, handled elsewhere)."""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text


_TAG_ALT = "|".join(re.escape(name) for name in THINK_TAG_NAMES)
_CLOSER_RE = re.compile(rf"</(?:{_TAG_ALT})>", re.IGNORECASE)
_OPENER_RE = re.compile(rf"<(?:{_TAG_ALT})\b[^>]*>", re.IGNORECASE)
_FENCE = r"|^(?P<fence>`{3,}(?!`)|~{3,}(?!~))[^\n]*\n.*?^(?P=fence)[`~]*[ \t]*$"
_FLAGS = re.IGNORECASE | re.DOTALL | re.MULTILINE
# Where a bare reasoning tag is a mention: a span wrapping one tag, or a fence with no bare closer.
_CODE_RE = re.compile(rf"(?<!`)(`+)(?:<(?:{_TAG_ALT})\b[^>]*>|</(?:{_TAG_ALT})>)\1(?!`)" + _FENCE, _FLAGS)
# Where any tag is a mention for the shared patterns: a whole one-line inline span, or such a fence.
_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`)[^\n]*?(?<!`)\1(?!`)" + _FENCE, _FLAGS)
_MASK_RE = re.compile(r"[^<\n]")


def _mention_spans(text: str, code_re: re.Pattern = _CODE_RE) -> list[tuple[int, int]]:
    """Code spans that are mentions; a fence with a bare closer inside protects nothing."""
    return [m.span() for m in code_re.finditer(text)
            if m.group("fence") is None or not _CLOSER_RE.search(text, *m.span())]


def _drop(text: str, cuts: list[tuple[int, int]]) -> str:
    starts = [0, *(b for _, b in cuts)]
    return "".join(text[s:a] for s, (a, _) in zip(starts, cuts)) + text[starts[-1]:]


def _strip_shared(text: str) -> str:
    """``strip_think_blocks`` minus its orphan-tag pattern; only closed reasoning pairs match inside code."""
    from agent.agent_runtime_helpers import (
        _ORPHAN_REASONING_TAG_PATTERN, _REASONING_BLOCK_PATTERNS, _THINK_STRIP_PATTERNS)

    # Closed pairs go unmasked first: a pair inside a span would otherwise leave a bare closer to cut at.
    for pattern in _REASONING_BLOCK_PATTERNS:
        text = pattern.sub("", text)
    masked, pos = [], 0
    for a, b in _mention_spans(text, _SPAN_RE):
        masked += [text[pos:a], _MASK_RE.sub("\x01", text[a:b])]  # keep `<` and `\n` for line patterns
        pos = b
    masked = "".join(masked) + text[pos:]  # same length as text, so match offsets apply to both
    for pattern in _THINK_STRIP_PATTERNS:
        if pattern is _ORPHAN_REASONING_TAG_PATTERN or pattern in _REASONING_BLOCK_PATTERNS:
            continue
        cuts = [m.span() for m in pattern.finditer(masked)]
        if cuts:
            text, masked = _drop(text, cuts), _drop(masked, cuts)
    return text


def summary_visible_text(content: Any) -> str:
    """Visible text of a summarizer reply, for the stored summary only.

    Removes closed reasoning pairs everywhere, then runs the other shared ``strip_think_blocks``
    patterns (tool-call blocks, named ``<function>`` blocks, an unclosed line-start reasoning opener,
    stray tool-call closers, unclosed tool calls and arg fragments) outside inline code spans and
    fences. Then it cuts text before a bare closing tag and from an unclosed opening tag through the
    end, even mid-line. A bare tag does not start a cut when it is inside an inline span that wraps
    exactly that one tag, or a fence with no bare closer.
    """
    from agent.agent_runtime_helpers import _flatten_content_text

    text = _flatten_content_text(content) if content else ""
    # Shared patterns first, so the bare-tag cuts below (strict spans, re-found per cut) get the last word.
    text = _strip_shared(text)
    spans = _mention_spans(text)
    closers = [m for m in _CLOSER_RE.finditer(text) if not any(a <= m.start() < b for a, b in spans)]
    if closers:
        text = text[closers[-1].end():]
    spans = _mention_spans(text)
    openers = [m for m in _OPENER_RE.finditer(text) if not any(a <= m.start() < b for a, b in spans)]
    if openers:
        text = text[:openers[0].start()]
    return text.strip()
