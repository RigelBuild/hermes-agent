"""Pre-call sanitizer pairing across Codex Responses composite tool ids.

Regression for the bug where EVERY tool result rendered as
``[Result unavailable — see context summary above]`` in the model's context
even though the real result was persisted and visible in the gateway session
log.

Root cause: the Codex Responses API stores a tool result's ``tool_call_id`` as
a composite ``<call_id>|<response_item_id>`` (e.g.
``call_ABC|fc_0f81…``), while the assistant ``tool_call`` carries the plain
``call_ABC``.  ``sanitize_api_messages`` compared the two id spaces as raw
strings, so they were always disjoint: every real result was dropped as
"orphaned" (pass 1) and a stub injected for every call (pass 2).  This is the
result-side completion of the #58168 fix, which taught the *assistant* side to
register both ``id`` and ``call_id`` but never split the composite a result
carries.

Observed live on mattfw (sessions 20260714_124859_153086e0 and two others):
PAIRED=0 with assistant id==call_id==``call_…`` and result
``tool_call_id``==``call_…|fc_…``.
"""

from agent.agent_runtime_helpers import (
    repair_message_sequence,
    sanitize_api_messages,
)
from agent.context_compressor import ContextCompressor


def _tool_msgs(messages):
    return [m for m in messages if m.get("role") == "tool"]


def _has_stub(messages):
    return any(
        "Result unavailable" in (m.get("content") or "") for m in _tool_msgs(messages)
    )


def test_composite_result_id_pairs_with_plain_call():
    """A composite ``call_|fc_`` result pairs with its plain-``call_`` call.

    The real result must survive and NO ``[Result unavailable]`` stub may be
    injected — this is the exact shape seen live on mattfw.
    """
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ABC",
                    "call_id": "call_ABC",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_ABC|fc_0f81deadbeef",
            "content": "REAL RESULT OUTPUT",
        },
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    tools = _tool_msgs(out)
    assert len(tools) == 1, f"expected exactly one tool result, got {tools}"
    assert tools[0]["content"] == "REAL RESULT OUTPUT"
    assert not _has_stub(out), "real result was replaced by an unavailable stub"


def test_composite_id_multiple_calls_all_pair():
    """Every call in a multi-tool turn pairs; none is stubbed (PAIRED==N)."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "call_id": "call_1",
                    "type": "function",
                    "function": {"name": "search_files", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "call_id": "call_2",
                    "type": "function",
                    "function": {"name": "memory", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1|fc_aaa", "content": "R1"},
        {"role": "tool", "tool_call_id": "call_2|fc_bbb", "content": "R2"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    contents = {m["content"] for m in _tool_msgs(out)}
    assert contents == {"R1", "R2"}
    assert not _has_stub(out)


def test_plain_ids_still_pair():
    """Non-composite (Chat Completions) ids keep pairing — no regression."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": "plain result"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    assert [m["content"] for m in _tool_msgs(out)] == ["plain result"]
    assert not _has_stub(out)


def test_genuinely_orphaned_result_still_dropped():
    """A result matching no assistant call is still dropped (safety net intact)."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_real",
                    "call_id": "call_real",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_real|fc_1", "content": "kept"},
        {"role": "tool", "tool_call_id": "call_ghost|fc_9", "content": "orphan"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    contents = [m["content"] for m in _tool_msgs(out)]
    assert "kept" in contents
    assert "orphan" not in contents


def test_genuinely_missing_result_still_stubbed():
    """A call with no result at all still gets a stub (safety net intact)."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_answered",
                    "call_id": "call_answered",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                },
                {
                    "id": "call_dropped",
                    "call_id": "call_dropped",
                    "type": "function",
                    "function": {"name": "g", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_answered|fc_1", "content": "ok"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    # answered call keeps its real result; the missing one is stubbed.
    assert any(m.get("content") == "ok" for m in _tool_msgs(out))
    assert _has_stub(out)


def test_composite_duplicate_results_deduped():
    """Two results sharing a call_ component collapse to one (strict-provider 400)."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_d",
                    "call_id": "call_d",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_d|fc_1", "content": "first"},
        {"role": "tool", "tool_call_id": "call_d|fc_2", "content": "second"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    assert len(_tool_msgs(out)) == 1, "duplicate results for one call not deduped"


# --- Composite tool-call id pairing: the fix is incomplete across the two SIBLING pairing passes ---
#
# The prior fix taught ONLY ``sanitize_api_messages`` (via
# ``_tool_id_match_keys`` / ``_assistant_tc_match_keys``) to split a composite
# ``call_|fc_`` result id and pair it with its plain-``call_`` assistant call.
# Two sibling passes still compare ids as raw strings, and one of them
# (``repair_message_sequence`` Pass 1) runs FIRST in the live loop and drops the
# real result before the composite-aware sanitizer ever sees it. A third gap:
# the set-based dedup in ``sanitize_api_messages`` keys on EVERY split component,
# so two distinct calls whose results share an ``fc_`` half collide.


def _composite_shape():
    """The exact live shape: plain-``call_`` assistant call, composite result."""
    return [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_ABC",
                    "call_id": "call_ABC",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_ABC|fc_0f81deadbeef",
            "content": "REAL RESULT OUTPUT",
        },
    ]


def test_repair_then_sanitize_live_order_keeps_composite_result():
    """T1 (RED) — production order repair→sanitize must keep the real result.

    The live loop runs ``repair_message_sequence`` first, then
    ``sanitize_api_messages`` on the result. Pass 1 of repair drops the
    composite result as an orphan (raw-string membership), so the downstream
    composite-aware sanitizer sees an unanswered call and stubs it. The real
    ``REAL RESULT OUTPUT`` must survive both passes and no stub may appear.
    """
    msgs = _composite_shape()
    repair_message_sequence(None, msgs)
    out = sanitize_api_messages([dict(m) for m in msgs])

    tools = _tool_msgs(out)
    assert len(tools) == 1, f"expected exactly one tool result, got {tools}"
    assert tools[0]["content"] == "REAL RESULT OUTPUT", (
        "real composite result did not survive repair->sanitize; "
        f"got {tools[0].get('content')!r}"
    )
    assert not _has_stub(out), "real result was replaced by an unavailable stub"


def test_repair_alone_keeps_composite_result():
    """T2 (RED) — repair Pass 1 must not drop a valid composite result.

    ``repair_message_sequence`` mutates ``msgs`` in place. Its orphan-drop pass
    uses raw-string membership (``tc_id in known_tool_ids``), so a composite
    ``call_ABC|fc_...`` result never matches its plain-``call_ABC`` call and is
    dropped. The invariant: the valid composite result stays in the list.
    """
    msgs = _composite_shape()
    repair_message_sequence(None, msgs)

    tools = _tool_msgs(msgs)
    assert any(m.get("content") == "REAL RESULT OUTPUT" for m in tools), (
        "repair dropped the valid composite tool result; "
        f"surviving tool msgs: {tools}"
    )


def test_repair_then_sanitize_plain_ids_survive():
    """T3 (GREEN) — plain Chat-Completions ids survive repair->sanitize.

    Regression guard: the fix for composite ids must not break the plain-id
    path, which already pairs (``tool_call_id == id``) in both passes.
    """
    msgs = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": "plain result"},
    ]
    repair_message_sequence(None, msgs)
    out = sanitize_api_messages([dict(m) for m in msgs])

    assert [m["content"] for m in _tool_msgs(out)] == ["plain result"]
    assert not _has_stub(out)


def test_compressor_sanitize_keeps_composite_result():
    """T4 (RED) — compressor _sanitize_tool_pairs must keep a composite result.

    ``_sanitize_tool_pairs`` computes orphans by raw set difference
    (``result_call_ids - surviving_call_ids``), so a composite result
    (``call_ABC|fc_...``) and its plain-``call_ABC`` call fall through as
    mutually orphaned: the result is dropped AND the assistant tool_call is
    stripped (replaced with a "(tool call removed)" placeholder).
    """
    compressor = ContextCompressor(
        model="test", quiet_mode=True, config_context_length=200000
    )
    msgs = _composite_shape()
    out = compressor._sanitize_tool_pairs([dict(m) for m in msgs])

    tools = _tool_msgs(out)
    assert any(m.get("content") == "REAL RESULT OUTPUT" for m in tools), (
        "compressor dropped the valid composite tool result; "
        f"surviving tool msgs: {tools}"
    )

    assistants = [m for m in out if m.get("role") == "assistant"]
    assert assistants, "assistant message vanished"
    a = assistants[0]
    call_ids = {
        tc.get("call_id") or tc.get("id") for tc in (a.get("tool_calls") or [])
    }
    assert "call_ABC" in call_ids, (
        "compressor stripped the assistant tool_call that the composite "
        f"result answers; assistant tool_calls: {a.get('tool_calls')}"
    )
    assert a.get("content") != "(tool call removed)", (
        "assistant turn was collapsed to the tool-call-removed placeholder"
    )


def test_compressor_sanitize_drops_genuine_orphan():
    """T5 (GREEN) — compressor still drops a genuinely orphaned result.

    A composite result whose ``call_`` half matches no assistant call
    (``call_ghost|fc_9``) must still be removed. Keeps the safety net honest.
    """
    compressor = ContextCompressor(
        model="test", quiet_mode=True, config_context_length=200000
    )
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_real",
                    "call_id": "call_real",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_real|fc_1", "content": "kept"},
        {"role": "tool", "tool_call_id": "call_ghost|fc_9", "content": "orphan"},
    ]
    out = compressor._sanitize_tool_pairs([dict(m) for m in msgs])

    contents = [m.get("content") for m in _tool_msgs(out)]
    assert "orphan" not in contents, "genuine orphan result was not dropped"


def test_dedup_keeps_distinct_calls_sharing_fc_component():
    """T6 (RED) — dedup must not collapse distinct calls sharing an fc_ half.

    Pass 3 of ``sanitize_api_messages`` keys dedup on EVERY split component of
    the composite id. Two distinct calls ``call_A`` / ``call_B`` whose results
    share the same ``fc_SAME`` response-item half collide on that shared key, so
    the second result is dropped as a "duplicate" and its call left unanswered.
    Both distinct results must survive.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_A",
                    "call_id": "call_A",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                },
                {
                    "id": "call_B",
                    "call_id": "call_B",
                    "type": "function",
                    "function": {"name": "g", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_A|fc_SAME", "content": "RA"},
        {"role": "tool", "tool_call_id": "call_B|fc_SAME", "content": "RB"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    contents = {m.get("content") for m in _tool_msgs(out)}
    assert contents == {"RA", "RB"}, (
        "distinct calls sharing an fc_ component were collapsed by dedup; "
        f"surviving tool contents: {contents}"
    )
    assert not _has_stub(out), "a distinct call was left unanswered and stubbed"


def test_dedup_collapses_true_duplicates_same_call():
    """T7 (GREEN) — true duplicates for the SAME call still collapse to one.

    Regression guard for the MED#3 narrowing: two results for the same call
    ``call_d`` (``call_d|fc_1`` / ``call_d|fc_2``) must still dedup to exactly
    one so a strict provider doesn't 400 on a duplicate tool_call_id.
    """
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_d",
                    "call_id": "call_d",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_d|fc_1", "content": "first"},
        {"role": "tool", "tool_call_id": "call_d|fc_2", "content": "second"},
    ]

    out = sanitize_api_messages([dict(m) for m in messages])

    assert len(_tool_msgs(out)) == 1, "duplicate results for one call not deduped"
