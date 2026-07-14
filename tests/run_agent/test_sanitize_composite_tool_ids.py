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

from agent.agent_runtime_helpers import sanitize_api_messages


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
