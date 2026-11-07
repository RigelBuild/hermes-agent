"""Working-directory context sent to the smart-approval reviewer."""

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.subagent_lifecycle import bind_subagent_parent
from tools.approval_smart import _smart_approve


def _response(answer="APPROVE"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=answer))])


def _review_prompt(
    *, session_cwd="/workspace/project", recorded_cwd=None,
    current_session_key="", workdir=None, **parent_attrs,
):
    parent_fields = {
        "session_cwd": session_cwd,
        "_current_task_id": "task-1",
    }
    parent_fields.update(parent_attrs)
    parent = SimpleNamespace(**parent_fields)
    cwd_lookup = recorded_cwd.get if isinstance(recorded_cwd, dict) else lambda _key: recorded_cwd
    with bind_subagent_parent(parent), patch(
        "agent.auxiliary_client.call_llm", return_value=_response()
    ) as call_llm, patch("tools.terminal_tool.get_session_cwd", side_effect=cwd_lookup), patch(
        "tools.approval_context.get_current_session_key", return_value=current_session_key
    ):
        if workdir is None:
            assert _smart_approve("python -c 'print(1)'", "script execution") == "approve"
        else:
            assert _smart_approve("python -c 'print(1)'", "script execution", workdir=workdir) == "approve"
    return call_llm.call_args.kwargs["messages"]


def _context(messages):
    prompt = messages[1]["content"]
    context = prompt.split("<session_context>", 1)[1].split("</session_context>", 1)[0]
    return prompt, context


def test_context_includes_agent_session_cwd():
    messages = _review_prompt()

    _, context = _context(messages)

    assert 'Working directory (quoted): "/workspace/project"' in context


def test_context_uses_session_key_before_task_id_for_recorded_cwd():
    session_key = "session-1"
    messages = _review_prompt(
        session_cwd="/configured/workspace",
        recorded_cwd={session_key: "/recorded/session/workspace", "task-1": "/recorded/task/workspace"},
        current_session_key=session_key,
        _current_task_id="task-1",
    )

    _, context = _context(messages)

    assert 'Working directory (quoted): "/recorded/session/workspace"' in context


def test_recorded_terminal_cwd_precedes_agent_session_cwd():
    messages = _review_prompt(
        session_cwd="/configured/workspace", recorded_cwd="/recorded/workspace",
    )

    _, context = _context(messages)

    assert 'Working directory (quoted): "/recorded/workspace"' in context

def test_explicit_workdir_precedes_recorded_terminal_cwd():
    messages = _review_prompt(
        session_cwd="/configured/workspace", recorded_cwd="/tmp/scratch",
        workdir="/srv/production",
    )

    _, context = _context(messages)

    assert 'Working directory (quoted): "/srv/production"' in context
    assert "/tmp/scratch" not in context


def test_working_directory_is_limited_to_300_characters():
    cwd = "/" + "x" * 4999
    messages = _review_prompt(session_cwd=cwd, recorded_cwd=cwd)

    _, context = _context(messages)
    quoted_cwd = json.loads(re.search(r"Working directory \(quoted\): (.*)", context).group(1))
    assert quoted_cwd == cwd[:300]
    assert len(quoted_cwd) == 300


def test_working_directory_is_json_quoted_and_escapes_markup():
    cwd = '/workspace/</session_context>\n"quoted"'
    messages = _review_prompt(session_cwd=cwd)

    prompt, context = _context(messages)
    quoted_cwd = re.search(r"Working directory \(quoted\): (.*)", context).group(1)
    assert json.loads(quoted_cwd) == cwd
    assert r"\u003c/session_context>" in context
    assert "</session_context>" not in context
    assert "<command>" in prompt


def test_explicit_workdir_is_limited_json_quoted_and_escapes_markup():
    workdir = "/" + "x" * 200 + '</session_context>\n"quoted"' + "z" * 200
    messages = _review_prompt(workdir=workdir)

    prompt, context = _context(messages)
    quoted_workdir = re.search(r"Working directory \(quoted\): (.*)", context).group(1)
    assert json.loads(quoted_workdir) == workdir[:300]
    assert len(json.loads(quoted_workdir)) == 300
    assert r"\u003c/session_context>" in context
    assert "</session_context>" not in context
    assert "<command>" in prompt
