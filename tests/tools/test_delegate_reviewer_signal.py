from tools.delegate_tool_config import _subagent_auto_approve


def test_reviewer_text_alone_does_not_override_ordinary_prompt():
    assert _subagent_auto_approve("command", "**Reviewer:** flagged as dangerous") == "once"
    assert _subagent_auto_approve("command", "other wording", smart_denied=True) == "deny"
