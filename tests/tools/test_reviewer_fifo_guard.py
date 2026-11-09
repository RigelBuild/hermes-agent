from tools import approval


def test_review_first_ignores_cached_approval(monkeypatch):
    monkeypatch.setattr(approval.approval_context, "is_plugin_origin_turn", lambda: False)
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "ask")
    monkeypatch.setattr(approval, "is_approved", lambda *args: True)
    monkeypatch.setattr(approval, "_presence", lambda cb: (cb, True, False, False))
    monkeypatch.setattr(approval, "_smart_verdict", lambda *args: "deny")
    monkeypatch.setattr(approval, "_record_denial", lambda *args: None)
    seen = {}
    monkeypatch.setattr(approval, "_human_decision", lambda *args, **kwargs: seen.update(kwargs) or {})
    approval._run_approval_gate(pattern_key="key", description="reason", display_target="action",
                                autoapprove_log_prefix="test", review_first=True)
    assert seen["smart_denied"] is True
