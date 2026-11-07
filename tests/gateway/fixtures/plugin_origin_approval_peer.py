"""Disposable real GatewayRunner fixture for plugin-origin approval gates."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import traceback


PLUGIN_ID = "approval-fixture"
PLUGIN_KEYS = ("allow_gateway_injection", "allow_origin_sessions")
CASES = {
    "review-approve": {"hook": True, "review": True, "answer": "once"},
    "review-escalate": {"hook": True, "review": True, "verdict": "escalate", "answer": "once"},
    "review-deny": {"hook": True, "review": True, "verdict": "deny", "answer": None},
    "human-deny": {"hook": True, "review": True, "answer": "deny"},
    "timeout": {"hook": True, "review": True, "answer": "timeout"},
    "stale-always": {"hook": True, "review": True, "answer": "always"},
    "terminal": {"tool": "terminal", "answer": "once"},
    "execute-code": {"tool": "execute_code", "answer": "once"},
    "yolo": {"hook": True, "review": True, "yolo": True, "answer": "once"},
    "session-yolo": {"hook": True, "review": True, "session_yolo": True, "answer": "once"},
    "mode-off": {"hook": True, "review": True, "mode": "off", "answer": "once"},
    "grants": {"hook": True, "review": True, "grants": True, "answer": "once"},
    "allowlist": {"tool": "terminal", "allowlist": True, "answer": "once"},
    "smart-approve": {"tool": "terminal", "mode": "smart", "verdict": "approve", "answer": "once"},
    "smart-deny": {"tool": "terminal", "mode": "smart", "verdict": "deny", "answer": None},
    "cotal-hook-deny": {"hook": True, "review": True, "peer": True, "answer": None},
    "cotal-code-deny": {"tool": "execute_code", "peer": True, "answer": None},
    "cotal-command-deny": {"tool": "terminal", "peer": True, "answer": None},
    "operator-bypasses": {"tool": "terminal", "operator": True, "yolo": True, "mode": "off",
                          "grants": True, "answer": None},
    "cotal-room-policy": {"hook": True, "review": True, "cotal_origin": True, "answer": "once"},
    "session-key-first": {"hook": True, "review": True, "mode": "off", "yolo": True,
                           "grants": True, "answer": "once", "session_key": 1},
    "session-key-second": {"hook": True, "review": True, "mode": "off", "yolo": True,
                            "grants": True, "answer": "once", "session_key": 2},
}


class ModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        messages = request.get("messages") or []
        system = next((str(message.get("content", "")) for message in messages if message.get("role") == "system"), "")
        if "security reviewer" in system:
            response = {"role": "assistant", "content": self.server.case.get("verdict_text", "APPROVE\nThe scoped fixture operation is expected.")}
            self._send_completion(request, response, "stop")
            return
        last_user = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1)
        user_text = str(messages[last_user].get("content", "")) if last_user >= 0 else ""
        is_seed = "seed turn" in user_text.lower() or "native_operator_seed" in user_text.lower()
        tools = request.get("tools") or []
        if tools and not is_seed:
            if self.server.action_user != user_text:
                self.server.action_user = user_text
                self.server.action_issued = False
            if not self.server.action_issued:
                tool = self.server.case.get("tool", "terminal")
                args = {"command": self.server.command} if tool == "terminal" else {"code": self.server.code}
                response = {"role": "assistant", "content": None, "tool_calls": [{
                    "index": 0, "id": f"fixture-call-{len(self.server.requests)}", "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
                self.server.action_issued = True
            else:
                response, finish = {"role": "assistant", "content": "fixture completed"}, "stop"
        else:
            response, finish = {"role": "assistant", "content": "fixture completed"}, "stop"
        self._send_completion(request, response, finish)

    def _send_completion(self, request, message, finish):
        base = {"id": "plugin-approval", "created": 1, "model": request.get("model") or "plugin-approval"}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk_base = {**base, "object": "chat.completion.chunk"}
            for delta, reason in ((message, None), ({}, finish)):
                frame = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(("data: " + json.dumps({**chunk_base, "choices": [], "usage": usage}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        response = {**base, "object": "chat.completion", "choices": [
            {"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


async def run_probe(peer, case_name):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy, bind_room_execution_policy, execution_policy_mapping,
        reset_room_execution_policy,
    )
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress import plugin_origin_turn
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_state_runtime import list_session_admissions
    from tests.waiting import async_wait_until
    from tools.approval import approve_permanent, approve_session, resolve_gateway_approval
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from hermes_cli import plugins
    from tools import approval, approval_smart

    case = CASES[case_name]
    home = Path(os.environ["HERMES_HOME"])
    effect = home / "effect"

    class RecordingDiscord(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="fixture-token",
                                            extra={"approval_mentions": True}), Platform.DISCORD)
            self.cards = []
            self.card_ready = asyncio.Event()
            self.chat_id = "9001"
            self._allowed_user_ids = {"12345"}
            self._client = type("Client", (), {"user": type("User", (), {"id": 999})()})()

        async def connect(self, *, is_reconnect=False):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="text")

        async def send_typing(self, chat_id, metadata=None):
            pass

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

        async def thread_owned_by_self(self, thread_id):
            return str(thread_id) == self.chat_id

        async def _send_exec_approval_prompt(self, prompt):
            self.cards.append(prompt)
            self.card_ready.set()
            return SendResult(success=True, message_id="card")

        def _approval_mention_content(self):
            return "<@12345>"

        def pause_typing_for_chat(self, chat_id):
            pass

        def resume_typing_for_chat(self, chat_id):
            pass

    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingDiscord()
    runner.adapters[Platform.DISCORD] = adapter
    adapter.gateway_runner = runner
    source = adapter.build_source(chat_id=adapter.chat_id, chat_name="fixture thread", chat_type="thread",
                                  user_id="12345", user_name="operator", thread_id=adapter.chat_id,
                                  parent_chat_id="8001", scope_id="7001")
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="approval-fixture")
    manager = PluginManager(scope_key=str(home))
    context = PluginContext(PluginManifest(name=PLUGIN_ID, key=PLUGIN_ID, source="user"), manager)
    if case.get("hook"):
        hook_result = {"action": "approve", "message": "fixture review rule", "rule_key": "fixture-rule",
                       "review": case.get("review", False)}
        context.register_hook("pre_tool_call", lambda **_: hook_result)
    manager._discovered = True
    plugins._plugin_manager = manager
    runner._wire_adapter_handlers(adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    authority.register(source)
    runner._install_plugin_message_injector()

    async def answer_cards(*, answer, expected, before, settled):
        seen = before
        while not settled.is_set() or (expected and seen == before):
            if len(adapter.cards) > seen:
                prompt = adapter.cards[seen]
                seen += 1
                adapter.card_ready.clear()
                assert set(prompt.choices) == {"once", "deny"}, prompt.choices
                assert "wants to run" in prompt.text.lower(), prompt.text
                assert adapter._approval_mention_content() == "<@12345>"
                if answer not in {None, "timeout"}:
                    assert answer in {"once", "deny", "always"}
                    assert resolve_gateway_approval(session_key, answer) == 1
                continue
            try:
                await asyncio.wait_for(adapter.card_ready.wait(), timeout=0.2)
            except TimeoutError:
                pass
        assert seen > before if expected else seen == before

    async def wait_for_turn(*, origin, answer, expected, text, session_key_injection=False, native=False):
        before = len(adapter.cards)
        adapter.card_ready.clear()
        settled = asyncio.Event()
        answer_task = asyncio.create_task(
            answer_cards(answer=answer, expected=expected, before=before, settled=settled))
        if origin:
            await asyncio.to_thread(context.inject_message, text, origin=source.to_dict())
        elif session_key_injection:
            await asyncio.to_thread(context.inject_message, text, session_key=session_key)
        else:
            event = MessageEvent(text=text, source=source, message_id=text)
            await adapter.handle_message(event)
        def admission_settled():
            for session_id in authority.sessions:
                for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False):
                    payload = row["payload"]
                    text_value = payload.get("text") or payload.get("native_text_v1", {}).get(
                        "event", {}).get("text")
                    if text_value == text and row["status"] == "terminal":
                        return True
            return False

        await async_wait_until(admission_settled, timeout=30, what=f"{case_name} admission settlement")
        settled.set()
        await answer_task
        assert len(adapter.cards) >= before + int(expected)

    session_key = runner._session_key_for_source(source)
    key_token = set_current_session_key(session_key)
    policy_token = None
    approval._YOLO_MODE_FROZEN = bool(case.get("yolo"))
    if case.get("session_yolo"):
        approval.enable_session_yolo(session_key)
    if case.get("grants") or case.get("operator"):
        for grant in ("plugin_rule:fixture-rule", "recursive delete", "execute_code"):
            approve_permanent(grant)
            approve_session(session_key, grant)
    if case.get("allowlist"):
        approval._permanent_approved.add("recursive delete")
    if case.get("mode"):
        from tools import approval_context
        approval_context._get_approval_mode = lambda: case["mode"]
    if case.get("verdict"):
        approval_smart._smart_approve = lambda *_: case["verdict"]
    if case.get("cotal_origin"):
        policy = execution_policy_mapping(
            target_profile="fixture", config={"approvals": {"mode": "off"}})
        policy_token = bind_room_execution_policy(RoomExecutionPolicy.from_mapping(policy))
    try:
        answer = case.get("answer")
        should_run = bool(case.get("operator") or answer in {"once", "always"})
        if case.get("peer"):
            from gateway.session_context import _SESSION_PLATFORM
            platform_token = _SESSION_PLATFORM.set("cotal")
            plugin_token = plugin_origin_turn.set(True)  # Directly mark this peer-mesh plugin-origin gate probe.
            try:
                if case.get("hook"):
                    result = approval.request_tool_approval(
                        "fixture_tool", "fixture peer review", rule_key="fixture-rule", review=True)
                elif case.get("tool") == "execute_code":
                    result = approval.check_execute_code_guard(peer.code, "local")
                else:
                    result = approval.check_all_command_guards(peer.command, "local")
            finally:
                plugin_origin_turn.reset(plugin_token)
                _SESSION_PLATFORM.reset(platform_token)
            assert result["approved"] is False, (case_name, result)
            assert not adapter.cards and not effect.exists(), case_name
        else:
            operator = case.get("operator")
            if operator:
                seed = MessageEvent(text="NATIVE_OPERATOR_SEED", source=source, message_id="operator-seed")
                await adapter.handle_message(seed)
                await async_wait_until(lambda: not adapter._active_sessions, timeout=30,
                                       what="operator seed turn")
            if case.get("session_key"):
                await wait_for_turn(origin=True, answer=None, expected=False, text="plugin seed turn")
                await wait_for_turn(origin=False, answer=None, expected=False,
                                    text="native operator seed", native=True)
                await wait_for_turn(origin=False, answer=answer, expected=answer is not None,
                                    text="plugin approval fixture", session_key_injection=True)
            else:
                await wait_for_turn(origin=not operator, answer=answer, expected=answer is not None,
                                    text="plugin approval fixture" if not operator else "operator approval fixture")
        assert effect.exists() is should_run, (case_name, effect.exists(), should_run)
        if answer == "always":
            assert not approval.is_approved(session_key, "plugin_rule:fixture-rule")
            assert not approval._is_permanently_approved("plugin_rule:fixture-rule")
        print(f"PASS {case_name}: cards={len(adapter.cards)} effect={effect.exists()}")
    finally:
        resolve_gateway_approval(session_key, "deny", resolve_all=True)
        if policy_token is not None:
            reset_room_execution_policy(policy_token)
        runner._clear_plugin_message_injector()
        if authority.db is not None:
            await asyncio.to_thread(authority.db.close)
        reset_current_session_key(key_token)
        if getattr(runner, "session_authority", None) is authority:
            runner.session_authority = None
        runner.config = GatewayConfig()
def main():
    case_name = os.environ["APPROVAL_PROBE_CASE"]
    home = Path(os.environ["HERMES_HOME"])
    case = CASES[case_name]
    effect = home / "effect"
    target = home / "target"
    target.mkdir()
    (target / "payload").write_text("fixture", encoding="utf-8")
    peer = ThreadingHTTPServer(("127.0.0.1", 0), ModelPeer)
    peer.requests = []
    peer.action_user = None
    peer.action_issued = False
    peer.responses = []
    peer.case = case
    peer.command = f"rm -r {target} && touch {effect}"
    peer.code = f"from pathlib import Path; Path({str(effect)!r}).touch()"
    peer.sent_tool_call = False
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    os.environ.update(OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=url, DISCORD_ALLOWED_USERS="12345",
                      DISCORD_APPROVAL_MENTIONS="true", TERMINAL_ENV="local")
    config = {"model": {"default": "plugin-approval", "provider": "custom", "base_url": url},
              "approvals": {"mode": case.get("mode", "manual"), "timeout": 8},
              "streaming": {"enabled": False}, "terminal": {"cwd": str(home)},
              "auxiliary": {"title_generation": {"enabled": False}},
              "platform_toolsets": {"cotal": ["discord", "terminal", "code_execution"],
                                     "discord": ["terminal", "code_execution"]},
              "plugins": {"entries": {PLUGIN_ID: {key: True for key in PLUGIN_KEYS}}}}
    if case.get("allowlist"):
        config["command_allowlist"] = ["recursive delete"]
    if case.get("yolo"):
        os.environ["HERMES_YOLO_MODE"] = "1"
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    try:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        asyncio.run(run_probe(peer, case_name))
        (home / "receipt.json").write_text(
            json.dumps({"case": case_name, "passed": True}), encoding="utf-8")
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
