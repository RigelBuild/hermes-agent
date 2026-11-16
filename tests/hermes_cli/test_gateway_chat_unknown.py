"""Resume must expose the admission needed by the existing fenced discard control."""
import argparse
import asyncio
from contextlib import nullcontext
import json

import pytest


@pytest.mark.asyncio
async def test_resume_displays_unknown_and_queue_then_discards_with_refreshed_generation(monkeypatch, capsys):
    from websockets.asyncio.server import serve

    from hermes_cli import gateway_chat

    unknown_id = "admission-unknown-0123456789abcdef0123456789abcdef"
    queued_id = "admission-queued-fedcba9876543210fedcba9876543210"
    snapshot = {
        "stored_session_id": "stored",
        "execution_generation": 9,
        "pending": [
            {"admission_id": unknown_id, "status": "unknown", "execution_generation": 4},
            {"admission_id": queued_id, "status": "queued", "execution_generation": None},
        ],
    }
    refreshed = {
        **snapshot,
        "execution_generation": 10,
        "pending": [{**snapshot["pending"][0], "execution_generation": 5}, snapshot["pending"][1]],
    }
    calls = []

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            method, params = request["method"], request["params"]
            calls.append((method, params))
            result = {}
            if method == "session.resume":
                resumes = sum(name == "session.resume" for name, _ in calls)
                result = snapshot if resumes == 1 else refreshed
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}))

    class Input:
        submitted = False

        def __init__(self, **_kwargs):
            pass

        async def prompt_async(self, _prompt):
            if self.submitted:
                return "/quit"
            output = capsys.readouterr()
            # The user must be able to copy the full admission ID, not guess it
            # from the session ID or inspect the gateway's storage.
            commands = [line.strip() for line in output.err.splitlines() if line.strip().startswith("/discard ")]
            assert commands == [f"/discard {unknown_id}"]
            assert "This chat is paused" in output.err
            assert f"    (queued message {queued_id})" in output.err.splitlines()
            assert "waiting" in output.err.lower()
            assert "won't re-run" in output.err.lower()
            # stdout holds the REPL banner only: no transcript or replay is rendered before
            # the operator has chosen what to do with the unknown admission.
            assert output.out.strip().splitlines() == ["Welcome to Hermes Agent! Type your message or /help for commands."]
            assert calls == [
                ("runtime.describe", {}),
                ("session.resume", {"session_id": snapshot["stored_session_id"]}),
            ], "Rendering recovery controls must not submit or resolve any work"
            self.submitted = True
            return commands[0]

    monkeypatch.setattr("prompt_toolkit.PromptSession", Input)
    monkeypatch.setattr("prompt_toolkit.patch_stdout.patch_stdout", nullcontext)
    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{port}")
        assert await asyncio.wait_for(
            gateway_chat.run_gateway_chat(argparse.Namespace(resume=snapshot["stored_session_id"])), 5
        ) == 0

    assert calls == [
        ("runtime.describe", {}),
        ("session.resume", {"session_id": snapshot["stored_session_id"]}),
        ("session.resume", {"session_id": snapshot["stored_session_id"]}),
        ("prompt.resolve_unknown", {
            "session_id": snapshot["stored_session_id"],
            "admission_id": unknown_id,
            "execution_generation": refreshed["pending"][0]["execution_generation"],
        }),
    ]


@pytest.mark.asyncio
async def test_discard_unknown_settles_the_lost_turn_before_the_oneshot_note(monkeypatch):
    """A gateway relaunch resumes with --discard-unknown: the lost turn is acknowledged first,
    so the finite continuation note is submitted instead of refused."""
    from contextlib import asynccontextmanager

    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView

    lost = {"admission_id": "admission-lost", "status": "unknown", "execution_generation": 4}
    snapshots = [{"stored_session_id": "stored", "execution_generation": 4, "pending": [lost], "info": {}},
                 {"stored_session_id": "stored", "execution_generation": 5, "pending": [], "info": {}}]
    calls, ran = [], []

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "session.resume":
                return snapshots.pop(0)
            return {}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def run(self, query, *, oneshot):
        ran.append((query, oneshot, self.unknown_admissions()))
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", run)
    args = argparse.Namespace(resume="stored", discard_unknown=True, query="continue", quiet=True)
    assert await gateway_chat.run_gateway_chat(args) == 0

    assert ("prompt.resolve_unknown", {"session_id": "stored", "admission_id": "admission-lost",
                                       "execution_generation": 4}) in calls
    assert ran == [("continue", True, [])]
