"""Real Discord approval gates for plugin-origin turns and cotal precedence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("case", [
    "review-approve", "review-escalate", "review-deny", "review-mixed-queue", "human-deny", "timeout", "stale-always",
    "terminal", "execute-code", "yolo", "session-yolo", "mode-off", "grants", "allowlist",
    "smart-approve", "smart-deny", "cotal-hook-deny", "cotal-command-deny", "cotal-code-deny",
    "operator-bypasses",
    "cotal-room-policy", "session-key-first", "session-key-second",
])
def test_plugin_origin_approval_round_trip(tmp_path, case):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    home.mkdir()
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(home), PYTHONPATH=str(repo),
               PYTHONUNBUFFERED="1", APPROVAL_PROBE_CASE=case)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "fixtures" / "plugin_origin_approval_peer.py")],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    receipt = json.loads((home / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["case"] == case
    assert receipt["passed"] is True
    print(result.stdout)
