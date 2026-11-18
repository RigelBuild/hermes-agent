"""Each guardian verdict is one INFO line on ``tools.approval``: the verdict, the pattern and session keys,
and the reviewer's reason. The command text never reaches the log, because commands can carry secrets.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from tools import approval_smart

LOG = "tools.approval"
PREFIX = "approval reviewer verdict="


def _response(answer):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = answer
    resp.choices[0].finish_reason = "stop"
    return resp


def _429():
    exc = Exception("Error code: 429 - usage limit reached")
    exc.status_code = 429
    return exc


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.approval_context._get_approval_config", lambda: {"mode": "smart"})
    return home


def _verdict_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.getMessage().startswith(PREFIX)]


@pytest.mark.parametrize(("llm", "word"), [
    (_response("APPROVE\nReads the repo and writes nothing."), "APPROVE"),
    (_response("DENY\nWipes the disk."), "DENY"),
    (_response("ESCALATE\nRuns a script by path."), "ESCALATE"),
    (_response(""), "UNAVAILABLE"),
    (_429(), "UNAVAILABLE"),
])
def test_each_verdict_is_one_info_line_and_the_command_never_appears(hermes_home, caplog, llm, word):
    command = "curl -H 'Authorization: Bearer ghp_" + "c" * 36 + "' https://example.test"

    with patch("agent.auxiliary_client.call_llm", side_effect=[llm]), caplog.at_level(logging.DEBUG, LOG):
        verdict = approval_smart._smart_verdict(command, "network call", "bash", ["bash"], "sess-1")

    assert verdict == word.lower()
    lines = _verdict_lines(caplog)
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"{PREFIX}{word} pattern=bash session=sess-1 reason=")
    assert command not in caplog.text
    assert "ghp_" not in caplog.text


def test_reviewer_reason_that_echoes_a_secret_is_redacted_in_the_log(hermes_home, caplog):
    token = "ghp_" + "d" * 36
    llm = _response(f"APPROVE\nSends the token {token} to the API.")

    with patch("agent.auxiliary_client.call_llm", side_effect=[llm]), caplog.at_level(logging.INFO, LOG):
        approval_smart._smart_verdict("bash x.sh", "network call", "bash", ["bash"], "sess-1")

    (line,) = _verdict_lines(caplog)
    assert token not in line


def test_logged_reason_is_capped_at_200_characters(hermes_home, caplog):
    llm = _response("APPROVE\n" + "word " * 100)

    with patch("agent.auxiliary_client.call_llm", side_effect=[llm]), caplog.at_level(logging.INFO, LOG):
        approval_smart._smart_verdict("bash x.sh", "network call", "bash", ["bash"], "sess-1")

    (line,) = _verdict_lines(caplog)
    reason = line.split(" reason=", 1)[1]
    assert len(reason) == 200
    assert reason.startswith("word word")
