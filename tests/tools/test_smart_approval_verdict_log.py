"""Each guardian verdict is one INFO line on ``tools.approval``: the verdict, the pattern and session keys,
and the length of the reviewer's note. The note text is logged only for ``unavailable``, whose notes are
fixed strings. A reviewer note often quotes the command, so approve, deny and escalate never log it.
The command text never reaches the log, because commands can carry secrets.
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


@pytest.mark.parametrize(("llm", "word", "reason"), [
    (_response("APPROVE\nReads the repo and writes nothing."), "APPROVE", None),
    (_response("DENY\nWipes the disk."), "DENY", None),
    (_response("ESCALATE\nRuns a script by path."), "ESCALATE", None),
    (_response(""), "UNAVAILABLE", "empty reply"),
    (_429(), "UNAVAILABLE", "rate limited (429)"),
])
def test_each_verdict_is_one_info_line_and_the_command_never_appears(hermes_home, caplog, llm, word, reason):
    command = "curl -H 'Authorization: Bearer ghp_" + "c" * 36 + "' https://example.test"

    with patch("agent.auxiliary_client.call_llm", side_effect=[llm]), caplog.at_level(logging.DEBUG, LOG):
        verdict = approval_smart._smart_verdict(command, "network call", "bash", ["bash"], "sess-1")

    assert verdict == word.lower()
    lines = _verdict_lines(caplog)
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"{PREFIX}{word} pattern=bash session=sess-1 reason_len=")
    if reason is None:
        assert " reason=" not in lines[0]
    else:
        assert lines[0].endswith(f" reason={reason}")
    assert command not in caplog.text
    assert "ghp_" not in caplog.text


def test_reason_len_counts_the_reviewers_note(hermes_home, caplog):
    note = "Reads the repo and writes nothing."

    with patch("agent.auxiliary_client.call_llm", return_value=_response(f"APPROVE\n{note}")), \
            caplog.at_level(logging.INFO, LOG):
        approval_smart._smart_verdict("bash x.sh", "network call", "bash", ["bash"], "sess-1")

    (line,) = _verdict_lines(caplog)
    assert line.endswith(f"reason_len={len(note)}")


@pytest.mark.parametrize("answer", [
    "DENY\nPipes curl -u admin:hunter2 to the API, leaking the password.",
    "APPROVE\nSends the token ghp_" + "d" * 36 + " to the API.",
    "ESCALATE\nReads credentials from admin:hunter2 in the URL.",
])
def test_reviewer_echoing_a_secret_is_not_logged(hermes_home, caplog, answer):
    note = answer.split("\n", 1)[1]

    with patch("agent.auxiliary_client.call_llm", return_value=_response(answer)), \
            caplog.at_level(logging.DEBUG, LOG):
        approval_smart._smart_verdict("bash x.sh", "network call", "bash", ["bash"], "sess-1")

    (line,) = _verdict_lines(caplog)
    assert "reason_len=" in line
    assert note not in caplog.text
    assert "hunter2" not in caplog.text
    assert "ghp_" not in caplog.text


def test_unavailable_reason_is_capped_at_200_characters(caplog):
    approval_smart._last_note.text = "x" * 300

    with caplog.at_level(logging.INFO, LOG):
        approval_smart._log_verdict("unavailable", "bash", "sess-1")

    (line,) = _verdict_lines(caplog)
    assert line.endswith(f"reason_len=300 reason={'x' * 200}")
