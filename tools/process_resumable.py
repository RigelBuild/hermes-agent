"""Admission of background chat commands that can be restarted without a shell."""

import os
import shlex


def resumable_chat_argv(command: str) -> list[str] | None:
    """Return chat argv only when it contains no shell syntax or inline secrets."""
    if any(char in command for char in "`$\n\r;|&<>()"):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv:
        return None
    if os.path.basename(argv[0]) == "hermes":
        start = 1
    elif os.path.basename(argv[0]) == "python" and argv[1:3] == ["-m", "hermes_cli"]:
        start = 3
    else:
        return None
    if argv[start:start + 1] != ["chat"]:
        return None
    for token in argv[start + 1:]:
        if token == "--api-key" or token.startswith("--api-key="):
            return None
        if token == "--query" or token.startswith("--query=") or token.startswith("-q"):
            return None
    return argv
