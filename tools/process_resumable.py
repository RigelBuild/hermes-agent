"""Admission of background chat commands that can be restarted without a shell."""

import contextlib
import io
import os
import shlex
from urllib.parse import urlsplit


def resumable_chat_argv(command: str) -> list[str] | None:
    """Return chat argv only when it contains no shell syntax or inline secrets."""
    if any(char in command for char in "`$\n\r;|&<>()~*?[{#!"):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if len(argv) < 2 or os.path.basename(argv[0]) != "hermes" or argv[1] != "chat":
        return None
    from hermes_cli._parser import build_top_level_parser
    parser = build_top_level_parser()[2]
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            args = parser.parse_args(argv[2:])
    except SystemExit:
        return None
    if any(getattr(args, name, None) is not None for name in ("api_key", "query", "request_id")):
        return None
    try:
        if getattr(args, "base_url", None) and urlsplit(args.base_url).username is not None:
            return None
    except ValueError:
        return None
    return argv
