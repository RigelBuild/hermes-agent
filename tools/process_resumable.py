"""Admission of background chat commands that can be restarted without a shell."""

import contextlib
import io
import os
import shlex
from functools import lru_cache
from urllib.parse import urlsplit


@lru_cache(maxsize=1)
def _chat_parser():
    from hermes_cli._parser import build_top_level_parser
    return build_top_level_parser()[2]


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
    parser = _chat_parser()
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            args = parser.parse_args(argv[2:])
    except SystemExit:
        return None
    if any(getattr(args, name, None) is not None for name in ("api_key", "query", "request_id")):
        return None
    if any(getattr(args, name, None) for name in
           ("resume", "continue_last", "create_if_missing", "tui", "tui_native")):
        return None
    try:
        if getattr(args, "base_url", None) and urlsplit(args.base_url).username is not None:
            return None
    except ValueError:
        return None
    return argv
