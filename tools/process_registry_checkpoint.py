"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import os
import shlex
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")

SHUTDOWN_RESUME_REASON = "gateway_shutdown"
SHUTDOWN_RESUME_CAP = 3
RESUME_NOTE = ("The gateway restarted while this run was in progress. "
               "Check what is already done, then continue the task.")
_DROPPED_DESTS = frozenset({"query", "query_file", "image"})


def checkpoint_entry(s) -> dict[str, Any]:
    """The checkpoint projection of one session (attempt fields only for an admitted attempt)."""
    from tools.process_registry import _CHECKPOINT_FIELDS
    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
    if not s.attempt_token:
        for field in ("argv", "request_id", "attempt_token", "resume_count", "resume_session_id", "attempt_status"):
            entry.pop(field, None)
    # Redact inline credentials before persisting (~/.hermes/processes.json). Recovery uses
    # command only for display (adoption re-validates the PID, never re-runs it). See #77484.
    entry["command"] = redact_sensitive_text(s.command, code_file=True)
    entry["owner_task_id"] = s.owner_task_id or s.task_id
    return entry


def _without_request_id(argv: list[str]) -> list[str]:
    """Spawn admits the command before it appends --request-id, so admit the same shape."""
    if "--request-id" not in argv:
        return []
    at = argv.index("--request-id")
    return argv[:at] + argv[at + 2:]


def _dropped_option(options: dict[str, str], arg: str) -> tuple[bool, bool]:
    """(dropped, value_glued) for one token, matched like argparse: exact, glued, or unique long prefix."""
    if not arg.startswith("-") or arg == "-" or arg == "--":
        return False, False
    if arg.startswith("--"):
        name, glued = arg.split("=", 1)[0], "=" in arg
        if name in options:
            return options[name] in _DROPPED_DESTS, glued
        matches = {dest for option, dest in options.items() if option.startswith("--") and option.startswith(name)}
        return len(matches) == 1 and matches <= _DROPPED_DESTS, glued
    if arg in options:
        return options[arg] in _DROPPED_DESTS, False
    return options.get(arg[:2]) in _DROPPED_DESTS, True


def resume_launch_argv(argv: list[str], chat_session_id: str) -> list[str]:
    """The stored argv without its query source or image, resuming the chat with the continuation note."""
    from tools.process_resumable import _chat_parser
    options = {option: action.dest for action in _chat_parser()._actions for option in action.option_strings}
    out, skip = list(argv[:2]), False
    for arg in argv[2:]:
        if skip:
            skip = False
            continue
        dropped, glued = _dropped_option(options, arg)
        if dropped:
            skip = not glued  # a value glued on (--query-file=x, -qtext) needs no second token
            continue
        out.append(arg)
    return [*out, "--resume", chat_session_id, "--discard-unknown", "--query", RESUME_NOTE]


class ProcessCheckpointMixin:
    def _pending_attempt_owner_is_live(self, pid, expected_start):
        from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match

        if not pid or not _pid_exists(pid):
            return False
        if expected_start is None:
            # Without a baseline, a live PID is the original owner until it exits.
            return True
        current_start = get_process_start_time(pid)
        if current_start is None:
            return True
        try:
            return start_time_fingerprints_match(expected_start, current_start)
        except (TypeError, ValueError):
            return False

    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[list[dict[str, Any]]] = None, *, checkpoint_path=None, preserve_ids=None):
        """Write running process metadata to the checkpoint file atomically."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS

        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    if s.exited:
                        continue
                    # Backfill the start time so recovery can detect PID recycling
                    # even for sessions spawned before this field existed.
                    if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                        s.host_start_time = self._safe_host_start_time(s.pid)
                    entries.append(checkpoint_entry(s))
                if extra_entries:
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
                known = {*self._running, *self._finished} - (preserve_ids or set())
            from hermes_cli.active_sessions import _FileLock
            from utils import atomic_json_write
            from pathlib import Path
            path = Path(checkpoint_path) if checkpoint_path else _checkpoint_path()
            # Every interpreter of the profile (owner, each managed worker) rewrites this file from
            # its own table; a worker's output-log handoff is its only path to the next one, so a
            # writer keeps the handoffs it does not track (read-merge-write under one file lock).
            with _FileLock(path.with_name(path.name + ".lock")):
                emitted = {entry.get('session_id') for entry in entries}
                entries.extend(entry for entry in self._foreign_handoffs(path, known)
                               if entry.get('session_id') not in emitted)
                atomic_json_write(path, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def _foreign_handoffs(self, path, known) -> list[dict[str, Any]]:
        """On-disk worker handoffs this registry never tracked whose log still decides an outcome:
        the process is alive (same start time) or its recorded exit is waiting to be read."""
        from pathlib import Path
        from tools.process_registry_output_log import _exit_path, adoptable_log
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return []
        return [entry for entry in on_disk if isinstance(entry, dict)
                and ((entry.get('attempt_token') and entry.get('attempt_status') == 'unknown'
                      and not ((log := adoptable_log(entry.get('output_log'))) and _exit_path(log).exists()))
                     or (entry.get('session_id') not in known
                         and ((entry.get('attempt_token') and entry.get('attempt_status') in {'pending', 'running', 'claimed'})
                              or ((log := adoptable_log(entry.get('output_log'))) and Path(log).exists()
                                  and (_exit_path(log).exists() or self._detached_host_fate(
                                      entry.get('pid'), entry.get('host_start_time')) == 'running')))))]

    def _write_resumable_entry(self, entry):
        """Persist an admitted attempt; corrupt checkpoint data disables admission."""
        from hermes_cli.active_sessions import _FileLock
        from tools.process_registry import _checkpoint_path
        from utils import atomic_json_write

        path = _checkpoint_path()
        with _FileLock(path.with_name(path.name + ".lock")):
            try:
                entries = json.loads(path.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                entries = []
            if not isinstance(entries, list):
                raise ValueError("invalid process checkpoint")
            owner_pid = os.getpid()
            attempt = {
                **entry,
                "attempt_owner_pid": owner_pid,
                "attempt_owner_started_at": self._safe_host_start_time(owner_pid),
            }
            entries = [
                item for item in entries
                if isinstance(item, dict) and item.get("session_id") != attempt["session_id"]
            ]
            entries.append(attempt)
            atomic_json_write(path, entries)
        entry.update(attempt)

    def _remove_resumable_entry(self, session_id, attempt_token, checkpoint_path=None):
        from hermes_cli.active_sessions import _FileLock
        from tools.process_registry import _checkpoint_path
        from utils import atomic_json_write

        from pathlib import Path
        path = Path(checkpoint_path) if checkpoint_path else _checkpoint_path()
        with _FileLock(path.with_name(path.name + ".lock")):
            try:
                entries = json.loads(path.read_text(encoding="utf-8-sig"))
            except FileNotFoundError:
                return
            entries = [entry for entry in entries if not (isinstance(entry, dict)
                       and entry.get("session_id") == session_id and entry.get("attempt_token") == attempt_token)]
            atomic_json_write(path, entries)

    def _mark_adopted_attempt_running(self, path, session_id, token):
        from hermes_cli.active_sessions import _FileLock
        from utils import atomic_json_write
        with _FileLock(path.with_name(path.name + '.lock')):
            entries = json.loads(path.read_text(encoding='utf-8-sig'))
            for entry in entries:
                if (isinstance(entry, dict) and entry.get('session_id') == session_id
                        and entry.get('attempt_token') == token and entry.get('attempt_status') == 'unknown'):
                    entry['attempt_status'] = 'running'
            atomic_json_write(path, entries)

    def _recover_interrupted_attempts(self, only_session_id=None, checkpoint_path=None):
        """Mark interrupted attempts unknown and replay notices at least once on each boot."""
        from pathlib import Path
        from hermes_cli.active_sessions import _FileLock
        from tools.process_registry import _checkpoint_path, _WATCHER_ROUTE_KEYS
        from tools.process_registry_output_log import _exit_path, adoptable_log
        from utils import atomic_json_write

        path = Path(checkpoint_path) if checkpoint_path else _checkpoint_path()
        notices = []
        with _FileLock(path.with_name(path.name + '.lock')):
            try:
                entries = json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                return
            changed = False
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get('attempt_token'):
                    continue
                if only_session_id is not None and entry.get('session_id') != only_session_id:
                    continue
                status = entry.get('attempt_status')
                if status not in {'pending', 'running', 'claimed', 'unknown'}:
                    continue
                log = adoptable_log(entry.get('output_log'))
                if log and Path(_exit_path(log)).exists():
                    continue
                pid = entry.get('pid')
                if pid and self._detached_host_fate(pid, entry.get('host_start_time')) == 'running':
                    continue
                if status == 'pending' and entry.get('attempt_owner_pid'):
                    # Without a baseline, retain the old wait-for-PID-death behavior.
                    owner_pid = entry['attempt_owner_pid']
                    if self._pending_attempt_owner_is_live(
                            owner_pid, entry.get('attempt_owner_started_at')):
                        continue
                if status != 'unknown':
                    entry['attempt_status'] = 'unknown'
                    changed = True
                notices.append({
                    'session_id': entry['session_id'], 'session_key': entry.get('session_key', ''),
                    'parent_session_id': entry.get('parent_session_id', ''),
                    'attempt_token': entry['attempt_token'],
                    'checkpoint_path': str(path),
                    'task_id': entry.get('task_id', ''), 'owner_task_id': entry.get('owner_task_id', ''),
                    'command': entry.get('command', ''),
                    'started_at': entry.get('started_at', 0),
                    'recovery_notice': 'outcome UNKNOWN, verify before retrying',
                    **{key: entry.get('watcher_' + key, '') for key in _WATCHER_ROUTE_KEYS},
                })
            if changed:
                atomic_json_write(path, entries)
        if only_session_id is None:
            self.pending_watchers.extend(notices)
        else:
            import sys
            runner_ref = getattr(sys.modules.get('gateway.run'), '_gateway_runner_ref', None)
            runner = runner_ref() if callable(runner_ref) else None
            for notice in notices:
                if runner is None or not runner.arm_process_watcher(notice):
                    self.pending_watchers.append(notice)

    @staticmethod
    def _chat_session_lookup(checkpoint_path):
        """A request-id -> chat-session-id lookup over this profile's state.db, opened once on first use."""
        from pathlib import Path
        from hermes_state import SessionDB
        from hermes_state_local import POLICY_PREFIX

        home_db = Path(checkpoint_path).parent / "state.db"
        opened: list = []

        def lookup(request_id) -> Optional[str]:
            if not request_id or not home_db.is_file():
                return None
            if not opened:
                opened.append(SessionDB(db_path=home_db, read_only=True))
            rows = opened[0]._read_all(
                "SELECT key FROM state_meta WHERE key GLOB ? AND json_valid(value) "
                "AND json_extract(value, '$.request_id') = ? LIMIT 1", (POLICY_PREFIX + "*", request_id))
            return rows[0][0][len(POLICY_PREFIX):] if rows else None

        def close():
            for db in opened:
                db.close()
        return lookup, close

    def _update_attempt(self, checkpoint_path, entry, decide) -> Optional[dict[str, Any]]:
        """Under the checkpoint lock, apply ``decide`` to this attempt's on-disk entry; returns the
        updated entry, or None when the entry is gone or ``decide`` refuses it."""
        from pathlib import Path
        from hermes_cli.active_sessions import _FileLock
        from utils import atomic_json_write
        path = Path(checkpoint_path)
        with _FileLock(path.with_name(path.name + ".lock")):
            entries = json.loads(path.read_text(encoding="utf-8-sig"))
            for item in entries:
                if (isinstance(item, dict) and item.get("session_id") == entry.get("session_id")
                        and item.get("attempt_token") == entry.get("attempt_token")):
                    if not decide(item):
                        return None
                    atomic_json_write(path, entries)
                    return dict(item)
        return None

    def _claim_shutdown_resume(self, checkpoint_path, entry, chat_session_id) -> Optional[dict[str, Any]]:
        """Claim one marked attempt for relaunch: only the claimer that moves it to pending spawns."""
        owner_pid = os.getpid()
        owner_started = self._safe_host_start_time(owner_pid)

        def claim(item):
            count = int(item.get("resume_count") or 0)
            if (item.get("resume_reason") != SHUTDOWN_RESUME_REASON or item.get("attempt_status") != "unknown"
                    or count >= SHUTDOWN_RESUME_CAP):
                return False
            item.pop("resume_reason")
            item.update(attempt_status="pending", resume_count=count + 1, resume_session_id=chat_session_id,
                        attempt_owner_pid=owner_pid, attempt_owner_started_at=owner_started)
            return True
        return self._update_attempt(checkpoint_path, entry, claim)

    def _give_back_shutdown_resume(self, checkpoint_path, claimed) -> None:
        """Undo a claim whose spawn failed, so the next boot can try again."""
        def give_back(item):
            if item.get("attempt_status") != "pending" or item.get("resume_count") != claimed["resume_count"]:
                return False
            item.update(attempt_status="unknown", resume_count=claimed["resume_count"] - 1,
                        resume_reason=SHUTDOWN_RESUME_REASON)
            return True
        self._update_attempt(checkpoint_path, claimed, give_back)

    def _relaunch_shutdown_killed(self, entry: dict[str, Any], checkpoint_path, lookup) -> bool:
        """Claim a dead attempt the gateway killed at shutdown, up to the cap, and queue its relaunch
        for spawn_queued_relaunches; False when it was not claimed (the unknown notice then applies)."""
        import contextvars
        from tools.process_resumable import resumable_chat_argv

        if (entry.get("resume_reason") != SHUTDOWN_RESUME_REASON
                or int(entry.get("resume_count") or 0) >= SHUTDOWN_RESUME_CAP):
            return False
        argv = list(entry.get("argv") or [])
        admitted = _without_request_id(argv)
        if not admitted or resumable_chat_argv(shlex.join(admitted)) is None:
            return False
        sid = entry.get("resume_session_id") or lookup(entry.get("request_id"))
        if not sid:
            logger.warning("Not resuming background chat %s: no chat session for its request", entry.get("session_id"))
            return False
        claimed = self._claim_shutdown_resume(checkpoint_path, entry, sid)
        if claimed is None:
            return False
        unknown = [w for w in self.pending_watchers
                   if w.get("recovery_notice") and w.get("session_id") == claimed["session_id"]
                   and w.get("attempt_token") == claimed.get("attempt_token")]
        self.pending_watchers = [w for w in self.pending_watchers if not any(w is u for u in unknown)]
        # The spawn runs after READY, in this profile's scope captured now.
        self.pending_relaunches.append((contextvars.copy_context(), claimed, str(checkpoint_path), unknown))
        return True

    def spawn_queued_relaunches(self) -> tuple[int, list[dict[str, Any]]]:
        """Spawn the claimed relaunches without touching the event loop; returns the notices to arm."""
        queued, self.pending_relaunches = self.pending_relaunches, []
        spawned, notices = 0, []
        for context, claimed, checkpoint_path, unknown in queued:
            session = context.run(self._spawn_claimed_relaunch, claimed, checkpoint_path)
            if session is None:
                notices.extend(unknown)
                continue
            spawned += 1
            try:
                notices.extend(self._resumed_watchers(session, checkpoint_path))
            except Exception:
                logger.exception("Could not queue the resumed notice for %s", session.id)
        return spawned, notices

    def give_back_pending_relaunches(self) -> int:
        """Undo the claims a startup that never reached READY did not spawn, so the next boot retries."""
        queued, self.pending_relaunches = self.pending_relaunches, []
        for _context, claimed, checkpoint_path, _unknown in queued:
            try:
                self._give_back_shutdown_resume(checkpoint_path, claimed)
            except Exception:
                logger.exception("Could not give back background chat %s", claimed["session_id"])
        return len(queued)

    def _spawn_claimed_relaunch(self, claimed, checkpoint_path):
        from tools.process_registry import ProcessSession, _CHECKPOINT_FIELDS, _CHECKPOINT_DEFAULTS
        from tools.environments.local import _append_dropped_path_entries
        try:
            argv = list(claimed["argv"])
            fields = {f: claimed.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            # A fresh spawn epoch gives this run its own completion identity.
            fields.update(command=claimed.get("command", "unknown"), argv=argv,
                          owner_task_id=claimed.get("owner_task_id", "") or claimed.get("task_id", ""),
                          started_at=time.time(), pid=None, pid_scope="host",
                          host_start_time=None, systemd_unit="", output_log="")
            session = ProcessSession(id=claimed["session_id"], **fields)
            session.recovered_checkpoint_path = checkpoint_path
            # Same login-PATH fix-up as the original terminal spawn.
            command = _append_dropped_path_entries(
                shlex.join(resume_launch_argv(argv, claimed["resume_session_id"])),
                self._spawn_env(None).get("PATH", ""))
            return self._spawn_local_admitted(session, command, None, False, session.persist_on_release)
        except Exception:
            logger.exception("Could not resume background chat %s; leaving it for the next boot", claimed["session_id"])
            try:
                self._give_back_shutdown_resume(checkpoint_path, claimed)
            except Exception:
                logger.exception("Could not restore background chat %s after a failed resume", claimed["session_id"])
            return None

    def _arm_watcher(self, notice) -> None:
        """Start a watcher on the serving gateway, else keep it for the next drain."""
        import sys
        runner_ref = getattr(sys.modules.get('gateway.run'), '_gateway_runner_ref', None)
        runner = runner_ref() if callable(runner_ref) else None
        if runner is None or not runner.arm_process_watcher(notice):
            self.pending_watchers.append(notice)

    def _resumed_watchers(self, session, checkpoint_path) -> list[dict[str, Any]]:
        """A resumed notice that keeps the new entry, plus the relaunched session's own watcher.
        The notice is keyed on resume_count, so it never shares the run's completion identity."""
        from tools.process_registry import _WATCHER_ROUTE_KEYS
        notice = {
            "session_id": session.id, "session_key": session.session_key,
            "parent_session_id": session.parent_session_id, "attempt_token": session.attempt_token,
            "checkpoint_path": str(checkpoint_path), "task_id": session.task_id,
            "owner_task_id": session.owner_task_id, "command": session.command,
            "recovery_notice": f"resumed after gateway restart ({session.resume_count}/{SHUTDOWN_RESUME_CAP})",
            "keep_entry": True, "resume_count": session.resume_count,
            **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
        }
        return [notice, *self._recovered_watchers(session)]

    def _recovered_watchers(self, session) -> list[dict[str, Any]]:
        """The watcher that resumes this session's notifications, if it needs one."""
        from tools.process_registry import _WATCHER_ROUTE_KEYS
        if not (session.watcher_interval > 0 or session.attempt_token):
            return []
        return [{
            "session_id": session.id,
            "check_interval": session.watcher_interval or 5,
            "session_key": session.session_key,
            **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
            "notify_on_complete": session.notify_on_complete,
            "death_only": session.watcher_interval <= 0,
            "parent_session_id": session.parent_session_id,
        }]

    def recover_from_checkpoint(self, *, resume_shutdown_killed: bool = False) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions. Only the gateway boot path passes
        ``resume_shutdown_killed`` to relaunch jobs its own shutdown killed."""
        from tools.process_registry import _checkpoint_path

        # This host adopts what a crashed predecessor left running, so the jobs it starts must
        # survive its own crash for the next one: they stay out of the owner-death supervisor.
        self._jobs_outlive_host = True
        checkpoint_path = _checkpoint_path()
        if not checkpoint_path.exists():
            return 0
        try:
            self._recover_interrupted_attempts()
        except Exception:
            logger.exception('Could not classify interrupted background jobs')
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return 0
        recovered = 0
        unresolved_scope_entries: list[dict[str, Any]] = []
        lookup, close_lookup = self._chat_session_lookup(checkpoint_path)
        try:
            recovered = self._recover_entries(
                entries, checkpoint_path, unresolved_scope_entries, lookup if resume_shutdown_killed else None)
        finally:
            close_lookup()
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        return recovered

    def _recover_entries(self, entries, checkpoint_path, unresolved_scope_entries, lookup) -> int:
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _CHECKPOINT_DEFAULTS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log
        recovered = 0
        for entry in entries:
            if entry.get('attempt_status') == 'unknown':
                from tools.process_registry_output_log import _exit_path
                log = adoptable_log(entry.get('output_log'))
                pid = entry.get('pid')
                if not ((log and _exit_path(log).exists()) or
                        (pid and self._detached_host_fate(pid, entry.get('host_start_time')) == 'running')):
                    if lookup is not None:
                        try:
                            if self._relaunch_shutdown_killed(entry, checkpoint_path, lookup):
                                recovered += 1
                        except Exception:
                            logger.exception('Could not resume background chat %s', entry.get('session_id'))
                    continue
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            if not pid:
                continue
            # The registry is process-global, so every profile's checkpoint carries every live
            # process; a multiplexer recovering several homes must adopt each session once.
            with self._lock:
                already_tracked = entry.get("session_id") in self._running
            if already_tracked:
                continue
            if pid_scope != "host":  # in-sandbox PIDs mean nothing once the env handle is gone
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60], pid, pid_scope)
                continue
            # Alive and still ours: re-attach. A start-time probe that cannot
            # be read is not proof the PID was reused — dropping it would leave
            # a live child unsupervised, and marking it exited would invent a
            # completion. A positive mismatch means the number was recycled:
            # do not adopt it and do not signal it.
            fate = self._detached_host_fate(pid, entry.get("host_start_time"))
            # A worker spawn's output log and recorded exit outlive both the spawner and the PID,
            # so its reader decides the outcome (a recycled PID is never signalled: fate != running).
            log = adoptable_log(entry.get("output_log"))
            if log:
                fate = "running"
            if fate == "reused":
                logger.info(
                    "Not recovering session %s: pid %d is alive but its "
                    "start time no longer matches — PID was recycled onto "
                    "an unrelated process; refusing to adopt or signal it.",
                    entry.get("session_id", "?"), pid)
                continue
            if fate != "running":
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit, pid)
                    unresolved_scope_entries.append(entry)
                continue
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()))
            # detached: no Popen handle, but can report status + kill; a worker spawn's output log
            # (and recorded exit) outlives its spawner, so tail it from the start.
            fields["output_log"] = log
            session = ProcessSession(id=entry["session_id"], detached=True, **fields)
            if session.attempt_status == 'unknown':
                try:
                    self._mark_adopted_attempt_running(checkpoint_path, session.id, session.attempt_token)
                except Exception:
                    logger.exception('Could not mark adopted background job %s running', session.id)
                session.attempt_status = 'running'
            session.recovered_checkpoint_path = str(checkpoint_path)
            if session.output_log:
                self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
            else:
                with self._lock:
                    self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
            self.pending_watchers.extend(self._recovered_watchers(session))
        return recovered
