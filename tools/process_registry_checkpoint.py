"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
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
                    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                    if not s.attempt_token:
                        for field in ("argv", "request_id", "attempt_token", "resume_count", "attempt_status"):
                            entry.pop(field, None)
                    # Redact inline credentials before persisting (~/.hermes/processes.json).
                    # Recovery uses command only for display (adoption re-validates the
                    # PID, never re-runs it), so masking is lossless.
                    # See #77484.
                    entry["command"] = redact_sensitive_text(s.command, code_file=True)
                    entry["owner_task_id"] = s.owner_task_id or s.task_id
                    entries.append(entry)
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
                entries.extend(self._foreign_handoffs(path, known))
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
                and ((entry.get('attempt_token') and entry.get('attempt_status') == 'unknown')
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
            entries = [item for item in entries if isinstance(item, dict) and item.get("session_id") != entry["session_id"]]
            entries.append(entry)
            atomic_json_write(path, entries)

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

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _checkpoint_path,
            _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log

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
        for entry in entries:
            if entry.get('attempt_status') == 'unknown':
                from tools.process_registry_output_log import _exit_path
                log = adoptable_log(entry.get('output_log'))
                pid = entry.get('pid')
                if not ((log and _exit_path(log).exists()) or
                        (pid and self._detached_host_fate(pid, entry.get('host_start_time')) == 'running')):
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
            session.recovered_checkpoint_path = str(checkpoint_path)
            if session.output_log:
                self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
            else:
                with self._lock:
                    self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
            # Re-enqueue watcher so gateway can resume notifications
            if session.watcher_interval > 0 or session.attempt_token:
                self.pending_watchers.append({
                    "session_id": session.id,
                    "check_interval": session.watcher_interval or 5,
                    "session_key": session.session_key,
                    **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                    "notify_on_complete": session.notify_on_complete,
                    "death_only": session.watcher_interval <= 0,
                    "parent_session_id": session.parent_session_id,
                })
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        return recovered
