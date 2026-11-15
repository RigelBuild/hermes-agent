"""Accounting at completion of an exact heartbeat admission attempt.

Execution means entering the gateway's agent runner after turn preparation,
not reserving an adapter slot, successful model completion, or outbound delivery.
The done callback retains the watch's profile ContextVars and manager claim.
"""
import logging


logger = logging.getLogger("gateway.run")


async def resolve_heartbeat_owner(runner, event, entry):
    """Keep normal reset/topic resolution, then admit only the original lineage."""
    expected = getattr(event, "_heartbeat_session_id", None)
    if not expected:
        return True
    resolved = entry.session_id
    if resolved != expected:
        def compression_tip():
            return runner.session_store._db_for_key(entry.session_key).get_compression_tip(expected)

        tip = await runner._run_in_executor_with_context(compression_tip)
        if tip != resolved:
            return False
    # Keep a value, not the mutable routing entry: preparation and hooks can yield
    # to /new or /stop before the agent runner starts.
    event._heartbeat_resolved_session_id = resolved
    return heartbeat_owner_is_current(runner, event, entry.session_key)


def heartbeat_owner_is_current(runner, event, session_key):
    expected = getattr(event, "_heartbeat_resolved_session_id", None)
    if not expected:
        return True
    current = runner.session_store.lookup_by_session_key(session_key)
    return current is not None and not current.suspended and current.session_id == expected


def settle_heartbeat_attempt(event, manager):
    if not getattr(event, "_heartbeat_execution_started", False):
        try:
            manager.abandon_fire()
        except Exception:
            logger.warning("Failed to refund unexecuted heartbeat", exc_info=True)


async def heartbeat_route_deliverable(runner, adapter, source, entry, *, check_envelope=True):
    """Whether a heartbeat can use this local, legacy, or committed messaging route.

    The poller uses the cheap adapter/local/legacy checks on every cycle; admission provenance
    is checked only for a slash command or when a due tick is actually admitted.
    """
    from gateway.session_local import LocalSessionAdapter

    if adapter is None:
        return False
    local = isinstance(adapter, LocalSessionAdapter)
    if not (getattr(adapter, "_message_handler", None) or local):
        return False
    if local or getattr(runner, "session_authority", None) is None or not check_envelope:
        return True
    from gateway.session_authorities import active_authority
    from gateway.session_automation import validated_committed_route_envelope

    authority = active_authority(runner)
    if authority is None or entry is None:
        return False

    await runner._run_in_executor_with_context(
        validated_committed_route_envelope, authority, adapter, source, entry
    )
    return True


def settle_busy_heartbeat(turn):
    """Refund a heartbeat steer the running agent never consumed."""
    claim, turn.heartbeat_claim = turn.heartbeat_claim, None
    if claim is None:
        return
    agent, ticket, manager = claim
    if agent.withdraw_steer(ticket):
        manager.abandon_fire()


def steer_busy_heartbeat(runner, source, session_id, route):
    """Claim a due steer only when this route has a running agent."""
    from agent.estop import check_paused
    from gateway.run import _AGENT_PENDING_SENTINEL
    from hermes_cli.heartbeat import HeartbeatManager

    state = runner._peek_session_state(route)
    agent = state.turn.agent if state else None
    if agent is None or agent is _AGENT_PENDING_SENTINEL or state.turn.heartbeat_steered:
        return
    manager = HeartbeatManager(session_id)
    if not manager.has_heartbeat() or manager.state.delivery_mode != "steer":
        return
    if check_paused("heartbeat", logger):
        return
    prompt = manager.due_prompt()
    if not prompt:
        return
    event = runner._synthetic_prompt_event(source, prompt)
    event._heartbeat_session_id = session_id
    event.metadata["gateway_session_key"] = route
    ticket = ("heartbeat", session_id, manager.state.fire_count)
    try:
        accepted = bool(agent.steer(runner._steer_text_with_origin(prompt, event), ticket=ticket))
    except Exception:
        logger.warning("Heartbeat steer failed for session %s", route, exc_info=True)
        manager.abandon_fire()
        return
    if accepted:
        state.turn.heartbeat_steered = True
        state.turn.heartbeat_claim = (agent, ticket, manager)
    else:
        manager.abandon_fire()


async def admit_heartbeat(runner, adapter, source, session_id, route):
    """Coalesce overdue ticks behind the canonical FIFO, not an adapter-local slot."""
    import json
    from gateway.platforms.event import MessageEvent
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_state_runtime import list_session_admissions

    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:
        return
    manager = HeartbeatManager(session_id)
    if manager.state and manager.state.delivery_mode == "steer" and runner._is_session_running(route):
        from gateway.run import _AGENT_PENDING_SENTINEL
        turn = runner._peek_session_state(route).turn
        if turn.agent is _AGENT_PENDING_SENTINEL or turn.heartbeat_steered:
            return
        authority._require_admission_open()
        if not manager.state.is_due():
            return
        from gateway.session_automation import snapshot_automation
        event = MessageEvent(text=manager.state.render_prompt(), source=source, internal=True,
            metadata={'gateway_session_key': route, 'gateway_session_id': session_id})
        await runner._run_in_executor_with_context(snapshot_automation, authority, adapter, event, 'heartbeat-validation')
        authority._require_admission_open()
        if runner.session_store.peek_session_id(route) == session_id and runner._is_session_running(route):
            steer_busy_heartbeat(runner, source, session_id, route)
        return
    # Admissions live on the route's logical owner (the compression root), not its transcript tip;
    # a LOCAL route's heartbeat rides local_automation_v1.
    pending = list_session_admissions(authority.db, session_id=authority.logical_owner(session_id))
    if any(row['payload'].get('native_text_v1', {}).get('automation', {}).get('heartbeat')
           or row['payload'].get('local_automation_v1', {}).get('heartbeat') for row in pending):
        return
    manager = HeartbeatManager(session_id)
    prompt = manager.due_prompt()
    if prompt is None:
        return
    state = manager.state
    identity = json.dumps(['heartbeat', session_id, state.created_at, state.fire_count], separators=(',', ':'))
    event = MessageEvent(text=prompt, source=source, internal=True, message_id=identity,
        metadata={'gateway_session_key': route, 'gateway_session_id': session_id})
    event._heartbeat_session_id = session_id
    try:
        await authority.admit_automation(adapter, event, identity)
    except BaseException:
        # Readiness is not a delivery attempt, and cancellation before commit is not ACK.
        if getattr(event, '_gateway_accepted', False) is not True:
            manager.abandon_fire()
        raise
