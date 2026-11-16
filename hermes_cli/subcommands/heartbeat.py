"""List persisted route heartbeats for the active profile."""

from datetime import UTC, datetime


def _list_heartbeats(_args):
    from hermes_cli.heartbeat import format_interval, list_heartbeats

    heartbeats = list_heartbeats()
    if not heartbeats:
        print("No heartbeats in the active profile.")
        return

    for route, state in heartbeats:
        last_fired = (
            datetime.fromtimestamp(state.last_fired_at, UTC).isoformat()
            if state.last_fired_at else "never"
        )
        print(f"Route: {route}")
        print(f"  Status: {state.status}")
        print(f"  {'Schedule' if state.schedule else 'Interval'}: {state.cadence_label() if state.schedule else format_interval(state.interval_seconds)}")
        next_fire = state.next_fire_or_none()
        next_label = datetime.fromtimestamp(next_fire, UTC).isoformat() if next_fire is not None else state.last_error
        print(f"  Next fire: {next_label if state.status == 'active' else 'paused'}")
        print(f"  Last fired: {last_fired}")
        print(f"  Fire count: {state.fire_count}")
        print(f"  Last error: {state.last_error or 'none'}")
        print()


def build_heartbeat_parser(subparsers):
    parser = subparsers.add_parser("heartbeat", help="Inspect route heartbeats")
    commands = parser.add_subparsers(dest="heartbeat_command", required=True)
    commands.add_parser("list", help="List heartbeats in the active profile").set_defaults(func=_list_heartbeats)
    return parser
