"""Durable admission for plugin turns that bootstrap bot-owned Discord threads."""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, admit_session_input


class PluginOriginRefused(Exception):
    """A plugin-origin admission no longer meets its live authorization predicate."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _plugin_config(plugin_home: Path) -> dict | None:
    """Read this plugin's effective config under its own home and fail closed."""
    try:
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli.plugins_loader import _plugin_home_scope
        home = Path(plugin_home)
        with _plugin_home_scope(home):
            config = load_user_config_effective(home / 'config.yaml', fail_closed=True)
        return config if isinstance(config, dict) else None
    except Exception:
        return None


def _plugin_entry(config: dict, plugin_id: str):
    plugins = config.get('plugins')
    entries = plugins.get('entries') if isinstance(plugins, Mapping) else None
    entry = entries.get(plugin_id) if isinstance(entries, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def allowed(plugin_home: Path, plugin_id: str) -> bool:
    """Whether this profile grants this plugin both gateway and origin injection."""
    config = _plugin_config(plugin_home)
    entry = _plugin_entry(config, plugin_id) if config is not None else None
    return bool(entry and entry.get('allow_gateway_injection') is True
                and entry.get('allow_origin_sessions') is True)


def origin_sessions_enabled(plugin_home: Path, plugin_id: str) -> bool:
    config = _plugin_config(plugin_home)
    if config is None:
        return True
    entry = _plugin_entry(config, plugin_id)
    return bool(entry and entry.get('allow_origin_sessions') is True)


def _thread_sessions_per_user(plugin_home: Path) -> bool | None:
    config = _plugin_config(plugin_home)
    if config is None:
        return None
    try:
        from gateway.config import GatewayConfig
        return GatewayConfig.from_dict(config).thread_sessions_per_user
    except Exception:
        return None


def eligible(runner, source) -> bool:
    """True when this source route has no committed native human message."""
    try:
        route = runner._session_key_for_source(source)
        entry = runner.session_store.lookup_by_session_key(route)
        if entry is None:
            return True
        from gateway.session_authorities import active_authority, authority_for_home
        from gateway.session_automation import committed_route_envelope
        from gateway.session_identity import identity_of
        identity = identity_of(source)
        authority = authority_for_home(runner, identity.runtime_home) if identity is not None else None
        if authority is None:
            authority = active_authority(runner)
        return authority is None or committed_route_envelope(authority, entry) is None
    except Exception:
        return False


def _source_for_plugin(source, plugin_id: str):
    from gateway.session_identity import replace_source
    label = f'plugin:{plugin_id}'
    return replace_source(source, user_id=label, user_name=label, is_bot=True)


async def admit(runner, authority, adapter, source, *, content: str, plugin_id: str, plugin_home: Path) -> bool:
    """Choose bootstrap or ordinary injection under the route's FIFO admission lock."""
    from functools import partial
    from gateway.session_runtime_workers import tracked_write

    route = runner._session_key_for_source(source)
    plugin_source = _source_for_plugin(source, plugin_id)
    identity = json.dumps(['plugin', plugin_id, route, uuid.uuid4().hex], separators=(',', ':'))
    event = MessageEvent(text=content, message_type=MessageType.TEXT, source=plugin_source,
                         message_id=identity, internal=True, allow_gateway_control=False)
    authority._require_admission_open()
    async with authority._route_turn(event):
        if not eligible(runner, source):
            if not runner._plugin_injection_authorized(
                    source, plugin_id=plugin_id, target=f'origin={source.platform.value}/{source.chat_id}'):
                return False
            entry = runner.session_store.lookup_by_session_key(route)
            if entry is None:
                return False
            return bool(await runner._admit_plugin_injection(
                adapter, source, session_key=route, session_id=entry.session_id,
                content=content, plugin_id=plugin_id))
        if (not allowed(plugin_home, plugin_id)
                or _thread_sessions_per_user(plugin_home) is not False
                or not await runner._plugin_origin_authorized(
                    source, plugin_id=plugin_id, plugin_home=plugin_home,
                    target=f'origin={source.platform.value}/{source.chat_id}')):
            return False
        ref = authority._publish_route(plugin_source, *await tracked_write(
            authority, partial(authority._reserve_route, plugin_source)))
        payload = {'text': content, 'plugin_origin_v1': {
            'plugin_id': plugin_id, 'route': route, 'target': ref.session_id}}
        await tracked_write(
            authority,
            partial(admit_session_input, authority.db, epoch=authority.epoch,
                    principal_id='automation:' + route, session_id=ref.session_id,
                    request_id=identity, payload=payload, _authorize_write=authority._admission_gate()),
            then=partial(authority._admitted, ref, event), ordered=True,
            after=partial(authority._schedule_admitted, ref),
        )
    tracker = getattr(adapter, '_threads', None)
    if tracker is not None and plugin_source.thread_id:
        await tracker.mark_async(plugin_source.thread_id)
    return True


def _permission_denied():
    raise RuntimeStoreError('permission_denied')


async def check(authority, ref: SessionRef, row: dict, *, source, adapter) -> str:
    """Revalidate the committed descriptor and return its route."""
    payload = row.get('payload') or {}
    descriptor = payload.get('plugin_origin_v1')
    if (set(payload) != {'text', 'plugin_origin_v1'} or not isinstance(descriptor, dict)
            or set(descriptor) != {'plugin_id', 'route', 'target'}
            or not all(isinstance(descriptor.get(key), str) and descriptor[key]
                       for key in ('plugin_id', 'route', 'target'))
            or row.get('principal_id') != 'automation:' + descriptor['route']
            or row.get('target_session_id') != ref.session_id
            or descriptor['target'] != ref.session_id):
        _permission_denied()
    try:
        identity = json.loads(row['request_id'])
    except (KeyError, TypeError, ValueError):
        _permission_denied()
    if (not isinstance(identity, list) or len(identity) != 4 or identity[:3] != [
            'plugin', descriptor['plugin_id'], descriptor['route']]
            or not isinstance(identity[3], str) or not identity[3]):
        _permission_denied()
    runner = authority.runner
    plugin_source = _source_for_plugin(source, descriptor['plugin_id'])
    route = runner._session_key_for_source(plugin_source)
    entry = runner.session_store.lookup_by_session_key(route)
    if (route != descriptor['route'] or entry is None
            or authority.logical_owner(entry.session_id) != ref.session_id
            or runner._session_key_for_source(source) != route):
        _permission_denied()
    if adapter is None or adapter is not runner._adapter_for_source(source):
        raise PluginOriginRefused('adapter')
    plugin_home = Path(authority.profile_id)
    if not allowed(plugin_home, descriptor['plugin_id']):
        raise PluginOriginRefused('authorization')
    if _thread_sessions_per_user(plugin_home) is not False:
        raise PluginOriginRefused('thread_sessions_per_user')
    if not await runner._plugin_origin_authorized(
            source, plugin_id=descriptor['plugin_id'], plugin_home=plugin_home,
            target=f'origin={source.platform.value}/{source.chat_id}'):
        raise PluginOriginRefused('authorization')
    return route


def restore(authority, ref: SessionRef, row: dict) -> MessageEvent:
    descriptor = row['payload']['plugin_origin_v1']
    live = authority.sessions[ref.session_id]
    source = _source_for_plugin(live.source, descriptor['plugin_id'])
    return MessageEvent(
        text=row['payload']['text'], message_type=MessageType.TEXT, source=source,
        message_id=row['request_id'], internal=True, allow_gateway_control=False,
        metadata={'gateway_session_key': descriptor['route'], 'gateway_session_id': ref.session_id},
    )


def narrowed(authority, row: dict) -> bool:
    payload = row.get('payload') or {}
    descriptor = payload.get('plugin_origin_v1')
    if isinstance(descriptor, dict) and isinstance(descriptor.get('plugin_id'), str):
        return True
    try:
        identity = json.loads(row.get('request_id', ''))
    except (TypeError, ValueError):
        return False
    if not (isinstance(identity, list) and len(identity) >= 2 and identity[0] == 'plugin'
            and isinstance(identity[1], str) and identity[1]):
        return False
    config = _plugin_config(Path(authority.profile_id))
    if config is None:
        return True
    entry = _plugin_entry(config, identity[1])
    return bool(entry and entry.get('allow_origin_sessions') is True)
