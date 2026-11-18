"""Client discovery of the Home Manager gateway job (org.nix-community.home.hermes-agent).

A client that finds the Home Manager job loaded and verified kickstarts it. It never spawns a
second gateway beside a Nix-owned one. A loaded job of another install root or profile is not ours and
is skipped. A job on this home that fails verification refuses discovery: it is ours, so no gateway spawns beside it.
"""
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

HM_LABEL = "org.nix-community.home.hermes-agent"
# nix/moduleCommon.nix gatewayArgv: `<package>/bin/hermes gateway`, no `run`.
HM_ARGV = ["/nix/store/hermes-agent/bin/hermes", "gateway"]


def _print_output(uid: int, home: Path, *, managed: str = "home-manager", argv: list[str] = HM_ARGV) -> str:
    """``launchctl print`` text for a loaded Home Manager job running ``argv`` for ``home``."""
    env = f"\t\tHERMES_HOME => {home}\n"
    if managed:
        env += f"\t\tHERMES_MANAGED => {managed}\n"
    return (
        f"gui/{uid}/{HM_LABEL} = {{\n"
        "\tstate = not running\n"
        f"\tprogram = {argv[0]}\n"
        "\targuments = {\n"
        + "".join(f"\t\t{arg}\n" for arg in argv)
        + "\t}\n"
        "\tenvironment = {\n"
        + env
        + "\t}\n"
        "}\n"
    )


def _fake_launchctl(tmp_path, monkeypatch, output: str) -> list[list[str]]:
    """Answer ``launchctl print gui/<uid>/HM_LABEL`` with ``output``; every other label is absent (113).

    Fakes the subprocess boundary only. Returns the recorded ``launchctl`` argv list.
    """
    uid = os.getuid()
    outfile = tmp_path / "launchctl_output.txt"
    outfile.write_text(output, encoding="utf-8")
    peer = tmp_path / "launchctl_fake.py"
    peer.write_text(
        "import sys\n"
        f"expected = 'gui/{uid}/{HM_LABEL}'\n"
        "if sys.argv[1:3] == ['print', expected]:\n"
        f"    sys.stdout.write(open({str(outfile)!r}).read())\n"
        "else:\n"
        "    sys.exit(113)\n",
        encoding="utf-8",
    )
    actual = subprocess.run
    calls: list[list[str]] = []

    def boundary(argv, **kwargs):
        if argv[0] != "launchctl":
            return actual(argv, **kwargs)
        calls.append(list(argv))
        return actual([sys.executable, str(peer), *argv[1:]], **kwargs)

    monkeypatch.setattr(subprocess, "run", boundary)
    return calls


def _home(tmp_path, monkeypatch):
    import pwd
    from hermes_cli import gateway_runtime_service as service

    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path), pw_name="owner"))
    monkeypatch.setattr(service, "service_suffix", lambda h: "")
    return home


def test_hm_job_loaded_and_verified_is_kickstarted_not_spawned(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_service as service

    home = _home(tmp_path, monkeypatch)
    calls = _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home))

    found = service._launchd(home, time.monotonic() + 5)

    assert found is not None and found.backend == "launchd"
    assert found.start_argv == ("launchctl", "kickstart", f"gui/{os.getuid()}/{HM_LABEL}")
    assert not any(call[1] == "bootstrap" for call in calls)


def test_hm_job_loaded_and_verified_ensure_never_spawns(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = _home(tmp_path, monkeypatch)
    _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home))
    kickstarts: list[tuple] = []
    monkeypatch.setattr(service, "discover_existing_gateway_service",
                        lambda h, deadline: service._launchd(h, deadline))
    monkeypatch.setattr(service, "start_existing_gateway_service",
                        lambda found, deadline: kickstarts.append(found.start_argv))
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    spawned: list[Path] = []
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    runtime.ensure_gateway_runtime(home, timeout=0.3)

    assert spawned == []
    assert kickstarts == [("launchctl", "kickstart", f"gui/{os.getuid()}/{HM_LABEL}")]


def test_hm_job_failing_identity_keeps_current_behaviour(tmp_path, monkeypatch):
    """The HM job is loaded but serves another home: it is not ours, so discovery finds no owner."""
    from hermes_cli import gateway_runtime_service as service

    home = _home(tmp_path, monkeypatch)
    calls = _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), tmp_path / "elsewhere" / ".hermes"))

    found = service._launchd(home, time.monotonic() + 5)

    assert found is None
    assert not any(call[1] == "kickstart" for call in calls)


def test_hm_job_on_our_home_without_managed_marker_refuses_and_never_spawns(tmp_path, monkeypatch):
    """A loaded HM-labelled job on our home without the HERMES_MANAGED marker is not a Home Manager job of ours.
    Its HERMES_HOME is ours, so the failure refuses: discovery must never spawn a second gateway beside it."""
    from hermes_cli import gateway_runtime_service as service

    home = _home(tmp_path, monkeypatch)
    calls = _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home, managed=""))

    with pytest.raises(service.RuntimeStartError, match="service_identity_unverified"):
        service._launchd(home, time.monotonic() + 5)
    assert not any(call[1] == "kickstart" for call in calls)


def test_hm_job_serving_a_non_default_home_is_kickstarted_not_spawned(tmp_path, monkeypatch):
    """The Home Manager label never carries a home suffix, so a home that has one still finds the bare job."""
    import pwd
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = tmp_path / "hermes-state"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path), pw_name="owner"))
    assert service.service_suffix(home)  # a suffixed label is what the old probe asked for
    _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home))
    kickstarts: list[tuple] = []
    spawned: list[Path] = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda h, deadline: service._launchd(h, deadline))
    monkeypatch.setattr(service, "start_existing_gateway_service",
                        lambda found, deadline: kickstarts.append(found.start_argv))
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    found = service._launchd(home, time.monotonic() + 5)
    assert found is not None and found.start_argv == ("launchctl", "kickstart", f"gui/{os.getuid()}/{HM_LABEL}")

    runtime.ensure_gateway_runtime(home, timeout=0.3)
    assert spawned == []
    assert kickstarts == [("launchctl", "kickstart", f"gui/{os.getuid()}/{HM_LABEL}")]


def test_hm_job_on_our_home_with_unverifiable_argv_refuses_and_never_spawns(tmp_path, monkeypatch):
    """Same home, unverifiable argv: the job is ours but its command is not. Refuse; never spawn beside it."""
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = _home(tmp_path, monkeypatch)
    _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home, argv=[*HM_ARGV, "--bogus"]))
    kickstarts: list[tuple] = []
    spawned: list[Path] = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda h, deadline: service._launchd(h, deadline))
    monkeypatch.setattr(service, "start_existing_gateway_service",
                        lambda found, deadline: kickstarts.append(found.start_argv))
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    with pytest.raises(service.RuntimeStartError) as refused:
        service._launchd(home, time.monotonic() + 5)
    assert refused.value.reason == "service_identity_unverified"

    result = runtime.ensure_gateway_runtime(home, timeout=0.3)
    assert result.reason_code == "service_identity_unverified"
    assert spawned == [] and kickstarts == []


def test_hm_job_with_extra_args_verifies_and_is_kickstarted(tmp_path, monkeypatch):
    """extraArgs follow ``gateway`` in the Home Manager job; ``--quiet`` is a flag the gateway accepts."""
    from hermes_cli import gateway_runtime_service as service

    home = _home(tmp_path, monkeypatch)
    calls = _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), home, argv=[*HM_ARGV, "--quiet"]))

    found = service._launchd(home, time.monotonic() + 5)

    assert found is not None and found.backend == "launchd"
    assert found.start_argv == ("launchctl", "kickstart", f"gui/{os.getuid()}/{HM_LABEL}")
    assert not any(call[1] == "bootstrap" for call in calls)


def test_hm_job_from_outside_the_nix_store_is_not_ours(tmp_path, monkeypatch):
    """Home Manager runs the store-pinned hermes. A hermes elsewhere, with our HERMES_HOME, is not our job."""
    from hermes_cli import gateway_runtime_service as service

    home = _home(tmp_path, monkeypatch)
    calls = _fake_launchctl(tmp_path, monkeypatch,
                            _print_output(os.getuid(), home, argv=["/tmp/evil/hermes", "gateway"]))

    assert service._launchd(home, time.monotonic() + 5) is None
    assert not any(call[1] == "kickstart" for call in calls)


def test_hm_argv_outside_the_nix_store_is_foreign(tmp_path):
    from hermes_cli.gateway_runtime_service_identity import ForeignRoot, verify_launchd_loaded

    home = tmp_path / ".hermes"
    output = _print_output(os.getuid(), home, argv=["/tmp/evil/hermes", "gateway"])

    with pytest.raises(ForeignRoot):
        verify_launchd_loaded(output, home, uid=os.getuid(), username="owner", home_manager=True)


def test_hm_job_serving_another_profile_of_our_root_is_not_ours(tmp_path, monkeypatch):
    """The job serves the root's default home; the requested home is another profile of the same root."""
    from hermes_cli import gateway_runtime_service as service

    _home(tmp_path, monkeypatch)
    profile = tmp_path / ".hermes" / "profiles" / "work"
    calls = _fake_launchctl(tmp_path, monkeypatch, _print_output(os.getuid(), tmp_path / ".hermes"))

    assert service._launchd(profile, time.monotonic() + 5) is None
    assert not any(call[1] == "kickstart" for call in calls)
