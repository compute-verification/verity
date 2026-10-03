"""vy-lean-sandbox (`pods/nebius/lean_sandbox.py`): the unit it starts, what it refuses before starting one, the probes and their
verdict, the `build` slot in lean_slot.py's files, the tree it checks out and calls clean; with stand-ins for systemd and the
host. And what deploys it: vy-lean-build-setup, run-daily's stub, the daily units and their entries in deploy.toml."""
from __future__ import annotations

import configparser
import fnmatch
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import deploy

NEBIUS = Path(__file__).resolve().parents[1] / "src/research/pods/nebius"
CHECK = Path(__file__).resolve().parents[3] / "tools/check"


def _load(name: str, path: Path):
    """``path`` as module ``name``, compiled from its source: tools/check's __pycache__ is no input of this suite."""
    old = sys.pycache_prefix, sys.dont_write_bytecode
    sys.pycache_prefix, sys.dont_write_bytecode = os.devnull, True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m                               # @dataclass looks its module up there
        spec.loader.exec_module(m)
    finally:
        sys.pycache_prefix, sys.dont_write_bytecode = old
    return m


S = _load("nebius_lean_sandbox", NEBIUS / "lean_sandbox.py")
LS = _load("check_lean_slot_for_sandbox", CHECK / "lean_slot.py")

NOW = datetime(2026, 10, 3, 6, 15, tzinfo=timezone.utc)

USERS = {"lean-build": (990, 990), "research": (1001, 1001), "toor": (0, 0), "dock": (991, 991)}
GROUPS = [SimpleNamespace(gr_name="root", gr_gid=0, gr_mem=[]), SimpleNamespace(gr_name="research", gr_gid=1001, gr_mem=[]),
          SimpleNamespace(gr_name="docker", gr_gid=986, gr_mem=["dock"]), SimpleNamespace(gr_name="lean-build", gr_gid=990, gr_mem=[])]
NET_UNIX = ("Num       RefCount Protocol Flags    Type St Inode Path\n"
            "0000000000000000: 00000002 00000000 00010000 0001 01 1001 /run/dbus/system_bus_socket\n"
            "0000000000000000: 00000003 00000000 00000000 0001 03 1002 /run/dbus/system_bus_socket\n"
            "0000000000000000: 00000002 00000000 00000000 0001 03 1003 /run/connected-only\n"
            "0000000000000000: 00000002 00000000 00010000 0001 01 1004 @abstract\n")


def getpwnam(name):
    if name not in USERS:
        raise KeyError(name)
    return SimpleNamespace(pw_name=name, pw_uid=USERS[name][0], pw_gid=USERS[name][1])


def prep(**kw) -> dict:
    """`prepare`'s view of node 1: run as root, /etc/verity's two files, a host /tmp of its own, mount points that exist."""
    return {"geteuid": lambda: 0, "getpwnam": getpwnam, "getgrall": lambda: GROUPS,
            "host": lambda d: SimpleNamespace(st_dev=1, st_ino=2), "listdir": lambda d: ["lean-cache.env", "hf-token"],
            "io_idle": lambda: False, "net_unix": lambda: NET_UNIX, "exists": lambda p: True, **kw}


@pytest.fixture
def host(tmp_path):
    tree = tmp_path / "build" / "verity"
    (tree / "pkg" / ".lake").mkdir(parents=True)
    elan = tmp_path / "elan"
    elan.mkdir()
    slots = tmp_path / "locks" / "lean-slots"
    slots.parent.mkdir()
    slots.write_text("check 2\nbuild 2\n")
    return SimpleNamespace(tmp=tmp_path, tree=tree, elan=elan, slots=slots, locks=slots.parent)


def args(h, *extra, cmd=("lake", "build")) -> list[str]:
    return ["run", "--tree", str(h.tree), "--elan", str(h.elan), "--slots", str(h.slots), "--cpus", "8-15", *extra, "--", *cmd]


HELD = [{"probe": p, "held": True, "detail": "ok"} for p in S.PROBES]


class Systemd:
    """systemd-run's stand-in: the probe unit prints `probes` (or `out`) and exits `probe_rc`; the build unit calls `during`
    and exits `build_rc`."""

    def __init__(self, probes=HELD, probe_rc=0, build_rc=0, out=None, during=None):
        self.calls, self.probes, self.probe_rc, self.build_rc, self.out, self.during = [], probes, probe_rc, build_rc, out, during

    def __call__(self, argv, capture, unit):
        self.calls.append((argv, capture, unit))
        if unit.startswith("vy-lean-sandbox-probe-"):
            return self.probe_rc, self.out if self.out is not None else S.MARK + json.dumps(self.probes) + "\n"
        if self.during:
            self.during()
        return self.build_rc, ""


def _opts(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv[:argv.index("--")]) if a == flag]


# ---- the unit

def test_the_build_runs_as_its_user_with_no_network_no_credentials_and_only_the_variables_passed(host):
    run = Systemd()
    assert S.main(args(host, "-E", "LEAN_NUM_THREADS=8", "-E", "LAKE_ARGS=-K a=b"), run=run, **prep(io_idle=lambda: True)) == S.OK
    (probe, captured, probe_unit), (build, piped, unit) = run.calls
    assert captured and not piped and probe_unit.startswith("vy-lean-sandbox-probe-") and unit.startswith("vy-lean-sandbox-build-")
    assert build[:9] == ["systemd-run", f"--unit={unit}", "--uid=lean-build", "--wait", "--pipe", "--collect", "--quiet",
                         "--service-type=exec", f"--working-directory={host.tree}"]
    props = _opts(build, "-p")
    for p in ("NoNewPrivileges=yes", "RestrictSUIDSGID=yes", "CapabilityBoundingSet=", "AmbientCapabilities=",
              "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes", "PrivateDevices=yes", "PrivateNetwork=yes",
              "PrivateIPC=yes", "ProtectProc=invisible", "RestrictAddressFamilies=AF_UNIX", "TemporaryFileSystem=/workspace:ro",
              "TemporaryFileSystem=/run:ro", "TemporaryFileSystem=/dev/shm", "InaccessiblePaths=-/etc/verity",
              "InaccessiblePaths=-/etc/rancher", f"BindReadOnlyPaths={host.elan}:/opt/elan", f"BindPaths={host.tree}:{host.tree}",
              "AllowedCPUs=8-15", "IOSchedulingClass=idle", f"MemoryMax={S.MEMORY}", f"RuntimeMaxSec={S.MAX_SEC}"):
        assert p in props, p
    assert not [p for p in props if p.startswith(("User=", "Group=", "EnvironmentFile=", "PassEnvironment=", "SupplementaryGroups="))]
    assert dict(e.split("=", 1) for e in _opts(build, "-E")) == {**S.BASE_ENV, "LEAN_NUM_THREADS": "8", "LAKE_ARGS": "-K a=b"}
    assert build[build.index("--") + 1:] == ["/usr/bin/env", "lake", "build"]       # resolved inside, on the unit's PATH
    same = lambda argv: [a for a in argv[:argv.index("--")] if not a.startswith("--unit=")]  # noqa: E731
    assert same(probe) == same(build)
    assert probe[probe.index("--") + 1:-1] == [S.PYTHON, "-I", "-c", (NEBIUS / "lean_sandbox.py").read_text(), "_probe"]
    checks = json.loads(probe[-1])
    assert checks["uid"] == 990 and checks["gids"] == [0, 986, 1001]
    assert {"/etc/verity/lean-cache.env", "/etc/verity/hf-token", "/home/research", "/etc/verity"} <= set(checks["secrets"])
    assert checks["sockets"] == ["/run/dbus/system_bus_socket"]                     # listening, by path, each once
    assert {str(host.tree), str(host.tree.parent), "/opt/elan", "/run"} <= set(checks["unwritable"])
    assert checks["private"] == {"/tmp": [1, 2], "/var/tmp": [1, 2], "/dev/shm": [1, 2]}
    assert set(checks["env"]) == {*S.BASE_ENV, "LEAN_NUM_THREADS", "LAKE_ARGS"}

    run = Systemd()
    assert S.main(args(host), run=run, **prep()) == S.OK
    assert not [p for p in _opts(run.calls[1][0], "-p") if p.startswith("IOSchedulingClass")]


def test_the_probe_source_runs_by_itself_in_isolated_mode():
    r = subprocess.run([sys.executable, "-I", "-c", (NEBIUS / "lean_sandbox.py").read_text(), "--help"], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "vy-lean-sandbox" in r.stdout


def test_deps_go_read_only_into_the_packages_lake_and_must_be_readable_by_others(host):
    deps = host.tmp / "deps"
    (deps / "mathlib").mkdir(parents=True)
    (deps / "mathlib" / "A.olean").write_text("o")
    for p, mode in ((deps, 0o755), (deps / "mathlib", 0o755), (deps / "mathlib" / "A.olean", 0o644)):
        p.chmod(mode)
    run = Systemd()
    assert S.main(args(host, "--deps", f"pkg={deps}"), run=run, **prep()) == S.OK
    inside = f"{host.tree}/pkg/.lake/packages"
    assert f"BindReadOnlyPaths={deps}:{inside}" in _opts(run.calls[1][0], "-p")
    assert (host.tree / "pkg" / ".lake" / "packages").is_dir() and inside in json.loads(run.calls[0][0][-1])["unwritable"]

    (deps / "mathlib" / "B.olean").write_text("o")
    (deps / "mathlib" / "B.olean").chmod(0o600)                 # as Mathlib's cache writes its builds
    run = Systemd()
    assert S.main(args(host, "--deps", f"pkg={deps}"), run=run, **prep()) == S.REFUSED and run.calls == []
    assert S.unreadable(str(deps)) == [str(deps / "mathlib" / "B.olean")]


REFUSALS = {
    "not root": (lambda h: args(h), {"geteuid": lambda: 1001}),
    "no such user": (lambda h: args(h, "--user", "ghost"), {}),
    "research": (lambda h: args(h, "--user", "research"), {}),
    "uid 0": (lambda h: args(h, "--user", "toor"), {}),
    "in docker": (lambda h: args(h, "--user", "dock"), {}),
    "a token": (lambda h: args(h, "-E", "HF_TOKEN=x"), {}),
    "the cache's key": (lambda h: args(h, "-E", "LEAN_CACHE_WRITE=x"), {}),
    "an R2 variable": (lambda h: args(h, "-E", "R2_ENDPOINT=x"), {}),
    "its own PATH": (lambda h: args(h, "-E", "PATH=/home/research/bin"), {}),
    "systemd's USER": (lambda h: args(h, "-E", "USER=root"), {}),
    "a bare name": (lambda h: args(h, "-E", "LEAN_NUM_THREADS"), {}),
    "no tree": (lambda h: args(h, "--tree", str(h.tmp / "build" / "none")), {}),
    "a top-level tree": (lambda h: args(h, "--tree", "/build"), {}),
    "no elan": (lambda h: args(h, "--elan", str(h.tmp / "none")), {}),
    "no mount point": (lambda h: args(h), {"exists": lambda p: p != S.ELAN_IN}),
    "a relative bind": (lambda h: args(h, "--ro", "deps"), {}),
    "deps outside the tree": (lambda h: args(h, "--deps", f"../x={h.elan}"), {}),
    "deps without a dir": (lambda h: args(h, "--deps", "pkg"), {}),
    "deps of no package": (lambda h: args(h, "--deps", f"nopkg={h.elan}"), {}),
    "no command": (lambda h: args(h, cmd=()), {}),
    "another pool's slot": (lambda h: args(h, "--held-slot", "check/0"), {}),
    "a slot with no lock": (lambda h: args(h, "--held-slot", "build/1"), {}),
}


@pytest.mark.parametrize("case", REFUSALS)
def test_a_refusal_exits_2_before_any_unit_starts(host, case):
    argv, kw = REFUSALS[case]
    run = Systemd()
    assert S.main(argv(host), run=run, **prep(**kw)) == S.REFUSED
    assert run.calls == [] and not (host.locks / "lean-line-build").exists()


@pytest.mark.parametrize("probes", [
    [{**r, "held": r["probe"] != "network", "detail": "1.1.1.1:443"} for r in HELD],
    [r for r in HELD if r["probe"] != "secrets"],                                   # one that didn't report
    [{**r, "held": "yes"} for r in HELD],                                           # only true holds
], ids=["one got through", "one didn't report", "not a boolean"])
def test_a_probe_that_got_through_stops_the_build(host, probes, capsys):
    run = Systemd(probes=probes)
    assert S.main(args(host), run=run, **prep()) == S.BREACH
    assert len(run.calls) == 1 and "no build, no upload" in capsys.readouterr().err
    i, fd = LS.take(host.locks, "build", 2)
    os.close(fd)
    assert i == 0                                                                   # its slot was given back


@pytest.mark.parametrize("probe_rc, out, build_rc, want", [
    (226, "", 0, S.REFUSED),                    # the probe unit's namespace couldn't be set up: no report
    (1, "Failed to start transient service unit\n", 0, S.REFUSED),
    (0, S.MARK + "{not json\n", 0, S.REFUSED),
    (0, None, 217, S.REFUSED),                  # the build unit couldn't become its user: never run as anyone else
    (0, None, 7, S.FAILED),
    (0, None, 0, S.OK),
])
def test_a_unit_that_cannot_start_is_a_refusal_and_the_builds_status_is_its_own(host, probe_rc, out, build_rc, want):
    run = Systemd(probe_rc=probe_rc, out=out, build_rc=build_rc)
    assert S.main(args(host), run=run, **prep()) == want
    assert len(run.calls) == (1 if out is not None else 2)


def test_a_stopped_wrapper_stops_its_unit_and_ends_with_the_signals_status(tmp_path, reap):
    """The units start outside the caller's cgroup, so stopping vy-lean-cache-daily.service reaches only the wrapper."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "systemctl").write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/systemctl.log\n'
                                    f'[ "$2" = --no-block ] && kill "$(cat {tmp_path}/unit.pid)"\nexit 0\n')
    (bin_ / "systemctl").chmod(0o755)
    unit = f"echo $$ > {tmp_path}/unit.pid; echo ready; exec cat"
    driver = (f"import importlib.util as u, sys\n"
              f"s = u.spec_from_file_location('m', {str(NEBIUS / 'lean_sandbox.py')!r}); m = u.module_from_spec(s)\n"
              f"sys.modules['m'] = m; s.loader.exec_module(m)\n"
              f"m.systemd_run(['sh', '-c', {unit!r}], False, 'vy-lean-sandbox-build-t')\n")
    p = subprocess.Popen([sys.executable, "-c", driver], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}, start_new_session=True)
    assert p.stdout.readline() == "ready\n"
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate()
    assert p.returncode == 128 + signal.SIGTERM, err
    assert (tmp_path / "systemctl.log").read_text().splitlines() == ["stop --no-block vy-lean-sandbox-build-t",
                                                                      "stop vy-lean-sandbox-build-t"]
    assert "stopped vy-lean-sandbox-build-t on signal 15" in err


# ---- the probes, inside the unit

UID = os.geteuid() + 7
STATUS = "".join(f"{k}:\t{v}\n" for k, v in (("Name", "python3"), ("NoNewPrivs", "1"), ("CapInh", "0" * 16), ("CapPrm", "0" * 16),
                                              ("CapEff", "0" * 16), ("CapBnd", "0" * 16), ("CapAmb", "0" * 16)))
ENV = {"HOME": "/tmp", "PATH": "/opt/elan/bin:/usr/bin:/bin", "INVOCATION_ID": "x", "LC_ALL": "C"}


def _world(tmp_path):
    """A spec and stand-ins under which every probe holds."""
    proc = tmp_path / "proc"
    proc.mkdir()

    def refuse(*_a):
        raise OSError("unreachable")

    spec = {"uid": UID, "gids": [4242], "env": ["HOME", "PATH"], "secrets": [str(tmp_path / "absent")],
            "unwritable": [str(tmp_path / "absent-dir")], "private": {str(tmp_path): [0, 0]}, "reach": [["192.0.2.1", 443]],
            "resolve": "example.invalid", "sockets": [str(tmp_path / "absent.sock")]}
    kw = {"environ": dict(ENV), "connect": refuse, "resolve": refuse, "sudo": lambda argv: 1, "ids": lambda: (UID, UID, {UID}),
          "status": lambda: STATUS, "proc": proc, "interfaces": lambda: [(1, "lo")], "which": lambda name, path=None: "/usr/bin/sudo"}
    return spec, kw


def test_every_probe_holds_in_a_sandbox(tmp_path):
    spec, kw = _world(tmp_path)
    got = S.probe(spec, **kw)
    assert [r["probe"] for r in got] == list(S.PROBES) and S.breaches(got) == [], got
    kw["which"] = lambda name, path=None: None
    assert S.breaches(S.probe(spec, **kw)) == []                                     # no sudo at all


def _sock(t, spec, kw):
    with socket.socket(socket.AF_UNIX) as s:
        s.bind("relay.sock")                                    # relative, in t: an AF_UNIX path is at most 107 bytes
    spec["sockets"] = [str(t / "relay.sock")]


def _timeout(argv):
    raise subprocess.TimeoutExpired(argv, 30)


def _boom():
    raise RuntimeError("no /proc/self/status")


BREACHES = {
    "user:forbidden group": lambda t, spec, kw: kw.update(ids=lambda: (UID, UID, {UID, 4242})),
    "user:root": lambda t, spec, kw: kw.update(ids=lambda: (0, 0, set())),
    "user:another uid": lambda t, spec, kw: kw.update(ids=lambda: (UID + 1, UID, {UID})),
    "privileges:new privileges": lambda t, spec, kw: kw.update(status=lambda: STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0")),
    "privileges:a capability": lambda t, spec, kw: kw.update(status=lambda: STATUS.replace("CapEff:\t" + "0" * 16, "CapEff:\t1")),
    "privileges:no answer": lambda t, spec, kw: kw.update(status=_boom),
    "sudo:allowed": lambda t, spec, kw: kw.update(sudo=lambda argv: 0),
    "sudo:hangs": lambda t, spec, kw: kw.update(sudo=_timeout),
    "network:a connection": lambda t, spec, kw: kw.update(connect=lambda host, port: None),
    "network:a name resolved": lambda t, spec, kw: kw.update(resolve=lambda name: None),
    "network:an interface": lambda t, spec, kw: kw.update(interfaces=lambda: [(1, "lo"), (2, "eth0")]),
    "sockets:a relay": _sock,
    "secrets:a key": lambda t, spec, kw: ((t / "key").write_text("k"), spec.update(secrets=[str(t / "key")])),
    "secrets:a home": lambda t, spec, kw: spec.update(secrets=[str(t)]),
    "writes:a directory": lambda t, spec, kw: spec.update(unwritable=[str(t)]),
    "private:the host's /tmp": lambda t, spec, kw: spec.update(private={str(t): [os.stat(t).st_dev, os.stat(t).st_ino]}),
    "environment:a credential": lambda t, spec, kw: kw["environ"].update(HF_HOME="/workspace/hf"),
    "environment:not passed": lambda t, spec, kw: kw["environ"].update(OTHER="1"),
    "processes:another user's": lambda t, spec, kw: (t / "proc" / "1").mkdir(),
}


@pytest.mark.parametrize("case", BREACHES)
def test_each_probe_catches_what_it_looks_for(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    spec, kw = _world(tmp_path)
    BREACHES[case](tmp_path, spec, kw)
    got = S.probe(spec, **kw)
    assert [r["probe"] for r in got if not r["held"]] == [case.split(":")[0]], got
    assert S.breaches(got) and S.breaches(got)[0].startswith(case.split(":")[0] + ": ")


def test_the_verdict_is_the_last_report_and_none_without_one():
    assert S.verdict("noise\n" + S.MARK + json.dumps(HELD) + "\n") == HELD
    assert S.verdict("Failed to set up mount namespacing: /etc/verity: No such file or directory\n") is None
    assert S.verdict(S.MARK + "{not json\n") is None
    assert S.breaches(HELD[:-1]) == ["processes: did not report"]


# ---- the `build` slot, in lean_slot.py's files

def test_a_run_holds_a_build_slot_that_lean_slot_sees_and_skips(host):
    seen = {}

    def during():
        seen["status"] = LS.status(LS.parse(host.slots.read_text()), host.locks, NOW, only="build")
        i, fd = LS.take(host.locks, "build", 2)
        os.close(fd)
        seen["took"] = i

    assert S.main(args(host, "--by", "lean"), run=Systemd(during=during), now=lambda: NOW, **prep()) == S.OK
    assert seen["took"] == 1
    assert seen["status"][0].startswith(f"build/0  held by pid {os.getpid()}") and "by lean, slot taken 2026-10-03T06:15Z" in seen["status"][0]
    who = json.loads((host.locks / "lean-build-0.who").read_text())
    assert who == {"pid": os.getpid(), "run": None, "tree": str(host.tree), "by": "lean", "took": "2026-10-03T06:15Z"}
    i, fd = LS.take(host.locks, "build", 2)
    os.close(fd)
    assert i == 0 and not list((host.locks / "lean-line-build").iterdir())         # given back, and out of the line


def test_a_run_waits_in_line_behind_lean_slots_waiters(host, capsys):
    ticket, tfd = LS.SLOT.join(host.locks, who={"pid": 1, "by": "audit"}, line="lean-line-build")
    waited = []

    def sleep(s):
        waited.append(s)
        ticket.unlink()                                         # it took its slot and left the line
        os.close(tfd)

    run = Systemd()
    assert S.main(args(host), run=run, now=lambda: NOW.replace(year=2099), sleep=sleep, **prep()) == S.OK
    assert waited == [S.POLL_S] and len(run.calls) == 2
    assert "1 ahead in lean-line-build" in capsys.readouterr().err


def test_slot_wait_gives_up_with_exit_4_and_nothing_run(host):
    host.slots.write_text("build 1\n")
    i, fd = LS.take(host.locks, "build", 1)
    clock = iter([NOW + timedelta(seconds=s) for s in (0, 0, 20, 40)])
    run = Systemd()
    try:
        assert S.main(args(host, "--slot-wait", "30"), run=run, now=lambda: next(clock), sleep=lambda s: None, **prep()) == S.NO_SLOT
    finally:
        os.close(fd)
    assert run.calls == [] and not list((host.locks / "lean-line-build").iterdir())


def test_a_held_slot_is_used_as_it_is_and_a_free_one_refused(host):
    i, fd = LS.take(host.locks, "build", 2)
    try:
        run = Systemd()
        assert S.main(args(host, "--held-slot", f"build/{i}"), run=run, **prep()) == S.OK
        assert len(run.calls) == 2 and not (host.locks / "lean-build-0.who").exists()
        assert not (host.locks / "lean-line-build").exists()
    finally:
        os.close(fd)
    run = Systemd()
    assert S.main(args(host, "--held-slot", "build/0"), run=run, **prep()) == S.REFUSED and run.calls == []


@pytest.mark.parametrize("text", [None, "build two\n", "check 2\naudit 1\n"], ids=["no file", "malformed", "no build pool"])
def test_no_slot_file_or_no_build_pool_is_a_refusal(host, text):
    if text is None:
        host.slots.unlink()
    else:
        host.slots.write_text(text)
    run = Systemd()
    assert S.main(args(host), run=run, **prep()) == S.REFUSED and run.calls == []


# ---- the tree

def _git(*a: str, cwd: Path) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def origin(tmp_path):
    work = tmp_path / "work"
    (work / "pkg" / "nested").mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=work)
    for p, text in {"pkg/lakefile.toml": 'name = "p"\n', "pkg/A.lean": "def a := 1\n", "pkg/nested/lakefile.lean": "-- n\n",
                    ".gitignore": "**/.lake/\n*.log\n", "README.md": "r\n"}.items():
        (work / p).write_text(text)
    _git("add", "-A", cwd=work)
    _git("commit", "-qm", "on main", cwd=work)
    on = _git("rev-parse", "HEAD", cwd=work)
    _git("checkout", "-qb", "side", cwd=work)
    (work / "side.txt").write_text("s\n")
    _git("add", "-A", cwd=work)
    _git("commit", "-qm", "off main", cwd=work)
    off = _git("rev-parse", "HEAD", cwd=work)
    _git("clone", "-q", "--bare", str(work), str(tmp_path / "verity.git"), cwd=tmp_path)
    return SimpleNamespace(bare=str(tmp_path / "verity.git"), on=on, off=off, tree=tmp_path / "build" / "verity")


def test_checkout_is_a_fresh_clone_of_a_commit_on_main_and_clean_holds_until_something_outside_lake_changes(origin):
    t = origin.tree
    t.mkdir(parents=True)
    (t / "stale").write_text("x")
    chowned = []
    S.checkout(origin.on, origin.bare, S.MAIN, str(t), 990, 990, chown=lambda p, u, g: chowned.append((Path(p), u, g)))
    assert not (t / "stale").exists() and (t / "pkg" / "A.lean").read_text() == "def a := 1\n"
    assert sorted(chowned) == [(t / "pkg" / ".lake", 990, 990), (t / "pkg" / "nested" / ".lake", 990, 990)]
    assert S.dirty(str(t), origin.on) == []
    (t / "pkg" / ".lake" / "build").mkdir()
    (t / "pkg" / ".lake" / "build" / "A.olean").write_text("o")
    assert S.dirty(str(t), origin.on) == [] and S.main(["clean", "--tree", str(t), "--sha", origin.on]) == S.OK
    (t / "build.log").write_text("l")
    (t / "pkg" / "B.lean").write_text("b")
    (t / "README.md").write_text("changed\n")
    assert sorted(S.dirty(str(t), origin.on)) == [" M README.md", "!! build.log", "?? pkg/B.lean"]
    assert S.dirty(str(t), origin.off)[0] == f"HEAD is {origin.on[:12]}, not {origin.off[:12]}"
    assert S.main(["clean", "--tree", str(t), "--sha", origin.on]) == S.FAILED
    empty = t.parent / "empty"
    _git("init", "-q", str(empty), cwd=t.parent)
    assert S.dirty(str(empty)) == [f"{empty} has no commit checked out"]


def test_checkout_refuses_and_leaves_no_tree(origin, tmp_path):
    t = str(origin.tree)
    for sha, repo, main, tree, why in [
        (origin.off, origin.bare, S.MAIN, t, "is not on refs/heads/main"),
        ("0" * 40, origin.bare, S.MAIN, t, "is not on refs/heads/main"),
        (origin.on, origin.bare, "refs/heads/nope", t, "has no refs/heads/nope"),
        (origin.on[:12], origin.bare, S.MAIN, t, "a full commit SHA"),
        (origin.on, str(tmp_path / "none.git"), S.MAIN, t, "no repository"),
        (origin.on, origin.bare, S.MAIN, "/build", "below a top-level one"),
    ]:
        with pytest.raises(S.Refused, match=why):
            S.checkout(sha, repo, main, tree, 990, 990, chown=lambda *a: None)
        assert not origin.tree.exists(), why

    def denied(*_a):
        raise PermissionError("chown")

    with pytest.raises(S.Refused, match="checkout failed"):
        S.checkout(origin.on, origin.bare, S.MAIN, t, 990, 990, chown=denied)
    assert not origin.tree.exists()
    origin.tree.parent.mkdir(exist_ok=True)
    origin.tree.symlink_to(tmp_path)
    with pytest.raises(S.Refused, match="below a top-level one"):
        S.checkout(origin.on, origin.bare, S.MAIN, t, 990, 990, chown=lambda *a: None)
    assert S.main(["checkout", origin.on, "--repo", origin.bare, "--tree", t], geteuid=lambda: 1001) == S.REFUSED


# ---- what deploys it

def _unit(name: str) -> configparser.ConfigParser:
    p = configparser.ConfigParser(interpolation=None)
    p.optionxform = str
    p.read_string((NEBIUS / name).read_text())
    return p


def test_the_daily_service_runs_the_publisher_as_root_bounded_and_its_timer_fires_at_0615_utc():
    svc, timer = _unit("vy-lean-cache-daily.service"), _unit("vy-lean-cache-daily.timer")
    assert dict(svc["Service"]) == {"Type": "oneshot", "ExecStart": "/usr/local/lib/vy-lean-cache/run-daily",
                                    "TimeoutStartSec": "3h", "Nice": "15", "IOSchedulingClass": "idle"}
    assert dict(timer["Timer"]) == {"OnCalendar": "*-*-* 06:15:00 UTC", "AccuracySec": "1min", "Persistent": "false"}
    assert dict(timer["Install"]) == {"WantedBy": "timers.target"}


def test_run_daily_is_a_stub_that_exits_2_until_leans_publisher_replaces_it():
    r = subprocess.run(["bash", str(NEBIUS / "lean_cache_daily.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and "publisher not installed (lean's PR)" in r.stderr


def test_the_setup_script_runs_only_as_root(tmp_path):
    (tmp_path / "id").write_text("#!/bin/sh\necho 1000\n")
    (tmp_path / "id").chmod(0o755)
    r = subprocess.run(["bash", str(NEBIUS / "lean_build_setup.sh")], capture_output=True, text=True,
                       env={"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert r.returncode == 2 and "run as root" in r.stdout


LEAN_CACHE = {"/usr/local/sbin/vy-lean-build-setup": "lean_build_setup.sh", "/usr/local/bin/vy-lean-sandbox": "lean_sandbox.py",
              "/usr/local/lib/vy-lean-cache/run-daily": "lean_cache_daily.sh",
              "/etc/systemd/system/vy-lean-cache-daily.service": "vy-lean-cache-daily.service",
              "/etc/systemd/system/vy-lean-cache-daily.timer": "vy-lean-cache-daily.timer"}


def test_deploy_puts_the_sandbox_its_setup_and_the_daily_units_on_node_1_only_and_starts_nothing():
    nodes = deploy.load()
    n1 = {e.path: e for e in nodes["vy-nebius-1"].entries}
    for path, src in LEAN_CACHE.items():
        assert n1[path].src == f"tools/research/src/research/pods/nebius/{src}" and n1[path].run == [], path
        assert any(fnmatch.fnmatch(path, m) for m in nodes["vy-nebius-1"].managed), path
    assert not set(LEAN_CACHE) & {e.path for e in nodes["vy-nebius-2"].entries}
    for src in ("lean_build_setup.sh", "lean_sandbox.py", "lean_cache_daily.sh"):
        assert os.access(NEBIUS / src, os.X_OK), src                                # installed with the repository's mode
