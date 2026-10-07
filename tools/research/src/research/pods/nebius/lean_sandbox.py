#!/usr/bin/env python3
"""`vy-lean-sandbox`: a Lean build on vy-nebius-1 as user `lean-build`, in a transient systemd unit with no network, none of the
node's credentials and nothing writable outside its build tree, probed from inside before every build.

    vy-lean-sandbox checkout SHA [--repo DIR] [--main REF] [--tree DIR]
    vy-lean-sandbox run [UNIT] [--slot-wait S | --held-slot build/I] [--by WHO] -- CMD...
    vy-lean-sandbox probe [UNIT]
    vy-lean-sandbox clean [--tree DIR] [--sha SHA] [--user NAME]

UNIT: --tree DIR[:DEST] (default /build/verity, bound read-write at DEST, the same path by default: a module built at a fixed
path is byte-identical), --deps PKG=DIR (DIR read-only at DEST/PKG/.lake/deps, restored outside first and readable by
others), --ro SRC[:DEST], --rw SRC[:DEST], -E NAME=VALUE (the only variables the build gets, beside PATH, HOME, ELAN_HOME and
the few below), --elan DIR (default ~research/.elan, read-only at /opt/elan), --cpus LIST (default /etc/vy/direct-cpus),
--memory SIZE, --max-sec N, --user NAME (default lean-build; never root, research or a member of research, sudo, docker or
adm). Every DEST must exist: systemd would make a missing one on the host's disk (`vy-lean-build-setup` makes /opt/elan).

Run it as root (`sudo -n`). `run` takes a slot of the `build` pool in node 1's Lean slot file, in `tools/verity/check/lean_slot.py`'s
format (`lean-build-I.lock` and `.who` beside the file, waiting in arrival order in `lean-line-build`), unless the caller holds
one (`--held-slot`, which must be held). The lock directory is research's, so root opens nothing there through a link, takes
only regular files with one name, and gives research only what it made itself; anything else there is refused. It holds the
slot while the unit runs, and stops the unit when it is stopped itself; a unit whose wrapper was killed ends at --max-sec.
Before the command, the probes run in a unit of their own with the same properties: who it runs as, its privileges, sudo, the
network, the host's Unix sockets, the credentials, writes outside the tree, a private /tmp, /var/tmp and /dev/shm, its
environment and other users' processes. If one gets through, nothing is built.

`checkout` makes the tree a fresh clone of SHA, which must be on REF (default refs/heads/main) of DIR (default node 1's bare
repository): root owns the sources and the build's user only each Lake package's `.lake`. `clean` is what a publisher waits
on: it passes when no process of the build's user is alive and no sandbox unit is active; when nothing in a `.lake` directory
is a link, a FIFO, socket or device, a file with a second name, setuid, writable by group or others, or owned by anyone but
root and the build's user; and when the tree is still SHA's outside them. Once it passes, nothing but root changes the tree
until the next `run`. A publisher still opens every directory from its parent's descriptor with O_NOFOLLOW and takes only
regular files with one name, by fstat: O_NOFOLLOW alone covers only a path's last component, and a build can make a directory
such as `.lake/build` a link to /etc/verity.

Exit status: 0 done (the command succeeded, the probes held, the tree is clean), 1 the command failed or the tree isn't clean,
2 refused or the unit couldn't start (the command never runs another way, nor as another user), 3 a probe got through and
nothing was built, 4 no `build` slot within --slot-wait and nothing ran.

Stdlib only: it runs as root from /usr/local/bin, and the probes run from its own source inside the unit (`python3 -I -c`).
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import grp
import json
import os
import pwd
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

OK, FAILED, REFUSED, BREACH, NO_SLOT = 0, 1, 2, 3, 4

USER = "lean-build"
TREE = "/build/verity"
SLOTS = "/workspace/research/locks/lean-slots"
POOL = "build"
ELAN = "/home/research/.elan"
ELAN_IN = "/opt/elan"
REPO = "/workspace/research/git/verity.git"
MAIN = "refs/heads/main"
DIRECT_CPUS = "/etc/vy/direct-cpus"
MEMORY = "256G"
TASKS = 16384
MAX_SEC = 3 * 3600
POLL_S = 20
PYTHON = "/usr/bin/python3"
UNIT_PREFIX = "vy-lean-sandbox-"
PROC = Path("/proc")
#: `clean` descends at most this far below the tree (Lake's deepest outputs are about fifteen)
DEPTH = 64

#: users the build never runs as, and groups its user is never in
FORBIDDEN_USERS = ("root", "research")
FORBIDDEN_GROUPS = ("root", "research", "sudo", "wheel", "docker", "adm")
#: hidden inside the unit (absent ones are skipped), beside ProtectHome's /home and /root, and empty /workspace and /run: /run
#: holds the gpu leases and sockets that act for any caller (systemd-resolved's resolver, D-Bus, snapd, docker)
HIDDEN = ("/etc/verity", "/etc/vy", "/etc/rancher", "/var/lib/rancher", "/var/lib/kubelet", "/var/lib/docker",
          "/var/lib/containerd")
#: what the probes must fail to read (beside every file in /etc/verity and every hidden path)
SECRETS = ("/home/research", "/root", "/run/user", "/run/gpu-lease", "/run/gpu-lease-circuits", "/run/credentials",
           "/workspace/research", "/workspace/jobs/dispatch/dispatch.env", "/etc/shadow", "/etc/rancher/k3s/k3s.yaml",
           "/var/lib/rancher/k3s/server/token")
#: at most this many of the host's listening Unix sockets are looked for from inside
SOCKETS_MAX = 400
#: where the probes must fail to create a file (beside the tree's root, its parent and every read-only bind)
UNWRITABLE = ("/", "/etc", "/usr/local/bin", "/var/lib", "/var/log", "/run", "/run/lock", "/home", "/workspace", ELAN_IN)
#: the unit's own: a probe fails when one is the host's
PRIVATE = ("/tmp", "/var/tmp", "/dev/shm")
#: a public resolver, the cloud metadata service, the host's own sshd and k3s API, and a public address over IPv6
REACH = (("1.1.1.1", 443), ("8.8.8.8", 53), ("169.254.169.254", 80), ("127.0.0.1", 22), ("127.0.0.1", 6443),
         ("2606:4700:4700::1111", 443))
RESOLVE = "github.com"
#: a variable a credential would travel in; never passed, and a probe fails on finding one
CREDENTIAL = re.compile(r"TOKEN|SECRET|PASSW|KEY|CREDENTIAL|^AWS_|^R2_|^LEAN_CACHE_|^GH_|^GITHUB_|^HF_|^SSH_|SLACK|NOTION|"
                        r"RUNPOD|NEBIUS|CURSOR", re.I)
#: what systemd itself puts in a service's environment
SYSTEMD_ENV = frozenset({"HOME", "LANG", "LANGUAGE", "LOGNAME", "USER", "SHELL", "PATH", "INVOCATION_ID", "JOURNAL_STREAM",
                         "SYSTEMD_EXEC_PID", "MEMORY_PRESSURE_WATCH", "MEMORY_PRESSURE_WRITE", "NOTIFY_SOCKET", "MANAGERPID",
                         "TERM", "PWD", "SHLVL", "_"})
#: every build's variables; -E can't replace them
BASE_ENV = {"PATH": f"{ELAN_IN}/bin:/usr/bin:/bin", "HOME": "/tmp", "ELAN_HOME": ELAN_IN, "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "GIT_CONFIG_PARAMETERS": "'safe.directory'='*'"}
PROBES = ("user", "privileges", "sudo", "network", "sockets", "secrets", "writes", "private", "environment", "processes")
MARK = "vy-lean-sandbox-probe/v1 "
#: systemd's own exit statuses for a unit it couldn't set up (200 CHDIR ... 226 NAMESPACE ... 245 BPF)
SETUP_FAILED = range(200, 246)


class Refused(Exception):
    """Nothing runs: a guard never fails open."""


def log(msg: str) -> None:
    print(f"vy-lean-sandbox: {msg}", file=sys.stderr, flush=True)


# ---- inside the unit, as the build's user: the probes

def _connect(host: str, port: int) -> None:
    socket.create_connection((host, port), timeout=3).close()


def _resolve(name: str) -> None:
    socket.getaddrinfo(name, 443)


def _sudo(argv: list[str]) -> int:
    return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=30).returncode


def _ids() -> tuple[int, int, set[int]]:
    return os.geteuid(), os.getegid(), set(os.getgroups())


def listening(text: str) -> list[str]:
    """The paths of the host's listening Unix sockets, from /proc/net/unix."""
    return sorted({f[7] for ln in text.splitlines()[1:] if len(f := ln.split()) >= 8 and f[7].startswith("/")
                   and int(f[3], 16) & 0x10000})[:SOCKETS_MAX]


def probe(spec: dict, environ=None, connect=_connect, resolve=_resolve, sudo=_sudo, ids=_ids,
          status: Callable[[], str] = lambda: Path("/proc/self/status").read_text(), proc: Path = Path("/proc"),
          interfaces=socket.if_nameindex, which=shutil.which) -> list[dict]:
    """Each probe of ``spec`` (what `prepare` passes into the unit) as {probe, held, detail}; a probe that raises didn't hold."""
    environ = os.environ if environ is None else environ
    euid, egid, groups = ids()

    def user():
        bad = sorted(g for g in spec["gids"] if g == egid or g in groups)
        held = euid == spec["uid"] and 0 not in (euid, egid) and not bad
        return held, f"uid {euid}, gid {egid}" + (f", in forbidden group(s) {bad}" if bad else "")

    def privileges():
        st = dict(ln.split(":", 1) for ln in status().splitlines() if ":" in ln)
        caps = {k: int(st.get(k, "1").strip(), 16) for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")}
        nnp = st.get("NoNewPrivs", "").strip() == "1"
        return nnp and not any(caps.values()), f"NoNewPrivs {int(nnp)}, " + ", ".join(f"{k} {v:x}" for k, v in caps.items() if v)

    def sudo_():
        path = which("sudo", path="/usr/bin:/bin:/usr/sbin:/sbin")
        if not path:
            return True, "no sudo"
        try:
            rc = sudo([path, "-n", "true"])
        except subprocess.TimeoutExpired:
            return False, "`sudo -n true` didn't answer"
        return rc != 0, f"`sudo -n true` exited {rc}"

    def network():
        got = []
        for host, port in spec["reach"]:
            try:
                connect(host, port)
                got.append(f"{host}:{port}")
            except OSError:
                pass
        try:
            resolve(spec["resolve"])
            got.append(f"resolved {spec['resolve']}")
        except OSError:
            pass
        try:
            got += [f"interface {n}" for _, n in interfaces() if n != "lo"]
        except OSError:
            pass
        return not got, ", ".join(got) or "nothing reachable"

    def sockets():                                              # seen, never connected: some start a service when one is
        got = []
        for path in spec["sockets"]:
            try:
                if stat.S_ISSOCK(os.stat(path).st_mode):
                    got.append(path)
            except OSError:
                pass
        return not got, ("visible: " + ", ".join(got)) if got else f"none of the host's {len(spec['sockets'])} visible"

    def secrets():
        got = []
        for p in spec["secrets"]:
            try:
                if os.path.isdir(p):
                    os.listdir(p)
                else:
                    with open(p, "rb") as f:
                        f.read(1)
                got.append(p)
            except OSError:
                pass
        return not got, ("readable: " + ", ".join(got)) if got else f"{len(spec['secrets'])} paths unreadable"

    def writes():
        got = []
        for d in spec["unwritable"]:
            p = os.path.join(d, f".vy-lean-sandbox-probe-{os.getpid()}")
            try:
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except OSError:
                continue
            os.close(fd)
            try:
                os.unlink(p)
            except OSError:
                pass
            got.append(d)
        return not got, ("writable: " + ", ".join(got)) if got else f"{len(spec['unwritable'])} directories unwritable"

    def private():
        got = []
        for d, ident in spec["private"].items():
            try:
                st = os.stat(d)
            except OSError:
                continue
            if [st.st_dev, st.st_ino] == list(ident):
                got.append(d)
        return not got, ("the host's: " + ", ".join(got)) if got else "own " + ", ".join(spec["private"])

    def environment():
        names = set(environ)
        cred = sorted(n for n in names if CREDENTIAL.search(n))
        extra = sorted(n for n in names if n not in spec["env"] and n not in SYSTEMD_ENV and not n.startswith("LC_"))
        return not cred and not extra, "; ".join([*(["credential-like: " + ", ".join(cred)] if cred else []),
                                                  *(["not passed: " + ", ".join(extra)] if extra else [])]) or "only what -E passed"

    def processes():
        others = []
        for pid in os.listdir(proc):
            if pid.isdigit():
                try:
                    if (proc / pid).stat().st_uid != euid:
                        others.append(pid)
                except OSError:
                    pass
        return not others, f"{len(others)} other users' processes visible" if others else "only its own processes"

    out = []
    for name, fn in zip(PROBES, (user, privileges, sudo_, network, sockets, secrets, writes, private, environment, processes)):
        try:
            held, detail = fn()
        except Exception as e:                                      # noqa: BLE001: a probe that can't answer didn't hold
            held, detail = False, f"the probe raised {type(e).__name__}: {e}"
        out.append({"probe": name, "held": bool(held), "detail": detail})
    return out


def verdict(stdout: str) -> list[dict] | None:
    """The probes' results from the probe unit's output, or None when it printed none (it didn't start)."""
    for ln in reversed(stdout.splitlines()):
        if ln.startswith(MARK):
            try:
                return json.loads(ln[len(MARK):])
            except ValueError:
                return None
    return None


def breaches(results: list[dict]) -> list[str]:
    """Every probe that didn't hold, or that didn't report."""
    seen = {r.get("probe"): r for r in results}
    return [f"{p}: did not report" if p not in seen else f"{p}: {seen[p].get('detail')}" for p in PROBES
            if p not in seen or seen[p].get("held") is not True]


# ---- outside the unit, as root

@dataclass
class Spec:
    user: str
    uid: int
    tree: str
    tree_in: str
    elan: str
    ro: list[tuple[str, str]] = field(default_factory=list)
    rw: list[tuple[str, str]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    hidden: list[str] = field(default_factory=list)
    cpus: str | None = None
    memory: str = MEMORY
    max_sec: int = MAX_SEC
    nice: int = 0
    io_idle: bool = False
    checks: dict = field(default_factory=dict)


def _bind(arg: str) -> tuple[str, str]:
    src, _, dst = arg.partition(":")
    if not os.path.isabs(src) or (dst and not os.path.isabs(dst)):
        raise Refused(f"{arg!r}: a bind is SRC[:DEST], both absolute")
    if not os.path.exists(src):
        raise Refused(f"{src} doesn't exist")
    return os.path.normpath(src), os.path.normpath(dst or src)


def unreadable(top: str, limit: int = 3) -> list[str]:
    """Up to ``limit`` paths in ``top`` that other users can't read (Mathlib's cache writes its builds 0600)."""
    out = []
    for d, dirs, files in os.walk(top):
        for p in [*([top] if d == top else []), *(os.path.join(d, n) for n in (*dirs, *files))]:
            mode = os.lstat(p).st_mode
            need = 0o005 if stat.S_ISDIR(mode) else 0o004
            if not stat.S_ISLNK(mode) and mode & need != need:
                out.append(p)
                if len(out) >= limit:
                    return out
    return out


def _io_idle() -> bool:
    """Whether this process's IO class is idle (the daily service's), so its units get the same; False when unknown."""
    try:
        r = subprocess.run(["ionice", "-p", str(os.getpid())], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and r.stdout.strip().startswith("idle")


def prepare(ns: argparse.Namespace, geteuid=os.geteuid, getpwnam=pwd.getpwnam, getgrall=grp.getgrall,
            host: Callable[[str], os.stat_result] = os.stat, listdir=os.listdir, io_idle=_io_idle,
            net_unix: Callable[[], str] = lambda: Path("/proc/net/unix").read_text(),
            exists: Callable[[str], bool] = os.path.exists) -> Spec:
    """The unit for ``ns`` (run's or probe's options), checked; Refused before anything starts."""
    if geteuid() != 0:
        raise Refused("run as root (`sudo -n vy-lean-sandbox ...`): it starts the unit as the build's user, and never "
                      "runs the build as anyone else")
    try:
        pw = getpwnam(ns.user)
    except KeyError:
        raise Refused(f"no user {ns.user!r}: `sudo -n vy-lean-build-setup` creates it") from None
    groups = {g.gr_name: g for g in getgrall()}
    member = sorted(n for n in FORBIDDEN_GROUPS if n in groups and (ns.user in groups[n].gr_mem or groups[n].gr_gid == pw.pw_gid))
    if pw.pw_uid == 0 or ns.user in FORBIDDEN_USERS or member:
        raise Refused(f"won't build as {ns.user!r}" + (f" (in {', '.join(member)})" if member else ""))
    tree, _, tree_in = ns.tree.partition(":")
    tree_in = tree_in or tree
    if not os.path.isabs(tree) or not os.path.isabs(tree_in) or len(Path(tree_in).parts) < 3:
        raise Refused(f"--tree {ns.tree!r}: an absolute directory below a top-level one, like {TREE}")
    if not os.path.isdir(tree):
        raise Refused(f"no tree {tree}: `vy-lean-sandbox checkout SHA` makes {TREE}")
    if not os.path.isdir(ns.elan):
        raise Refused(f"no elan toolchains at {ns.elan}")
    ro = [_bind(b) for b in ns.ro]
    rw = [_bind(b) for b in ns.rw]
    deps = []
    for d in ns.deps:
        pkg, _, src = d.partition("=")
        if not pkg or os.path.isabs(pkg) or ".." in Path(pkg).parts or not src:
            raise Refused(f"--deps {d!r}: PKG=DIR, PKG relative to the tree")
        src, _ = _bind(src)
        if bad := unreadable(src):
            raise Refused(f"--deps {d!r}: the build's user can't read {', '.join(bad)}: restore with `chmod -R a+rX` after")
        mount = Path(tree) / pkg / ".lake" / "deps"
        if not mount.is_dir():
            if not mount.parent.is_dir():
                raise Refused(f"--deps {d!r}: {mount.parent} isn't a Lake package's .lake (checkout makes one per package)")
            mount.mkdir(mode=0o755)
        deps.append((src, str(Path(tree_in) / pkg / ".lake" / "deps")))
    for dst in (ELAN_IN, tree_in, *(d for _, d in (*ro, *rw))):
        if not exists(dst):
            raise Refused(f"no mount point {dst}: systemd would make it on the host's disk (`sudo -n vy-lean-build-setup` "
                          f"makes {ELAN_IN} and {Path(TREE).parent})")
    env = dict(BASE_ENV)
    for e in ns.setenv:
        k, eq, v = e.partition("=")
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            raise Refused(f"-E {e!r}: NAME=VALUE (a bare NAME would pass the caller's value)")
        if k in BASE_ENV or k in SYSTEMD_ENV:
            raise Refused(f"-E {k}: the sandbox sets it")
        if CREDENTIAL.search(k):
            raise Refused(f"-E {k}: a credential's name; a build gets none")
        env[k] = v
    cpus = ns.cpus
    if cpus is None and os.path.isfile(DIRECT_CPUS):
        cpus = Path(DIRECT_CPUS).read_text().strip() or None
    try:
        verity = sorted(os.path.join("/etc/verity", n) for n in listdir("/etc/verity"))
    except FileNotFoundError:
        verity = []
    hidden = list(HIDDEN)
    leases = sorted(glob.glob("/run/gpu-lease*"))
    private = {}
    for d in PRIVATE:
        try:
            st = host(d)
            private[d] = [st.st_dev, st.st_ino]
        except OSError:
            pass
    ro_in = [dst for _, dst in (*ro, *deps)]
    unwritable = sorted({*UNWRITABLE, tree_in, str(Path(tree_in).parent), *ro_in})
    gids = sorted({groups[n].gr_gid for n in FORBIDDEN_GROUPS if n in groups})
    checks = {"uid": pw.pw_uid, "gids": gids, "env": sorted(env), "secrets": sorted({*SECRETS, *verity, *hidden, *leases}),
              "unwritable": unwritable, "private": private, "reach": [list(r) for r in REACH], "resolve": RESOLVE,
              "sockets": listening(net_unix())}
    return Spec(ns.user, pw.pw_uid, tree, tree_in, ns.elan, ro=[*ro, *deps], rw=rw, env=env, hidden=hidden,
                cpus=cpus, memory=ns.memory, max_sec=ns.max_sec, nice=os.nice(0), io_idle=io_idle(), checks=checks)


def unit_argv(spec: Spec, unit: str, cmd: list[str]) -> list[str]:
    """`systemd-run` for ``cmd`` in the sandbox: the probe unit and the build unit differ only in ``unit`` and ``cmd``."""
    props = [
        "NoNewPrivileges=yes", "RestrictSUIDSGID=yes", "CapabilityBoundingSet=", "AmbientCapabilities=",
        "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes", "PrivateDevices=yes", "PrivateNetwork=yes",
        "PrivateIPC=yes", "ProtectProc=invisible", "ProtectKernelTunables=yes", "ProtectKernelModules=yes",
        "ProtectKernelLogs=yes", "ProtectControlGroups=yes", "ProtectClock=yes", "ProtectHostname=yes",
        "RestrictNamespaces=yes", "LockPersonality=yes", "RestrictRealtime=yes", "RestrictAddressFamilies=AF_UNIX",
        "SystemCallArchitectures=native", "KeyringMode=private", "UMask=0022",
        "TemporaryFileSystem=/workspace:ro", "TemporaryFileSystem=/run:ro", "TemporaryFileSystem=/dev/shm",
        *(f"InaccessiblePaths=-{p}" for p in spec.hidden),
        f"BindReadOnlyPaths={spec.elan}:{ELAN_IN}",
        f"BindPaths={spec.tree}:{spec.tree_in}",
        *(f"BindReadOnlyPaths={s}:{d}" for s, d in spec.ro),
        *(f"BindPaths={s}:{d}" for s, d in spec.rw),
        f"MemoryMax={spec.memory}", f"TasksMax={TASKS}", f"RuntimeMaxSec={spec.max_sec}", f"Nice={spec.nice}",
        # `--wait` returns when the unit is inactive: with these, only once every process in its cgroup is gone
        "KillMode=control-group", "SendSIGKILL=yes",
        *([f"AllowedCPUs={spec.cpus}"] if spec.cpus else []),
        *(["IOSchedulingClass=idle"] if spec.io_idle else []),
    ]
    return ["systemd-run", f"--unit={unit}", f"--uid={spec.user}", "--wait", "--pipe", "--collect",
            "--quiet", "--service-type=exec", f"--working-directory={spec.tree_in}",
            *(x for p in props for x in ("-p", p)), *(x for k, v in sorted(spec.env.items()) for x in ("-E", f"{k}={v}")),
            "--", *cmd]


def probe_cmd(spec: Spec) -> list[str]:
    return [PYTHON, "-I", "-c", Path(__file__).read_text(), "_probe", json.dumps(spec.checks)]


Runner = Callable[[list[str], bool, str], "tuple[int, str]"]


def systemd_run(argv: list[str], capture: bool, unit: str) -> tuple[int, str]:
    """Run `systemd-run` and wait; on SIGTERM, SIGINT or SIGHUP stop the unit, then end with that signal's status."""
    got: list[int] = []

    def stop(signum, _frame):
        got.append(signum)
        subprocess.run(["systemctl", "stop", "--no-block", unit], capture_output=True)

    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE if capture else None)
        out, _ = p.communicate()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    if got:
        subprocess.run(["systemctl", "stop", unit], capture_output=True)
        log(f"stopped {unit} on signal {got[0]}")
        sys.exit(128 + got[0])
    return p.returncode, (out or b"").decode(errors="replace")


def sandboxed(spec: Spec, cmd: list[str] | None, run: Runner = systemd_run, tag: str | None = None) -> int:
    """The probes, then ``cmd`` (None: the probes only), each in its own unit."""
    tag = tag or f"{os.getpid()}-{os.urandom(3).hex()}"
    unit = f"{UNIT_PREFIX}probe-{tag}"
    rc, out = run(unit_argv(spec, unit, probe_cmd(spec)), True, unit)
    results = verdict(out)
    if results is None:
        log(f"the sandbox didn't start (systemd-run exited {rc}, the probes reported nothing): nothing ran")
        return REFUSED
    bad = breaches(results)
    for r in results:
        log(f"probe {r['probe']}: {'held' if r['held'] else 'GOT THROUGH'} ({r['detail']})")
    if bad:
        log(f"{len(bad)} probe(s) got through: no build, no upload")
        return BREACH
    if cmd is None:
        return OK
    unit = f"{UNIT_PREFIX}build-{tag}"
    log(f"building in {unit} as {spec.user} in {spec.tree_in}: {' '.join(cmd)}")
    rc, _ = run(unit_argv(spec, unit, ["/usr/bin/env", *cmd]), False, unit)
    if rc in SETUP_FAILED:
        log(f"the build's unit didn't start (status {rc}): nothing ran")
        return REFUSED
    if rc != 0:
        log(f"the build exited {rc}")
        return FAILED
    return OK


# ---- the `build` slot, in lean_slot.py's files

def _owned(fd: int, d: Path, mode: int) -> None:
    """Give what root just made in the shared lock directory its owner, so lean_slot.py (as research) can use it."""
    if os.geteuid() != 0:
        return
    st = d.stat()
    os.fchown(fd, st.st_uid, st.st_gid)
    os.fchmod(fd, mode)


def _plain(fd: int, where: Path) -> int:
    """``fd`` if it is a regular file with one name; otherwise closed, and refused."""
    st = os.fstat(fd)
    if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
        return fd
    os.close(fd)
    raise Refused(f"{where} isn't a plain file with one name ({stat.filemode(st.st_mode)}, {st.st_nlink} names): move it aside")


def _open(where: Path, d: Path, mode: int, dir_fd: int | None = None) -> int:
    """``where`` (by its name in ``dir_fd``, if given) read-write, never through a link nor stuck on a FIFO, and only a plain
    file: ``d``'s owner gets it if root made it here and now; one that was there stays as it was."""
    path = where.name if dir_fd is not None else where
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        try:
            fd, made = os.open(path, flags | os.O_CREAT | os.O_EXCL, mode, dir_fd=dir_fd), True
        except FileExistsError:
            fd, made = os.open(path, flags, dir_fd=dir_fd), False
    except OSError as e:
        raise Refused(f"can't open {where} as a plain file ({e.strerror}): move it aside") from None
    _plain(fd, where)
    if made:
        _owned(fd, d, mode)
    return fd


def pools(slots: Path) -> dict[str, int]:
    try:
        text = slots.read_text()
    except FileNotFoundError:
        raise Refused(f"no Lean slot file {slots}: a sandboxed build never runs uncapped") from None
    out = {}
    for ln in text.splitlines():
        f = ln.split("#")[0].split()
        if not f:
            continue
        if len(f) != 2 or not f[1].isdigit():
            raise Refused(f"{slots}: {ln.strip()!r} is not `POOL N`")
        out[f[0]] = int(f[1])
    if POOL not in out:
        raise Refused(f"{slots} lists no pool {POOL!r}, the only one a sandboxed build takes; the steward adds `{POOL} N`")
    return out


def _try(d: Path, i: int) -> int | None:
    fd = _open(d / f"lean-{POOL}-{i}.lock", d, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _join(d: Path, who: dict, ns: int) -> tuple[int, str, int]:
    """slot.py's `join`: a ticket in the pool's line, named by arrival and locked before it is visible: (the line's
    descriptor, the ticket's name in it, the ticket's descriptor)."""
    line = d / f"lean-line-{POOL}"
    try:
        os.mkdir(line, 0o775)
        made = True
    except FileExistsError:
        made = False
    try:
        lfd = os.open(line, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise Refused(f"can't open {line} as a directory ({e.strerror}): move it aside") from None
    try:
        if made:
            _owned(lfd, d, 0o775)
        name = f"{ns:020d}-{os.getpid()}"
        fd = _open(line / f".{name}", d, 0o664, dir_fd=lfd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, (json.dumps(who) + "\n").encode())
        os.rename(f".{name}", name, src_dir_fd=lfd, dst_dir_fd=lfd)
    except BaseException:
        os.close(lfd)
        raise
    return lfd, name, fd


def _ahead(line: int, ticket: str) -> int:
    """slot.py's `ahead`: the live waiters before ``ticket`` in ``line``; a dead waiter's ticket is removed, and what isn't a
    plain file is no waiter's."""
    n = 0
    for p in sorted(os.listdir(line)):
        if p >= ticket:
            break
        if p.startswith("."):
            continue
        try:
            fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=line)
        except OSError:
            continue
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                continue
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _unlink(p, line)
        except BlockingIOError:
            n += 1
        finally:
            os.close(fd)
    return n


def _unlink(name: str, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass


def take(slots: Path, who: dict, wait_s: float | None = None, now=lambda: datetime.now(timezone.utc),
         sleep=time.sleep) -> tuple[int, int] | None:
    """A slot of the `build` pool, in arrival order with lean_slot.py's waiters: (its index, the descriptor holding it), or
    None when ``wait_s`` passed first."""
    d = slots.parent
    n = pools(slots)[POOL]
    start = now()
    line, ticket, tfd = _join(d, who, int(start.timestamp() * 1e9))
    told = None
    try:
        while True:
            try:
                n = pools(slots)[POOL]
            except (OSError, Refused):
                pass                                            # mid-rewrite: the last N it read
            ahead = _ahead(line, ticket)
            got = None if ahead else next(((i, fd) for i in range(n) if (fd := _try(d, i)) is not None), None)
            if got:
                break
            if wait_s is not None and (now() - start).total_seconds() >= wait_s:
                log(f"no {POOL} slot within {wait_s:.0f} s: nothing ran")
                return None
            why = f"{ahead} ahead in lean-line-{POOL}" if ahead else f"all {n} {POOL} slot(s) taken"
            if why != told:
                log(f"waiting for a {POOL} slot ({why}), retrying every {POLL_S} s ({slots})")
                told = why
            sleep(POLL_S)
    finally:
        _unlink(ticket, line)
        os.close(tfd)
        os.close(line)
    i, fd = got
    path = d / f"lean-{POOL}-{i}.who"
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        wfd = _open(tmp, d, 0o664)
    except Refused:
        os.close(fd)
        raise
    try:
        os.ftruncate(wfd, 0)
        os.write(wfd, (json.dumps({**who, "took": f"{now():%Y-%m-%dT%H:%MZ}"}) + "\n").encode())
    finally:
        os.close(wfd)
    os.replace(tmp, path)
    log(f"{POOL} slot {i} of {n} ({slots})")
    return i, fd


def held(slots: Path, slot: str) -> None:
    """Refused unless ``slot`` (POOL/I) is a `build` slot someone holds."""
    pool, _, i = slot.partition("/")
    if pool != POOL or not i.isdigit():
        raise Refused(f"--held-slot {slot!r}: {POOL}/I")
    path = slots.parent / f"lean-{pool}-{i}.lock"
    try:
        fd = _plain(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), path)
    except FileNotFoundError:
        raise Refused(f"--held-slot {slot}: no {path}") from None
    except OSError as e:
        raise Refused(f"--held-slot {slot}: can't open {path} as a plain file ({e.strerror})") from None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    finally:
        os.close(fd)
    raise Refused(f"--held-slot {slot}: nobody holds it; take it first, or leave --held-slot out")


# ---- the tree

def _git(*args: str, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *(["-C", cwd] if cwd else []), *args], capture_output=True, text=True, check=check)


def checkout(sha: str, repo: str, main: str, tree: str, uid: int, gid: int, chown=os.chown) -> None:
    """``tree``: a fresh clone of ``sha``, which must be on ``main`` of ``repo``; root owns it, the build each `.lake`."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Refused(f"{sha!r}: a full commit SHA")
    t = Path(tree)
    if not t.is_absolute() or len(t.parts) < 3 or t.is_symlink():
        raise Refused(f"--tree {tree!r}: an absolute directory below a top-level one, like {TREE}")
    if not os.path.isdir(repo):
        raise Refused(f"no repository {repo}")
    umask = os.umask(0o022)
    try:
        if t.exists():
            shutil.rmtree(t)
        t.parent.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", str(t))
        # the local transport drops `-c` settings before upload-pack, which runs in another user's repository
        r = _git("-c", "protocol.file.allow=always", "fetch", "-q", "--no-tags",
                 f"--upload-pack=git -c safe.directory={repo} upload-pack", f"file://{repo}",
                 f"+{main}:refs/remotes/origin/main", cwd=tree, check=False)
        if r.returncode != 0:
            raise Refused(f"{repo} has no {main} to check {sha[:12]} against: {r.stderr.strip()[-300:]}")
        tip = _git("rev-parse", "refs/remotes/origin/main", cwd=tree).stdout.strip()
        if _git("merge-base", "--is-ancestor", sha, "refs/remotes/origin/main", cwd=tree, check=False).returncode != 0:
            raise Refused(f"{sha[:12]} is not on {main} of {repo} (at {tip[:12]})")
        _git("checkout", "-q", "--detach", sha, cwd=tree)
        pkgs = sorted({str(Path(p).parent) for p in _git("ls-files", "-z", cwd=tree).stdout.split("\0")
                       if Path(p).name in ("lakefile.lean", "lakefile.toml")})
        for pkg in pkgs:
            lake = t / pkg / ".lake"
            lake.mkdir(exist_ok=True)
            chown(lake, uid, gid)
    except (Refused, subprocess.CalledProcessError, OSError) as e:
        shutil.rmtree(t, ignore_errors=True)
        if isinstance(e, subprocess.CalledProcessError):
            raise Refused(f"`git {' '.join(e.cmd[1:])}` failed: {(e.stderr or '').strip()[-300:]}") from None
        if isinstance(e, OSError):
            raise Refused(f"checkout failed: {e}") from None
        raise
    finally:
        os.umask(umask)
    log(f"{tree}: {sha[:12]} on {main} (at {tip[:12]}), {len(pkgs)} Lake package(s) writable by uid {uid}")


def dirty(tree: str, sha: str | None = None) -> list[str]:
    """What keeps ``tree`` from being ``sha``'s (default its HEAD's): changed, added or ignored paths outside `.lake`."""
    git = ("-c", f"safe.directory={tree}")
    r = _git(*git, "rev-parse", "--verify", "-q", "HEAD^{commit}", cwd=tree, check=False)
    if r.returncode != 0:
        return [f"{tree} has no commit checked out"]
    head = r.stdout.strip()
    out = [f"HEAD is {head[:12]}, not {sha[:12]}"] if sha and head != sha else []
    entries = _git(*git, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching", cwd=tree).stdout.split("\0")
    skip = False
    for e in entries:
        if skip or not e:
            skip = False
            continue
        xy, path = e[:2], e[3:]
        skip = xy[0] in "RC"
        if not (xy == "!!" and ".lake" in Path(path).parts):
            out.append(f"{xy} {path}")
    return out


Systemctl = Callable[[list[str]], "tuple[int, str]"]


def _systemctl(argv: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(["systemctl", *argv], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)
    return r.returncode, r.stdout if r.returncode == 0 else r.stderr


def alive(uid: int, proc: Path = PROC, systemctl: Systemctl = _systemctl) -> list[str]:
    """What of a build may still run: a process with ``uid`` as its real, effective, saved or filesystem uid, and a sandbox unit
    neither inactive nor failed. A process that ends while it is read is gone, not a reason."""
    out = []
    for pid in sorted((p for p in os.listdir(proc) if p.isdigit()), key=int):
        try:
            st = dict(ln.split(":", 1) for ln in (proc / pid / "status").read_text().splitlines() if ":" in ln)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as e:
            out.append(f"pid {pid}: its status is unreadable ({e})")
            continue
        uids = st.get("Uid", "").split()
        if len(uids) != 4:
            out.append(f"pid {pid}: no uids in its status")
        elif str(uid) in uids:
            out.append(f"still running: pid {pid} uid {uid} ({st.get('Name', '').strip()})")
    rc, text = systemctl(["list-units", "--all", "--plain", "--no-legend", "--full", f"{UNIT_PREFIX}*"])
    if rc != 0:
        return [*out, f"the sandbox's units are unknown (`systemctl list-units` exited {rc}: {text.strip()[-200:]})"]
    for ln in text.splitlines():
        f = ln.split()
        f = f[1:] if f[:1] == ["●"] else f
        if not f:
            continue
        if len(f) < 3 or not f[0].startswith(UNIT_PREFIX):
            out.append(f"systemctl listed {ln.strip()!r}")
        elif f[2] not in ("inactive", "failed"):
            out.append(f"unit {f[0]} is {f[2]}")
    return out


KINDS = {stat.S_IFLNK: "a symbolic link", stat.S_IFIFO: "a FIFO", stat.S_IFSOCK: "a socket", stat.S_IFBLK: "a block device",
         stat.S_IFCHR: "a character device", stat.S_IFREG: "a file", stat.S_IFDIR: "a directory"}


def _untrusted(st: os.stat_result, uid: int) -> str | None:
    kind = stat.S_IFMT(st.st_mode)
    if kind not in (stat.S_IFREG, stat.S_IFDIR):
        return KINDS.get(kind, f"of type {kind:o}")
    if kind == stat.S_IFREG and st.st_nlink > 1:
        return f"a file with {st.st_nlink} names"
    if st.st_uid not in (0, uid):
        return f"owned by uid {st.st_uid}"
    if st.st_mode & 0o022:                                      # an ACL granting another user write shows in the group bits
        return f"writable by group or others (mode {stat.S_IMODE(st.st_mode):o})"
    if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
        return f"setuid or setgid (mode {stat.S_IMODE(st.st_mode):o})"
    return None


def unsafe(tree: str, uid: int) -> list[str]:
    """What in ``tree``'s `.lake` directories a publisher can't trust (see `clean`), and a `.lake` that isn't a directory;
    walked from directory descriptors, never following a link nor leaving the tree's filesystem."""
    out: list[str] = []

    def walk(fd: int, rel: str, inside: bool, depth: int) -> None:
        try:
            names = sorted(os.listdir(fd))
        except OSError as e:
            out.append(f"{rel or '.'}: {e.strerror}")
            return
        for name in names:
            p = rel + name
            if p == ".git":
                continue
            try:
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except OSError as e:
                out.append(f"{p}: {e.strerror}")
                continue
            lake = inside or name == ".lake"
            if name == ".lake" and not inside and not stat.S_ISDIR(st.st_mode):
                out.append(f"{p}: {KINDS.get(stat.S_IFMT(st.st_mode), 'not a directory')}, not a directory")
                continue
            if lake and (why := _untrusted(st, uid)):
                out.append(f"{p}: {why}")
                continue
            if not stat.S_ISDIR(st.st_mode):
                continue
            if st.st_dev != dev:
                out.append(f"{p}: another filesystem")
                continue
            if depth >= DEPTH:
                out.append(f"{p}: more than {DEPTH} directories down")
                continue
            try:
                sub = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as e:
                out.append(f"{p}: {e.strerror}")
                continue
            try:
                walk(sub, p + "/", lake, depth + 1)
            finally:
                os.close(sub)

    try:
        top = os.open(tree, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        return [f"can't open {tree}: {e.strerror}"]
    try:
        dev = os.fstat(top).st_dev
        walk(top, "", False, 0)
    finally:
        os.close(top)
    return out


# ---- the command line

def _unit_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tree", default=TREE, help=f"DIR[:DEST], default {TREE}")
    p.add_argument("--deps", action="append", default=[], metavar="PKG=DIR")
    p.add_argument("--ro", action="append", default=[], metavar="SRC[:DEST]")
    p.add_argument("--rw", action="append", default=[], metavar="SRC[:DEST]")
    p.add_argument("-E", "--setenv", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--elan", default=ELAN)
    p.add_argument("--user", default=USER)
    p.add_argument("--cpus", default=None, help=f"AllowedCPUs, default {DIRECT_CPUS}'s")
    p.add_argument("--memory", default=MEMORY)
    p.add_argument("--max-sec", type=int, default=MAX_SEC)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vy-lean-sandbox", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkout", help="a fresh clone of SHA, which must be on main")
    c.add_argument("sha")
    c.add_argument("--repo", default=REPO)
    c.add_argument("--main", default=MAIN)
    c.add_argument("--tree", default=TREE)
    c.add_argument("--user", default=USER)
    r = sub.add_parser("run", help="the probes, then CMD")
    _unit_opts(r)
    r.add_argument("--slots", default=SLOTS)
    r.add_argument("--slot-wait", type=float, default=None, metavar="S", help="give up (exit 4) without a slot after S seconds")
    r.add_argument("--held-slot", default=None, metavar="build/I", help="the caller holds this slot")
    r.add_argument("--by", default=None)
    r.add_argument("command", nargs=argparse.REMAINDER)
    _unit_opts(sub.add_parser("probe", help="the probes only"))
    k = sub.add_parser("clean", help="whether the build is over and its tree is safe to publish")
    k.add_argument("--tree", default=TREE)
    k.add_argument("--sha", default=None)
    k.add_argument("--user", default=USER)
    return ap


def main(argv: list[str], run: Runner = systemd_run, now=lambda: datetime.now(timezone.utc), sleep=time.sleep,
         geteuid=os.geteuid, getpwnam=pwd.getpwnam, proc: Path = PROC, systemctl: Systemctl = _systemctl, **prep) -> int:
    if argv[:1] == ["_probe"]:
        print(MARK + json.dumps(probe(json.loads(argv[1]))), flush=True)
        return OK
    ns = parser().parse_args(argv)
    try:
        if ns.cmd == "checkout":
            if geteuid() != 0:
                raise Refused("run as root (`sudo -n vy-lean-sandbox checkout ...`)")
            try:
                pw = getpwnam(ns.user)
            except KeyError:
                raise Refused(f"no user {ns.user!r}: `sudo -n vy-lean-build-setup` creates it") from None
            checkout(ns.sha, ns.repo, ns.main, ns.tree, pw.pw_uid, pw.pw_gid)
            if bad := dirty(ns.tree, ns.sha):
                raise Refused(f"{ns.tree} isn't {ns.sha[:12]}'s tree after the checkout: {'; '.join(bad[:5])}")
            return OK
        if ns.cmd == "clean":
            if geteuid() != 0:
                raise Refused("run as root (`sudo -n vy-lean-sandbox clean ...`): only root sees every process and file of "
                              "the build")
            try:
                uid = getpwnam(ns.user).pw_uid
            except KeyError:
                raise Refused(f"no user {ns.user!r}") from None
            bad = [*alive(uid, proc, systemctl), *unsafe(ns.tree, uid), *dirty(ns.tree, ns.sha)]
            for b in bad[:20]:
                log(f"{ns.tree}: {b}")
            if bad:
                log(f"{ns.tree} isn't clean ({len(bad)} reason(s)): don't upload it")
            return FAILED if bad else OK
        spec = prepare(ns, geteuid=geteuid, getpwnam=getpwnam, **prep)
        if ns.cmd == "probe":
            return sandboxed(spec, None, run)
        cmd = ns.command[1:] if ns.command[:1] == ["--"] else ns.command
        if not cmd:
            raise Refused("run ... -- CMD")
        slots = Path(ns.slots)
        fd = None
        if ns.held_slot:
            held(slots, ns.held_slot)
        else:
            got = take(slots, {"pid": os.getpid(), "run": None, "tree": spec.tree, "by": ns.by}, ns.slot_wait, now, sleep)
            if got is None:
                return NO_SLOT
            fd = got[1]
        try:
            return sandboxed(spec, cmd, run)
        finally:
            if fd is not None:
                os.close(fd)
    except Refused as e:
        log(f"refused: {e}")
        return REFUSED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
