#!/usr/bin/python3 -I
"""vy-lockout: the Nebius nodes' lock-out, step by step (root, stdlib only; run as vy-admin with sudo, never as research).

    vy-lockout plan NODE                      every step in order, with its check and its rollback
    vy-lockout preflight NODE [--json]        each check the sudoers step needs, pass or fail, and how to fix a failure
    vy-lockout switch NODE (--unit U | --all) [--apply] [--back]     a tmux loop -> its systemd unit (--back: unit -> tmux)
    vy-lockout root-units NODE [--apply] [--back]                    the loops that need cluster-admin, as root
    vy-lockout roots [--apply] [--back]       the runs roots root's and sticky, so a sealed run can't be renamed or replaced
    vy-lockout kubeconfig [--apply] [--back]  research's kubeconfig: cluster-admin out, a read-only token in
    vy-lockout harden [--apply] [--back]      ubuntu out of lxd, py-spy without cap_sys_ptrace, research in systemd-journal
    vy-lockout sudoers [--apply] [--skip CHECK]...   research's ALL -> the allowlist, with the 24 h fallback
    vy-lockout fallback on [--hours H] | off | expire                the fallback; `on` only for vy-admin
    vy-lockout audit [--since -24h]           what research ran under sudo that the allowlist would refuse (no arguments printed)
    vy-lockout rollback [--apply]             the sudoers step undone: cloud-init's grant back, the allowlist and fallback out
    vy-lockout render                         the allowlist (ALLOW) as sudoers: sudoers/50-vy-research is this, byte for byte

Without --apply every step says what it would do and changes nothing. A step with --apply stops at its first failure. Each
step's --back undoes it; what a step moves aside goes to BACKUP (root's), never deleted.

The allowlist is ALLOW below, one Rule per command form with the evidence for it (a count in the nodes' sudo journal on
2026-10-07, over the 3 days before, or the repository file that runs it); EXCLUDED says why each other form research ran is
not in it. The sudoers step installs it beside a fallback that grants research ALL again until its `# expires` line, which
vy-sudo-fallback-expire.timer (root, every 10 min and at boot) enforces: past it, or without a readable one, the file goes.
"""
from __future__ import annotations

import argparse
import base64
import calendar
import fnmatch
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LIB = Path("/usr/local/lib/vy-lockout")
SUDOERS_D = Path("/etc/sudoers.d")
ALLOWLIST = "50-vy-research"
FALLBACK = "59-vy-research-fallback"
ADMIN = "10-vy-admin"
CLOUD_INIT = "90-cloud-init-users"
BACKUP = Path("/root/vy-lockout")
UNITS = Path("/etc/systemd/system")
ENVS = Path("/etc/vy/env")
DROPIN = "50-vy-root.conf"
RUNS_ROOTS = ("/workspace/research/runs", "/workspace/jobs/runs")
K3S_CONFIG = "/etc/rancher/k3s/k3s.yaml"
RESEARCH_KUBE = Path("/home/research/.kube")
PY_SPY = "/usr/local/bin/py-spy"
FALLBACK_HOURS = 24
SHELLS = ("bash", "sh", "zsh", "dash")
REFUSED = 3
TS = "%Y-%m-%dT%H:%M:%SZ"


# ---- the allowlist ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    cmd: str            # absolute path
    args: str           # sudoers argument pattern ("" = no arguments; `*` matches anything, `\*` a literal star)
    evidence: str


#: research -> root. Every variable-argument form goes through vy-sudo or vy-seal-runs, which check what sudoers can't.
ALLOW = (
    Rule("/usr/local/sbin/vy-sudo", "renice *", "journal n1 926 `renice -n -N -p N` (benchmarks/network_traces/active_replay.py:1115, "
         "its own threads); vy-sudo: own pids only"),
    Rule("/usr/local/sbin/vy-sudo", "taskset *", "journal n2 491 + n1 217 `taskset -pc|-apc CPUS PID`; vy-sudo: own pid only"),
    Rule("/usr/local/sbin/vy-sudo", "systemd-run *", "journal n1 43 + n2 31 `systemd-run --scope ... --slice=`, n1 6 `--uid= --pipe` "
         "(jobuser.py:196, benchmarks/pouw/hidden_zk/scope.sh:14); vy-sudo: runs as the caller or a job user"),
    Rule("/usr/local/sbin/vy-sudo", "env *", "journal n2 48 `env PATH=* systemd-run --quiet --scope` (provers.sh:8, pous_quiet.sh:8, "
         "device.sh:17, fill_runner.py:472)"),
    Rule("/usr/local/sbin/vy-sudo", "systemctl *", "journal n2 2 `systemctl stop hidden-zk-*.scope` (scope.sh:18), fill_runner.py:399 "
         "(fill-timing-*.scope); vy-sudo: a scope of the caller's processes only"),
    Rule("/usr/local/sbin/vy-sudo", "hf-ns *", "n2_build.sh:143, n2_commit.sh:288 (`sudo -n unshare --mount ...`, which ran bash as root)"),
    Rule("/usr/local/sbin/vy-seal-runs", "delete *", "research.retention gc/rm on a sealed run (this change)"),
    Rule("/usr/local/sbin/vy-seal-runs", "evict *", "research data evict --runs on a sealed run (vy-store-evict-research.service, as research)"),
    Rule("/usr/local/bin/vy-lean-sandbox", "*", "journal n1 5 (`vy-lean-sandbox probe`); lean_sandbox.py:17 is meant for `sudo -n` "
         "and never runs a build as root, research or a privileged group"),
    Rule("/usr/bin/systemctl", "start vy-steward-watch.service", "journal n1 3 (infra's duty pass; the watch is read-only)"),
    Rule("/usr/bin/true", "", "journal n2 21 + n1 11 `sudo -n true` (infra/cluster/src/cluster/status.py:54)"),
    Rule("/usr/local/sbin/vy-sudo", "du-bundles", "journal n1 23,140 each `find /workspace/jobs/cov /workspace/jobs/probe-jit -xdev "
         "-name replay_bundle_p* -prune -print0` and `xargs -0r du -sb` (infra/cluster/src/cluster/status.py:54, sky/release.py's: a "
         "root unit after the switch); vy-sudo runs that find with -exec du, since `xargs du` as root takes du's options from stdin"),
)
#: who besides root research may run as, for anything: itself (journal n1 1,471 tmux, 512 env), and the job user ubuntu (journal n1
#: 311 `env HOME=* PYTHONPATH=* ...`, 27 mv: n1_custody.sh:245, n2_build.sh:98,202, n2_commit.sh:50,197,215) once ubuntu has no root
RUNAS_ALL = ("ubuntu", "research")
#: ubuntu -> root: vy-store-evict.service runs `research data evict --runs` on /workspace/jobs/runs as ubuntu
UBUNTU_ALLOW = (Rule("/usr/local/sbin/vy-seal-runs", "evict *", "vy-store-evict.service (User=ubuntu) on a sealed job run"),
                Rule("/usr/local/sbin/vy-seal-runs", "delete *", "research retention gc/rm as ubuntu on a sealed job run"))

EXCLUDED = {
    "k3s kubectl, kubectl": "journal n1 410,831 + 106: the loops (dispatch, n1_lease, pool_n1, gpu_stray, release) become root units; "
                            "reads go through research's read-only kubeconfig (research-view.yaml); no cluster-admin",
    "find, xargs, du, readlink, grep, ls, cat, awk, stat, head, sha256sum, lsof, fuser": "journal n1 ~35,000: other users' /proc and "
        "files (gpu_stray's /proc/PID/environ and root, release.py's bundle sizes: root units after the switch); as root, any of "
        "them reads every secret on the node (`xargs du` too: `du --files0-from=FILE` prints FILE in its errors)",
    "rm, mv, cp, chown, chmod, install, tee, sed, rsync, tar, mkdir": "writes as root (n1 423 rm, n2 111 rm, 6 install, 7 tee): "
        "vy-admin's; n2_*.sh's `--rsync-path=\"sudo -n rsync\" --chown=1000:1000` becomes `sudo -n -u ubuntu rsync`",
    "python3 -c": "journal n1 48 + n2 22: research deploy's install/record/attic, which write root's files: vy-admin's",
    "journalctl, dmesg": "journal n1 55 + n2 14: research joins systemd-journal instead (vy-lockout harden)",
    "systemctl daemon-reload|enable|set-property|stop|start (other units)": "journal n1 26 + n2 12: vy-admin's",
    "docker, crontab, xfs_quota, bpftrace, py-spy, grub-*, update-grub, sysctl, setcap, systemd-tmpfiles, git, dmesg":
        "root by another name (a container, a kernel probe, ptrace, boot config): vy-admin's",
    "env, sh, bash, nohup, nice, timeout, tmux (as root)": "a wrapper runs any command: never allowlisted",
    "kill, pkill": "journal n2 8 + n1 4: a process of the caller's needs no sudo",
    "unshare": "n2_build.sh:143 and n2_commit.sh:288 move to `vy-sudo hf-ns`",
    "vy-lean-build-setup, vy-device-cores": "setup that creates users or edits boot config: vy-admin's",
}


def _alias_lines(name: str, rules: tuple[Rule, ...]) -> list[str]:
    specs = [f"{r.cmd} {r.args}".rstrip() if r.args else f'{r.cmd} ""' for r in rules]
    return [f"Cmnd_Alias {name} = \\"] + [f"    {s}, \\" for s in specs[:-1]] + [f"    {specs[-1]}"]


def render() -> str:
    """The allowlist as sudoers (sudoers/50-vy-research)."""
    out = ["# /etc/sudoers.d/50-vy-research: what research (every lane and infra loop) may run as root after the lock-out.",
           "# Generated by `vy-lockout render` from ALLOW in tools/research/src/research/pods/nebius/lockout/lockout.py, which",
           "# gives each line's evidence; edit there. vy-sudo and vy-seal-runs check the arguments sudoers can't.",
           "", *_alias_lines("VY_RESEARCH", ALLOW), "", *_alias_lines("VY_UBUNTU", UBUNTU_ALLOW), "",
           "research ALL=(root) NOPASSWD: VY_RESEARCH",
           f"research ALL=({', '.join(RUNAS_ALL)}) NOPASSWD: ALL",
           "ubuntu ALL=(root) NOPASSWD: VY_UBUNTU", ""]
    return "\n".join(out)


def fallback_text(expires: float) -> str:
    return ("# /etc/sudoers.d/59-vy-research-fallback: research's blanket sudo, for the first day after the allowlist went in.\n"
            "# vy-sudo-fallback-expire.timer removes this file once the time below has passed (or it can't read it);\n"
            "# only vy-admin puts it back (`vy-lockout fallback on`).\n"
            f"# expires {time.strftime(TS, time.gmtime(expires))}\n"
            "research ALL=(ALL) NOPASSWD: ALL\n")


def expiry(text: str) -> float | None:
    m = re.search(r"^# expires (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$", text, re.M)
    if not m:
        return None
    try:
        return float(calendar.timegm(time.strptime(m.group(1), TS)))
    except ValueError:
        return None


def _pattern(args: str) -> str:
    """A sudoers argument pattern as fnmatch's: `*` matches anything (spaces and slashes too), `\\*` a star."""
    out, i = "", 0
    while i < len(args):
        c = args[i]
        if c == "\\" and i + 1 < len(args):
            nxt = args[i + 1]
            out += "[*]" if nxt == "*" else "[?]" if nxt == "?" else re.escape(nxt) if nxt == "[" else nxt
            i += 2
            continue
        out += c
        i += 1
    return out


def allowed(user: str, runas: str, command: str, rules: tuple[Rule, ...] = ALLOW) -> bool:
    """Whether the allowlist alone lets `user` run `command` (path and arguments, as sudo logs it) as `runas`."""
    if user == "research" and runas in RUNAS_ALL:
        return True
    rules = rules if user == "research" else UBUNTU_ALLOW if user == "ubuntu" else ()
    if runas != "root":
        return False
    path, _, args = command.partition(" ")
    return any(r.cmd == path and (args == "" if r.args == "" else fnmatch.fnmatchcase(args, _pattern(r.args))) for r in rules)


SUDO_LINE = re.compile(r"^\s*(?P<user>[a-z_][a-z0-9_-]*) : .*?USER=(?P<runas>[a-z_][a-z0-9_-]*) ; .*?COMMAND=(?P<cmd>.*)$")


def shape(command: str, words: int = 3) -> str:
    """A command as the audit prints it: its path and first words, numbers as N, every VAR=value as VAR=*, long words cut."""
    out = []
    for w in command.split()[: words + 1]:
        w = re.sub(r"=.*", "=*", w) if "=" in w else w
        w = re.sub(r"\d+", "N", w)
        out.append(w[:40] + "..." if len(w) > 40 else w)
    return " ".join(out)


def audit(journal: str) -> list[tuple[int, str]]:
    """(count, shape) of each command form research or ubuntu ran under sudo that the allowlist would refuse, most frequent first."""
    counts: dict[str, int] = {}
    for ln in journal.splitlines():
        m = SUDO_LINE.match(ln)
        if not m or m["user"] not in ("research", "ubuntu") or "command not allowed" in ln:
            continue
        if not allowed(m["user"], m["runas"], m["cmd"].strip()):
            k = f"{m['user']}->{m['runas']} {shape(m['cmd'].strip())}"
            counts[k] = counts.get(k, 0) + 1
    return sorted(((n, k) for k, n in counts.items()), key=lambda t: (-t[0], t[1]))


# ---- the units switch ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Loop:
    unit: str                     # the unit that replaces it
    tmux: str | None              # research's tmux session it replaces (None: nothing ran it before)
    root: bool = False            # a root unit in the root-units step (it needs cluster-admin or other users' /proc)
    code: tuple[str, ...] = ()    # what a root unit runs: files and trees that must be root's and nobody else's to write
    writes: tuple[str, ...] = ()  # directories it writes in: root's too, or a symlink there redirects a root write anywhere


#: Each node's loops in switch order: observers first, then the workers, the GPU leases, and the dispatcher last, which drives
#: the rest. Stop order for --back is the reverse. Node 1's other tmux sessions (fp-feed, gm-*, kueue-shadow, q30b-backstop) are
#: lanes' and stay research's; node 2's pouw-infra-* units are held in deploy.toml (compute accounting's freeze list).
SWITCH = {
    "vy-nebius-1": (
        Loop("vy-write-probe@vy-nebius-1.service", "write-probe"),
        Loop("vy-gpu-stray.service", "gpu-stray", root=True,
             code=("/usr/bin/python3", "/workspace/verity-guest/bin"), writes=("/workspace/verity-guest",)),
        Loop("vy-n1-pool.service", "n1-pool", root=True,
             code=("/usr/bin/python3", "/workspace/verity-guest/bin", "/workspace/jobs/dispatch/infra/nebius"),
             writes=("/workspace/verity-guest", "/workspace/usage")),
        Loop("vy-commit-release.service", "commit-release", root=True,
             code=("/usr/bin/python3", "/workspace/jobs/dispatch/infra/nebius"),
             writes=("/workspace/jobs/dispatch", "/home/research/commit-release")),
        Loop("vy-n2-offload.service", "n2-offload"),
        Loop("vy-n2-commit-offload.service", "n2-commit-offload"),
        Loop("vy-n1-lease@circuits.service", "n1-lease-circuits", root=True,
             code=("/usr/bin/python3", "/workspace/verity-guest/bin", "/workspace/verity-guest/cluster-policy"),
             writes=("/workspace/verity-guest", "/run/gpu-lease-circuits")),
        Loop("vy-n1-lease@provers.service", "n1-lease", root=True,
             code=("/usr/bin/python3", "/workspace/verity-guest/bin", "/workspace/verity-guest/cluster-policy"),
             writes=("/workspace/verity-guest", "/run/gpu-lease")),
        Loop("vy-node1-dispatch.service", "node1-dispatch", root=True,
             code=("/workspace/jobs/venv312", "/workspace/jobs/dispatch/infra/nebius", "/workspace/jobs/dispatch/dispatch.env",
                   "/workspace/verity-guest/cluster-src", "/workspace/verity-guest/bin"),
             writes=("/workspace/jobs/dispatch", "/workspace/verity-guest", "/workspace/monitoring/alerts/outbox")),
        Loop("vy-custody.timer", None, root=True, code=("/usr/local/bin/vy-custody", "/workspace/custody"),
             writes=("/workspace/research/custody-n1",)),
        Loop("vy-monitors.timer", None),
    ),
    "vy-nebius-2": (
        Loop("vy-write-probe@vy-nebius-2.service", "write-probe"),
    ),
}
COMMON_CODE = ("/etc/vy/env", K3S_CONFIG)


def unit_file(unit: str) -> str:
    """The unit file a unit loads: a template's for an instance (vy-n1-lease@provers.service -> vy-n1-lease@.service)."""
    name, _, kind = unit.rpartition(".")
    base, at, _inst = name.partition("@")
    return f"{base}@.{kind}" if at else unit


def service_of(unit: str) -> str:
    return unit.rsplit(".", 1)[0] + ".service" if unit.endswith(".timer") else unit


def tmux_command(unit_text: str, unit: str) -> str:
    """The unit's command as a tmux pane runs it (switch --back): its environment files sourced, its directory, its output file."""
    name = unit.rsplit(".", 1)[0]
    inst = name.partition("@")[2]
    lines = [ln.strip() for ln in unit_text.splitlines()]
    get = lambda k: [ln.split("=", 1)[1] for ln in lines if ln.startswith(k + "=")]
    res = lambda s: s.replace("%N", name).replace("%i", inst).replace("$$", "$").replace("%%", "%")
    envs = [res(e.lstrip("-")) for e in get("EnvironmentFile")]
    start = res(get("ExecStart")[0].lstrip("-@+!"))
    wd = res((get("WorkingDirectory") or ["/"])[0])
    out = next((res(o.split(":", 1)[1]) for o in get("StandardOutput") if o.startswith("append:")), None)
    src = "".join(f". {shlex.quote(e)}; " for e in envs)
    return f"set -a; {src}set +a; cd {shlex.quote(wd)} && {start}" + (f" >> {shlex.quote(out)} 2>&1" if out else "")


# ---- checks -------------------------------------------------------------------------------------------------------------------------

def untrusted(paths: tuple[str, ...] | list[str], lstat: Callable = os.lstat, walk: Callable = os.walk) -> list[str]:
    """Each path (a file, or a tree and all it holds), or a directory above it, that someone other than root can change: not root's,
    or writable by group or others (a sticky root-owned directory above is fine). Symlinks are judged as links and by target."""
    bad: list[str] = []

    def check(p: str, above: bool = False) -> None:
        try:
            st = lstat(p)
        except FileNotFoundError:
            bad.append(f"{p}: missing")
            return
        if st.st_uid != 0:
            bad.append(f"{p}: owner uid {st.st_uid}")
        elif st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode) and not (above and st.st_mode & stat.S_ISVTX):
            bad.append(f"{p}: mode {stat.S_IMODE(st.st_mode):o}")

    for p in paths:
        parts = Path(p).parts
        for i in range(1, len(parts)):
            check(str(Path(*parts[:i])), above=True)
        check(p)
        try:
            is_dir = stat.S_ISDIR(lstat(p).st_mode)
        except OSError:
            continue
        if is_dir:
            for base, dirs, files in walk(p):
                for n in sorted(dirs + files):
                    check(os.path.join(base, n))
        if os.path.islink(p):
            check(os.path.realpath(p))
    seen: set[str] = set()
    return [b for b in bad if not (b in seen or seen.add(b))]


def kubeconfig_admin(text: str) -> bool:
    """Whether a kubeconfig carries a client certificate's key (k3s's admin identity), rather than a token."""
    return bool(re.search(r"^\s*client-key(-data)?:", text, re.M))


def groups_of(user: str, group_file: str = "/etc/group") -> set[str]:
    out = set()
    for ln in Path(group_file).read_text().splitlines():
        f = ln.split(":")
        if len(f) >= 4 and user in f[3].split(","):
            out.add(f[0])
    return out


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def preflight(node: str, run: Runner = _run, root: Path = Path("/")) -> list[Check]:
    """What the sudoers step needs, each as a Check."""
    at = lambda p: root / str(p).lstrip("/")
    out: list[Check] = []
    keys = at("/home/vy-admin/.ssh/authorized_keys")
    admin_sudo = at(SUDOERS_D / ADMIN)
    out.append(Check("vy-admin", keys.is_file() and keys.stat().st_size > 0 and admin_sudo.is_file(),
                     f"authorized_keys {'present' if keys.is_file() else 'missing'}, {ADMIN} {'present' if admin_sudo.is_file() else 'missing'}",
                     "vy-admin-setup KEYS_FILE (Dan's and infra's lead's public keys)"))
    login = run(["journalctl", "-u", "ssh", "-u", "sshd", "--since", "-7d", "-o", "cat", "-g", "Accepted publickey for vy-admin"])
    out.append(Check("vy-admin-login", login.returncode == 0 and "Accepted publickey for vy-admin" in login.stdout,
                     "an ssh login as vy-admin in the last 7 days" if login.stdout.strip() else "no ssh login as vy-admin in the journal",
                     "log in as vy-admin and run `sudo -n true`: root must work through vy-admin before research loses it"))
    try:
        g = groups_of("ubuntu", str(at("/etc/group")))
    except OSError:
        g = set()
    out.append(Check("ubuntu-unprivileged", not g & {"lxd", "docker", "disk"}, f"ubuntu's groups: {', '.join(sorted(g)) or 'none'}",
                     "vy-lockout harden --apply (lxd is root for its members, and research may act as ubuntu)"))
    cap = run(["getcap", PY_SPY])
    out.append(Check("py-spy", "cap_sys_ptrace" not in cap.stdout, cap.stdout.strip() or f"{PY_SPY}: no capability",
                     "vy-lockout harden --apply (cap_sys_ptrace lets anyone read any process's memory)"))
    admin = [str(p) for p in sorted(at(RESEARCH_KUBE).glob("*")) if p.is_file() and kubeconfig_admin(p.read_text(errors="replace"))] \
        if at(RESEARCH_KUBE).is_dir() else []
    out.append(Check("kubeconfig", not admin, "research holds a cluster-admin kubeconfig: " + ", ".join(admin) if admin else
                     "no client key under ~research/.kube", "vy-lockout kubeconfig --apply, then rotate k3s's client certificates"))
    out.append(Check("journal-group", "systemd-journal" in (groups_of("research", str(at("/etc/group"))) if at("/etc/group").exists()
                                                             else set()), "research reads the journal without sudo",
                     "vy-lockout harden --apply"))
    for p in RUNS_ROOTS:
        try:
            st = at(p).lstat()
            ok = st.st_uid == 0 and bool(st.st_mode & stat.S_ISVTX)
            d = f"{p}: owner {st.st_uid}, mode {stat.S_IMODE(st.st_mode):o}"
        except FileNotFoundError:
            ok, d = True, f"{p}: none"
        out.append(Check(f"root:{p}", ok, d, "vy-lockout roots --apply"))
    for loop in SWITCH.get(node, ()):
        if loop.tmux:
            ls = run(["runuser", "-u", "research", "--", "tmux", "has-session", "-t", f"={loop.tmux}"])
            out.append(Check(f"switched:{loop.unit}", ls.returncode != 0, f"tmux {loop.tmux} {'still runs' if ls.returncode == 0 else 'gone'}",
                             f"vy-lockout switch {node} --unit {loop.unit} --apply"))
        if loop.root:
            drop = at(UNITS / f"{unit_file(service_of(loop.unit))}.d" / DROPIN)
            out.append(Check(f"root-unit:{loop.unit}", drop.is_file(), "runs as root" if drop.is_file() else "runs as research",
                             f"vy-lockout root-units {node} --apply"))
    for tool in ("/usr/local/sbin/vy-sudo", "/usr/local/sbin/vy-seal-runs", "/usr/local/sbin/vy-lockout"):
        bad = untrusted([str(at(tool))]) if root == Path("/") else []
        out.append(Check(f"tool:{tool}", at(tool).is_file() and not bad, "; ".join(bad) or "root's", "research deploy install"))
    return out


# ---- steps --------------------------------------------------------------------------------------------------------------------------

class Failed(Exception):
    pass


@dataclass
class Ctx:
    apply: bool
    run: Runner = _run
    say: Callable[[str], None] = print
    log: list[str] = field(default_factory=list)

    def do(self, argv: list[str], what: str = "") -> subprocess.CompletedProcess | None:
        self.say(("+ " if self.apply else "would: ") + (what or shlex.join(argv)))
        if not self.apply:
            return None
        r = self.run(argv)
        if r.returncode != 0:
            raise Failed(f"{shlex.join(argv)}: exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}")
        return r


def visudo_ok(text: str, run: Runner = _run) -> str | None:
    """None when `visudo -cf` accepts `text` as a sudoers file; else its complaint."""
    with tempfile.NamedTemporaryFile("w", suffix=".sudoers", delete=False) as fh:
        fh.write(text)
    try:
        r = run(["visudo", "-cf", fh.name])
        return None if r.returncode == 0 else (r.stderr or r.stdout).strip()
    finally:
        os.unlink(fh.name)


def install_sudoers(ctx: Ctx, name: str, text: str, sudoers_d: Path = SUDOERS_D) -> None:
    if why := visudo_ok(text, ctx.run):
        raise Failed(f"{name}: visudo -cf refuses it: {why}")
    ctx.say(("+ " if ctx.apply else "would: ") + f"install {sudoers_d / name} (0440 root, visudo -cf checked)")
    if ctx.apply:
        tmp = sudoers_d / f".{name}.tmp"
        tmp.write_text(text)
        os.chmod(tmp, 0o440)
        os.replace(tmp, sudoers_d / name)


def step_switch(ctx: Ctx, node: str, units: list[str], back: bool, wait: Callable[[float], None] = time.sleep,
                tries: int = 60) -> None:
    loops = [lp for lp in SWITCH[node] if lp.unit in units]
    if not loops:
        raise Failed(f"no loop {units} on {node}: {', '.join(lp.unit for lp in SWITCH[node])}")
    for lp in reversed(loops) if back else loops:
        as_research = ["runuser", "-u", "research", "--", "tmux"]
        if back:
            ctx.do(["systemctl", "disable", "--now", lp.unit])
            if lp.tmux:
                text = (UNITS / unit_file(service_of(lp.unit))).read_text() if ctx.apply else ""
                cmd = tmux_command(text, service_of(lp.unit)) if text else f"<{lp.unit}'s ExecStart, with its environment files>"
                ctx.do([*as_research, "new-session", "-d", "-s", lp.tmux, cmd], f"tmux new-session -d -s {lp.tmux} '{cmd}' (as research)")
            continue
        if ctx.apply and not (UNITS / unit_file(lp.unit)).is_file():
            raise Failed(f"{UNITS / unit_file(lp.unit)} is not installed: research deploy install {node} first")
        if lp.tmux:
            has = ctx.run([*as_research, "has-session", "-t", f"={lp.tmux}"]) if ctx.apply else None
            if has is None or has.returncode == 0:
                ctx.do([*as_research, "send-keys", "-t", f"={lp.tmux}", "C-c"], f"tmux send-keys -t {lp.tmux} C-c (as research)")
                for _ in range(tries if ctx.apply else 0):
                    r = ctx.run([*as_research, "list-panes", "-t", f"={lp.tmux}", "-F", "#{pane_current_command}"])
                    if r.returncode != 0 or r.stdout.strip() in SHELLS:
                        break
                    wait(2)
                else:
                    if ctx.apply:
                        raise Failed(f"tmux {lp.tmux}: the loop didn't stop on C-c; stop it by hand, then rerun")
                ctx.do([*as_research, "kill-session", "-t", f"={lp.tmux}"], f"tmux kill-session -t {lp.tmux} (as research)")
        ctx.do(["systemctl", "enable", "--now", lp.unit])
        if ctx.apply:
            for _ in range(5):
                if ctx.run(["systemctl", "is-active", "--quiet", lp.unit]).returncode == 0:
                    break
                wait(2)
            else:
                raise Failed(f"{lp.unit} isn't active: journalctl -u {service_of(lp.unit)}; back with --back --unit {lp.unit} --apply")
        ctx.say(f"{lp.unit}: {'active' if ctx.apply else 'would be active'}" + (f", replacing tmux {lp.tmux}" if lp.tmux else ""))


def step_root_units(ctx: Ctx, node: str, back: bool, lib: Path = LIB, units: Path = UNITS, envs: Path = ENVS) -> None:
    loops = [lp for lp in SWITCH[node] if lp.root]
    if not back:
        bad = untrusted([*COMMON_CODE, *sorted({p for lp in loops for p in lp.code + lp.writes})])
        if bad:
            for b in bad:
                ctx.say(f"  not root's alone: {b}")
            raise Failed(f"{len(bad)} paths a root unit would run or write are not root's alone: chown root:root and chmod go-w "
                         "each (or move what writes there), then rerun")
        if ctx.apply and not (envs / "root.env").is_file():
            raise Failed(f"{envs / 'root.env'} is not installed: research deploy install first")
    for lp in loops:
        d = units / f"{unit_file(service_of(lp.unit))}.d"
        if back:
            ctx.do(["rm", "-f", str(d / DROPIN)])
        else:
            ctx.do(["install", "-D", "-m", "0644", "-o", "root", "-g", "root", str(lib / "vy-root.conf"), str(d / DROPIN)])
    ctx.do(["systemctl", "daemon-reload"])
    for lp in loops:
        ctx.do(["systemctl", "try-restart", service_of(lp.unit)])


def step_roots(ctx: Ctx, back: bool, backup: Path = BACKUP) -> None:
    rec = backup / "roots.json"
    if back:
        saved = json.loads(rec.read_text()) if ctx.apply else {p: {"uid": "?", "gid": "?", "mode": "?"} for p in RUNS_ROOTS}
        for p, s in saved.items():
            ctx.do(["chown", f"{s['uid']}:{s['gid']}", p])
            ctx.do(["chmod", str(s["mode"]), p])
        return
    saved = {}
    for p in RUNS_ROOTS:
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        saved[p] = {"uid": st.st_uid, "gid": st.st_gid, "mode": f"{stat.S_IMODE(st.st_mode):o}"}
        ctx.do(["chown", "root:root", p])
        ctx.do(["chmod", "1777", p])
    if ctx.apply and not rec.exists():
        backup.mkdir(mode=0o700, parents=True, exist_ok=True)
        rec.write_text(json.dumps(saved, indent=1) + "\n")


def research_kubeconfig(token: str, ca_b64: str, server: str = "https://127.0.0.1:6443") -> str:
    return ("apiVersion: v1\nkind: Config\nclusters:\n- name: vy-nebius-1\n  cluster:\n"
            f"    server: {server}\n    certificate-authority-data: {ca_b64}\n"
            "users:\n- name: research-view\n  user:\n"
            f"    token: {token}\n"
            "contexts:\n- name: research-view\n  context:\n    cluster: vy-nebius-1\n    user: research-view\n"
            "current-context: research-view\n")


def step_kubeconfig(ctx: Ctx, back: bool, lib: Path = LIB, kube: Path = RESEARCH_KUBE, backup: Path = BACKUP) -> None:
    saved = backup / "kube"
    if back:
        ctx.do(["cp", "-a", f"{saved}/.", str(kube)], f"restore {kube} from {saved}")
        return
    ctx.do(["k3s", "kubectl", "apply", "-f", str(lib / "research-view.yaml")])
    if not ctx.apply:
        ctx.say(f"would: move every file of {kube} to {saved}, and write {kube / 'config'} with research-view's token (0600 research)")
        return
    r = ctx.run(["k3s", "kubectl", "-n", "vy-lockout", "get", "secret", "research-view-token", "-o", "jsonpath={.data.token}"])
    token = base64.b64decode(r.stdout.strip()).decode() if r.returncode == 0 and r.stdout.strip() else ""
    m = re.search(r"certificate-authority-data:\s*(\S+)", Path(K3S_CONFIG).read_text())
    if not token or not m:
        raise Failed("research-view's token or the cluster CA isn't there yet: rerun in a few seconds")
    saved.mkdir(mode=0o700, parents=True, exist_ok=True)
    for p in sorted(kube.glob("*")):
        if p.is_file():
            shutil.move(str(p), saved / p.name)
    cfg = kube / "config"
    fd = os.open(cfg, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(research_kubeconfig(token, m.group(1)))
    shutil.chown(cfg, "research", "research")
    ctx.say(f"+ {cfg}: research-view (read-only); the old files are in {saved}")


def step_harden(ctx: Ctx, back: bool) -> None:
    if back:
        ctx.do(["usermod", "-aG", "lxd", "ubuntu"])
        ctx.do(["setcap", "cap_sys_ptrace+ep", PY_SPY])
        ctx.do(["gpasswd", "-d", "research", "systemd-journal"])
        return
    if "lxd" in groups_of("ubuntu"):
        ctx.do(["gpasswd", "-d", "ubuntu", "lxd"])
    if "cap_sys_ptrace" in ctx.run(["getcap", PY_SPY]).stdout:
        ctx.do(["setcap", "-r", PY_SPY])
    ctx.do(["usermod", "-aG", "systemd-journal", "research"])


def step_sudoers(ctx: Ctx, node: str, skip: list[str], now: float, lib: Path = LIB, sudoers_d: Path = SUDOERS_D,
                 backup: Path = BACKUP, checks: list[Check] | None = None) -> None:
    checks = preflight(node, ctx.run) if checks is None else checks
    failing = [c for c in checks if not c.ok and c.name not in skip]
    for c in checks:
        ctx.say(f"  {'ok  ' if c.ok else 'SKIP' if c.name in skip else 'FAIL'} {c.name}: {c.detail}")
    if failing:
        raise Failed(f"preflight fails: {', '.join(c.name for c in failing)} (each check says its fix; --skip NAME only with Daniel's yes)")
    text = (lib / ALLOWLIST).read_text()
    if text != render():
        raise Failed(f"{lib / ALLOWLIST} isn't `vy-lockout render`'s: install the files of one commit")
    install_sudoers(ctx, ALLOWLIST, text, sudoers_d)
    install_sudoers(ctx, FALLBACK, fallback_text(now + FALLBACK_HOURS * 3600), sudoers_d)
    ctx.do(["install", "-d", "-m", "0700", str(backup / "sudoers")])
    ctx.do(["mv", str(sudoers_d / CLOUD_INIT), str(backup / "sudoers" / CLOUD_INIT)],
           f"move {sudoers_d / CLOUD_INIT} (research's and ubuntu's NOPASSWD: ALL) to {backup / 'sudoers'}")
    if ctx.apply and ctx.run(["visudo", "-c"]).returncode != 0:
        shutil.move(str(backup / "sudoers" / CLOUD_INIT), sudoers_d / CLOUD_INIT)
        raise Failed("visudo -c refuses the whole configuration: cloud-init's file is back; fix, then rerun")
    ctx.do(["systemctl", "enable", "--now", "vy-sudo-fallback-expire.timer"])


def step_rollback(ctx: Ctx, sudoers_d: Path = SUDOERS_D, backup: Path = BACKUP) -> None:
    src = backup / "sudoers" / CLOUD_INIT
    if not ctx.apply or src.exists():
        ctx.do(["cp", "-p", str(src), str(sudoers_d / CLOUD_INIT)], f"restore {sudoers_d / CLOUD_INIT} from {src}")
    ctx.do(["rm", "-f", str(sudoers_d / ALLOWLIST), str(sudoers_d / FALLBACK)])
    ctx.do(["visudo", "-c"])


def fallback(ctx: Ctx, action: str, now: float, hours: float = FALLBACK_HOURS, sudoers_d: Path = SUDOERS_D,
             environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    path = sudoers_d / FALLBACK
    if action == "on":
        if env.get("SUDO_USER") != "vy-admin":
            raise Failed("only vy-admin puts the fallback back (sudo as vy-admin)")
        if not 0 < hours <= FALLBACK_HOURS:
            raise Failed(f"--hours: more than 0, at most {FALLBACK_HOURS}")
        install_sudoers(ctx, FALLBACK, fallback_text(now + hours * 3600), sudoers_d)
        return f"fallback on until {time.strftime(TS, time.gmtime(now + hours * 3600))}"
    if not path.exists():
        return "no fallback"
    if action == "expire":
        try:
            exp = expiry(path.read_text())
        except OSError:
            exp = None
        if exp is not None and exp > now:
            return f"fallback holds until {time.strftime(TS, time.gmtime(exp))}"
    ctx.do(["rm", "-f", str(path)])
    return "fallback removed" + ("" if action == "off" else " (expired)" if exp is not None else " (no readable `# expires` line)")


def plan_text(node: str) -> str:
    loops = SWITCH[node]
    rows = [
        ("deploy", "from an agent VM: `research deploy install NODE --by WHO --apply` (stages /usr/local/lib/vy-lockout, the "
                   "tools and node 1's /etc/vy/env/root.env; installs nothing into sudoers.d or a unit's .d)", "research deploy attic"),
        ("vy-admin", "copy the public keys to the node, then `sudo vy-admin-setup KEYS`; log in as vy-admin, `sudo -n true`; every "
                     "step below runs as vy-admin", "userdel -r vy-admin; rm /etc/sudoers.d/10-vy-admin"),
        ("seal", "`sudo vy-seal-runs seal --dry-run`, read it, then `sudo systemctl enable --now vy-seal-runs.timer`; "
                 "`sudo vy-lockout roots --apply`", "systemctl disable --now vy-seal-runs.timer; vy-lockout roots --back --apply "
                 "(sealed dirs stay sealed: /var/lib/vy-seal-runs/sealed.jsonl has each one's former owner and mode)"),
        ("switch", "one loop at a time, in this order: " + ", ".join(f"{lp.unit}" + (f" (tmux {lp.tmux})" if lp.tmux else "")
                                                               for lp in loops) + f"; `vy-lockout switch {node} --unit U --apply`",
         f"vy-lockout switch {node} --unit U --back --apply, newest first"),
        ("root-units", f"`vy-lockout root-units {node}` lists what isn't root's alone; fix each, then --apply",
         f"vy-lockout root-units {node} --back --apply"),
        ("kubeconfig", "`vy-lockout kubeconfig --apply`; then `k3s certificate rotate` in a quiet window (copies of the old admin "
                       "key elsewhere stop working)", "vy-lockout kubeconfig --back --apply"),
        ("harden", "`vy-lockout harden --apply`", "vy-lockout harden --back --apply"),
        ("sudoers", f"`vy-lockout preflight {node}` all ok, then `vy-lockout sudoers --node {node} --apply`: allowlist + 24 h "
                    "fallback, cloud-init's grant moved aside. From here `research deploy install` (sudo python3) needs vy-admin's "
                    "login", "vy-lockout rollback --apply"),
        ("watch", "during the 24 h: `vy-lockout audit --since -24h`; fix each refused form (call site or allowlist, by PR) before the "
                  "fallback expires; vy-admin may `vy-lockout fallback on`", "vy-lockout fallback on (vy-admin)"),
    ]
    return "\n".join(f"{i}. {name}: {do}\n   rollback: {back}" for i, (name, do, back) in enumerate(rows, 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vy-lockout", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("node", choices=sorted(SWITCH))
        p.add_argument("--json", action="store_true")
    p = sub.add_parser("switch")
    p.add_argument("node", choices=sorted(SWITCH))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--unit", action="append")
    g.add_argument("--all", action="store_true")
    p.add_argument("--back", action="store_true")
    p.add_argument("--apply", action="store_true")
    p = sub.add_parser("root-units")
    p.add_argument("node", choices=sorted(SWITCH))
    for name in ("roots", "kubeconfig", "harden"):
        sub.add_parser(name)
    for name in ("root-units", "roots", "kubeconfig", "harden"):
        sp = sub.choices[name]
        sp.add_argument("--back", action="store_true")
        sp.add_argument("--apply", action="store_true")
    p = sub.add_parser("sudoers")
    p.add_argument("--node", default=os.environ.get("VY_NODE", ""))
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--apply", action="store_true")
    p = sub.add_parser("fallback")
    p.add_argument("action", choices=("on", "off", "expire"))
    p.add_argument("--hours", type=float, default=FALLBACK_HOURS)
    p = sub.add_parser("audit")
    p.add_argument("--since", default="-24h")
    p = sub.add_parser("rollback")
    p.add_argument("--apply", action="store_true")
    sub.add_parser("render")
    ns = ap.parse_args(argv)
    if ns.cmd == "render":
        sys.stdout.write(render())
        return 0
    if ns.cmd == "plan":
        print(plan_text(ns.node))
        return 0
    if ns.cmd == "audit":
        r = _run(["journalctl", "SYSLOG_IDENTIFIER=sudo", "--since", ns.since, "-o", "cat"])
        rows = audit(r.stdout)
        for n, k in rows:
            print(f"{n:8d}  {k}")
        print(f"vy-lockout audit: {sum(n for n, _ in rows)} sudo calls since {ns.since} the allowlist would refuse, in {len(rows)} forms")
        return 1 if rows else 0
    if os.geteuid() != 0:
        print("vy-lockout: run as root (vy-admin, with sudo)", file=sys.stderr)
        return REFUSED
    if ns.cmd == "preflight":
        checks = preflight(ns.node)
        if ns.json:
            print(json.dumps([c.__dict__ for c in checks], indent=1))
        for c in [] if ns.json else checks:
            print(f"{'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}" + ("" if c.ok else f"\n     fix: {c.fix}"))
        return 0 if all(c.ok for c in checks) else 1
    ctx = Ctx(apply=getattr(ns, "apply", False))
    try:
        if ns.cmd == "switch":
            step_switch(ctx, ns.node, [lp.unit for lp in SWITCH[ns.node]] if ns.all else ns.unit, ns.back)
        elif ns.cmd == "root-units":
            step_root_units(ctx, ns.node, ns.back)
        elif ns.cmd == "roots":
            step_roots(ctx, ns.back)
        elif ns.cmd == "kubeconfig":
            step_kubeconfig(ctx, ns.back)
        elif ns.cmd == "harden":
            step_harden(ctx, ns.back)
        elif ns.cmd == "sudoers":
            if ns.node not in SWITCH:
                raise Failed(f"--node: one of {', '.join(SWITCH)}")
            step_sudoers(ctx, ns.node, ns.skip, time.time())
        elif ns.cmd == "rollback":
            step_rollback(ctx)
        elif ns.cmd == "fallback":
            ctx.apply = True
            print(fallback(ctx, ns.action, time.time(), ns.hours))
    except Failed as e:
        print(f"vy-lockout {ns.cmd}: {e}", file=sys.stderr)
        return 1
    if not ctx.apply:
        print("(nothing changed: --apply does it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
