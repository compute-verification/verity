"""A run's workload as an unprivileged job user (`research run --job-user USER`), opt-in.

On a shared node the runner and its workload run as the operators' account, which has sudo, a user systemd, tmux sessions, ssh
keys and a kubeconfig, so anything the workload runs (a build dependency included) can act as the operators.  With --job-user
the runner keeps its records as itself, and the harness spawns, in the workload's place (inside a queued GPU phase's lease),

    research job-exec --user USER --unit UNIT [--pass NAME]... -- CMD...

which runs CMD as USER in a transient system service (`sudo -n systemd-run --uid=USER --wait --pipe`, unit_argv): no new
privileges and no capabilities; the host read-only but for its run dir, and no other run's dir, no launch request, no /home or
/root; its own /tmp, /dev/shm and IPC; an empty /run but for DNS and the GPU lease dirs; /etc/verity and the k3s state hidden;
link-local addresses (the cloud metadata service) unreachable; the CPUs, memory cap and nice it was started with; and only the
variables job_env passes, from a file only root reads (no value travels on a command line).  SIGTERM, SIGINT or SIGHUP stops
the unit (SIGTERM, then SIGKILL after TIMEOUT_STOP_S) and exits 128+signal.

The runner opens its run dir and its out/ to the job (prepare: sticky and world-writable, so the kernel's protected_symlinks and
protected_regular keep it from following or opening what the job leaves at the top) and, once the harness returns and before
anything reads the run dir, finish() waits for the unit to be gone, deletes as root every entry the job left that is neither a
file nor a directory (a symlink, a FIFO, a socket), gives the rest to the runner's user and restores the mode.  admit refuses a
job user that is root, is the runner, or is in a privileged group or the runner's own; nothing falls back to running the
workload as the runner.
"""
from __future__ import annotations

import argparse
import grp
import os
import pwd
import re
import signal
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .telemetry.run import OUT_DIR

SUDO = ("sudo", "-n")
UNIT_PREFIX = "research-job-"
ENV_FILE = Path("control") / "job.env"          # run-dir relative, in the runner's control/ (the job can't write there)
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LEASE_DIRS = ("/run/gpu-lease", "/run/gpu-lease-circuits")      # gpu-lease's locks, when the workload takes its own lease
HIDDEN = ("/etc/verity", "/etc/rancher", "/var/lib/rancher", "/var/lib/kubelet")
#: where a run dir is out of the unit's reach (ProtectHome, PrivateTmp, the emptied /run)
UNREACHABLE = ("/home", "/root", "/run", "/tmp", "/var/tmp", "/dev/shm")
TIMEOUT_STOP_S = 10                 # the harness's own --grace: the SIGTERM-to-SIGKILL window a workload has without --job-user
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
FORBIDDEN_GROUPS = frozenset({"root", "sudo", "admin", "wheel", "adm", "docker", "lxd", "systemd-journal"})
#: what the workload gets beside --pass: the run's identity and its OUT, the tree's import path, the GPUs its lease gave it
PASS_NAMES = frozenset({"OUT", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LANG", "LC_ALL", "TZ"})
PASS_PREFIXES = ("RESEARCH_", "GPU_LEASE_")
#: a name a credential would travel in: passed only when the run's own --env names it
CREDENTIAL = re.compile(r"TOKEN|SECRET|PASSW|KEY|CREDENTIAL|^AWS_|^R2_|^GH_|^GITHUB_|^HF_|^SSH_|SLACK|NOTION|RUNPOD|NEBIUS|CURSOR",
                        re.IGNORECASE)
SAFE_PATH = re.compile(r"/[A-Za-z0-9._/@+-]*")                    # a path property splits on spaces and expands % specifiers
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
EX_CONFIG = 78


class Refused(Exception):
    """Nothing runs: the workload never falls back to the runner's own user."""


def log(msg: str) -> None:
    print(f"research job-exec: {msg}", file=sys.stderr, flush=True)


def check_user(name: str, *, getpwnam=pwd.getpwnam, getgrall=grp.getgrall, uid: int | None = None,
               gid: int | None = None) -> pwd.struct_passwd:
    """The job user's passwd entry; Refused when it's missing, root, the runner itself, or in a group that is privileged or the
    runner's primary group (which owns the runner's trees, its store and the machine's git repository)."""
    uid = os.getuid() if uid is None else uid
    gid = os.getgid() if gid is None else gid
    try:
        pw = getpwnam(name)
    except KeyError:
        raise Refused(f"--job-user {name}: no such user here (sudo useradd --system --no-create-home --shell /usr/sbin/nologin "
                      f"{name})") from None
    if pw.pw_uid in (0, uid):
        raise Refused(f"--job-user {name}: uid {pw.pw_uid} is {'root' if pw.pw_uid == 0 else 'the runner'}")
    groups = getgrall()
    names = {g.gr_gid: g.gr_name for g in groups}
    gids = {pw.pw_gid} | {g.gr_gid for g in groups if name in g.gr_mem}
    bad = sorted(names.get(g, str(g)) for g in gids if g in (0, gid) or names.get(g) in FORBIDDEN_GROUPS)
    if bad:
        raise Refused(f"--job-user {name}: in group(s) {', '.join(bad)}")
    return pw


def unit_name(run_id: str) -> str:
    return UNIT_PREFIX + run_id


def check_run_dir(run_dir: str | Path, run_id: str) -> str:
    """The run dir's real path, which the unit's properties carry; Refused when they can't carry it or the unit can't reach it."""
    real = os.path.realpath(run_dir)
    if not SAFE_PATH.fullmatch(real) or not SAFE_NAME.fullmatch(run_id):
        raise Refused(f"run dir {real!r} or run id {run_id!r} has a character a unit's property can't carry")
    if any(real == p or real.startswith(p + "/") for p in UNREACHABLE):
        raise Refused(f"run dir {real} is under {'/'.join(real.split('/')[:2])}, which the job's unit doesn't see; use a runs root "
                      f"elsewhere (--runs-dir)")
    return real


def spawn_argv(command: list[str], user: str, unit: str, passed: list[str] | tuple[str, ...] = ()) -> list[str]:
    """What the harness spawns in the workload's place (`research job-exec`, run by this interpreter, so its package is this one)."""
    return [sys.executable, "-m", "research", "job-exec", "--user", user, "--unit", unit,
            *(x for k in sorted(passed) for x in ("--pass", k)), "--", *command]


def job_env(environ: Mapping[str, str], passed: list[str] | tuple[str, ...] = ()) -> dict[str, str]:
    """The workload's whole environment: the --pass names, PASS_NAMES and PASS_PREFIXES (never a credential-shaped name unless
    passed), and a system PATH, a HOME in the unit's own /tmp and a UTF-8 locale; nothing else of the runner's."""
    keep = {k: v for k, v in environ.items()
            if k in passed or ((k in PASS_NAMES or k.startswith(PASS_PREFIXES)) and not CREDENTIAL.search(k))}
    return {"PATH": SYSTEM_PATH, "HOME": "/tmp", "LANG": "C.UTF-8", **keep}


def env_file_text(env: Mapping[str, str]) -> str:
    """`NAME="VALUE"` lines in systemd's EnvironmentFile= syntax (a double-quoted value keeps everything but its escapes)."""
    def q(v: str) -> str:
        return '"' + re.sub(r'(["\\`$])', r"\\\1", v) + '"'

    return "".join(f"{k}={q(v)}\n" for k, v in sorted(env.items()))


def allowed_cpus() -> str | None:
    """This process's CPU affinity as a cpu list (`0-31,64-95`): the CPUs the queue, taskset or the machine gave the run."""
    cpus = sorted(os.sched_getaffinity(0))
    out, start = [], None
    for i, c in enumerate(cpus):
        start = c if start is None else start
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            out.append(f"{start}-{c}" if c != start else str(c))
            start = None
    return ",".join(out) or None


def memory_max(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> int | None:
    """The tightest memory.max on this process's cgroup and its ancestors (the queue's scope, gpu-lease's), in bytes; None if none."""
    try:
        path = next(ln.split(":", 2)[2].strip() for ln in Path(proc_cgroup).read_text().splitlines() if ln.startswith("0::"))
    except (OSError, StopIteration):
        return None
    caps = []
    p = Path(path)
    while True:
        try:
            v = (Path(root) / p.relative_to("/") / "memory.max").read_text().strip()
        except (OSError, ValueError):
            v = "max"
        if v.isdigit():
            caps.append(int(v))
        if p == p.parent:
            break
        p = p.parent
    return min(caps) if caps else None


@dataclass
class Unit:
    name: str
    user: str
    run_dir: str
    cwd: str
    env_file: str
    cpus: str | None = None
    memory: int | None = None
    nice: int = 0


def unit_argv(u: Unit) -> list[str]:
    """`sudo -n systemd-run` for the unit, up to the `--` its command follows."""
    runs_root = os.path.dirname(u.run_dir)
    props = [
        "NoNewPrivileges=yes", "RestrictSUIDSGID=yes", "CapabilityBoundingSet=", "AmbientCapabilities=",
        "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes", "PrivateIPC=yes", "ProtectProc=invisible",
        "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectKernelLogs=yes", "ProtectControlGroups=yes",
        "ProtectClock=yes", "ProtectHostname=yes", "RestrictNamespaces=yes", "LockPersonality=yes", "RestrictRealtime=yes",
        "SystemCallArchitectures=native", "KeyringMode=private", "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "IPAddressDeny=link-local", "UMask=0022",
        # /run holds sockets that act for any caller; DNS (resolv.conf points into it) and gpu-lease's dirs come back
        "TemporaryFileSystem=/run:ro", "TemporaryFileSystem=/dev/shm:mode=1777,nosuid,nodev",
        "BindReadOnlyPaths=-/run/systemd/resolve", *(f"BindPaths=-{d}" for d in LEASE_DIRS),
        *(f"InaccessiblePaths=-{p}" for p in HIDDEN),
        # every other run's records and every launch request (its --env values) out of sight; this run's dir back, writable
        f"TemporaryFileSystem={runs_root}:ro", f"InaccessiblePaths=-{os.path.dirname(runs_root)}/requests",
        f"BindPaths={u.run_dir}", f"ReadWritePaths={u.run_dir}", f"EnvironmentFile={u.env_file}",
        "KillMode=control-group", "SendSIGKILL=yes", f"TimeoutStopSec={TIMEOUT_STOP_S}",
        *([f"AllowedCPUs={u.cpus}"] if u.cpus else []),
        *([f"MemoryMax={u.memory}"] if u.memory else []),
        *([f"Nice={u.nice}"] if u.nice else []),
    ]
    return [*SUDO, "systemd-run", f"--unit={u.name}", f"--uid={u.user}", "--wait", "--pipe", "--collect", "--quiet",
            "--service-type=exec", "--expand-environment=no", f"--working-directory={u.cwd}",
            *(x for p in props for x in ("-p", p)), "--"]


def supervise(argv: list[str], unit: str, popen=subprocess.Popen, run=subprocess.run) -> tuple[int, int | None]:
    """Run `systemd-run` and wait; on SIGTERM, SIGINT or SIGHUP stop the unit.  Returns its exit status and the signal, if any.
    sudo runs in its own session, so a signal to this process group reaches this process alone."""
    got: list[int] = []

    def stop(signum, _frame):
        if not got:
            run([*SUDO, "systemctl", "stop", "--no-block", unit], capture_output=True)
        got.append(signum)

    old = {s: signal.signal(s, stop) for s in STOP_SIGNALS}
    try:
        rc = popen(argv, start_new_session=True).wait()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    if got or rc < 0:
        run([*SUDO, "systemctl", "stop", unit], capture_output=True)       # blocks until the unit is gone
    return rc, (got[0] if got else None)


def prepare(run_dir: str) -> int:
    """Runner side, before the harness: the runner's control/ in place, then the run dir and its out/ ($OUT) opened to the job.
    Returns the run dir's mode."""
    st = os.lstat(run_dir)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise Refused(f"{run_dir} is not a directory this user owns")
    os.makedirs(os.path.join(run_dir, "control"), exist_ok=True)    # the harness's timeout control and the env file
    os.makedirs(os.path.join(run_dir, OUT_DIR), exist_ok=True)
    for d in (run_dir, os.path.join(run_dir, OUT_DIR)):
        os.chmod(d, 0o1777)
    return stat.S_IMODE(st.st_mode)


def finish(run_dir: str, unit: str, user: str, uid: int, mode: int, run=subprocess.run) -> tuple[list[str], list[str]]:
    """Runner side, once the harness returns and before anything reads the run dir: wait for the unit to be gone (however the
    workload ended, job-exec killed included), then as root delete what the job left that is neither a file nor a directory, give
    the rest to this user, and give the run dir and its out/ that mode back.  Returns what was deleted and what the job user still
    owns."""
    run([*SUDO, "systemctl", "stop", unit], capture_output=True)            # waits out a stop in progress; a no-op once gone
    (Path(run_dir) / ENV_FILE).unlink(missing_ok=True)
    r = run([*SUDO, "find", "-P", run_dir, "-mindepth", "1", "-user", user, "!", "-type", "d", "!", "-type", "f", "-print",
             "-delete"], capture_output=True, text=True)
    removed = [os.path.relpath(p, run_dir) for p in (r.stdout or "").splitlines() if p]
    run([*SUDO, "chown", "-R", "-P", f"--from={user}", f"{os.getuid()}:{os.getgid()}", "--", run_dir], capture_output=True)
    for d in (os.path.join(run_dir, OUT_DIR), run_dir):
        os.chmod(d, mode)
    still = [os.path.relpath(os.path.join(base, n), run_dir) for base, dirs, files in os.walk(run_dir) for n in dirs + files
             if os.lstat(os.path.join(base, n)).st_uid == uid]
    return removed, still


def main(argv: list[str], *, popen=subprocess.Popen, run=subprocess.run, environ: Mapping[str, str] | None = None,
         check=check_user) -> int:
    ap = argparse.ArgumentParser(prog="research job-exec")
    ap.add_argument("--user", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--pass", dest="passed", action="append", default=[], metavar="NAME", help="a variable of the run's own --env")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    cmd = ns.cmd[1:] if ns.cmd[:1] == ["--"] else ns.cmd
    environ = os.environ if environ is None else environ
    run_dir = environ.get("RESEARCH_RUN_DIR", "")
    if not cmd or not run_dir:
        log("give the command after -- (the harness sets RESEARCH_RUN_DIR)")
        return 2
    cwd = os.getcwd()
    try:
        check(ns.user)
        st = os.lstat(run_dir)
        if not all(SAFE_PATH.fullmatch(p) for p in (run_dir, cwd)) or not SAFE_NAME.fullmatch(ns.unit):
            raise Refused(f"run dir {run_dir!r}, cwd {cwd!r} or unit {ns.unit!r} has a character a unit's property can't carry")
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise Refused(f"{run_dir} is not a directory this user owns")
    except (Refused, OSError) as e:
        log(f"refused: {e}; nothing ran")
        return EX_CONFIG
    env_file = os.path.join(run_dir, ENV_FILE)
    os.makedirs(os.path.dirname(env_file), exist_ok=True)
    fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)   # root's service manager reads it
    with os.fdopen(fd, "w") as f:
        f.write(env_file_text(job_env(environ, ns.passed)))
    unit = Unit(name=ns.unit, user=ns.user, run_dir=run_dir, cwd=cwd, env_file=env_file, cpus=allowed_cpus(), memory=memory_max(),
                nice=os.nice(0))
    try:
        rc, sig = supervise([*unit_argv(unit), *cmd], unit.name, popen=popen, run=run)
    finally:
        os.unlink(env_file)
    if sig:
        log(f"stopped {unit.name} on signal {sig}")
        return 128 + sig
    return rc


__all__ = ["Refused", "Unit", "check_run_dir", "check_user", "finish", "job_env", "main", "prepare", "spawn_argv", "supervise",
           "unit_argv", "unit_name"]
