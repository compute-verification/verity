#!/usr/bin/python3 -I
"""vy-sudo: the narrow forms of root that `research` keeps after the lock-out (root, stdlib only; sudoers names each form).

    sudo -n vy-sudo renice -n PRIO -p PID...                  only the caller's own processes (threads too)
    sudo -n vy-sudo taskset [-a] -p -c CPUS PID               also -pc / -apc; only the caller's own process
    sudo -n vy-sudo [env VAR=VAL...] systemd-run ARGS -- CMD  a slot's scope (--scope), or a job user's unit (--uid=USER --wait)
    sudo -n vy-sudo systemctl stop|kill|freeze|thaw [--signal=SIG] NAME.scope    a scope whose processes are all the caller's
    sudo -n vy-sudo systemctl stop [--no-block] research-job-RUN  a job's unit (`research run --job-user`), which never runs as root
    sudo -n vy-sudo job-reclaim RUN_DIR USER                  what job user USER left in the caller's run dir, back to the caller
    sudo -n vy-sudo hf-ns [VAR=VAL...] -- SCRIPT ARGS...     SCRIPT as the caller, with /workspace/jobs/hf bound on /workspace/hf
    sudo -n vy-sudo du-bundles                                the bytes of each replay bundle under BUNDLE_ROOTS (`du -sb` lines)

sudoers can't check arguments (a `*` there matches spaces, so `cmd x *` lets through any further options), so the allowlist
(/etc/sudoers.d/50-vy-research, vy-lockout) names this helper per form and the checks are here. The caller is sudo's
SUDO_UID / SUDO_GID; run without sudo, or by root, it refuses. Each form is parsed into the exact argv that runs, and anything
not listed is refused (exit 3) with why, before anything runs:

  systemd-run runs the command as the caller (`--uid`/`--gid` are added when absent) or, for a job's unit, as a job user: an
  account root has listed in the group JOB_GROUP (vy-job), that is also not root, not vy-admin and in none of
  PRIVILEGED_GROUPS. Any other account is refused, a daemon's (polkitd, messagebus) among them: some are root by another
  name. Root (vy-admin, after the lock-out) lists an account made for jobs and nothing else:
  `sudo groupadd -f --system vy-job && sudo usermod -aG vy-job USER`. Only the options and unit properties listed in
  SCOPE_OPTIONS, UNIT_OPTIONS and PROPERTIES pass: a property such as ExecStartPre=+..., User= or a foreign EnvironmentFile=
  would act as root.
  An EnvironmentFile= must be a regular file the caller owns. A leading `setpriv --reuid=UID --regid=GID --init-groups --`
  that drops to the caller is removed, since `--uid`/`--gid` do the same (systemd-run calls initgroups).
  `env` takes only the variables ENV_OK matches (PATH, HOME, CUDA_*, ...): never LD_*, PYTHON* or BASH_ENV, which act on root.
  job-reclaim runs here, not in a command of its own: under a directory the caller owns it gives the caller each file and
  directory USER owns and removes what USER left that is neither (a symlink, a FIFO, a socket), printing each removed path. It
  changes the owner only of an inode it holds open and has checked, and works only inside directories the caller or USER owns,
  so it does nothing the caller couldn't do as itself or as USER, whom a job's unit already lets it act as.
"""
from __future__ import annotations

import grp
import os
import pwd
import re
import stat
import sys
from pathlib import Path

REFUSED = 3
PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")
SYSTEMCTL = "/usr/bin/systemctl"
PRIVILEGED_GROUPS = frozenset({"root", "sudo", "admin", "wheel", "adm", "docker", "lxd", "systemd-journal", "vy-admin", "disk", "shadow"})
NO_JOB_USERS = frozenset({"root", "vy-admin"})
#: the job users' group. An account is a job user only once root lists it here, so the accounts a caller may act as are the ones
#: `getent group vy-job` prints and no other
JOB_GROUP = "vy-job"
ENV_OK = re.compile(r"^(PATH|HOME|USER|LOGNAME|LANG|LC_[A-Z]+|TZ|CUDA_[A-Z_]+|NVIDIA_[A-Z_]+|OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|"
                    r"MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|FILL[A-Z_]*|GPU_LEASE_[A-Z_]+|VY_[A-Z0-9_]+)$")
CPUS = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")
UNIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,200}$")
SLICE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.slice$")
FORBIDDEN_SLICES = frozenset({"-.slice", "system.slice", "init.scope", "machine.slice"})
SIGNALS = frozenset({"SIGTERM", "SIGKILL", "SIGINT", "SIGHUP", "SIGSTOP", "SIGCONT", "SIGUSR1", "SIGUSR2", "TERM", "KILL", "INT", "HUP",
                     "STOP", "CONT", "USR1", "USR2"})
SCOPE_OPTIONS = frozenset({"--scope", "--quiet", "--collect"})
UNIT_OPTIONS = frozenset({"--wait", "--pipe", "--quiet", "--collect", "--service-type=exec", "--expand-environment=no"})
# unit properties a caller may set: resource limits and sandboxing that only narrow what the command can do
PROPERTIES = frozenset({
    "AllowedCPUs", "MemoryMax", "MemoryHigh", "MemorySwapMax", "CPUWeight", "CPUQuota", "IOWeight", "TasksMax", "Nice",
    "NoNewPrivileges", "RestrictSUIDSGID", "PrivateTmp", "PrivateIPC", "ProtectProc", "ProtectSystem", "ProtectHome",
    "ProtectKernelTunables", "ProtectKernelModules", "ProtectKernelLogs", "ProtectControlGroups", "ProtectClock", "ProtectHostname",
    "RestrictNamespaces", "LockPersonality", "RestrictRealtime", "SystemCallArchitectures", "KeyringMode", "RestrictAddressFamilies",
    "IPAddressDeny", "UMask", "TemporaryFileSystem", "BindReadOnlyPaths", "BindPaths", "InaccessiblePaths", "ReadWritePaths",
    "KillMode", "SendSIGKILL", "TimeoutStopSec", "EnvironmentFile",
})
EMPTY_ONLY = frozenset({"CapabilityBoundingSet", "AmbientCapabilities"})
#: the namespace types RestrictNamespaces= may name here: only `yes` (make none) or a space-separated list out of these narrows the
#: unit; `no`, a `~` list or any other value widens it. jobuser.check_namespaces (NAMESPACES) holds the same set for the launcher's
#: side; the two agree by test_nebius_lockout, since this file is installed on a node as a lone `python3 -I` script and cannot import it
NAMESPACES = frozenset({"user", "mnt", "net"})
HF_BIND = ("/workspace/jobs/hf", "/workspace/hf")
BUNDLE_ROOTS = ("/workspace/jobs/cov", "/workspace/jobs/probe-jit")
#: every form, as the usage line lists them; jobuser.py asks for root through this helper once it reads "job-reclaim" here
FORMS = ("renice", "taskset", "systemd-run", "env", "systemctl", "hf-ns", "du-bundles", "job-reclaim")
JOB_UNIT = re.compile(r"^research-job-[A-Za-z0-9._-]{1,180}$")      # jobuser.py's unit_name: research-job-RUN


class Refused(Exception):
    pass


def caller(environ: dict[str, str]) -> tuple[int, int]:
    try:
        uid, gid = int(environ["SUDO_UID"]), int(environ["SUDO_GID"])
    except (KeyError, ValueError):
        raise Refused("run through sudo: the caller is SUDO_UID") from None
    if uid == 0:
        raise Refused("root needs no vy-sudo")
    return uid, gid


def owner(pid: str, proc: Path = PROC) -> int:
    if not pid.isdigit():
        raise Refused(f"{pid!r} is not a pid")
    try:
        return (proc / pid).stat().st_uid
    except OSError:
        raise Refused(f"no process {pid}") from None


def own(pids: list[str], uid: int, proc: Path = PROC) -> None:
    if not pids:
        raise Refused("no pid")
    for p in pids:
        if owner(p, proc) != uid:
            raise Refused(f"process {p} is not the caller's (uid {uid})")


def renice(args: list[str], uid: int, proc: Path = PROC) -> list[str]:
    if len(args) < 4 or args[0] != "-n" or args[2] != "-p":
        raise Refused("renice -n PRIO -p PID...")
    try:
        prio = int(args[1])
    except ValueError:
        raise Refused(f"priority {args[1]!r}") from None
    if not -20 <= prio <= 19:
        raise Refused(f"priority {prio} outside -20..19")
    own(args[3:], uid, proc)
    return ["/usr/bin/renice", "-n", str(prio), "-p", *args[3:]]


def taskset(args: list[str], uid: int, proc: Path = PROC) -> list[str]:
    long = {"--all-tasks": "a", "--pid": "p", "--cpu-list": "c"}
    flags, rest = set(), list(args)
    while rest and rest[0].startswith("-"):
        f = rest.pop(0)
        if f in long:
            flags.add(long[f])
        elif re.fullmatch(r"-[apc]+", f):
            flags |= set(f[1:])
        else:
            raise Refused(f"taskset option {f}")
    if not {"p", "c"} <= flags or len(rest) != 2 or not CPUS.match(rest[0]):
        raise Refused("taskset [-a] -p -c CPUS PID")
    own([rest[1]], uid, proc)
    return ["/usr/bin/taskset", *(["-a"] if "a" in flags else []), "-p", "-c", rest[0], rest[1]]


def env_vars(pairs: list[str]) -> dict[str, str]:
    out = {}
    for kv in pairs:
        k, eq, v = kv.partition("=")
        if not eq or not ENV_OK.match(k):
            raise Refused(f"environment variable {k!r} is not one vy-sudo passes")
        out[k] = v
    return out


def job_user(name: str) -> pwd.struct_passwd:
    """The passwd entry of the job user `name` (a name or a uid); Refused for any account that isn't one. An allow list: the
    account's name is in JOB_GROUP's member list, and it is still not root, not one of NO_JOB_USERS and in none of
    PRIVILEGED_GROUPS. Only the member list counts, not a primary group: it is the one line root writes to grant it
    (`usermod -aG`) and reads to see every job user, an account's gid can come to equal the group's with nobody granting it,
    and as a primary group it would be the group of every file a job writes."""
    try:
        pw = pwd.getpwnam(name) if not name.isdigit() else pwd.getpwuid(int(name))
    except KeyError:
        raise Refused(f"no user {name}") from None
    if pw.pw_uid == 0 or pw.pw_name in NO_JOB_USERS:
        raise Refused(f"{pw.pw_name} can't be a job user")
    try:
        primary = grp.getgrgid(pw.pw_gid).gr_name
    except KeyError:
        raise Refused(f"{pw.pw_name}'s group {pw.pw_gid} has no entry: not a job user") from None
    listed = {g.gr_name for g in grp.getgrall() if pw.pw_name in g.gr_mem}
    if bad := (listed | {primary}) & PRIVILEGED_GROUPS:
        raise Refused(f"{pw.pw_name} is in {', '.join(sorted(bad))}: not a job user")
    if JOB_GROUP not in listed:
        only = f" ({JOB_GROUP} as its primary group doesn't count)" if primary == JOB_GROUP else ""
        raise Refused(f"{pw.pw_name} is not a job user: it is not listed in group {JOB_GROUP}{only}. If the account was made to run "
                      f"jobs and nothing else, root (vy-admin, after the lock-out) lists it: "
                      f"sudo groupadd -f --system {JOB_GROUP} && sudo usermod -aG {JOB_GROUP} {pw.pw_name}")
    return pw


def _restrict_namespaces(value: str) -> None:
    """Refuse a RestrictNamespaces= that would widen the unit. systemd reads this property as a boolean or a whitespace-separated
    list of namespace types: `yes` makes none (the launcher's default), a list of NAMESPACES makes only those, but `no`/`false`/`0`
    make every type, and a `~`-prefixed list all but those. The launcher (jobuser.check_namespaces) only ever sends `yes` or a
    sorted subset of NAMESPACES, so hold a direct caller to the same: `yes`, or a non-empty list naming NAMESPACES only. A property
    repeated with different values is checked here on each occurrence, so a widening one is caught even when systemd would take the
    last."""
    types = value.split()
    if value == "yes" or (types and all(t in NAMESPACES for t in types)):
        return
    raise Refused(f"RestrictNamespaces={value!r} would widen the unit: only `yes` (make no namespace) or a space-separated list of "
                  f"{', '.join(sorted(NAMESPACES))} is allowed (systemd reads anything else, e.g. no or a ~list, as every namespace "
                  "type or all but some)")


def _strip_setpriv(cmd: list[str], uid: int, gid: int) -> list[str]:
    drop = ["setpriv", f"--reuid={uid}", f"--regid={gid}", "--init-groups", "--"]
    return cmd[len(drop):] if cmd[:len(drop)] == drop else cmd


def systemd_run(args: list[str], uid: int, gid: int, user=job_user) -> list[str]:
    if "--" not in args:
        raise Refused("systemd-run ARGS -- CMD...")
    i = args.index("--")
    opts, cmd = args[:i], _strip_setpriv(args[i + 1:], uid, gid)
    if not cmd:
        raise Refused("no command")
    scope, out, it, as_uid, as_gid = "--scope" in opts, [], iter(opts), None, None
    allowed = SCOPE_OPTIONS if scope else UNIT_OPTIONS
    for o in it:
        if o in ("-p", "--property"):
            o = "--property=" + next(it, "")
        if o.startswith("--property="):
            k, eq, v = o.split("=", 1)[1].partition("=")
            if not eq or not (k in PROPERTIES or (k in EMPTY_ONLY and v == "")):
                raise Refused(f"unit property {k!r} is not one vy-sudo passes")
            if k == "EnvironmentFile":
                p = Path(v.lstrip("-"))
                try:
                    st = p.lstat()
                except OSError:
                    raise Refused(f"EnvironmentFile {p}: no such file") from None
                if not p.is_file() or p.is_symlink() or st.st_uid != uid:
                    raise Refused(f"EnvironmentFile {p} is not a file of the caller's")
            elif k == "RestrictNamespaces":
                _restrict_namespaces(v)
            out += ["-p", f"{k}={v}"]
        elif o == "--unit":
            out.append(f"--unit={_unit(next(it, ''))}")
        elif o.startswith("--unit="):
            out.append(f"--unit={_unit(o.split('=', 1)[1])}")
        elif o.startswith("--slice="):
            s = o.split("=", 1)[1]
            if not SLICE.match(s) or s in FORBIDDEN_SLICES:
                raise Refused(f"slice {s!r}")
            out.append(o)
        elif o.startswith("--nice="):
            n = int(o.split("=", 1)[1]) if re.fullmatch(r"--nice=-?\d+", o) else 99
            if not -20 <= n <= 19:
                raise Refused(f"{o}")
            out.append(o)
        elif o.startswith("--uid="):
            as_uid = o.split("=", 1)[1]
        elif o.startswith("--gid="):
            as_gid = o.split("=", 1)[1]
        elif o.startswith("--working-directory=") and not scope:
            out.append(o)
        elif o in allowed:
            out.append(o)
        else:
            raise Refused(f"systemd-run option {o!r} is not one vy-sudo passes")
    if scope:
        if as_uid not in (None, str(uid)) or as_gid not in (None, str(gid)):
            raise Refused(f"a scope runs as the caller (uid {uid}, gid {gid})")
        out += [f"--uid={uid}", f"--gid={gid}"]
    else:
        if as_uid is None:
            raise Refused("a unit runs as a job user: --uid=USER")
        pw = user(as_uid) if as_uid != str(uid) else None
        if as_gid is not None and as_gid not in (str(gid), str(pw.pw_gid) if pw else str(gid)):
            raise Refused(f"--gid={as_gid}")
        out.append(f"--uid={as_uid}")
        if as_gid is not None:
            out.append(f"--gid={as_gid}")
    return ["/usr/bin/systemd-run", *out, "--", *cmd]


def _unit(name: str) -> str:
    if not UNIT.match(name):
        raise Refused(f"unit name {name!r}")
    return name


def scope_pids(unit: str, show=None, cgroup: Path = CGROUP) -> list[str]:
    """Every pid in the scope's cgroup and the cgroups beneath it."""
    import subprocess

    show = show or (lambda u: subprocess.run([SYSTEMCTL, "show", "-p", "ControlGroup", "--value", u], capture_output=True,
                                             text=True).stdout.strip())
    cg = show(unit)
    if not cg.startswith("/") or ".." in cg.split("/"):
        return []
    pids = []
    for base, _dirs, files in os.walk(cgroup / cg.lstrip("/")):
        if "cgroup.procs" in files:
            pids += [ln for ln in Path(base, "cgroup.procs").read_text().split() if ln.isdigit()]
    return pids


def unit_props(unit: str) -> dict[str, str]:
    import subprocess

    out = subprocess.run([SYSTEMCTL, "show", "-p", "LoadState", "-p", "User", unit], capture_output=True, text=True).stdout
    return dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)


def job_unit(args: list[str], props=unit_props) -> list[str]:
    """`stop [--no-block] research-job-RUN[.service]`: the unit a job runs in. Only vy-sudo's systemd-run makes a unit for
    research, and never one that runs as root, so a loaded unit of that name that does is somebody else's and is refused."""
    names = [a for a in args if a != "--no-block"]
    if len(names) != 1 or len(args) - len(names) > 1:
        raise Refused("systemctl stop [--no-block] research-job-RUN")
    name = names[0][:-len(".service")] if names[0].endswith(".service") else names[0]
    if not JOB_UNIT.match(name):
        raise Refused(f"unit name {names[0]!r}")
    unit = name + ".service"
    p = props(unit)
    if p.get("LoadState") == "loaded" and p.get("User", "") in ("", "root", "0"):
        raise Refused(f"{unit} runs as root: not a job's unit")
    return [SYSTEMCTL, "stop", *(a for a in args if a == "--no-block"), unit]


def systemctl(args: list[str], uid: int, show=None, cgroup: Path = CGROUP, proc: Path = PROC, props=unit_props) -> list[str]:
    if args[:1] == ["stop"] and any(a.startswith("research-job-") and not a.endswith(".scope") for a in args[1:]):
        return job_unit(args[1:], props)
    if not args or args[0] not in ("stop", "kill", "freeze", "thaw"):
        raise Refused("systemctl stop|kill|freeze|thaw [--signal=SIG] NAME.scope")
    verb, rest = args[0], args[1:]
    sig = [a for a in rest if a.startswith("--signal=")]
    units = [a for a in rest if not a.startswith("--signal=")]
    if len(units) != 1 or not units[0].endswith(".scope") or (sig and verb != "kill") or len(sig) > 1:
        raise Refused("systemctl stop|kill|freeze|thaw [--signal=SIG] NAME.scope")
    if sig and sig[0].split("=", 1)[1] not in SIGNALS:
        raise Refused(f"{sig[0]}")
    unit = _unit(units[0])
    for p in scope_pids(unit, show, cgroup):
        try:
            if (proc / p).stat().st_uid != uid:
                raise Refused(f"{unit} holds process {p}, which is not the caller's")
        except FileNotFoundError:
            continue
    return [SYSTEMCTL, verb, *sig, unit]


def hf_ns(args: list[str], uid: int, gid: int) -> list[str]:
    if "--" not in args:
        raise Refused("hf-ns [VAR=VAL...] -- SCRIPT ARGS...")
    i = args.index("--")
    env, rest = env_vars(args[:i]), args[i + 1:]
    if not rest:
        raise Refused("no script")
    inner = 'mount --bind "$1" "$2" && exec setpriv --reuid="$3" --regid="$4" --init-groups -- bash "${@:5}"'
    return ["/usr/bin/unshare", "--mount", "--propagation", "private", "/usr/bin/env", "-i", "VY_N2_NS=1",
            *(f"{k}={v}" for k, v in sorted(env.items())), "/bin/bash", "-c", inner, "hf-ns", *HF_BIND, str(uid), str(gid), *rest]


def reclaim(args: list[str], uid: int, gid: int, user=job_user, say=print, warn=None) -> int:
    """job-reclaim RUN_DIR USER, as the docstring has it. 0 once everything of USER's it reached is the caller's or gone, 1
    when something couldn't be read or changed (each named on stderr); Refused before anything changes."""
    warn = warn or (lambda m: print(f"vy-sudo: job-reclaim: {m}", file=sys.stderr))
    if len(args) != 2:
        raise Refused("job-reclaim RUN_DIR USER")
    top, job = args[0], user(args[1]).pw_uid
    if not top.startswith("/") or os.path.normpath(top) != top:
        raise Refused(f"{top!r}: an absolute, normalized path")
    try:
        fd = os.open(top, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise Refused(f"{top}: {e.strerror}") from None
    if os.fstat(fd).st_uid != uid:
        os.close(fd)
        raise Refused(f"{top} is not a directory of the caller's")
    failed = 0
    stack = [(fd, top, iter(sorted(os.listdir(fd))))]      # one open directory per level: nothing is reached by a path again
    while stack:
        dfd, base, names = stack[-1]
        name = next(names, None)
        if name is None:
            os.close(dfd)
            stack.pop()
            continue
        path = f"{base}/{name}"
        try:
            st = os.lstat(name, dir_fd=dfd)
            if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                if st.st_uid == job:
                    os.unlink(name, dir_fd=dfd)
                    say(path)
                continue
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_DIRECTORY if stat.S_ISDIR(st.st_mode) else 0),
                         dir_fd=dfd)
        except OSError as e:
            failed += 1
            warn(f"{path}: {e.strerror}")
            continue
        try:
            st = os.fstat(fd)                               # the inode this descriptor holds, whatever the name points at now
            is_dir = stat.S_ISDIR(st.st_mode)
            if st.st_uid == job and (is_dir or stat.S_ISREG(st.st_mode)):
                os.fchown(fd, uid, gid)
            if is_dir and st.st_uid in (job, uid):
                stack.append((fd, path, iter(sorted(os.listdir(fd)))))
                fd = -1
        except OSError as e:
            failed += 1
            warn(f"{path}: {e.strerror}")
        finally:
            if fd >= 0:
                os.close(fd)
    return 1 if failed else 0


def plan(argv: list[str], environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """(the argv to exec, its environment) for vy-sudo's arguments; Refused with why."""
    uid, gid = caller(environ)
    if not argv:
        raise Refused(f"usage: vy-sudo {'|'.join(FORMS)} ...")
    base = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
    head, rest = argv[0], argv[1:]
    if head == "renice":
        return renice(rest, uid), base
    if head == "taskset":
        return taskset(rest, uid), base
    if head == "env":
        k = next((i for i, a in enumerate(rest) if "=" not in a), len(rest))
        extra = env_vars(rest[:k])
        if rest[k:k + 1] != ["systemd-run"]:
            raise Refused("env VAR=VAL... systemd-run ...")
        return systemd_run(rest[k + 1:], uid, gid), {**base, **extra}
    if head == "systemd-run":
        return systemd_run(rest, uid, gid), base
    if head == "systemctl":
        return systemctl(rest, uid), base
    if head == "hf-ns":
        return hf_ns(rest, uid, gid), base
    if head == "du-bundles":
        if rest:
            raise Refused("du-bundles takes no arguments")
        # every path find hands du starts with a root, never `-`: nothing of the caller's reaches du as an option
        return ["/usr/bin/find", *BUNDLE_ROOTS, "-xdev", "-name", "replay_bundle_p*", "-prune", "-exec", "/usr/bin/du", "-sb", "{}", "+"], base
    raise Refused(f"{head}: not a form vy-sudo has")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if os.geteuid() != 0:
        print("vy-sudo: run through sudo", file=sys.stderr)
        return REFUSED
    try:
        if argv[:1] == ["job-reclaim"]:
            return reclaim(argv[1:], *caller(dict(os.environ)))
        cmd, env = plan(argv, dict(os.environ))
    except Refused as e:
        print(f"vy-sudo: refused: {e}", file=sys.stderr)
        return REFUSED
    os.execve(cmd[0], cmd, env)
    return 1


if __name__ == "__main__":
    sys.exit(main())
