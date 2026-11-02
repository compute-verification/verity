"""The training job's sandbox (fixed): one candidate's training command in new mount, PID, network, IPC and UTS
namespaces, as the run's user with no privileges, seeing only what it is given.

    sudo -n /usr/bin/python3 -I sandbox.py --root DIR --uid U --gid G --tree TREE.json --cwd DIR \\
        [--ro PATH[:DEST]]... [--rw PATH]... [--tmpfs DEST]... [--dev PATH]... [--env NAME=VALUE]... -- CMD...

The command's root is a fresh tmpfs at DIR holding only: each `--ro` path, read-only, at DEST (default the same path);
each `--rw` path, read-write; an empty read-only tmpfs at each `--tmpfs` (mounted before the `--ro` paths below it, so
`--tmpfs ISLANDS --ro SRC:ISLANDS/NAME` shows one island where the source tree had all of them); /proc of its own PID
namespace, /sys read-only without its submounts (no cgroup tree), a /dev of `--dev`'s nodes and null, zero, full,
random and urandom, and private /tmp and /dev/shm. The host's /bin, /lib, /lib64 and /sbin links are repeated. There is
no network (a new namespace with only loopback, down), no other process (its PID namespace), and nothing else on the
node: the held-out sets, the sealed side, other runs and the home directories aren't mounted. The command runs with
`--env`'s variables only, after setgroups, setgid and setuid to U and G and no_new_privs, so it holds no capability and
can't gain one.

TREE gets the job's accounting, written by root outside the sandbox: the CPU seconds of every process the job ran
(the namespace's init reaps every one, so its children's rusage is the whole tree; the main process's own figure
includes the children it waited for), how many processes other than the main one init reaped, the main process's
exit, wall seconds, and each GPU compute process `nvidia-smi` saw in the namespace while it ran (sampled every
`--poll` seconds: its pid inside the namespace and its GPU's UUID). A SIGTERM or SIGINT is passed to the main process;
when it exits, whatever else is left in the namespace is killed. The sandbox dies with the process that started it.

Stdlib only: it runs as root from the run's source (`python3 -I`), before the command, which never runs another way.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

PR_SET_PDEATHSIG, PR_SET_NO_NEW_PRIVS = 1, 38
#: CLONE_NEWNS, CLONE_NEWUTS, CLONE_NEWIPC, CLONE_NEWPID, CLONE_NEWNET (linux/sched.h)
FLAGS = 0x00020000 | 0x04000000 | 0x08000000 | 0x20000000 | 0x40000000
DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")
LINKS = ("/bin", "/lib", "/lib64", "/lib32", "/sbin")
#: a refused setup or sandbox failure (the command never ran)
REFUSED = 125

libc = ctypes.CDLL(None, use_errno=True)


def prctl(option: int, value: int) -> None:
    if libc.prctl(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), f"prctl({option})")


def forward(pid: int):
    def handler(sig, _frame):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    return handler


def mount(*args: str) -> None:
    r = subprocess.run(["mount", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"sandbox: mount {' '.join(args)}: {r.stderr.strip()}")


def target(root: Path, dest: str, is_dir: bool) -> Path:
    t = root / dest.lstrip("/")
    if is_dir:
        t.mkdir(parents=True, exist_ok=True)
    else:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.touch(exist_ok=True)
    return t


def bind(root: Path, src: str, dest: str, ro: bool) -> None:
    # root may not follow a link another user owns in a sticky world-writable directory (fs.protected_symlinks; a uv
    # Python's minor-version link in /workspace/jobs/cache/python is one), but realpath's lstat and readlink may
    src = os.path.realpath(src)
    if not os.path.exists(src):
        raise SystemExit(f"sandbox: {src} does not exist")
    t = target(root, dest, os.path.isdir(src))
    mount("--bind", src, str(t))
    mount("-o", f"remount,bind,{'ro,' if ro else ''}nosuid,nodev", str(t))


def setup(a: argparse.Namespace) -> None:
    """The new root, from inside the new namespaces (this process is their init)."""
    root = Path(a.root)
    mount("-t", "tmpfs", "-o", "mode=755,nosuid,nodev,size=64m", "cw-sandbox", str(root))
    for link in LINKS:
        if os.path.islink(link):
            os.symlink(os.readlink(link), root / link.lstrip("/"))
    mount("-t", "tmpfs", "-o", "mode=1777,nosuid,nodev,size=16g", "cw-tmp", str(target(root, "/tmp", True)))
    home = root / "tmp" / "home"
    home.mkdir()
    os.chown(home, a.uid, a.gid)
    ro = [(s, d or s) for s, _, d in (x.partition(":") for x in a.ro)]
    tmpfs = sorted(a.tmpfs, key=len)
    for src, dest in sorted(ro, key=lambda sd: len(sd[1])):
        for t in [t for t in tmpfs if len(t) <= len(dest) and (dest + "/").startswith(t + "/")]:
            mount("-t", "tmpfs", "-o", "mode=755,nosuid,nodev,size=1m", "cw-empty", str(target(root, t, True)))
            tmpfs.remove(t)
        bind(root, src, dest, True)
    for t in tmpfs:
        mount("-t", "tmpfs", "-o", "mode=755,nosuid,nodev,size=1m", "cw-empty", str(target(root, t, True)))
    for t in a.tmpfs:
        mount("-o", "remount,ro", str(root / t.lstrip("/")))
    for path in a.rw:
        bind(root, path, path, False)
    mount("-t", "proc", "-o", "nosuid,nodev,noexec", "proc", str(target(root, "/proc", True)))
    sys_ = target(root, "/sys", True)
    mount("--bind", "/sys", str(sys_))
    mount("-o", "remount,bind,ro,nosuid,nodev,noexec", str(sys_))
    dev = target(root, "/dev", True)
    mount("-t", "tmpfs", "-o", "mode=755,nosuid,size=1m", "cw-dev", str(dev))
    for node in (*DEVICES, *a.dev):
        if os.path.exists(node):
            t = target(root, node, False)
            mount("--bind", node, str(t))
    for name, to in (("fd", "/proc/self/fd"), ("stdin", "/proc/self/fd/0"), ("stdout", "/proc/self/fd/1"),
                     ("stderr", "/proc/self/fd/2")):
        os.symlink(to, dev / name)
    mount("-t", "tmpfs", "-o", "mode=1777,nosuid,nodev,size=16g", "cw-shm", str(target(root, "/dev/shm", True)))
    mount("-o", "remount,ro,nosuid,nodev", str(root))


def run_main(a: argparse.Namespace, env: dict[str, str]) -> None:
    """The command, in the new root, unprivileged (never returns)."""
    try:
        os.chroot(a.root)
        os.chdir(a.cwd)
        os.setgroups([a.gid])
        os.setresgid(a.gid, a.gid, a.gid)
        os.setresuid(a.uid, a.uid, a.uid)
        prctl(PR_SET_NO_NEW_PRIVS, 1)
        os.execve(a.cmd[0], a.cmd, env)
    except BaseException as e:  # noqa: BLE001 - nothing of the command ran
        print(f"sandbox: could not start {a.cmd[0]}: {e}", file=sys.stderr, flush=True)
        os._exit(REFUSED)


def init(a: argparse.Namespace, env: dict[str, str], record: Path) -> int:
    """PID 1 of the new namespace: set up, run the command, reap everything, account."""
    prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    try:
        mount("--make-rprivate", "/")
        setup(a)
    except SystemExit as e:
        print(e, file=sys.stderr, flush=True)
        return REFUSED
    base = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    main = os.fork()
    if main == 0:
        run_main(a, env)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward(main))
    status, main_ru, others = None, None, 0
    while True:
        try:
            pid, st, ru = os.wait4(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
        if pid == main:
            status, main_ru = st, ru
            try:
                os.kill(-1, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            others += 1
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    rc = os.waitstatus_to_exitcode(status) if status is not None else REFUSED
    record.write_text(json.dumps({
        "main_pid": main, "rc": rc, "wall_s": time.time() - t0, "started": t0,
        "cpu_s": (ru.ru_utime - base.ru_utime) + (ru.ru_stime - base.ru_stime),
        "main_cpu_s": main_ru.ru_utime + main_ru.ru_stime if main_ru else None,
        "other_processes": others, "max_rss_kb": ru.ru_maxrss}) + "\n")
    return rc if rc >= 0 else 128 - rc


def hold(a: argparse.Namespace, env: dict[str, str], record: Path, r: int, w: int) -> int:
    """The new namespaces' holder: unshares them, forks their init (its first child, so PID 1), tells the caller init's
    pid, and waits for it. It starts nothing else: any other child would be PID 1."""
    os.close(r)
    rc = REFUSED
    try:
        prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        if libc.unshare(FLAGS) != 0:
            raise SystemExit(f"sandbox: unshare: {os.strerror(ctypes.get_errno())}")
        pid1 = os.fork()
        if pid1 == 0:
            os.close(w)
            try:
                rc = init(a, env, record)
            except BaseException as e:  # noqa: BLE001 - init never returns into the caller's code
                print(f"sandbox: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            finally:
                os._exit(rc)
        os.write(w, f"{pid1}\n".encode())
        os.close(w)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, forward(pid1))
        _, status = os.waitpid(pid1, 0)
        rc = os.waitstatus_to_exitcode(status)
        rc = rc if rc >= 0 else 128 - rc
    except BaseException as e:  # noqa: BLE001 - the holder never returns into the caller's code
        print(f"sandbox: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return rc


def gpu_apps(ns: str) -> list[dict]:
    """The GPU compute processes in the PID namespace `ns` (its link target), by their pid inside it."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        pid, _, uuid = (x.strip() for x in line.partition(","))
        try:
            if os.readlink(f"/proc/{pid}/ns/pid") != ns:
                continue
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            continue
        nspid = next((x.split()[-1] for x in status.splitlines() if x.startswith("NSpid:")), "")
        out.append({"pid": int(nspid), "gpu_uuid": uuid})
    return out


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--uid", type=int, required=True)
    ap.add_argument("--gid", type=int, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--ro", action="append", default=[])
    ap.add_argument("--rw", action="append", default=[])
    ap.add_argument("--tmpfs", action="append", default=[])
    ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not a.cmd:
        ap.error("no command")
    if a.uid == 0 or a.gid == 0:
        ap.error("the command never runs as root")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    if os.geteuid() != 0:
        raise SystemExit("sandbox: run it as root (sudo -n)")
    env = dict(x.partition("=")[::2] for x in a.env)
    if not os.path.isdir(a.root) or os.listdir(a.root):
        raise SystemExit(f"sandbox: {a.root} must be an empty directory")
    prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    record = a.tree.with_name(a.tree.name + ".init")
    r, w = os.pipe()
    holder = os.fork()
    if holder == 0:
        os._exit(hold(a, env, record, r, w))
    os.close(w)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward(holder))
    with os.fdopen(r) as f:
        pid1 = int(f.read().strip() or 0)
    ns = os.readlink(f"/proc/{pid1}/ns/pid") if pid1 else ""
    smi = subprocess.run(["sh", "-c", "command -v nvidia-smi"], capture_output=True).returncode == 0
    seen: dict[tuple[int, str], int] = {}
    samples = 0
    while True:
        if smi and ns:
            for app in gpu_apps(ns):
                key = (app["pid"], app["gpu_uuid"])
                seen[key] = seen.get(key, 0) + 1
            samples += 1
        done, status = os.waitpid(holder, os.WNOHANG)
        if done:
            break
        time.sleep(a.poll)
    rc = os.waitstatus_to_exitcode(status)
    rec = json.loads(record.read_text()) if record.is_file() else {"rc": rc, "setup": "failed"}
    record.unlink(missing_ok=True)
    rec |= {"gpu_samples": samples if smi else None, "poll_s": a.poll,
            "gpu_apps": [{"pid": p, "gpu_uuid": u, "samples": n} for (p, u), n in sorted(seen.items())]}
    a.tree.write_text(json.dumps(rec, indent=1) + "\n")
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
