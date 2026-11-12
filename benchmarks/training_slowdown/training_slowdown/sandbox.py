"""The training job's sandbox: the process a route's code runs in has no network, no final set, none of the job's
kept state, and no socket of the machine's services or of its user's own.

    python -m training_slowdown.sandbox [--hide DIR]… -- COMMAND…

`enter` puts the calling process in a user, a mount and a network namespace of its own, with its user mapped to itself,
and mounts an empty read-only directory over each DIR and over each of `operators`; the entry then becomes COMMAND. The
new network namespace holds one loopback interface, down, so nothing COMMAND starts reaches another host, or a
service of this one that listens on an address or on a socket with no path; and COMMAND,
started by a user other than root, holds no capability after the exec and gains none from a setuid program (no root
is mapped), so it can't take the mounts off. It needs a machine that lets an unprivileged process make a user
namespace (`REQUIRES`): where one doesn't, the entry stops with that and COMMAND never starts.

`refusals` is COMMAND's own half, read before it runs any code it didn't write: what it can still reach, an interface
that is up, an entry of a hidden directory or a socket of `ways_out`, each by name, and a capability it holds, as
root's COMMAND does, with which it could undo the rest. A job that finds one stops before it trains (`run.inside`),
so a sandbox that wasn't made is a failed job, never a job that trained outside one.

A socket at a path is reached wherever the path is, whatever the network namespace, and the process behind it is
outside the sandbox: the user's systemd manager starts what it is asked to there, with the network up and nothing
hidden. So the sandbox hides, on every machine, the directories those sockets are in (`operators`): /run whole, where
the user's systemd manager and session bus are, and the system bus and the machine's daemons beside them, and the
directory of the user's tmux server. COMMAND finds each empty, so it is one that reads nothing there.

What it hides besides is the directories the registry names (`final_sets`), the job's own state directory
(`run.Kept`) and, where the job's run holds a key to the store, the directory the key lies in
(`run.routes_environment`): a copy of a final set at another path stays as readable as the job's user finds it, and
so does another job's state. The rest of the file system is the user's as before, a socket at another path among it. The entry
imports the standard library alone, since a process makes a user namespace only while it has one thread.
"""
from __future__ import annotations

import argparse
import array
import ctypes
import fcntl
import os
import socket
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWNET = 0x00020000, 0x10000000, 0x40000000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_REC, MS_PRIVATE = 1, 2, 4, 1 << 14, 1 << 18
SIOCGIFFLAGS, IFF_UP = 0x8913, 1
LINUX_CAPABILITY_VERSION_3 = 0x20080522

#: what a machine that runs training jobs gives them
REQUIRES = ("a machine that runs a training job lets an unprivileged process make a user namespace: sysctl "
            "kernel.unprivileged_userns_clone=1 and kernel.apparmor_restrict_unprivileged_userns=0, each where the "
            "kernel has it, and user.max_user_namespaces above 0")


def final_sets(reg: Mapping) -> list[Path]:
    """Every registered data's final set: what no training job's process may read."""
    return sorted({Path(d["final"]) for d in reg["data"].values()})


def operators(uid: int | None = None) -> list[Path]:
    """The directories the sandbox hides on every machine, for user `uid` (default this process's): /run, and the
    directory of the user's tmux server."""
    uid = os.getuid() if uid is None else uid
    return [Path("/run"), Path(f"/tmp/tmux-{uid}")]


def ways_out(uid: int | None = None) -> list[Path]:
    """The sockets through which a process of user `uid` (default this process's) has one started outside the sandbox:
    its systemd manager's, its session bus's and its tmux server's. Each is under one of `operators`."""
    uid = os.getuid() if uid is None else uid
    return [Path(f"/run/user/{uid}/systemd/private"), Path(f"/run/user/{uid}/bus"), Path(f"/tmp/tmux-{uid}/default")]


def enter(hide: Iterable[Path]) -> None:
    """Move this process into namespaces of its own with no network, an empty directory over each of `hide` and of
    `operators` that exists here."""
    uid, gid = os.getuid(), os.getgid()
    libc = ctypes.CDLL(None, use_errno=True)

    def call(what: str, rc: int) -> None:
        if rc != 0:
            raise SystemExit(f"sandbox: {what}: {os.strerror(ctypes.get_errno())}. {REQUIRES}")

    def write(path: str, text: str) -> None:
        try:
            Path(path).write_text(text)
        except OSError as e:
            raise SystemExit(f"sandbox: writing {path}: {e.strerror}. {REQUIRES}") from None

    call("unshare(user, mount, network)", libc.unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET))
    write("/proc/self/setgroups", "deny")
    write("/proc/self/uid_map", f"{uid} {uid} 1")
    write("/proc/self/gid_map", f"{gid} {gid} 1")
    call("making / private", libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None))
    for d in dict.fromkeys([*hide, *operators(uid)]):
        if d.is_dir():
            call(f"mounting an empty directory over {d}", libc.mount(
                b"tmpfs", os.fsencode(d), b"tmpfs", MS_RDONLY | MS_NOSUID | MS_NODEV, b"size=4k,mode=555"))
        elif d.exists():
            raise SystemExit(f"sandbox: {d} is not a directory: what the sandbox hides is one")


def interfaces() -> dict[str, bool]:
    """This process's network interfaces, and whether each is up."""
    out = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _, name in socket.if_nameindex():
            req = array.array("B", name.encode()[:15].ljust(16, b"\0") + bytes(24))
            fcntl.ioctl(s, SIOCGIFFLAGS, req)
            out[name] = bool(int.from_bytes(req[16:18], sys.byteorder) & IFF_UP)
    return out


def capabilities() -> int:
    """This process's effective capabilities, the kernel's bit set (`capget`, its third version: two words of each)."""
    header, sets = (ctypes.c_uint32 * 2)(LINUX_CAPABILITY_VERSION_3, 0), (ctypes.c_uint32 * 6)()
    if ctypes.CDLL(None, use_errno=True).capget(header, sets) != 0:
        raise OSError(ctypes.get_errno(), "capget")
    return sets[0] | sets[3] << 32


def refusals(hidden: Iterable[Path], up: Mapping[str, bool] | None = None, held: int | None = None) -> list[str]:
    """Why this process is not inside the sandbox: each interface that is up (`up`, default this process's), each of
    `hidden` and of `operators` it can read an entry of, each socket of `ways_out` it finds, and any capability it
    holds (`held`, default this process's). Empty inside it. A hidden path that exists and can't be listed raises:
    nothing here takes a failed read for an empty directory."""
    up = interfaces() if up is None else up
    held = capabilities() if held is None else held
    out = [f"the network is reachable (interface {name} is up)" for name in sorted(up) if up[name]]
    for d in dict.fromkeys([*hidden, *operators()]):
        if d.exists() and (seen := sorted(os.listdir(d))):
            out.append(f"{d}, which the sandbox hides, is readable ({', '.join(seen[:3])})")
    out += [f"a process outside the sandbox is reachable (socket {s}): what it starts has the network and every hidden "
            f"directory" for s in ways_out() if s.exists()]
    if held:
        out.append(f"the process holds capabilities ({held:016x}) with which it could undo the sandbox: run the job as "
                   f"a user other than root")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hide", type=Path, action="append", default=[], metavar="DIR",
                    help="a directory COMMAND finds empty")
    ap.add_argument("command", nargs="+", metavar="COMMAND")
    a = ap.parse_args(argv)
    enter(a.hide)
    os.execvp(a.command[0], a.command)


if __name__ == "__main__":
    sys.exit(main())
