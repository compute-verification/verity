"""What keeps `sandbox.sh`'s command from every unix socket it did not make, run as `python3 -I -S sockets.py MODE`.

A read-only mount doesn't stop `connect`, and a new network namespace hides only abstract names, so a socket file the
sandbox can see (`/run/user/UID/bus`, through which `systemd-run --user` starts anything outside it) is reachable. The
sandbox hides `/run` and the TMPDIR under private tmpfs, and three modes do the rest:

- `exec -- CMD...`: CMD under a seccomp filter that refuses (EACCES) `socket(AF_UNIX, ...)`, `io_uring_setup` (an
  io_uring makes sockets without `socket`) and every unix `socketpair` but a `SOCK_STREAM` one for x86-64 and arm64, and
  every system call of any other ABI. A datagram pair would reach any datagram socket by its address (`sendto`, or
  a `connect` that replaces its peer), and the kernel makes a unix `SOCK_RAW` pair a datagram one; a stream pair is
  connected for good (`EISCONN`). No unix socket made, no socket file or abstract name is reached, whenever it was
  bound.
- `inventory`: before the sandbox mounts its own /proc, every unix socket name the kernel lists in the network
  namespace of a process it can see (`/proc/PID/net/unix`), one per line: what a walk of every mount would find (a walk
  takes minutes on a node), and also names in a directory no walk can list.
- `probe [ADDRESS...]`: run as CMD runs (under `exec`), tries to connect to each name the inventory gave (on stdin),
  each D-Bus address or agent socket given, the well-known ones, and the abstract names of the sandbox's network
  namespace; exit 3, naming each, if any connects or the sandbox's own list can't be read.
"""

from __future__ import annotations

import ctypes
import os
import socket
import struct
import sys

PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
RET_ALLOW, RET_ERRNO = 0x7FFF0000, 0x00050000
EACCES, AF_UNIX = 13, 1
#: per ABI: its audit arch, the first number of a foreign ABI it also accepts or None, and the numbers of socket,
#: socketpair and io_uring_setup
ABI = {"x86_64": (0xC000003E, 0x40000000, 41, 53, 425), "aarch64": (0xC00000B7, None, 198, 199, 425)}
SOCK_STREAM, SOCK_TYPE_MASK = 1, 0xF
#: names a process of the host listens on wherever they exist ({uid}: the user the sandbox's root is)
KNOWN = ("/run/user/{uid}/bus", "/run/user/{uid}/systemd/private", "/run/user/{uid}/systemd/notify",
         "/run/user/{uid}/gnupg/S.gpg-agent", "/run/user/{uid}/gnupg/S.gpg-agent.ssh", "/run/user/{uid}/pipewire-0",
         "/run/dbus/system_bus_socket", "/var/run/dbus/system_bus_socket", "/run/systemd/private",
         "/run/systemd/notify", "/run/systemd/journal/socket", "/run/systemd/journal/stdout", "/dev/log",
         "/run/systemd/userdb/io.systemd.Multiplexer", "/run/systemd/resolve/io.systemd.Resolve", "/run/snapd.socket",
         "/run/snapd-snap.socket", "/run/docker.sock", "/var/run/docker.sock", "/run/containerd/containerd.sock",
         "/run/uuidd/request", "/run/rpcbind.sock", "/tmp/.X11-unix/X0")
#: why the kernel's list of unix sockets can't be read whole, and what to do
UNPARSED = ("holds a name no line parses: another process on this node holds a unix socket whose name holds a newline,"
            " which the list can't carry, so the sockets the sandbox must hide can't all be known. Find it with"
            " `ss -xap` (its name runs onto a second line) and stop the process that holds it, or run the build on"
            " another node")


def program() -> bytes:
    """The filter: refuse a foreign ABI's calls, and this ABI's `socket(AF_UNIX, ...)`, `io_uring_setup` and
    `socketpair(AF_UNIX, ...)` of any type but `SOCK_STREAM`."""
    try:
        arch, foreign, sock, pair, uring = ABI[os.uname().machine]
    except KeyError:
        raise SystemExit(f"sockets.py: no seccomp filter for {os.uname().machine}, so the sandbox can't keep its command"
                         f" from the host's unix sockets: run the build on an x86-64 or arm64 machine, or add this"
                         f" machine's audit arch and its `socket`, `socketpair` and `io_uring_setup` numbers to"
                         f" `ABI`") from None
    deny = RET_ERRNO | EACCES
    ld, jeq, jge, ret, and_ = 0x20, 0x15, 0x35, 0x06, 0x54
    ins = [(ld, 0, 0, 4), (jeq, 1, 0, arch), (ret, 0, 0, deny), (ld, 0, 0, 0)]
    if foreign is not None:
        ins += [(jge, 0, 1, foreign), (ret, 0, 0, deny)]
    ins += [(jeq, 0, 1, uring), (ret, 0, 0, deny),
            (jeq, 0, 4, sock), (ld, 0, 0, 16), (jeq, 0, 1, AF_UNIX), (ret, 0, 0, deny), (ret, 0, 0, RET_ALLOW),
            # socketpair: its domain (args[0]), then its type (args[1]) without SOCK_NONBLOCK and SOCK_CLOEXEC
            (jeq, 0, 6, pair), (ld, 0, 0, 16), (jeq, 0, 4, AF_UNIX), (ld, 0, 0, 24), (and_, 0, 0, SOCK_TYPE_MASK),
            (jeq, 1, 0, SOCK_STREAM), (ret, 0, 0, deny), (ret, 0, 0, RET_ALLOW)]
    return b"".join(struct.pack("HBBI", *i) for i in ins)


def confine() -> None:
    code = program()
    buf = ctypes.create_string_buffer(code)
    prog = struct.pack("HxxxxxxP", len(code) // 8, ctypes.addressof(buf))
    pbuf = ctypes.create_string_buffer(prog)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) or libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                                                                 ctypes.addressof(pbuf), 0, 0):
        raise SystemExit(f"sockets.py: the seccomp filter wasn't installed ({os.strerror(ctypes.get_errno())}), so the"
                         f" sandbox can't keep its command from the host's unix sockets: run the build on a kernel with"
                         f" seccomp filters (CONFIG_SECCOMP_FILTER), outside a container whose own filter refuses"
                         f" `prctl(PR_SET_SECCOMP)`")


def names(text: str) -> tuple[list[str], bool]:
    """The socket names of one `/proc/.../net/unix`, and whether every line parsed (a name holding a newline doesn't)."""
    out, ok = [], True
    for line in text.splitlines()[1:]:
        w = line.split(None, 7)
        if len(w) == 7:
            continue
        if len(w) != 8 or not w[0].endswith(":") or not w[6].isdigit():
            ok = False
            continue
        out.append(w[7])
    return out, ok


def inventory() -> int:
    seen, found, ok, read, unread = set(), set(), True, 0, []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/net/unix") as f:
                text = f.read()
        except OSError as e:
            unread.append(f"{pid} ({e.strerror})")
            continue
        read += 1
        if text in seen:
            continue
        seen.add(text)
        got, parsed = names(text)
        ok &= parsed
        found.update(n for n in got if not n.startswith("@"))
    if not read or not ok:
        print("sockets.py: the kernel's list of unix sockets (/proc/PID/net/unix) "
              + (f"could be read for no process ({', '.join(unread[:5])}): run the build where /proc is mounted and"
                 f" readable" if not read else UNPARSED), file=sys.stderr)
        return 3
    sys.stdout.write("".join(f"{n}\n" for n in sorted(found)))
    return 0


def host_uid() -> str:
    with open("/proc/self/uid_map") as f:
        for line in f:
            inside, outside, _ = line.split()
            if inside == "0":
                return outside
    return str(os.getuid())


def connects(addr: str, kind: int) -> bool:
    """Whether a unix socket of ``kind`` that we make connects to ``addr``; under `exec` none can be made."""
    try:
        s = socket.socket(socket.AF_UNIX, kind)
    except OSError:
        return False
    try:
        s.settimeout(2)
        s.connect(addr)
        return True
    except OSError:
        return False
    finally:
        s.close()


def reached(name: str) -> str | None:
    """How ``name`` was reached, or None: a stream, a datagram or a seqpacket socket of ours that connected to it."""
    addr = "\0" + name[1:] if name.startswith("@") else name
    kinds = ((socket.SOCK_STREAM, "stream"), (socket.SOCK_DGRAM, "datagram"), (socket.SOCK_SEQPACKET, "seqpacket"))
    return next((how for kind, how in kinds if connects(addr, kind)), None)


def probe(addresses: list[str]) -> int:
    """``addresses``: the command's D-Bus addresses and agent sockets (`DBUS_SESSION_BUS_ADDRESS`, `SSH_AUTH_SOCK`, ...),
    as `sandbox.sh` passes them; the inventory comes on stdin."""
    uid = host_uid()
    want = {*filter(None, sys.stdin.read().split("\n")), *(k.format(uid=uid) for k in KNOWN)}
    for a in (p for v in addresses for p in v.split(";")):
        if a.startswith("unix:path="):
            want.add(a[len("unix:path="):].split(",")[0])
        elif a.startswith("/"):
            want.add(a)
    try:
        with open("/proc/self/net/unix") as f:
            own, ok = names(f.read())
    except OSError as e:
        print(f"sockets.py: the sandbox's own list of unix sockets can't be read ({e}): run `sandbox.sh` under"
              f" `unshare --pid --fork`, as its header says, so it mounts a /proc of its own", file=sys.stderr)
        return 3
    if not ok:
        print(f"sockets.py: the sandbox's own list of unix sockets {UNPARSED}", file=sys.stderr)
        return 3
    want |= {n for n in own if n.startswith("@")}
    hits = [f"{n} ({how})" for n in sorted(want) if (how := reached(n))]
    if hits:
        print("sandbox.sh: the command could connect to these unix sockets, which it did not make:", *hits, "- the"
              " seccomp filter should have refused this, so check that `sockets.py exec` installed it, and hide each"
              " socket's directory from the sandbox (or stop what listens on it)", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str]) -> int:
    match argv:
        case ["exec", "--", cmd, *args]:
            confine()
            os.execvp(cmd, [cmd, *args])
        case ["inventory"]:
            return inventory()
        case ["probe", *addresses]:
            return probe(addresses)
    print("usage: sockets.py exec -- CMD... | inventory | probe [ADDRESS...] < INVENTORY", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
