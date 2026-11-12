"""A live run's parties, each a process of its own. The server and the certifier share a host and its monotonic
clock; the observer is on that host too, over loopback, or on another, where it reads its own clock alone:

    python parties.py server|certifier|observer|waker|awake CONFIG.json [KEY=VALUE ...]

- **The server** is the adversary. It samples a seed of its own, and interval t's bit and lead are SHA-512's of the
  seed and t (`coin`). It acts only through when and how it writes its own bytes: `Plan.writes` is every time it
  writes at, a function of the bit, the lead and the public schedule. The bytes it writes are the same whatever the
  bit, `unit` bytes an interval.
- **The certifier** is the device under test: the network certifier (`network`, which imports it from whatever
  tree is first on the import path), on its loop (`paced_ms` 0) or with its pacer, or `relay`, a stand-in that paces
  nothing and passes each side's bytes on as they come, which is where a planted channel is put.
- **The observer** is the client side. It reads the certifier's bytes a `chunk` at most a read and timestamps the
  first byte of each interval's `chunk` twice: on its own clock once the read returns, waiting in the kernel for it
  (`plain`) or polling without waiting (`spin`), and with the stamp the kernel gave the bytes as they came off the
  network (`kernel_stamp`), which no scheduling of the observer's process moves. It keeps each chunk's last byte,
  which names its interval. It never has the seed: `run.py` joins its timestamps with the server's bits once both
  have ended.
  Its process counts, before and after its case, how many times the host's cores entered an idle state that stops
  them (`halts`): a host that keeps its cores from idling shows none.
- **The waker** (`waker_ms`) and **the awake cores** (`awake`) are candidate fixes beside the certifier, not parties
  to the channel. The waker is a reader asleep on a loopback connection and a writer that wakes it every `waker_ms`,
  so the host's network path is as recently used at every release whatever the server does. `awake` is a process
  spinning at idle priority on each of the run's CPUs, so none of them halts: what a guest's cores do under
  `idle=poll`, without root.

**The attacks** (`Plan.attack`), over interval t's release time R_t = t·τ + `aim` after the origin, `aim` the pacer's
offset, or with a launch gate (`run.py --launch`) the gate's opening:
- `delay`: interval t's bytes at t·τ, or half an interval later with bit 1. Only the relay passes it on: the network
  certifier releases a late interval as `FAIL`, in its own interval, and the connection ends.
- `write`: the next interval's bytes in one write, a lead of at most `lead_ms` before R_t with bit 1, and with bit 0
  at the quiet time, midway between R_t and the next interval's boundary. Both are in time for the next interval.
- `burst`: with bit 1 the next interval's bytes in `burst` writes spread from `lead_ms` before R_t to an eighth of
  that after it; with bit 0 as `write`.
With `ignore` the server flips its bits and writes as for bit 0 whatever they are: the control.

The server takes the admission bucket s of a network certifier's connection from when its request arrives, the
boundary after the one it was admitted at (`admit_lag`), and writes bucket s + D's bytes at once; a request that
comes before the origin it takes as the first interval's. Interval t's bytes
are all the byte t mod 256, so `run.py` checks which interval each of the observer's chunks is, and on one host its
first arrival's time as well. Nothing here but the certifier's process imports `verity`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import hashlib
import json
import math
import os
import select
import signal
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ATTACKS = ("delay", "write", "burst")
CERTIFIERS = ("relay", "network")
LOOPBACK = "127.0.0.1"
OBSERVERS = ("plain", "spin")
#: how long before a time the server stops sleeping and spins on the clock
SPIN_S = 0.003
#: a cell's header, `verity.protocols.certifier.verifier.active.HEADER`'s size: the certifier checks it at its start
HEADER_BYTES = 9
#: what the observer sends, a request of no meaning
HELLO = b"go\n"
#: the socket priority of a certifier's client connections under a launch gate (`run.py --launch`): one no type of
#: service maps to, so the gate's class holds the certifier's packets and no other of the host's
LAUNCH_PRIORITY = 5
#: Linux's SO_TIMESTAMPNS and the ancillary message it adds to a read, SCM_TIMESTAMPNS (one number on x86-64 and
#: arm64, which Python's `socket` doesn't name), and the room a read leaves for it
SO_TIMESTAMPNS = 35
ANCILLARY = 64
#: where the kernel keeps each core's idle states and their entry counts (`halts`)
CPUIDLE = Path("/sys/devices/system/cpu")
#: how many loopback reads `stamping` waits for the kernel's first stamp
STAMP_TRIES = 20000


@dataclass(frozen=True)
class Plan:
    """The schedule and the attack, all public: τ and the release's offset `paced_s` after each boundary (0 for a
    certifier that writes on its loop, and for the relay; a launch gate's opening where there is one), in seconds."""

    attack: str
    interval_s: float
    paced_s: float = 0.0
    lead_s: float = 0.002
    burst: int = 32
    ignore: bool = False

    def __post_init__(self) -> None:
        if self.attack not in ATTACKS:
            raise ValueError(f"an attack is one of {', '.join(ATTACKS)}, not {self.attack!r}")
        if not 0 <= self.paced_s < self.interval_s or not 0 < self.lead_s < self.interval_s / 4 or self.burst < 2:
            raise ValueError("a release is within its interval, a lead under a quarter of it, a burst two writes or more")

    def release(self, t: int) -> float:
        """R_t: when interval t leaves, in seconds after the origin."""
        return t * self.interval_s + self.paced_s

    def quiet(self, t: int) -> float:
        """Midway between R_t and interval t + 1's boundary."""
        return self.release(t) + (self.interval_s - self.paced_s) / 2

    def carries(self, t: int) -> int:
        """The interval whose bytes the server writes on bit t's choice: its own under `delay`, else the next."""
        return t if self.attack == "delay" else t + 1

    def writes(self, t: int, bit: int, u: float) -> list[float]:
        """When the server writes interval `carries(t)`'s bytes, in seconds after the origin, in order: one time a
        write, the bytes split evenly over them. `u` in [0, 1) places `write`'s lead."""
        bit = 0 if self.ignore else bit
        if self.attack == "delay":
            return [t * self.interval_s + (self.interval_s / 2 if bit else 0.0)]
        if not bit:
            return [self.quiet(t)]
        if self.attack == "write":
            return [self.release(t) - self.lead_s * (1 - u)]
        return [self.release(t) - self.lead_s + (k + 1) * self.lead_s * 1.125 / self.burst for k in range(self.burst)]


def coin(seed: bytes, t: int) -> tuple[int, float]:
    """Interval t's bit and u in [0, 1)."""
    h = hashlib.sha512(seed + t.to_bytes(8, "big")).digest()
    return h[0] & 1, int.from_bytes(h[1:5], "big") / 2 ** 32


def schedule(plan: Plan, seed: bytes, first: int, intervals: int, unit: int):
    """The server's writes from interval `first` on, for `intervals` bits: (t, its bit, [(when after the origin, the
    bytes)]), `unit` bytes a bit, the same whatever the bit, split evenly over its writes."""
    for t in range(first, first + intervals):
        bit, u = coin(seed, t)
        times = plan.writes(t, bit, u)
        data, m = bytes([plan.carries(t) & 0xFF]) * unit, len(times)
        yield t, bit, [(when, data[unit * k // m:unit * (k + 1) // m]) for k, when in enumerate(times)]


def first_interval(cfg: dict, arrived: int) -> int:
    """The first interval the server writes for, on a connection whose request reached it in interval `arrived`: a
    network certifier's releases the request `admit_lag` boundaries after admission and the first of the server's
    bytes `delay` after it; the relay's two intervals on."""
    return arrived - cfg["admit_lag"] + cfg["delay"] if cfg["certifier"] == "network" else arrived + 2


def plan_of(cfg: dict) -> Plan:
    """The server's plan: it aims at `aim_ms` after each boundary, the pacer's offset unless the case says another."""
    return Plan(cfg["attack"], cfg["interval_ms"] / 1000, cfg.get("aim_ms", cfg["paced_ms"]) / 1000,
                cfg["lead_ms"] / 1000, cfg["burst"], cfg["ignore"])


def span_s(cfg: dict) -> float:
    """How long a connection's intervals take, from its request to its last interval and a margin, in seconds."""
    return (cfg["delay"] + cfg["intervals"] + 12) * cfg["interval_ms"] / 1000 + 2.0


def distinct_cores(cpus: list[int]) -> list[int]:
    """One CPU of each physical core among `cpus`, in order (every CPU where the topology can't be read)."""
    seen, out = set(), []
    for cpu in cpus:
        try:
            core = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
        except OSError:
            core = str(cpu)
        if core not in seen:
            seen.add(core)
            out.append(cpu)
    return out


def _pin(cfg: dict, role: int) -> list[int]:
    """With `pin`, this process on a core of its own among the run's (0 the certifier's loop, 1 its pacer, 2 the
    server, 3 the observer); the CPUs it may run on."""
    if cfg["pin"]:
        os.sched_setaffinity(0, {cfg["cores"][role]})
    return sorted(os.sched_getaffinity(0))


def _announce(doc: dict) -> None:
    print(json.dumps(doc), flush=True)


def _wait(when: float) -> None:
    d = when - time.monotonic()
    if d > SPIN_S:
        time.sleep(d - SPIN_S)
    while time.monotonic() < when:
        pass


def server(cfg: dict) -> dict:
    """The adversary: one connection at a time, until it has chosen `intervals` bits or the run's `stop`. Returns its
    seed, each connection's first interval and, per bit, [interval, bit, how late its last write ended in µs]."""
    plan, origin, tau, unit = plan_of(cfg), cfg["origin"], cfg["interval_ms"] / 1000, cfg["unit"]
    cpus = _pin(cfg, 2)
    seed, rows, connections = os.urandom(32), [], []
    listener = socket.create_server((LOOPBACK, 0))
    _announce({"port": listener.getsockname()[1]})
    gc.disable()
    while len(rows) < cfg["intervals"] and (left := cfg["stop"] - time.monotonic()) > 0:
        if not select.select([listener], [], [], left)[0]:
            break
        conn, _ = listener.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(max(0.001, cfg["stop"] - time.monotonic()))
            try:
                if not conn.recv(1 << 16):
                    continue
                at = max(0, math.floor((time.monotonic() - origin) / tau))
                first = first_interval(cfg, at)
                if cfg["certifier"] == "network":
                    conn.sendall(bytes([first & 0xFF]) * unit)
                connections.append({"arrived": at, "first": first})
                for t, bit, writes in schedule(plan, seed, first, cfg["intervals"] - len(rows), unit):
                    if origin + writes[-1][0] > cfg["stop"]:
                        break
                    for when, data in writes:
                        _wait(origin + when)
                        conn.sendall(data)
                    rows.append([t, bit, round((time.monotonic() - origin - writes[-1][0]) * 1e6, 1)])
            except OSError as e:
                connections.append({"error": f"{type(e).__name__}: {e}"})
    return {"seed": seed.hex(), "connections": connections, "columns": ["interval", "bit", "late_us"], "rows": rows,
            "cpus": cpus}


def starts(total: int, size: int, chunk: int) -> int:
    """The chunks whose first byte is among `size` bytes read after `total`."""
    return -(-(total + size) // chunk) - -(-total // chunk)


def last_bytes(stream: bytes, chunk: int) -> list[int]:
    """Each chunk's last byte, of a chunk cut short the last it has."""
    return [stream[min(len(stream), end) - 1] for end in range(chunk, len(stream) + chunk, chunk)]


def kernel_stamp(ancillary: list[tuple[int, int, bytes]]) -> int:
    """The kernel's receive stamp among a read's ancillary data, in ns on the host's wall clock; 0 where it holds
    none. Under SO_TIMESTAMPNS a read of a TCP stream carries the stamp of the last segment it took bytes from: the
    wall clock's time when the kernel's network stack took the segment in, before any socket or process saw it (over
    loopback or a veth pair, as the sender's write reached the device). Segments the kernel merged while they waited
    to be read carry the last one's stamp."""
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS and len(data) >= 16:
            seconds, nanoseconds = struct.unpack("@ll", data[:16])
            return seconds * 1_000_000_000 + nanoseconds
    return 0


def stamping(tries: int = STAMP_TRIES) -> tuple[socket.socket | None, str]:
    """A socket that keeps the kernel stamping what this host receives: the reading end of a loopback connection,
    returned once one of its reads carried a stamp. The kernel starts stamping a moment after the first socket asks
    and stops with the last, so an observer asks here before it connects and holds this socket to its end. Where no
    stamp came (a kernel without SO_TIMESTAMPNS, or no loopback), no socket and why."""
    try:
        with socket.create_server((LOOPBACK, 0)) as listener:
            writer = socket.create_connection(listener.getsockname())
            reader, _ = listener.accept()
        with writer:
            reader.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            for _ in range(tries):
                writer.sendall(b"x")
                if kernel_stamp(reader.recvmsg(1, ANCILLARY)[1]):
                    return reader, ""
    except OSError as e:
        return None, f"asking this host's kernel for its receive stamp failed ({type(e).__name__}: {e})"
    return None, f"this host's kernel stamped none of {tries} loopback reads"


def halts(root: Path = CPUIDLE) -> int | str:
    """How many times this host's cores have entered an idle state that stops them, since boot: the entries the
    kernel counts for every cpuidle state of every core but the polling one's (`POLL`). Where there is no count, why:
    a host with no cpuidle driver, whose cores stop in the kernel's default idle uncounted, or one whose driver left
    while its states were read."""
    states = list(root.glob("cpu[0-9]*/cpuidle/state[0-9]*"))
    if not states:
        return "this host has no cpuidle driver, so its kernel counts no idle state's entries"
    try:
        return sum(int((state / "usage").read_text()) for state in states
                   if (state / "name").read_text().strip() != "POLL")
    except (OSError, ValueError) as e:
        return f"an idle state's count could not be read ({type(e).__name__}: {e})"


def wall_less_monotonic(reads: int = 5) -> int:
    """The host's wall clock less its monotonic one, in ns: of `reads` readings of the wall clock, each between two
    of the monotonic, the one whose two were closest. The two clocks run at one rate, so it changes only when the
    wall clock is set."""
    best = None
    for _ in range(reads):
        before, wall, after = time.monotonic_ns(), time.time_ns(), time.monotonic_ns()
        if best is None or after - before < best[0]:
            best = (after - before, wall - (before + after) // 2)
    return best[1]


def observer(cfg: dict) -> dict:
    """The client side: one connection after another until it has seen `intervals` intervals or its stop, each a
    segment: the monotonic time in ns at which the read holding the first byte of each `chunk` returned, and each
    chunk's last byte (`labels`); a connection that ended in an error leaves it in `errors`. A read takes one
    chunk's bytes at most, so the kernel's stamp on it (`kernel_stamp`) is of that chunk's bytes: `kernel` holds it
    for each chunk, moved from the wall clock to the monotonic one by their difference as the observer started
    (`wall_less_monotonic_ns`: then, and at its end), 0 for a read that carried none; None where the kernel stamps
    nothing, with why in `unstamped` (`stamping`). On the certifier's host the observer starts at the run's origin
    and ends at its stop. On another host (`connect`, and no `origin`) it has its own clocks alone: it tries the
    certifier for `wait_s`, past any connection that gave it nothing, and ends once a connection's intervals have had
    their time."""
    cpus = _pin(cfg, 3)
    spin, chunk, tau = cfg["observer"] == "spin", int(cfg["chunk"]), cfg["interval_ms"] / 1000
    address, remote = (cfg.get("connect", LOOPBACK), int(cfg["port"])), cfg.get("origin") is None
    segments: list[list[int]] = []
    kernel: list[list[int]] = []
    labels: list[list[int]] = []
    errors: list[str] = []
    keeper, unstamped = stamping()
    wall = [wall_less_monotonic()]
    gc.disable()
    if remote:
        retry = time.monotonic() + cfg["wait_s"]
        stop = retry + span_s(cfg)
    else:
        time.sleep(max(0.0, cfg["origin"] - time.monotonic()))
        retry, stop = 0.0, cfg["stop"]
    while sum(map(len, segments)) < cfg["intervals"] and time.monotonic() < stop - 4 * tau:
        try:
            sock = socket.create_connection(address, timeout=5.0)
        except OSError:
            if segments or time.monotonic() > retry:
                break
            time.sleep(1.0)
            continue
        arrivals: list[int] = []
        stamps: list[int] = []
        blobs: list[bytes] = []
        with sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if keeper is not None:
                sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            sock.sendall(bytes.fromhex(cfg["hello"]))
            sock.settimeout(0.0 if spin else 0.05)
            total = polls = 0
            while True:
                try:
                    data, ancillary, _, _ = sock.recvmsg(chunk - total % chunk, ANCILLARY)
                except (BlockingIOError, TimeoutError):
                    polls += 1
                    if (not spin or not polls & 0xFFF) and time.monotonic() > stop:
                        break
                    continue
                except OSError as e:
                    errors.append(f"{type(e).__name__}: {e}")
                    break
                now = time.monotonic_ns()
                if not data:
                    break
                if starts(total, len(data), chunk):
                    arrivals.append(now)
                    stamps.append(stamp - wall[0] if (stamp := kernel_stamp(ancillary)) else 0)
                total += len(data)
                blobs.append(data)
        if len(arrivals) < 2:
            if remote and not segments and time.monotonic() < retry:
                time.sleep(1.0)
                continue
            break
        segments.append(arrivals)
        kernel.append(stamps)
        labels.append(last_bytes(b"".join(blobs), chunk))
    wall.append(wall_less_monotonic())
    return {"observer": cfg["observer"], "chunk": chunk, "remote": remote, "segments": segments,
            "kernel": kernel if keeper is not None else None, "unstamped": unstamped,
            "wall_less_monotonic_ns": wall, "labels": labels, "errors": errors, "cpus": cpus}


def waker(cfg: dict) -> dict:
    """The waker's two processes, until the run's stop: this one writes a byte every `waker_ms` to a loopback
    connection, and its child sleeps in a read of it."""
    listener = socket.create_server((LOOPBACK, 0))
    if os.fork() == 0:
        conn, _ = listener.accept()
        while conn.recv(4096):
            pass
        os._exit(0)
    cpus, period, sent = sorted(os.sched_getaffinity(0)), cfg["waker_ms"] / 1000, 0
    stopped: list[int] = []
    with socket.create_connection(listener.getsockname()) as conn:
        _announce({"waker_ms": cfg["waker_ms"]})
        signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
        while not stopped and time.monotonic() < cfg["stop"]:
            conn.sendall(b"x")
            sent += 1
            time.sleep(period)
    return {"waker_ms": cfg["waker_ms"], "sent": sent, "cpus": cpus}


def awake(cfg: dict) -> dict:
    """A process on each of the run's CPUs, spinning at idle priority until the run's stop."""
    cpus, children = sorted(os.sched_getaffinity(0)), []
    for cpu in cpus:
        if (pid := os.fork()) == 0:
            os.sched_setaffinity(0, {cpu})
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
            while time.monotonic() < cfg["stop"]:
                pass
            os._exit(0)
        children.append(pid)
    _announce({"awake": len(cpus)})
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        for pid in children:
            os.waitpid(pid, 0)
    except SystemExit:
        for pid in children:
            os.kill(pid, signal.SIGKILL)
    return {"awake": cpus}


def gated(writer: asyncio.StreamWriter, cfg: dict) -> None:
    """Under a launch gate, put a certifier's client connection at `LAUNCH_PRIORITY`, whoever then writes to it. Each
    connection is set: one takes no priority from the socket it was accepted on."""
    if cfg.get("launch"):
        writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, LAUNCH_PRIORITY)


def _listener(cfg: dict) -> socket.socket:
    """Where a certifier takes the observer: loopback, or `listen` and `port`."""
    return socket.create_server((cfg.get("listen", LOOPBACK), int(cfg.get("port", 0))))


async def relay(cfg: dict) -> dict:
    """The stand-in that paces nothing: each side's bytes go on as they come. With `listen` it takes one connection,
    the case's observer."""
    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(1 << 16):
                writer.write(data)
        finally:
            writer.close()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if cfg.get("listen"):
            listener.close()
        up_reader, up_writer = await asyncio.open_connection(LOOPBACK, int(cfg["upstream"]))
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer), return_exceptions=True)

    cpus = _pin(cfg, 0)
    listener = await asyncio.start_server(handle, sock=_listener(cfg))
    _announce({"port": listener.sockets[0].getsockname()[1], "hello": HELLO.hex(), "chunk": cfg["unit"]})
    stopped, loop = asyncio.Event(), asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stopped.set)
    at_stop = loop.call_later(max(0.0, cfg["stop"] - time.monotonic()), stopped.set)
    try:
        await stopped.wait()
    finally:
        at_stop.cancel()
        listener.close()
    return {"certifier": "relay", "cpus": cpus}


async def network(cfg: dict) -> dict:
    """The network certifier of the tree on the import path, relaying any bytes, one lane for the observer and one
    spare; its records are dropped as they close. It runs for the intervals of one connection and, before them, the
    `wait_s` an observer on another host may take to come (`listen`), the one connection it then takes. With
    `gate_ms` [before, after] it reads no server from
    `before` ms ahead of each release to `after` ms past it: the server's bytes for the next interval wait in the
    kernel. Returns each tick's lateness, and from a paced one each bucket's release start and write end after its
    time."""
    from verity.protocols.certifier.verifier.active import HEADER
    from verity.protocols.certifier.verifier.warden import Warden as Certifier, WardenParams as Parameters, cells

    plan = plan_of(cfg)
    if HEADER.size != HEADER_BYTES:
        raise SystemExit(f"this tree's cells have a {HEADER.size}-byte header, not {HEADER_BYTES}")
    held = cfg["delay"] + cfg["intervals"] + 16
    total = held + math.ceil(cfg.get("wait_s", 0.0) * 1000 / cfg["interval_ms"])
    fields = {f.name for f in dataclasses.fields(Parameters)}
    params = Parameters(**{
        "cell_bytes": cfg["cell_bytes"], "slots": cfg["slots"], "bucket_ms": cfg["interval_ms"], "delay": cfg["delay"],
        "lanes": 2, "queue": 4, "window": cfg["window"], "ingress_cells": 4, "ingress_buckets": 8,
        "egress_cells": cfg["slots"] * held} | ({"egress": "bytes"} if "egress" in fields else {}))
    if params.payload * params.slots != cfg["unit"]:
        raise SystemExit(f"an interval holds {params.payload * params.slots} of the server's bytes, not {cfg['unit']}")
    cpus = _pin(cfg, 0)
    upstreams: list = []

    async def connect():
        reader, writer = await asyncio.open_connection(LOOPBACK, int(cfg["upstream"]))
        upstreams.append(writer.transport)
        return reader, writer

    def drop(record, grid) -> None:
        pass

    w = Certifier(params, connect, origin=cfg["origin"], windows=-(-total // cfg["window"]), publish=drop,
                  withhold=drop, paced=cfg["paced_ms"] / 1000 if cfg["paced_ms"] else None)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if cfg.get("listen"):
            listener.close()
        gated(writer, cfg)
        await w.handle(reader, writer)

    listener = await asyncio.start_server(handle, sock=_listener(cfg))
    _announce({"port": listener.sockets[0].getsockname()[1], "hello": cells(HELLO, params.cell_bytes).hex(),
               "chunk": params.cell_bytes * params.slots})
    if cfg["origin"] - time.monotonic() < 0.5:
        raise SystemExit("the certifier was ready under half a second before the origin: raise --start-s")
    run = asyncio.create_task(w.run())
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, run.cancel)
    pacer_cpus = None
    if cfg["pin"] and cfg["paced_ms"]:
        while w._pacer is None and not run.done():
            await asyncio.sleep(0.005)
        if w._pacer is not None:
            os.sched_setaffinity(w._pacer.process.pid, {cfg["cores"][1]})
            pacer_cpus = sorted(os.sched_getaffinity(w._pacer.process.pid))

    async def gate(before: float, after: float) -> None:
        for t in range(total):
            for when, act in ((plan.release(t) - before, "pause_reading"), (plan.release(t) + after, "resume_reading")):
                await asyncio.sleep(max(0.0, cfg["origin"] + when - time.monotonic()))
                for transport in upstreams:
                    if not transport.is_closing():
                        getattr(transport, act)()

    gating = asyncio.create_task(gate(*(x / 1000 for x in cfg["gate_ms"]))) if cfg["gate_ms"] else None
    try:
        await asyncio.wait([run])
    finally:
        listener.close()
        if gating is not None:
            gating.cancel()
    if not run.cancelled():
        run.result()
    return {"certifier": "network", "cpus": cpus, "pacer_cpus": pacer_cpus, "params": dataclasses.asdict(params),
            "slack_ms": w.slack * 1000, "ticks": w.ticks, "refused": w.refused, "connections": w.log,
            "tick_late_us": [round(x * 1e6, 1) for x in w.lateness],
            "releases": {"columns": ["bucket", "arrived_us", "start_us", "end_us", "written"],
                         "rows": [[r.t, round(r.arrived * 1e6, 1), round(r.start * 1e6, 1), round(r.end * 1e6, 1),
                                   r.written] for r in w.releases]}}


def main(argv: list[str]) -> int:
    role, cfg = argv[0], json.loads(Path(argv[1]).read_text())
    cfg |= dict(a.split("=", 1) for a in argv[2:])
    if role == "server":
        doc = server(cfg)
    elif role == "observer":
        doc = observer(cfg)
    elif role == "waker":
        doc = waker(cfg)
    elif role == "awake":
        doc = awake(cfg)
    elif role == "certifier":
        before = halts()
        doc = asyncio.run(relay(cfg) if cfg["certifier"] == "relay" else network(cfg))
        doc["halts"] = [before, halts()]
    else:
        raise SystemExit(f"a party is server, certifier, observer, waker or awake, not {role!r}")
    Path(cfg["out"], f"{role}.json").write_text(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
