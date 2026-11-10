"""A live run's three parties, each a process of its own on one host, over loopback, all on the host's monotonic clock:

    python parties.py server|certifier|observer CONFIG.json [KEY=VALUE ...]

- **The server** is the adversary. It samples a seed of its own, and interval t's bit and lead are SHA-512's of the
  seed and t (`draw`). It acts only through when and how it writes its own bytes: `Plan.writes` is every time it
  writes at, a function of the bit, the lead and the public schedule. The bytes it writes are the same whatever the
  bit, `unit` bytes an interval.
- **The certifier** is the device under test: the network certifier (`warden.Warden`, of whatever tree is first on
  the import path), on its loop (`paced_ms` 0) or with its pacer, or `relay`, a stand-in that paces nothing and passes
  each side's bytes on as they come, which is where a planted channel is put.
- **The observer** is the client side. It reads the certifier's bytes and timestamps the first byte of each
  interval's `chunk`, waiting in the kernel for it (`plain`) or polling without waiting (`spin`). It never has the
  seed: `run.py` joins its timestamps with the server's bits once both have ended.

**The attacks** (`Plan.attack`), over interval t's release time R_t = t·τ + `paced` after the origin:
- `delay`: interval t's bytes at t·τ, or half an interval later with bit 1. Only the relay passes it on: the network
  certifier releases a late interval as `FAIL`, in its own interval, and the connection ends.
- `write`: the next interval's bytes in one write, a lead of at most `lead_ms` before R_t with bit 1, and with bit 0
  at the quiet time, midway between R_t and the next interval's boundary. Both are in time for the next interval.
- `burst`: with bit 1 the next interval's bytes in `burst` writes spread from `lead_ms` before R_t to an eighth of
  that after it; with bit 0 as `write`.
With `ignore` the server draws its bits and writes as for bit 0 whatever they are: the control.

The server takes the admission bucket s of a network certifier's connection from when its request arrives, the
boundary after the one it was admitted at (`admit_lag`), and writes bucket s + D's bytes at once; `run.py` checks
that against the observer's first arrival. Nothing here but `certifier` imports `verity`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ATTACKS = ("delay", "write", "burst")
CERTIFIERS = ("relay", "warden")
OBSERVERS = ("plain", "spin")
#: how long before a time the server stops sleeping and spins on the clock
SPIN_S = 0.003
#: a cell's header, `verity.core.protocols.certifier.active.HEADER`'s size: the certifier checks it at its start
HEADER_BYTES = 9
#: what the observer sends, a request of no meaning
HELLO = b"go\n"


@dataclass(frozen=True)
class Plan:
    """The schedule and the attack, all public: τ and the release's offset `paced_s` after each boundary (0 for a
    certifier that writes on its loop, and for the relay), in seconds."""

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


def draw(seed: bytes, t: int) -> tuple[int, float]:
    """Interval t's bit and u in [0, 1)."""
    h = hashlib.sha512(seed + t.to_bytes(8, "big")).digest()
    return h[0] & 1, int.from_bytes(h[1:5], "big") / 2 ** 32


def schedule(plan: Plan, seed: bytes, first: int, intervals: int, unit: int):
    """The server's writes from interval `first` on, for `intervals` bits: (t, its bit, [(when after the origin, the
    bytes)]), `unit` bytes a bit, the same whatever the bit, split evenly over its writes."""
    for t in range(first, first + intervals):
        bit, u = draw(seed, t)
        times = plan.writes(t, bit, u)
        data, m = bytes([plan.carries(t) & 0xFF]) * unit, len(times)
        yield t, bit, [(when, data[unit * k // m:unit * (k + 1) // m]) for k, when in enumerate(times)]


def first_interval(cfg: dict, arrived: int) -> int:
    """The first interval the server writes for, on a connection whose request reached it in interval `arrived`: a
    network certifier's releases the request `admit_lag` boundaries after admission and the first of the server's
    bytes `delay` after it; the relay's two intervals on."""
    return arrived - cfg["admit_lag"] + cfg["delay"] if cfg["certifier"] == "warden" else arrived + 2


def plan_of(cfg: dict) -> Plan:
    return Plan(cfg["attack"], cfg["interval_ms"] / 1000, cfg["paced_ms"] / 1000, cfg["lead_ms"] / 1000,
                cfg["burst"], cfg["ignore"])


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
    listener = socket.create_server(("127.0.0.1", 0))
    _announce({"port": listener.getsockname()[1]})
    gc.disable()
    while len(rows) < cfg["intervals"] and (left := cfg["stop"] - time.monotonic()) > 0:
        listener.settimeout(left)
        try:
            conn, _ = listener.accept()
        except TimeoutError:
            break
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(max(0.001, cfg["stop"] - time.monotonic()))
            try:
                if not conn.recv(1 << 16):
                    continue
                at = math.floor((time.monotonic() - origin) / tau)
                first = first_interval(cfg, at)
                if cfg["certifier"] == "warden":
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


def observer(cfg: dict) -> dict:
    """The client side: from the origin, one connection after another until it has seen `intervals` intervals or the
    run's `stop`, each a segment: the monotonic time in ns at which the first byte of each `chunk` arrived."""
    cpus = _pin(cfg, 3)
    spin, chunk, stop = cfg["observer"] == "spin", int(cfg["chunk"]), cfg["stop"]
    segments: list[list[int]] = []
    gc.disable()
    time.sleep(max(0.0, cfg["origin"] - time.monotonic()))
    while sum(map(len, segments)) < cfg["intervals"] and time.monotonic() < stop - 4 * cfg["interval_ms"] / 1000:
        arrivals: list[int] = []
        segments.append(arrivals)
        try:
            sock = socket.create_connection(("127.0.0.1", int(cfg["port"])))
        except OSError:
            break
        with sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(bytes.fromhex(cfg["hello"]))
            sock.settimeout(0.0 if spin else 0.05)
            total = polls = 0
            while True:
                try:
                    data = sock.recv(1 << 16)
                except (BlockingIOError, TimeoutError):
                    polls += 1
                    if (not spin or not polls & 0xFFF) and time.monotonic() > stop:
                        break
                    continue
                except OSError:
                    break
                now = time.monotonic_ns()
                if not data:
                    break
                arrivals += [now] * ((total + len(data) - 1) // chunk + 1 - -(-total // chunk))
                total += len(data)
        if len(arrivals) < 2:
            break
    return {"observer": cfg["observer"], "chunk": chunk, "segments": [s for s in segments if s], "cpus": cpus}


async def relay(cfg: dict) -> dict:
    """The stand-in that paces nothing: each side's bytes go on as they come."""
    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(1 << 16):
                writer.write(data)
        finally:
            writer.close()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", int(cfg["upstream"]))
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer), return_exceptions=True)

    cpus = _pin(cfg, 0)
    listener = await asyncio.start_server(handle, "127.0.0.1", 0)
    _announce({"port": listener.sockets[0].getsockname()[1], "hello": HELLO.hex(), "chunk": cfg["unit"]})
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    try:
        await asyncio.wait_for(stopped.wait(), max(0.0, cfg["stop"] - time.monotonic()))
    except TimeoutError:
        pass
    finally:
        listener.close()
    return {"certifier": "relay", "cpus": cpus}


async def warden(cfg: dict) -> dict:
    """The network certifier of the tree on the import path, relaying any bytes, one lane for the observer and one
    spare; its records are dropped as they close. With `gate_ms` [before, after] it reads no server from `before`
    ms ahead of each release to `after` ms past it: the server's bytes for the next interval wait in the kernel.
    Returns each tick's lateness, and from a paced one each bucket's release start and write end after its time."""
    from verity.core.protocols.certifier.active import HEADER
    from verity.core.protocols.certifier.warden import Warden, WardenParams, cells

    plan = plan_of(cfg)
    if HEADER.size != HEADER_BYTES:
        raise SystemExit(f"this tree's cells have a {HEADER.size}-byte header, not {HEADER_BYTES}")
    total = cfg["delay"] + cfg["intervals"] + 16
    fields = {f.name for f in dataclasses.fields(WardenParams)}
    params = WardenParams(**{
        "cell_bytes": cfg["cell_bytes"], "slots": cfg["slots"], "bucket_ms": cfg["interval_ms"], "delay": cfg["delay"],
        "lanes": 2, "queue": 4, "window": cfg["window"], "ingress_cells": 4, "ingress_buckets": 8,
        "egress_cells": cfg["slots"] * total} | ({"egress": "bytes"} if "egress" in fields else {}))
    if params.payload * params.slots != cfg["unit"]:
        raise SystemExit(f"an interval holds {params.payload * params.slots} of the server's bytes, not {cfg['unit']}")
    cpus = _pin(cfg, 0)
    upstreams: list = []

    async def connect():
        reader, writer = await asyncio.open_connection("127.0.0.1", int(cfg["upstream"]))
        upstreams.append(writer.transport)
        return reader, writer

    def drop(record, grid) -> None:
        pass

    w = Warden(params, connect, origin=cfg["origin"], windows=-(-total // cfg["window"]), publish=drop, withhold=drop,
               paced=cfg["paced_ms"] / 1000 if cfg["paced_ms"] else None)
    listener = await asyncio.start_server(w.handle, "127.0.0.1", 0)
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
        await run
    except asyncio.CancelledError:
        pass
    finally:
        listener.close()
        if gating is not None:
            gating.cancel()
    return {"certifier": "warden", "cpus": cpus, "pacer_cpus": pacer_cpus, "params": dataclasses.asdict(params),
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
    elif role == "certifier":
        doc = asyncio.run(relay(cfg) if cfg["certifier"] == "relay" else warden(cfg))
    else:
        raise SystemExit(f"a party is server, certifier or observer, not {role!r}")
    Path(cfg["out"], f"{role}.json").write_text(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
