"""Serve a workload on the mock datacenter and check it: each block's server in its network namespace, behind the network
certifier's proxy on its uplink, and the client on the outside host, with the lab's Program (`lab.Lab`) on every link;
then the certifier's audit, N3 (`lab.check`) and the answers.

    python -m workload.lab_run run --out DIR [--layers 4] [--tamper b1:flip-bit --expect refused] [--local]

Run it from `benchmarks/network_sim/`, or with it on `PYTHONPATH`, in a Python with torch. The workload is placed on the
lab's declaration as `mockdc.lab` builds it, at the certifiers' cadence (`hardware`): its blocks are the units, their
GPUs compute and the outside host is the client. It needs mockdc's lab up in that shape (`mockdc.py up`; by default mockdc's own,
`--blocks 4 --leaves 2 --spines 1`) and idle: a lab whose namespaces run any process is refused, since `mockdc.py down`,
or another run, would kill or collide with this one. `--local` runs the same processes on loopback addresses instead,
with no namespaces and no sudo.

**Processes.** Each starts as `mockdc.Process` starts one (in its namespace, as the invoking user), prints one line when it
is ready and reads the origin from its stdin; all are timed from that origin on the host's monotonic clock.
- Per block, in `na-bI`, the block (`block` role). It holds its values (the registered inputs from the seed and every
  constant) and evaluates its gates once on zeros to warm up. It then serves each session's request in session order on
  its uplink's address at `mockdc.SERVER_PORT`, writes the session's events as soon as its gates are evaluated, and logs
  each one as `active_replay.LiveServer` does (`server.events.gz`). Told to stop, it commits its values (`result.json`),
  none if a session never came.
- Per block, in `na-bI-cert`, mockdc's proxy (`mockdc.Process(ns, "proxy", cfg)`: `active.Proxy` unmodified), whose
  upstream is the block.
- On the outside host, the client (`client` role), with one link to each proxy. It reads every link to its end, keeping
  `active_replay.Peer`'s digest of each window. It relays each message once its last chunk is in, and sends each block's
  requests in session order as soon as their messages are in, the first ones half a bucket after the origin. At the end
  it commits the run inputs and reads the answers (`outside.json`).

**Calibration**, before the processes start, as a trace is calibrated. The orchestrator evaluates the Program session by
session to time it, and sets the first-frame latency D, in buckets, to four times the slowest session plus half a second
(`--delta-buckets` sets it). It models the run (a request is released in the bucket after it is sent, and its response is
at the client D + 1 buckets later) and calibrates the certifier on the model (`calibration.calibrate`, one configuration
per block): the egress allocation, Σ_sync with δ up to D, and the ingress allocation. Every window declares the envelope,
the largest δ and the smallest ρ declared, so a response is on time whenever it is ready within D buckets of its
request's release. The run lasts the windows that cover the model's last response with 2D + 20 buckets to spare.

**Checks.** The certifier's audit (`active_replay.audit_configuration` with `audit.Audit`) on each block's raw and committed
records, in its live mode, whose Program is the block's event log anchored at each request's release as the ingress
records show it: the timing. N3 over the same raw records (`lab.check`): the content. And the answers against the Program
evaluated in one process.

**Tampering** (`--tamper BLOCK:MODE`): `flip-bit` flips the first bit of the block's first outbound message, which N3
refuses while the certifier, whose timing is honest, accepts; `late` holds the block's first response past every δ of
Σ_sync, which the certifier refuses (`coverage-never-established`). A block can't write into a frame's fill: the proxy
frames its events.

**Outputs** in `--out`: `report.json` (the calibration); per block `BLOCK/`, with `plan.json`, the proxy's records and log,
the block's `server.json`, `server.events.gz` and `result.json`, the client's `peer.json` and the certifier's `audit.json`;
`outside.json`, `public.json`, `witness.json`, `n3.json`, and `summary.json`: per block the certifier's verdicts and timing
and N3's verdict, and the answers against the reference. The exit status is 0 when every process exited cleanly and the
run is what `--expect` says: `accepted` (every block's certifier audit accepts, N3 holds and the answers are the
reference's) or `refused` (the tampered block's certifier audit or N3 refuses).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mockdc  # puts network_traces, datacenter and the repository root on sys.path, for the next three

import active_replay as ar
import graph
import records
from datacenter import Cadence
from verity.core.protocols.certifier import calibration
from verity.core.protocols.certifier.active import HEADER, PAD, decode_frame
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.schedule import ClockSync

from . import framing as F
from . import lab as L
from . import model as M
from . import values as V
from .block import Context
from .hardware import OUTSIDE, Hardware
from .run import statement

BUCKET_MS, WINDOW_BUCKETS, FRAME_BYTES, QUEUE = 100, 60, 4096, 1 << 16
FABRIC_MS = 10  # the model's delay between a proxy and the client, each way
PATHS = [*mockdc.PATHS, str(mockdc.ROOT / "integrations" / "vllm")]
TAMPERS = ("flip-bit", "late")


# The plan: the Program timed, the run modeled, the certifier calibrated on the model.

def time_sessions(ctx: Context, lab: L.Lab) -> dict[str, float]:
    """Each session's evaluation in seconds: the whole Program evaluated tick by tick (once the evaluator is warm)."""
    values = dict(ctx.inputs) | ctx.constants
    tick = ctx.placement.tick_of
    by_tick: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for b in lab.blocks:
        for s in lab.sessions[b]:
            for g in s.gates:
                by_tick[tick[g]][s.name].append(g)
    out = {s.name: 0.0 for b in lab.blocks for s in lab.sessions[b]}
    for t in sorted(by_tick):
        for name, gates in by_tick[t].items():
            t0 = time.perf_counter()
            ctx.model.evaluate(values, gates)
            out[name] += time.perf_counter() - t0
    return out


def model_run(lab: L.Lab, D: int) -> dict[str, list[tuple[float, int]]]:
    """Each block's requests in the model, as (send time in ms, the bucket it is released in) per session: the first ones
    half a bucket after the origin, each released in the bucket after it is sent, and its response at the client
    `FABRIC_MS` after bucket release + D + 1 starts. A request goes as soon as its messages are in and the block's
    previous one has gone."""
    tau = BUCKET_MS
    ready = {m.stream: 0.0 for m in lab.messages.values() if m.src == OUTSIDE}
    sent: dict[str, list[tuple[float, int]]] = {b: [] for b in lab.blocks}
    moved = True
    while moved:
        moved = False
        for b in lab.blocks:
            for s in lab.sessions[b][len(sent[b]):]:
                if any(m.stream not in ready for m in s.inbound):
                    break
                t = max([tau / 2, *(ready[m.stream] + 1 for m in s.inbound), *(x for x, _ in sent[b][-1:])])
                release = math.ceil(t / tau)
                sent[b].append((t, release))
                ready.update({m.stream: (release + D + 1) * tau + FABRIC_MS for m in s.outbound})
                moved = True
    if stuck := [s.name for b in lab.blocks for s in lab.sessions[b][len(sent[b]):]]:
        raise ValueError(f"sessions {stuck} wait on each other")
    return sent


def request_frames(lab: L.Lab, s: L.Session) -> int:
    return max(1, math.ceil(lab.request_length(s) / (FRAME_BYTES - HEADER.size)))


def calibrate_run(lab: L.Lab, D: int, name: str) -> tuple[dict, ar.Setup, dict[str, dict]]:
    """The calibration report on the modeled run, one configuration per block, and each block's plan for the proxy and
    the audit."""
    sent = model_run(lab, D)
    configs = []
    for b in lab.blocks:
        frames, arrival = [], []
        for s, (t, release) in zip(lab.sessions[b], sent[b], strict=True):
            frames += [{"session": s.name, "index": k, "anchor": release, "ready": release + D}
                       for k in range(len(lab.positions(b, s.index)))]
            arrival += [t + FABRIC_MS / 2] * request_frames(lab, s)
        configs.append({"name": f"{name}/{b}", "egress": [{"link": L.EGRESS, "frames": frames}],
                        "ingress": [{"link": L.INGRESS, "arrival_ms": arrival}]})
    report = calibration.calibrate({"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS, "frame_bytes": FRAME_BYTES,
                                    "queue_capacity": QUEUE, "configurations": configs})
    setup = ar.Setup(report)
    envelope, sync_set, T = setup.envelope, setup.sync_set, WINDOW_BUCKETS
    if envelope.delta < D:
        raise ValueError(f"the calibration's envelope δ = {envelope.delta} is below the model's D = {D}")
    end = max(release + D + 1 + len(lab.events[b]) for b in lab.blocks for _, release in sent[b])
    windows = math.ceil((end + 2 * D + 20) / T)
    eager = ClockSync(min(sync_set.deltas), max(sync_set.rhos))
    plans = {}
    for b in lab.blocks:
        sessions = [[s.name, s.stream, [D] * len(lab.positions(b, s.index))] for s in lab.sessions[b]]
        requests = [{"stream": s.stream, "session": s.name, "frames": request_frames(lab, s), "trace_ms": t,
                     "enqueue": math.ceil(t / setup.ingress_params.bucket_ms), "send_ms": t}
                    for s, (t, _) in zip(lab.sessions[b], sent[b], strict=True)]
        release = ar.model_releases(requests, setup.ingress_params, windows, setup.egress_params.bucket_ms)
        chosen, _, misses = calibration.choose_syncs([ar.model_frames(sessions, release, windows * T)], sync_set, T)
        chosen = (list(chosen) + [eager] * windows)[:windows]
        plans[b] = {"name": f"{name}/{b}", "windows": windows, "sessions": sessions, "requests": requests,
                    "model_release": {str(k): v for k, v in release.items()}, "model_misses": misses,
                    "syncs": [[envelope.delta, str(envelope.rho)]] * windows, "sync_policy": "envelope",
                    "model_syncs": [[s.delta, str(s.rho)] for s in chosen], "live_model": name, "seed": 0}
    return report, setup, plans


# The roles. Each prints one line when it is ready and reads the origin from its stdin; the block then reads one more
# line before it stops.

def _ready(**info) -> None:
    print(json.dumps({"ready": True} | info), flush=True)


async def _line() -> str:
    return await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)


def _commitment(unit: str, gates, values: dict[int, int], width: int) -> dict | None:
    """The unit's values committed, or None when a request or a message it needed never came."""
    if any(g not in values for g in gates):
        return None
    c = V.commit(unit, gates, values, width)
    return {"root": c.root.hex(), "count": len(c.gates), "rows": [r.hex() for r in c.registration.rows],
            "salts": [s.hex() for s in c.registration.salts]}


class Server:
    """A block's server: each session's request in session order, its gates evaluated, its events written and logged."""

    def __init__(self, ctx: Context, lab: L.Lab, block: str, tau: float, tamper: str | None, hold: int) -> None:
        self.ctx, self.lab, self.block, self.tau, self.tamper, self.hold = ctx, lab, block, tau, tamper, hold
        self.sessions = lab.sessions[block]
        held = set(lab.held[block])
        self.values = {g: v for g, v in ctx.inputs.items() if g in held} | ctx.constants
        loop = asyncio.get_running_loop()
        self.requests = [loop.create_future() for _ in self.sessions]
        self.origin = 0.0
        self.events: list[tuple[int, str, int, int, bytes]] = []
        self.log = {"receipt_late_ms": [], "batch_late_ms": [], "late_registrations": 0, "requests": 0, "events": 0,
                    "failed": [], "cut_at_horizon": 0, "first_event_ms": [], "compute_ms": [], "served": 0,
                    "server": "lab", "tamper": tamper}

    def warm(self) -> None:
        """Evaluate every gate of the block once on zeros, so that no session pays for the evaluator's start-up."""
        scratch = dict(self.values)
        for s in self.sessions:
            scratch.update({e.value: 0 for m in s.inbound for e in m.edges})
            self.ctx.model.evaluate(scratch, s.gates)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            j, n = L.REQUEST.unpack(await reader.readexactly(L.REQUEST.size))
            body = await reader.readexactly(n)
        except (asyncio.IncompleteReadError, ConnectionError) as e:
            self.log["failed"].append(repr(e)[:200])
            writer.close()
            return
        if j >= len(self.sessions) or self.requests[j].done() or \
                L.REQUEST.size + n != self.lab.request_length(self.sessions[j]):
            self.log["failed"].append(f"a request for session {j} of {n} bytes")
            writer.close()
            return
        received = time.monotonic() - self.origin
        self.log["requests"] += 1
        self.log["receipt_late_ms"].append(round((received % self.tau) * 1000, 3))
        self.requests[j].set_result((body, writer, received))

    async def serve(self, origin: float) -> None:
        self.origin = origin
        tampered = False
        for s in self.sessions:
            body, writer, received = await self.requests[s.index]
            at = 0
            for m in s.inbound:
                n = (m.bits + 7) // 8
                self.values.update(F.decode(m, body[at:at + n]))
                at += n
            t0 = time.monotonic()
            await asyncio.to_thread(self.ctx.model.evaluate, self.values, s.gates)
            self.log["compute_ms"].append(round((time.monotonic() - t0) * 1000, 3))
            payloads = {m.stream: F.encode(m, self.values) for m in s.outbound}
            if self.tamper and s.outbound and not tampered:
                tampered = True
                if self.tamper == "flip-bit":
                    first = payloads[s.outbound[0].stream]
                    payloads[s.outbound[0].stream] = bytes([first[0] ^ 1]) + first[1:]
                else:
                    await asyncio.sleep(max(0.0, origin + received + self.hold * self.tau - time.monotonic()))
            for k, p in enumerate(self.lab.positions(self.block, s.index)):
                event = self.lab.event(self.block, p, payloads[self.lab.events[self.block][p][1].stream])
                writer.write(event)
                now = time.monotonic() - origin
                if k == 0:
                    self.log["first_event_ms"].append(round((now - received) * 1000, 3))
                self.events.append((p, s.name, k, math.floor(now / self.tau), event))
                self.log["events"] += 1
            try:
                await writer.drain()
            except ConnectionError as e:
                self.log["failed"].append(repr(e)[:200])
            writer.close()
            self.log["served"] += 1

    def write(self, out: Path) -> None:
        (out / "server.json").write_text(json.dumps(self.log | ar._usage()))
        with gzip.open(out / "server.events.gz", "wt", compresslevel=1) as fh:
            for p, name, k, bucket, event in self.events:
                fh.write(json.dumps([p, name, k, bucket, event.decode("latin-1")]) + "\n")
        values = _commitment(self.block, self.lab.held[self.block], self.values, self.ctx.model.value_bits)
        (out / "result.json").write_text(json.dumps({"unit": self.block, "values": values}))


async def block_role(cfg: dict) -> None:
    ctx = Context(cfg["config"])
    server = Server(ctx, L.Lab.of(ctx, FRAME_BYTES), cfg["block"], cfg["bucket_ms"] / 1000, cfg["tamper"],
                    cfg["hold_buckets"])
    server.warm()
    listening = await asyncio.start_server(server.handle, cfg["host"], mockdc.SERVER_PORT)
    _ready()
    serving = asyncio.create_task(server.serve(float(await _line())))
    await _line()
    serving.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await serving
    listening.close()
    server.write(Path(cfg["out"]))


class Client:
    """The outside host: a link to every block's proxy, the run inputs, the relay between blocks, and the answers."""

    def __init__(self, ctx: Context, lab: L.Lab, params: dict) -> None:
        self.ctx, self.lab = ctx, lab
        held = set(lab.held[OUTSIDE])
        self.values = {g: v for g, v in ctx.inputs.items() if g in held} | ctx.constants
        self.payloads = {s: F.encode(m, self.values) for s, m in lab.messages.items() if m.src == OUTSIDE}
        self.when = dict.fromkeys(self.payloads, 0.0)
        self.chunks: dict[int, dict[int, bytes]] = defaultdict(dict)
        self.arrived = asyncio.Event()
        self.B, self.tau = params["frame_bytes"], params["bucket_ms"] / 1000
        self.bucket_bytes = params["allocation"] * self.B
        self.window_bytes = params["buckets"] * self.bucket_bytes
        self.logs = {b: {"send_late_ms": [], "window_sha256": [], "bucket_lag_ms": [], "bytes": 0} for b in lab.blocks}
        self.strays: list[str] = []

    async def read(self, block: str, reader: asyncio.StreamReader, origin: float) -> None:
        """The block's link to its end: a SHA-256 of each window and each bucket's arrival after its tick, as
        `active_replay.Peer` keeps them, and each event's chunk."""
        log, h, buffer = self.logs[block], hashlib.sha256(), bytearray()
        while data := await reader.read(1 << 22):
            now, mv, before = time.monotonic(), memoryview(data), log["bytes"] // self.bucket_bytes
            while mv:
                take = min(len(mv), self.window_bytes - log["bytes"] % self.window_bytes)
                h.update(mv[:take])
                log["bytes"], mv = log["bytes"] + take, mv[take:]
                if log["bytes"] % self.window_bytes == 0:
                    log["window_sha256"].append(h.hexdigest())
                    h = hashlib.sha256()
            for t in range(before, log["bytes"] // self.bucket_bytes):
                log["bucket_lag_ms"].append(round((now - origin - t * self.tau) * 1000, 3))
            buffer += data
            whole = len(buffer) - len(buffer) % self.B
            for i in range(0, whole, self.B):
                self._frame(block, bytes(buffer[i:i + self.B]), now)
            del buffer[:whole]

    def _frame(self, block: str, frame: bytes, now: float) -> None:
        kind, stream, event = decode_frame(frame)
        if kind == PAD:
            return
        parsed = L.parse(event)
        if parsed is None or parsed[0] >= len(self.lab.events[block]):
            self.strays.append(f"{block}: a frame of stream {stream} that is no event of the Program")
            return
        p, chunk = parsed
        _, m, j = self.lab.events[block][p]
        got = self.chunks[m.stream]
        got[j] = chunk
        if len(got) == self.lab.chunks(m) and m.stream not in self.payloads:
            self.payloads[m.stream] = b"".join(got[i] for i in range(len(got)))
            self.when[m.stream] = now
            self.arrived.set()
            self.arrived = asyncio.Event()

    async def send(self, block: str, writer: asyncio.StreamWriter, origin: float) -> None:
        """The block's requests in session order, each as soon as its messages are in, none before half a bucket after
        the origin; `send_late_ms` is how long after that each went."""
        log, last = self.logs[block], origin + self.tau / 2
        await asyncio.sleep(max(0.0, last - time.monotonic()))
        for s in self.lab.sessions[block]:
            while any(m.stream not in self.payloads for m in s.inbound):
                await self.arrived.wait()
            last = max([last, *(self.when[m.stream] for m in s.inbound)])
            writer.write(ar.request_frames(s.stream, self.lab.request(s, self.payloads), self.B))
            log["send_late_ms"].append(round((time.monotonic() - last) * 1000, 3))

    async def run(self, links: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]], origin: float) -> None:
        readers = [asyncio.create_task(self.read(b, r, origin)) for b, (r, _) in links.items()]
        senders = [asyncio.create_task(self.send(b, w, origin)) for b, (_, w) in links.items()]
        await asyncio.gather(*readers)
        for task in senders:
            task.cancel()
        for task in senders:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for _, w in links.values():
            w.close()

    def answers(self) -> list[int] | None:
        for s, m in self.lab.messages.items():
            if m.dst == OUTSIDE and s in self.payloads:
                self.values.update(F.decode(m, self.payloads[s]))
        try:
            return self.ctx.model.output_words(self.values)
        except KeyError:
            return None

    def write(self, out: Path) -> None:
        for b, log in self.logs.items():
            (out / b / "peer.json").write_text(json.dumps(log | ar._usage()))
        answers = self.answers()
        values = _commitment(OUTSIDE, self.lab.held[OUTSIDE], self.values, self.ctx.model.value_bits)
        (out / "outside.json").write_text(json.dumps({"unit": OUTSIDE, "values": values, "answers": answers,
                                                      "relayed": len(self.payloads), "strays": self.strays}))


async def client_role(cfg: dict) -> None:
    ctx = Context(cfg["config"])
    client = Client(ctx, L.Lab.of(ctx, FRAME_BYTES), cfg["params"])
    links = {b: await asyncio.open_connection(host, mockdc.LINK_PORT, limit=1 << 22) for b, host in cfg["links"].items()}
    _ready()
    await client.run(links, float(await _line()))
    client.write(Path(cfg["out"]))


ROLES = {"block": block_role, "client": client_role}


def role(args) -> None:
    ar._quiet_runtime()
    asyncio.run(ROLES[args.name](json.loads(Path(args.config).read_text())))


# The run.

class Role(mockdc.Process):
    """One of this module's roles, or mockdc's run with no namespace (`ns` None, for `--local`), started as
    `mockdc.Process` starts mockdc's: its config in `<out>/<name>.json` and its stderr in `<out>/<name>.log`."""

    def __init__(self, ns: str | None, name: str, cfg: dict, argv: list[str]) -> None:
        out = Path(cfg["out"])
        config = out / f"{name}.json"
        config.write_text(json.dumps(cfg))
        self.name, self.log = f"{ns or 'local'}/{name}", out / f"{name}.log"
        cmd = ["env", f"PYTHONPATH={os.pathsep.join(PATHS)}", sys.executable, "-B", "-u", *argv, str(config)]
        if ns is not None:
            cmd = ["sudo", "-n", "ip", "netns", "exec", ns, "setpriv", f"--reuid={os.getuid()}",
                   f"--regid={os.getgid()}", "--clear-groups", "--", *cmd]
        with open(self.log, "w") as err:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True)


@dataclass(frozen=True)
class Place:
    """Where a block's processes run: the block's and its certifier's namespaces (None locally), the block's server
    address and the proxy's link address."""

    block_ns: str | None
    cert_ns: str | None
    server: str
    link: str


def hardware(dc) -> Hardware:
    """The workload's hardware: the lab's declaration at the certifiers' cadence (`records.cadence`'s bucket of
    `BUCKET_MS` ticks of `records.TICK_MS`), so that an engine step is one of their buckets."""
    cadence = Cadence(frame_bits=8 * FRAME_BYTES, bucket_ticks=BUCKET_MS // records.TICK_MS, buckets=WINDOW_BUCKETS)
    return Hardware.from_datacenter(dataclasses.replace(dc, cadence=cadence))


def shape_of(args) -> argparse.Namespace:
    """The lab's shape in `mockdc.py`'s options."""
    return argparse.Namespace(declaration=args.declaration, blocks=args.blocks, leaves=args.leaves, spines=args.spines,
                              fabric_latency_us=args.fabric_latency_us, fabric_jitter_us=args.fabric_jitter_us)


def lab_places(shape: argparse.Namespace, dc, blocks: list[str]) -> tuple[dict[str, Place], str]:
    """Each block's place in mockdc's lab, and the outside host's namespace, once the lab is up in its shape and idle."""
    plan = mockdc.plan_of(shape, dc)
    if diff := mockdc.differences(plan):
        raise SystemExit(f"{'; '.join(diff)}: {mockdc.RESHAPE}")
    if busy := {ns: pids for ns in mockdc.namespaces() if (pids := mockdc.sudo(["ip", "netns", "pids", ns]).stdout.split())}:
        raise SystemExit(f"the lab is busy, processes {busy} run in its namespaces: wait until its run ends")
    places = {}
    for b in blocks:
        _, cert = records.uplink(dc, b)
        places[b] = Place(graph.netns_name(b), graph.netns_name(cert), plan.address(b, cert), plan.fabric_address(cert))
    return places, graph.netns_name(mockdc.outside(dc))


def local_places(blocks: list[str]) -> tuple[dict[str, Place], None]:
    return {b: Place(None, None, f"127.0.1.{i + 1}", f"127.0.2.{i + 1}") for i, b in enumerate(blocks)}, None


def _stop(procs: list[mockdc.Process], namespaces: list[str]) -> None:
    """Stop what is left of a failed run: the processes in its namespaces through sudo (the lab was idle when the run
    began, so they are all the run's), a local one directly."""
    for ns in namespaces:
        for pid in mockdc.sudo(["ip", "netns", "pids", ns], check=False).stdout.split():
            mockdc.sudo(["kill", "-TERM", pid], check=False)
    for p in procs:
        if p.proc.poll() is None and not namespaces:
            p.proc.terminate()
        p.proc.wait()


def spec_of(args) -> dict:
    if args.workload == "mlp":
        return {"d": 4, "layers": args.layers, "requests": 3, "groups": 1}
    return {"layers": args.layers, "prompt": args.prompt, "steps": args.steps}


def launch(args, dc, out: Path, config: dict, lab: L.Lab, report: dict, setup: ar.Setup, plans: dict,
           tamper: tuple[str, str] | None) -> dict[str, int]:
    """Start every process, give them one origin, wait for them, and return their exit codes."""
    places, client_ns = local_places(lab.blocks) if args.local else lab_places(shape_of(args), dc, lab.blocks)
    namespaces = [] if args.local else [ns for p in places.values() for ns in (p.block_ns, p.cert_ns)] + [client_ns]
    me = ["-m", "workload.lab_run", "role"]
    procs: list[mockdc.Process] = []
    codes: dict[str, int] = {}
    try:
        blocks = {}
        for b in lab.blocks:
            (out / b).mkdir(exist_ok=True)
            (out / b / "plan.json").write_text(json.dumps(plans[b]))
            cfg = {"config": config, "block": b, "host": places[b].server, "out": str(out / b), "bucket_ms": BUCKET_MS,
                   "tamper": tamper[1] if tamper and tamper[0] == b else None,
                   "hold_buckets": max(setup.sync_set.deltas) + 2}
            procs.append(blocks.setdefault(b, Role(places[b].block_ns, "block", cfg, [*me, "block"])))
        for p in blocks.values():
            p.ready()
        proxies = {}
        for b in lab.blocks:
            cfg = {"plan": plans[b], "report": report, "host": places[b].link, "upstream": places[b].server,
                   "out": str(out / b)}
            proxies[b] = (mockdc.Process(places[b].cert_ns, "proxy", cfg) if places[b].cert_ns else
                          Role(None, "proxy", cfg, [str(Path(mockdc.__file__)), "role", "proxy"]))
            procs.append(proxies[b])
            proxies[b].ready()
        client = Role(client_ns, "client", {"config": config, "links": {b: places[b].link for b in lab.blocks},
                                            "params": ar.wire_params(setup), "out": str(out)}, [*me, "client"])
        procs.append(client)
        client.ready()
        origin = time.monotonic() + mockdc.STARTUP_S
        for p in procs:
            p.send(repr(origin))
        print(f"running {lab.blocks} {'locally' if args.local else 'in the lab'} from origin {origin:.3f}: "
              f"{plans[lab.blocks[0]]['windows']} windows of {WINDOW_BUCKETS} buckets of {BUCKET_MS} ms", flush=True)
        codes["client"] = client.wait()
        for b in lab.blocks:
            codes[f"{b}/proxy"] = proxies[b].wait()
        for b in lab.blocks:
            blocks[b].send("stop")
            codes[f"{b}/block"] = blocks[b].wait()
    finally:
        if any(p.proc.poll() is None for p in procs):
            _stop(procs, namespaces)
    return codes


def run(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dc = mockdc.lab(shape_of(args))
    config = {"workload": args.workload, "spec": spec_of(args), "hardware": hardware(dc).to_json(),
              "frame_bytes": FRAME_BYTES, "seed": args.seed}
    ctx = Context(config)
    lab = L.Lab.of(ctx, FRAME_BYTES)
    tamper = None
    if args.tamper:
        unit, _, mode = args.tamper.partition(":")
        if unit not in lab.blocks or mode not in TAMPERS:
            raise SystemExit(f"--tamper is BLOCK:MODE, a block of {lab.blocks} and a mode of {TAMPERS}")
        tamper = (unit, mode)
    reference = ctx.model.reference(ctx.seed)
    compute = time_sessions(ctx, lab)
    D = args.delta_buckets or max(10, math.ceil((4 * max(compute.values()) + 0.5) * 1000 / BUCKET_MS))
    name = f"lab/{args.workload}"
    report, setup, plans = calibrate_run(lab, D, name)
    (out / "report.json").write_text(json.dumps(report))
    print(f"{name} on {dc.name} ({len(lab.blocks)} blocks of {', '.join(str(len(ctx.hw.devices(b))) for b in lab.blocks)} "
          f"GPUs): {ctx.model.n} gates, {len(lab.messages)} messages, "
          f"{sum(len(ss) for ss in lab.sessions.values())} sessions, {sum(len(e) for e in lab.events.values())} events; "
          f"slowest session {max(compute.values()) * 1000:.0f} ms, D = {D} buckets; envelope δ = {setup.envelope.delta}, "
          f"ρ = {setup.envelope.rho}, r = {setup.egress_params.allocation}, r_in = {setup.ingress_params.allocation}",
          flush=True)
    codes = launch(args, dc, out, config, lab, report, setup, plans, tamper)

    published, blocks = {}, {}
    for b in lab.blocks:
        a = ar.audit_configuration(plans[b], report, out / b, audit=Audit)
        (out / b / "audit.json").write_text(json.dumps(a))
        published[b] = [ar.unpack_record(x, FRAME_BYTES) for f in (f"{L.EGRESS}.records", f"{L.INGRESS}.records")
                        for x in ar.read_blobs(out / b / f)]
        rows = [[r["link"], r["window"], r["raw"], r["committed"]] for r in a["rows"]]
        t, c = a["timing"], a["comparisons"]
        blocks[b] = {"tamper": tamper[1] if tamper and tamper[0] == b else None,
                     "certifier": {"accepted": all(raw is None and com is None for _, _, raw, com in rows),
                                   "verdicts": rows},
                     "slack_ms": t["slack_ms"],
                     "tick_late_ms_max": {k: v.get("max") for k, v in t["tick_late_ms"].items()},
                     "ticks_over_slack": {k: v["over_slack"] for k, v in t["tick_late_ms"].items()},
                     "wire_bucket_lag_ms": t["wire_bucket_lag_ms"], "first_event_ms": t["server_first_event_ms"],
                     "egress": a["egress"],
                     "comparisons": {k: c[k] for k in ("records_are_run_shapers", "hands_are_the_rules",
                                                       "frames_log_is_the_programs", "wire_is_the_records",
                                                       "released_not_the_programs", "releases_off_model")}}
    results = {u: json.loads(p.read_text()) if p.exists() else {"values": None, "answers": None}
               for u, p in [*((b, out / b / "result.json") for b in lab.blocks), (OUTSIDE, out / "outside.json")]}
    public, witness = statement(ctx, results)
    n3 = L.check(public, witness, published)
    for key, obj in (("public", public), ("witness", witness), ("n3", n3.to_json())):
        (out / f"{key}.json").write_text(json.dumps(obj, indent=1))
    for b in lab.blocks:
        mine = [v for v in n3.violations if v.where.split(":")[0] in (b, "")]
        blocks[b]["n3"] = {"accepted": not mine, "violations": [f"{v.code} {v.where}: {v.detail}" for v in mine]}
    answers = results[OUTSIDE]["answers"]
    accepted = all(x["certifier"]["accepted"] and x["n3"]["accepted"] for x in blocks.values()) and answers == reference
    summary = {"workload": args.workload, "spec": config["spec"], "blocks": blocks, "n3": n3.to_json() | {"violations": None},
               "answers": answers, "reference": reference, "answers_equal": answers == reference, "accepted": accepted,
               "delta_buckets": D, "session_compute_ms": {k: round(v * 1000, 3) for k, v in compute.items()},
               "windows": plans[lab.blocks[0]]["windows"], "exit": codes, "local": args.local,
               "grid": {"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS, "frame_bytes": FRAME_BYTES,
                        "allocation": setup.egress_params.allocation, "ingress_allocation": setup.ingress_params.allocation,
                        "envelope": [setup.envelope.delta, str(setup.envelope.rho)], "sync_set_size": setup.sync_set.size}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    for b, x in blocks.items():
        failed = [f"{link}[{w}] {raw or com}" for link, w, raw, com in x["certifier"]["verdicts"] if raw or com]
        late = x["tick_late_ms_max"]
        print(f"{b}{' (' + x['tamper'] + ')' if x['tamper'] else ''} certifier audit "
              f"{'ACCEPTED' if x['certifier']['accepted'] else 'REFUSED (' + ', '.join(failed) + ')'}; "
              f"N3 {'holds' if x['n3']['accepted'] else 'REFUSED (' + '; '.join(x['n3']['violations'][:3]) + ')'}; "
              f"latest tick {max((v for v in late.values() if v is not None), default=0):.1f} ms late against "
              f"{x['slack_ms']:.0f} ms of slack, "
              f"{sum(x['ticks_over_slack'].values())} over")
    s = n3.stats
    print(f"N3 {'holds' if n3.ok else 'fails'}: {s['frames']} frames recorded, {s['explained']} of {s['events']} events "
          f"and {s['requests']} of {s['sessions']} requests explained, against {s['values']} opened values")
    print(f"answers {answers} {'equal' if answers == reference else 'differ from'} the reference {reference}")
    clean = not any(codes.values())
    if not clean:
        print(f"a process failed: {codes}", file=sys.stderr)
    if args.expect == "refused":
        hit = tamper is not None and not (blocks[tamper[0]]["certifier"]["accepted"] and blocks[tamper[0]]["n3"]["accepted"])
        print(f"run {'REFUSED' if not accepted else 'ACCEPTED'}; the tampered block {'is' if hit else 'is not'} refused")
        return 0 if clean and hit else 1
    print(f"run {'ACCEPTED' if accepted else 'REFUSED'}")
    return 0 if clean and accepted else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="serve the workload on the lab (or locally) and check it")
    r.set_defaults(fn=run)
    r.add_argument("--out", required=True)
    r.add_argument("--workload", choices=M.WORKLOADS, default="served")
    r.add_argument("--layers", type=int, default=4)
    r.add_argument("--prompt", type=int, default=2, help="served: prompt tokens")
    r.add_argument("--steps", type=int, default=1, help="served: greedy decode steps")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tamper", metavar="BLOCK:MODE", help=f"MODE one of {', '.join(TAMPERS)}")
    r.add_argument("--expect", choices=("accepted", "refused"), default="accepted")
    r.add_argument("--delta-buckets", type=int, help="the first-frame latency D (default: from the sessions' timing)")
    r.add_argument("--local", action="store_true", help="every process on loopback addresses, no namespaces")
    r.add_argument("--declaration", metavar="FILE.py:NAME", help="the lab's declaration, as mockdc.py takes it")
    r.add_argument("--blocks", type=int, default=4, help="the lab's blocks, as mockdc.py takes them")
    r.add_argument("--leaves", type=int, default=2)
    r.add_argument("--spines", type=int, default=1)
    r.add_argument("--fabric-latency-us", type=int, default=2_000)
    r.add_argument("--fabric-jitter-us", type=int, default=500)
    q = sub.add_parser("role")
    q.set_defaults(fn=role)
    q.add_argument("name", choices=sorted(ROLES))
    q.add_argument("config")
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
