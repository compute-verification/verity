"""Serve a workload on the mock datacenter and check it: each block's server in its network namespace, behind the network
certifier's proxy on its uplink, and the client on the outside host, with the lab's Program (`lab.Lab`) on every link;
then the certifier's audit, N2's served verdict on each link-window and the answers.

    python -m workload.lab_run run --out DIR [--layers 4] [--tamper b2:late --expect refused] [--local]

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
  `active_replay.Peer`'s digest of each window, and takes the k-th event on a link for position k, the order the
  certifier's shaper hands in. It relays each message once its last chunk is in, and sends each block's requests in
  session order as soon as their messages are in (`active.request_frames`), the first ones half a bucket after the
  origin. At the end
  it commits the run inputs and reads the answers (`outside.json`).

**Calibration**, before the processes start, as a trace is calibrated. The orchestrator evaluates the Program session by
session to time it, and sets the first-frame latency D, in buckets, to four times the slowest session plus half a second
(`--delta-buckets` sets it). It models the run (a request is released in the bucket after it is sent, and its response is
at the client D + 1 buckets later) and calibrates the certifier on the model (`calibration.calibrate`, one configuration
per block): the egress allocation, Σ_sync with δ up to D, and the ingress allocation. Every window declares the envelope,
the largest δ and the smallest ρ declared, so a response is on time whenever it is ready within D buckets of its
request's release. The run lasts the windows that cover the model's last response with 2D + 20 buckets to spare.

**Served.** Every block's link is a served link (`plan["served"]`, the block's (F, F_in), `lab.Lab.fixlen`): its
responses are padded to F events with empty ones, and its requests to F_in frames, so its egress counts derive from its
ingress counts. Its proxy admits a request only while the request's window, with it, still ends idle under the
envelope (`served.Admission`, `admissions`), and holds it for the next window otherwise. That moves the model's
releases, so the calibration is repeated on the admitted model until it gives the envelope and allocations it was
modeled under (`calibrate_run`).

**Checks.** The certifier's audit (`active_replay.audit_configuration` with `audit.Audit`) on each block's raw and committed
records, in its live mode, whose Program is the block's event log anchored at each request's release as the ingress
records show it: the timing. N2's served verdict on each block's link (`n2_verdicts`): each link-window's published
records, their heads signed by a stand-in certifier (the lab's proxy signs nothing), through the proof service's stream
on live coins and N2's verifier of record, so the counts only. And the answers against the Program evaluated in one
process.

**Tampering** (`--tamper BLOCK:MODE`): `late` holds the block's first response past every δ of Σ_sync, which the
certifier refuses (`coverage-never-established`) and N2 refuses on its window; `extra` writes one more empty event after
the block's last response, which N2 refuses on its window, since its egress counts aren't the ones its ingress counts
derive. A block can't write into a frame's fill: the proxy frames its events.

**Outputs** in `--out`: `report.json` (the calibration); per block `BLOCK/`, with `plan.json`, the proxy's records and log,
the block's `server.json`, `server.events.gz` and `result.json`, the client's `peer.json` and the certifier's `audit.json`;
`outside.json`, `n2.json` (N2's served verdict on each block's link) and `summary.json`: per block the certifier's
verdicts and timing and N2's, and the answers against the reference. The exit status is 0 when every process exited
cleanly and the run is what `--expect` says: `accepted` (every block's certifier audit accepts, N2 accepts every window
and the answers are the reference's) or `refused` (the tampered block's certifier audit or N2 refuses).
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
from fractions import Fraction
from pathlib import Path

import mockdc  # puts network_traces, datacenter and the repository root on sys.path, for the next three

import active_replay as ar
import graph
import records
from datacenter import Cadence
from verity.core.protocols.certifier import calibration
from verity.core.protocols.certifier import certificate as CT
from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, PAD, decode_frame, event_data, \
    request_frames
from verity.core.protocols.certifier.active import event as canonical_event
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.certificate import Direction, parameters_digest, record_head
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.certifier.served import Admission, ServedProfile, derive
from verity.core.protocols.certifier.service import LEAVES, LinkWindows
from verity.ml.commitments import hm96
from verity.ml.crypto import lms

from . import framing as F
from . import lab as L
from . import model as M
from . import values as V
from .block import Context
from .hardware import OUTSIDE, Hardware

BUCKET_MS, WINDOW_BUCKETS, FRAME_BYTES, QUEUE = 100, 60, 4096, 1 << 16
FABRIC_MS = 10  # the model's delay between a proxy and the client, each way
PATHS = [*mockdc.PATHS, str(mockdc.ROOT / "integrations" / "vllm")]
TAMPERS = ("late", "extra")


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


def _idle(anchors: list[int], admission: Admission, window: int) -> bool:
    counts = [0] * WINDOW_BUCKETS
    for a in anchors:
        counts[a] += admission.profile.request_frames
    return derive(counts, admission.sync(window), admission.profile, admission.allocation) is not None


def model_run(lab: L.Lab, D: int, admission: dict[str, Admission] | None = None) -> dict[str, list[tuple[float, int]]]:
    """Each block's requests in the model, as (send time in ms, the bucket it is released in) per session: the first ones
    half a bucket after the origin, each released in the bucket after it is sent, and its response at the client
    `FABRIC_MS` after bucket release + D + 1 starts. A request goes as soon as its messages are in and the block's
    previous one has gone. With `admission`, each block's proxy admits a request only while its window, with it, still
    ends idle (`served.admits`): one it doesn't is released at the next window's first bucket."""
    tau, T = BUCKET_MS, WINDOW_BUCKETS
    ready = {m.stream: 0.0 for m in lab.messages.values() if m.src == OUTSIDE}
    sent: dict[str, list[tuple[float, int]]] = {b: [] for b in lab.blocks}
    anchored: dict[tuple[str, int], list[int]] = defaultdict(list)
    moved = True
    while moved:
        moved = False
        for b in lab.blocks:
            for s in lab.sessions[b][len(sent[b]):]:
                if any(m.stream not in ready for m in s.inbound):
                    break
                t = max([tau / 2, *(ready[m.stream] + 1 for m in s.inbound), *(x for x, _ in sent[b][-1:])])
                release = math.ceil(t / tau)
                while admission and not _idle(anchored[b, release // T] + [release % T], admission[b], release // T):
                    if release % T == 0 and not anchored[b, release // T]:
                        raise ValueError(f"{b}: a request and its response of {admission[b].profile.response_frames} "
                                         f"frames don't fit one window of {T} buckets")
                    release += T - release % T
                anchored[b, release // T].append(release % T)
                sent[b].append((t, release))
                ready.update({m.stream: (release + D + 1) * tau + FABRIC_MS for m in s.outbound})
                moved = True
    if stuck := [s.name for b in lab.blocks for s in lab.sessions[b][len(sent[b]):]]:
        raise ValueError(f"sessions {stuck} wait on each other")
    return sent


def admissions(lab: L.Lab, setup: ar.Setup) -> dict[str, Admission]:
    """Each block's admission rule (`served.Admission`) under the envelope every window declares."""
    return {b: Admission(ServedProfile(*lab.fixlen[b], 1), setup.egress_params.allocation, lambda w: setup.envelope)
            for b in lab.blocks}


def _calibrate(lab: L.Lab, D: int, name: str, admission: dict[str, Admission] | None) -> tuple[dict, dict, ar.Setup]:
    sent = model_run(lab, D, admission)
    configs = []
    for b in lab.blocks:
        frames, arrival = [], []
        for s, (t, release) in zip(lab.sessions[b], sent[b], strict=True):
            frames += [{"session": s.name, "index": k, "anchor": release, "ready": release + D}
                       for k in range(len(lab.positions(b, s.index)))]
            arrival += [t + FABRIC_MS / 2] * lab.request_frames(s)
        configs.append({"name": f"{name}/{b}", "egress": [{"link": L.EGRESS, "frames": frames}],
                        "ingress": [{"link": L.INGRESS, "arrival_ms": arrival}]})
    report = calibration.calibrate({"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS, "frame_bytes": FRAME_BYTES,
                                    "queue_capacity": QUEUE, "configurations": configs})
    return sent, report, ar.Setup(report)


def calibrate_run(lab: L.Lab, D: int, name: str) -> tuple[dict, ar.Setup, dict[str, dict]]:
    """The calibration report on the modeled run, one configuration per block, and each block's plan for the proxy and
    the audit. Each block's link is served (`plan["served"]`): its proxy admits requests by the admission rule
    (`admissions`), which moves the model's releases, so the calibration is repeated on the admitted model until the
    envelope and the allocations it gives are the ones it was modeled under. Every request of the exact model
    (`active_replay.model_releases`) must then be admitted within the run."""
    T, admission, key = WINDOW_BUCKETS, None, None
    for _ in range(8):
        sent, report, setup = _calibrate(lab, D, name, admission)
        settled = (setup.envelope, setup.egress_params.allocation, setup.ingress_params.allocation)
        if settled == key:
            break
        admission, key = admissions(lab, setup), settled
    else:
        raise ValueError(f"the calibration on the admitted model doesn't settle: (envelope, r, r_in) {key}")
    envelope, sync_set = setup.envelope, setup.sync_set
    if envelope.delta < D:
        raise ValueError(f"the calibration's envelope δ = {envelope.delta} is below the model's D = {D}")
    end = max(release + D + 1 + len(lab.events[b]) for b in lab.blocks for _, release in sent[b])
    windows = math.ceil((end + 2 * D + 20) / T)
    eager = ClockSync(min(sync_set.deltas), max(sync_set.rhos))
    plans = {}
    for b in lab.blocks:
        F, FIN = lab.fixlen[b]
        sessions = [[s.name, s.stream, [D] * len(lab.positions(b, s.index))] for s in lab.sessions[b]]
        requests = [{"stream": s.stream, "session": s.name, "frames": lab.request_frames(s), "trace_ms": t,
                     "enqueue": math.ceil(t / setup.ingress_params.bucket_ms), "send_ms": t}
                    for s, (t, _) in zip(lab.sessions[b], sent[b], strict=True)]
        release = ar.model_releases(requests, setup.ingress_params, windows, setup.egress_params.bucket_ms,
                                    admission[b])
        if len(release) < len(requests):
            raise ValueError(f"{b}: the admission rule admits {len(release)} of its {len(requests)} requests within "
                             f"the run's {windows} windows")
        chosen, _, misses = calibration.choose_syncs([ar.model_frames(sessions, release, windows * T)], sync_set, T)
        chosen = (list(chosen) + [eager] * windows)[:windows]
        plans[b] = {"name": f"{name}/{b}", "windows": windows, "sessions": sessions, "requests": requests,
                    "model_release": {str(k): v for k, v in release.items()}, "model_misses": misses,
                    "syncs": [[envelope.delta, str(envelope.rho)]] * windows, "sync_policy": "envelope",
                    "model_syncs": [[s.delta, str(s.rho)] for s in chosen], "live_model": name, "seed": 0,
                    "served": {"response_frames": F, "request_frames": FIN}}
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
            at = self.lab.request_pad(s)
            for m in s.inbound:
                n = (m.bits + 7) // 8
                self.values.update(F.decode(m, body[at:at + n]))
                at += n
            t0 = time.monotonic()
            await asyncio.to_thread(self.ctx.model.evaluate, self.values, s.gates)
            self.log["compute_ms"].append(round((time.monotonic() - t0) * 1000, 3))
            payloads = {m.stream: F.encode(m, self.values) for m in s.outbound}
            if self.tamper == "late" and s.outbound and not tampered:
                tampered = True
                await asyncio.sleep(max(0.0, origin + received + self.hold * self.tau - time.monotonic()))
            positions = self.lab.positions(self.block, s.index)
            events = [self.lab.event(self.block, p, self.lab.payload(self.block, p, payloads)) for p in positions]
            if self.tamper == "extra" and s is self.sessions[-1]:
                positions = [*positions, len(self.lab.events[self.block])]
                events.append(b"id: %d\n" % positions[-1] + canonical_event(b""))
            for k, (p, event) in enumerate(zip(positions, events)):
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
        self.position = dict.fromkeys(lab.blocks, 0)
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
        """A link frame: the block's next event, the certifier's shaper handing them in position order, a padding
        event carrying no message."""
        kind, stream, body = decode_frame(frame)
        if kind == PAD:
            return
        p, self.position[block] = self.position[block], self.position[block] + 1
        chunk = event_data(body) if (kind, stream) == (DATA, CANONICAL_STREAM) else None
        if chunk is None or p >= len(self.lab.events[block]):
            self.strays.append(f"{block}: frame {p} is no event of the Program")
            return
        _, m, j = self.lab.events[block][p]
        if m is None:
            return
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
            writer.write(request_frames(self.lab.request(s, self.payloads), self.B))
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


def _seeded(seed: bytes, n: int) -> bytes:
    return hashlib.shake_256(seed).digest(n)


def n2_verdicts(lab: L.Lab, setup: ar.Setup, plans: dict[str, dict], published: dict[str, list], run: str) -> dict:
    """N2's served verdict on each block's link (`service.LinkWindows` with the served profile of the block's (F, F_in)
    and the run's windows): each window's two records as the proxy published them, their heads signed under the block's
    name by a stand-in certifier (an LMS key of the deployed N = 64, W = 1 drawn from a seed, in a zero session: the
    lab's proxy signs nothing), committed, registered, drawn and delivered on the challenger's live coins, and checked by
    N2's verifier of record under the window's declared sync; then the link's network verdict, the AND over its windows.
    A window without both records ends the registrations, so it and every later window are refused. `served_counts` is
    the served counts' check alone on each window registered. The verdicts are computed after the run, on the host, with
    no proof to time: the stream's clock is held at 0, so no outcome is late."""
    windows = plans[lab.blocks[0]]["windows"]
    H = max(4, (2 * windows * len(lab.blocks) - 1).bit_length())
    key = lms.keygen(lms.Params(64, 1, H), hashlib.sha512(b"lab-n2/lms").digest())
    public, session, signed = CT.LmsKey(key.params, lms.public_key(key)), bytes(lms.SESSION_BYTES), [0]

    def certify(head: CT.RecordHead) -> CT.CertifiedRecord:
        n, signed[0] = signed[0], signed[0] + 1
        salt = _seeded(b"lab-n2/salt_c/" + n.to_bytes(8, "big"), CT.SALT_BYTES)
        digest = CT.certificate_digest(salt, head.digest())
        message = CT.certificate_message(public.key_id, session, head.link, head.direction, head.window, digest)
        signature = lms.sign_at(key, n, message).encode()
        return CT.CertifiedRecord(head, salt, CT.Certificate(public.key_id, head.link, head.direction, head.window,
                                                             digest, signature))

    ends = ((L.EGRESS, Direction.EGRESS, setup.egress_params, parameters_digest(setup.egress_params, setup.sync_set)),
            (L.INGRESS, Direction.INGRESS, setup.ingress_params, parameters_digest(setup.ingress_params)))
    out = {}
    for b in lab.blocks:
        lw = LinkWindows(b, setup.egress_params, setup.sync_set, public, run, session,
                         ingress=setup.ingress_params, served=ServedProfile(*lab.fixlen[b], windows),
                         proof_deadline=WINDOW_BUCKETS * BUCKET_MS / 1000)
        stream = lw.stream(budget=windows, delta=2.0 ** -40, challenger=f"{run}/{b}", clock=lambda: 0.0)
        syncs = [ClockSync(d, Fraction(rho)) for d, rho in plans[b]["syncs"]]
        records = {(r.link, r.window): r for r in published[b]}
        previous, counts = None, []
        for w in range(windows):
            if any((link, w) not in records for link, *_ in ends):
                break
            heads = [record_head(b, d, w, records[link, w].grid, records[link, w].status, g, digest)
                     for link, d, g, digest in ends]
            salts = [_seeded(f"lab-n2/{b}/{w}".encode() + i.to_bytes(8, "big"), hm96.SHA512.salt_bytes)
                     for i in range(LEAVES)]
            opening = lw.commit(w, *map(certify, heads), salts)
            reg = lw.register(stream, lw.registration(opening, previous))
            previous, draw = reg.digest, stream.select(reg)
            stream.deliver(draw, [lw.session(opening, draw)])
            lw.unproved_outcome(stream, draw, opening, syncs)
            counts.append(lw.served_check(opening, syncs))
        verdict = lw.network_verdict(stream)
        out[b] = {"accepted": verdict["accepted"], "windows": verdict["windows"], "refused": verdict["refused"],
                  "served_counts": counts, "profile": list(lab.fixlen[b]), "digest": verdict["digest"]}
    return out


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
          f"ρ = {setup.envelope.rho}, r = {setup.egress_params.allocation}, r_in = {setup.ingress_params.allocation}; "
          "served (F, F_in): " + ", ".join(f"{b} {tuple(plans[b]['served'].values())}" for b in lab.blocks),
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
    outside = out / "outside.json"
    answers = json.loads(outside.read_text())["answers"] if outside.exists() else None
    n2 = n2_verdicts(lab, setup, plans, published, name)
    (out / "n2.json").write_text(json.dumps(n2, indent=1))
    for b in lab.blocks:
        blocks[b]["n2"] = n2[b]
    accepted = all(x["certifier"]["accepted"] and x["n2"]["accepted"] for x in blocks.values()) and answers == reference
    summary = {"workload": args.workload, "spec": config["spec"], "blocks": blocks, "answers": answers, "reference": reference, "answers_equal": answers == reference, "accepted": accepted,
               "delta_buckets": D, "session_compute_ms": {k: round(v * 1000, 3) for k, v in compute.items()},
               "windows": plans[lab.blocks[0]]["windows"], "exit": codes, "local": args.local,
               "grid": {"bucket_ms": BUCKET_MS, "window_buckets": WINDOW_BUCKETS, "frame_bytes": FRAME_BYTES,
                        "allocation": setup.egress_params.allocation, "ingress_allocation": setup.ingress_params.allocation,
                        "envelope": [setup.envelope.delta, str(setup.envelope.rho)], "sync_set_size": setup.sync_set.size},
               "served": {b: plans[b]["served"] for b in lab.blocks}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    for b, x in blocks.items():
        failed = [f"{link}[{w}] {raw or com}" for link, w, raw, com in x["certifier"]["verdicts"] if raw or com]
        late = x["tick_late_ms_max"]
        print(f"{b}{' (' + x['tamper'] + ')' if x['tamper'] else ''} certifier audit "
              f"{'ACCEPTED' if x['certifier']['accepted'] else 'REFUSED (' + ', '.join(failed) + ')'}; "
              f"N2 served {'ACCEPTED' if x['n2']['accepted'] else 'REFUSED (windows ' + str(x['n2']['refused']) + ')'} "
              f"on {x['n2']['windows']} windows at F = {x['n2']['profile'][0]}, F_in = {x['n2']['profile'][1]}; "
              f"latest tick {max((v for v in late.values() if v is not None), default=0):.1f} ms late against "
              f"{x['slack_ms']:.0f} ms of slack, "
              f"{sum(x['ticks_over_slack'].values())} over")
    print(f"answers {answers} {'equal' if answers == reference else 'differ from'} the reference {reference}")
    clean = not any(codes.values())
    if not clean:
        print(f"a process failed: {codes}", file=sys.stderr)
    if args.expect == "refused":
        t = blocks[tamper[0]] if tamper is not None else None
        hit = t is not None and not (t["certifier"]["accepted"] and t["n2"]["accepted"])
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
