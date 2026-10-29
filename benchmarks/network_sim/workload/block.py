"""A block's process: evaluate the gates placed in the block tick by tick, and send and receive crossing values as framed
streams. The client is the same process for `OUTSIDE`: it holds the requests and reads the answers.

**What a block derives** (`Context`, from the run's config, so every process of one run derives the same): the Program
(`config["workload"]`: the MLP, or verity_vllm's serving Program, `served`), the hardware (`config["hardware"]`, a
`DeviceGraph`'s JSON, as `DeviceGraph.from_datacenter` reads a declaration, or `mock`), the placement (one engine step
every `step_ticks` ticks, one bucket by default) and φ. The prover runs every block, so the private placement is shared
among them. A block holds the values the placement puts at its leaves: the registered inputs from the weights' seed,
the client the run inputs from the requests' seed. Every block has every constant, an operation's own bits.

**The loop.** At each tick, in order: take every message read at this tick (waiting for its frames), evaluate this tick's
gates with the model's evaluator (verity's), then send every message whose last value this tick produced. A
read is always at a later tick than the value's (`placement.causal`), so the blocks never wait on each other in a cycle.
After its last tick a block closes its connections, waits until every connection into it is closed, and commits its values
(`values.commit`).

**Connections.** One per (source unit, destination unit), opened with φ's `HELLO` frame, then the frames of the messages in
send order. No frame names its message (`framing`), so `Inbox` cuts each connection's frames into its messages by φ: the
next message on the connection takes the next `framing.frame_count` frames. Frames no message of φ takes (one without the
canonical header, a frame after the connection's last message) stay in the inbox and go into the record at the end.

**Records.** A block records, on each of its certified boundary link directions, the frames that crossed it: what it sent on
egress, in send order, and what it took on ingress, in the order it took them, then what was left over. Each frame is tagged
with its bucket: that of the tick at which the message is read, at which the occupancy profile counts it (a `HELLO` takes
its connection's opener's, `framing.Phi.openers`, and left-over frames the block's last tick's). These logs stand in for the certifiers' records; under certifiers the bucket
comes from each certifier's clock.

**Transports.** `MemoryNet` runs every unit in one event loop and hands frames over directly; `serve_tcp` runs one unit as an
asyncio TCP server and client, one process per unit (`python -m workload.block`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from functools import cached_property

from verity.core.protocols.certifier.active import HEADER

from . import framing as F
from . import model as M
from . import placement as P
from . import values as V
from .hardware import OUTSIDE, DeviceGraph, mock


class Context:
    """What every process of a run derives from its config."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.workload = config.get("workload", "mlp")
        self.model = M.build(self.workload, config["spec"])
        self.spec = self.model.spec
        if "hardware" in config:
            self.hw = DeviceGraph.from_json(config["hardware"])
        else:
            self.hw = mock(config["blocks"], config.get("devices", 1), bucket_ticks=config.get("bucket_ticks", 1))
        self.placement = P.place(self.model, self.hw, P.PLANS[config.get("plan", "pipeline")](self.model, self.hw),
                                 config.get("step_ticks", self.hw.bucket_ticks))
        bad = P.causal(self.model, self.placement, self.hw)
        if bad:
            raise ValueError(f"the placement is not causal: gate {bad[0][0]} reads gate {bad[0][1]} too early")
        self.edges = P.crossings(self.model, self.placement, self.hw)
        self.phi = F.phi(self.hw, self.edges, config["frame_bytes"])
        self.seed = config.get("seed", 0)

    @cached_property
    def inputs(self) -> dict[int, int]:
        return self.model.inputs(self.seed)

    @cached_property
    def constants(self) -> dict[int, int]:
        return self.model.constants()

    def held(self, unit: str) -> list[int]:
        return self.placement.at(self.hw.leaves_of(unit))

    def units(self) -> list[str]:
        return self.hw.unit_names()


class Inbox:
    """The frames arriving on every connection into one unit, cut into φ's messages."""

    def __init__(self, ctx: Context, unit: str, senders: set[str]) -> None:
        self.ctx, self.unit = ctx, unit
        self.waiting = set(senders)
        self.closed = asyncio.Event()
        if not self.waiting:
            self.closed.set()
        self.hellos: dict[str, bytes] = {}
        self.pending = {src: ctx.phi.connection(src, unit) for src in senders}
        self.partial: dict[int, list[tuple[str, bytes]]] = defaultdict(list)
        self.complete: dict[int, list[tuple[str, bytes]]] = {}
        self.ready: dict[int, asyncio.Event] = defaultdict(asyncio.Event)
        self.stray: list[tuple[str, bytes]] = []

    def connection(self) -> "Sink":
        return Sink(self)

    def add(self, src: str, frame: bytes) -> None:
        """The frame is the next of the connection's current message, unless it has no canonical header or the
        connection has no message left."""
        B = self.ctx.phi.frame_bytes
        pending = self.pending.get(src, [])
        if len(frame) != B or frame[:HEADER.size] != F.head(B) or not pending:
            self.stray.append((src, frame))
            return
        m = pending[0]
        self.partial[m.stream].append((src, frame))
        if len(self.partial[m.stream]) == F.frame_count(m, B):
            pending.pop(0)
            self.complete[m.stream] = self.partial.pop(m.stream)
            self.ready[m.stream].set()

    def eof(self, src: str | None) -> None:
        self.waiting.discard(src)
        if not self.waiting:
            self.closed.set()

    async def take(self, stream: int) -> list[tuple[str, bytes]]:
        await self.ready[stream].wait()
        return self.complete.pop(stream)

    def leftovers(self) -> list[tuple[str, bytes]]:
        """Every frame no message took: strays, streams never taken, streams never ended."""
        out = list(self.stray)
        for stream in sorted(self.complete):
            out += self.complete[stream]
        for stream in sorted(self.partial):
            out += self.partial[stream]
        return out


class Sink:
    """One connection into a unit: its first frame names the sender."""

    def __init__(self, inbox: Inbox) -> None:
        self.inbox = inbox
        self.src: str | None = None

    def feed(self, frame: bytes) -> None:
        if self.src is None:
            ends = F.read_hello(self.inbox.ctx.hw, frame)
            if ends is None or ends[1] != self.inbox.unit:
                raise ConnectionError(f"a connection into {self.inbox.unit} didn't open with its HELLO")
            self.src = ends[0]
            self.inbox.hellos[self.src] = frame
            return
        self.inbox.add(self.src, frame)

    def eof(self) -> None:
        self.inbox.eof(self.src)


class Block:
    def __init__(self, ctx: Context, unit: str, connect) -> None:
        self.ctx, self.unit, self.connect = ctx, unit, connect
        hw, phi = ctx.hw, ctx.phi
        self.inbound: dict[int, list[F.Message]] = defaultdict(list)
        self.outbound: dict[int, list[F.Message]] = defaultdict(list)
        for m in phi.messages:
            if m.dst == unit:
                self.inbound[m.read_tick].append(m)
            if m.src == unit:
                self.outbound[m.ready_tick].append(m)
        self.openers = phi.openers()
        self.inbox = Inbox(ctx, unit, {m.src for m in phi.messages if m.dst == unit})
        self.held = ctx.held(unit)
        held = set(self.held)
        self.transcript = {g: v for g, v in ctx.inputs.items() if g in held}
        self.transcript.update(ctx.constants)
        self.compute: dict[int, list[int]] = defaultdict(list)
        ticks, kind = ctx.placement.tick_of, ctx.model.kind
        for g in self.held:
            if kind[g] == "computed":
                self.compute[ticks[g]].append(g)
        self.mine = set() if unit == OUTSIDE else {hw.links[e].name for e in hw.boundary(unit)}
        self.records: dict[str, list[tuple[int, bytes]]] = defaultdict(list)
        self.writers: dict[str, object] = {}
        self.taken: set[str] = set()
        self.last_tick = 0

    def _keys(self, m: F.Message) -> list[str]:
        return [k for k in self.ctx.phi.route(m) if k.split(">")[0] in self.mine]

    def _record(self, m: F.Message, frames: list[bytes], hello: bytes | None = None) -> None:
        hw = self.ctx.hw
        b = hw.bucket(m.read_tick)
        for k in self._keys(m):
            if hello is not None:
                self.records[k].append((hw.bucket(self.openers[(m.src, m.dst)].read_tick), hello))
            self.records[k] += [(b, f) for f in frames]

    async def _send(self, m: F.Message) -> None:
        B = self.ctx.phi.frame_bytes
        frames = self.ctx.phi.frames(m, self.transcript)
        hello = None
        if m.dst not in self.writers:
            self.writers[m.dst] = await self.connect(self.unit, m.dst)
            hello = F.hello(self.ctx.hw, self.unit, m.dst, B)
        self._record(m, frames, hello)
        w = self.writers[m.dst]
        for f in ([hello] if hello else []) + frames:
            await w.write(f)

    async def _take(self, m: F.Message) -> None:
        got = await self.inbox.take(m.stream)
        if any(src != m.src for src, _ in got):
            raise ConnectionError(f"stream {m.stream} arrived from another unit than {m.src}")
        hello = None
        if m.src not in self.taken:
            self.taken.add(m.src)
            hello = self.inbox.hellos[m.src]
        self._record(m, [f for _, f in got], hello)
        payload = b"".join(f[HEADER.size:] for _, f in got)[:m.nbytes]
        self.transcript.update(F.decode(m, payload))

    async def run(self) -> dict:
        ticks = sorted(set(self.compute) | set(self.inbound) | set(self.outbound))
        for t in ticks:
            self.last_tick = t
            for m in self.inbound.get(t, []):
                await self._take(m)
            if self.compute.get(t):
                self.ctx.model.evaluate(self.transcript, self.compute[t])
            for m in self.outbound.get(t, []):
                await self._send(m)
        for w in self.writers.values():
            await w.close()
        await self.inbox.closed.wait()
        self._leftovers()
        out = {"unit": self.unit,
               "records": {k: [[b, f.hex()] for b, f in fs] for k, fs in sorted(self.records.items())},
               "values": None}
        if self.held:
            c = V.commit(self.unit, self.held, self.transcript, self.ctx.model.value_bits)
            out["values"] = {"root": c.root.hex(), "count": len(c.gates),
                             "rows": [r.hex() for r in c.registration.rows],
                             "salts": [s.hex() for s in c.registration.salts]}
        if self.unit == OUTSIDE:
            out["answers"] = self.ctx.model.output_words(self.transcript)
        return out

    def _leftovers(self) -> None:
        """Frames no message took, on the boundary link direction into this unit (the one the client's route enters by)."""
        hw = self.ctx.hw
        b = hw.bucket(self.last_tick)
        into = [hw.key(e, to) for e, to in hw.watched(hw.client, min(hw.leaves_of(self.unit)))
                if hw.links[e].name in self.mine]
        for _src, f in self.inbox.leftovers():
            for k in into:
                self.records[k].append((b, f))


class _MemoryWriter:
    def __init__(self, sink: Sink) -> None:
        self.sink = sink

    async def write(self, frame: bytes) -> None:
        self.sink.feed(frame)

    async def close(self) -> None:
        self.sink.eof()


class MemoryNet:
    """Every unit of a run in one event loop, frames handed over directly."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.blocks = {u: Block(ctx, u, self._connect) for u in ctx.units()}

    async def _connect(self, src: str, dst: str) -> _MemoryWriter:
        return _MemoryWriter(self.blocks[dst].inbox.connection())

    async def run(self) -> dict[str, dict]:
        results = await asyncio.gather(*(b.run() for b in self.blocks.values()))
        return {r["unit"]: r for r in results}


class _TcpWriter:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer

    async def write(self, frame: bytes) -> None:
        self.writer.write(frame)
        await self.writer.drain()

    async def close(self) -> None:
        self.writer.write_eof()
        await self.writer.drain()
        self.writer.close()
        await self.writer.wait_closed()


async def serve_tcp(ctx: Context, unit: str, addresses: dict[str, tuple[str, int]], ready, go: asyncio.Event) -> dict:
    """Run one unit over TCP: listen at its address, call `ready()` once listening, and start when `go` is set."""
    B = ctx.phi.frame_bytes

    async def connect(src: str, dst: str) -> _TcpWriter:
        host, port = addresses[dst]
        _, writer = await asyncio.open_connection(host, port)
        return _TcpWriter(writer)

    block = Block(ctx, unit, connect)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sink = block.inbox.connection()
        try:
            while True:
                try:
                    frame = await reader.readexactly(B)
                except asyncio.IncompleteReadError as e:
                    if e.partial:
                        block.inbox.stray.append((sink.src or "?", e.partial))
                    break
                sink.feed(frame)
        finally:
            sink.eof()
            writer.close()

    host, port = addresses[unit]
    server = await asyncio.start_server(handle, host, port)
    async with server:
        ready()
        await go.wait()
        return await block.run()


def _main() -> int:
    ap = argparse.ArgumentParser(description="Run one unit of a simulated datacenter over TCP.")
    ap.add_argument("--config", required=True, help="the run's config JSON")
    ap.add_argument("--unit", required=True, help="the unit to run: a block's name, or 'outside' for the client")
    ap.add_argument("--out", required=True, help="where to write the unit's result JSON")
    args = ap.parse_args()
    config = json.loads(open(args.config).read())
    ctx = Context(config)
    addresses = {u: (h, p) for u, (h, p) in config["addresses"].items()}

    async def main() -> dict:
        go = asyncio.Event()
        loop = asyncio.get_running_loop()

        def ready() -> None:
            print("ready", flush=True)
            loop.run_in_executor(None, sys.stdin.readline).add_done_callback(lambda _f: go.set())

        return await serve_tcp(ctx, args.unit, addresses, ready, go)

    result = asyncio.run(main())
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
