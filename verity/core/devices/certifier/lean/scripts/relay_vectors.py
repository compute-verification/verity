"""The relay difftest's data: connections the relay form of the network certifier serves, and what `warden.Warden` does
with each, for `CertifierDevice/RelayDifftest.lean`.

    python scripts/relay_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/relay/instances.jsonl`, one instance per line as `CertifierDevice.RelayDifftest.evaluate` reads it: the
relay's parameters (`WardenParams`, as `NetworkCheck.Period.scheduleOf` reads them), its windows, and each connection in
arrival order as `[ready, last]`, beside the answer its server gives (which the Lean side doesn't read); and
`DIR/relay/python.jsonl`, the same line's decisions from the checkout's real `Warden`, driven tick by tick on a
`VirtualClock`: `"refused"` (its `REFUSED` cell), `"cut"` (queued when the run ended) or `[admitted, cells]`, the cells
its lane sent with its `END` or `FAIL`.

Each connection is a client whose request is one `END` cell naming it, connected just before the tick `ready`, so that
tick is the first to find it queued, and an in-memory server that answers it with chunks of bytes, each a number of
buckets after the request reached it, and then its end of stream, or none. Its `last` is the index of its lane's `END`
or `FAIL` in a run of its own with no cap and no cut (one lane, `egress_cells` past any answer, and windows past its
end): a lane's cells are timed from its admission, so that is the index the run under test reaches were the lane
neither capped nor cut. The parameters are small (one to four slots, one to three lanes, a queue of one to eight, one
to three windows of six to sixteen buckets, caps of one to forty cells) and the arrivals come in bursts, so that runs
refuse connections, hold them queued past their `ready`, cap lanes at `egress_cells`, fail them late, and cut them
queued, in their delay and while sending. The Lean audit runs `CertifierDevice/RelayDifftestMain.lean` on both files
(`runs` in `lean-audit.json`, which pins their SHA-256), and `verity/core/protocols/certifier/tests/
test_network_certifier_relay_lean_difftest.py` checks that this script still writes them.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
from pathlib import Path

import verity.core.protocols.certifier
from verity.core.protocols.certifier.active import VirtualClock
from verity.core.protocols.certifier.warden import REFUSED, Warden, WardenParams, cells, read_cell

COUNT = 300
#: a server's answer is a list of (buckets after the request reached it, bytes); `EOF` in place of the bytes ends it
EOF = None
SHAPES = ["whole", "chunks", "late", "open", "long"]


def request(i: int) -> bytes:
    return b"req#%d\n" % i


class Upstream:
    """One server connection: once its request arrives, it answers with that connection's answer, each chunk fed by
    `release(t)` once due."""

    def __init__(self, answers: list[list]) -> None:
        self.answers, self.reader, self.received = answers, asyncio.StreamReader(), bytearray()
        self.answer: list | None = None
        self.at: int | None = None  # the bucket its request reached it in
        self.fed, self.eof, self.closed = 0, False, False

    def write(self, data: bytes) -> None:
        self.received += data
        if self.at is None and (m := re.search(rb"req#(\d+)\n", bytes(self.received))):
            self.answer, self.at = list(self.answers[int(m.group(1))]), -1

    def release(self, t: int) -> None:
        if self.at == -1:
            self.at = t - 1
        while self.at is not None and self.answer and self.at + self.answer[0][0] <= t and not self.closed:
            _, data = self.answer.pop(0)
            if data is EOF:
                self.eof = True
                self.reader.feed_eof()
            else:
                self.fed += len(data)
                self.reader.feed_data(data)

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


class Server:
    def __init__(self, answers: list[list]) -> None:
        self.answers, self.conns = answers, []

    async def connect(self):
        conn = Upstream(self.answers)
        self.conns.append(conn)
        return conn.reader, conn

    def release(self, t: int) -> None:
        for conn in self.conns:
            conn.release(t)


class Client:
    """One client connection: its request's cells in its reader, and every write the relay makes to it."""

    def __init__(self, warden: Warden, data: bytes) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self.transport = self
        self.reader = asyncio.StreamReader()
        self.reader.feed_data(data)
        self.reader.feed_eof()
        self.task = asyncio.create_task(warden.handle(self.reader, self))

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def settled(warden: Warden, server: Server) -> bool:
    """Every live connection's client cells read, its server connected, and what its server has sent read."""
    cap = warden.params.egress_cells * warden.params.payload
    for c in warden.active:
        if c.server is None and c.upstream is None:
            return False
        if c.problem is None and not c.ingress_ended and c.reader._buffer:
            return False
        if c.server is not None:
            conn = c.server[1]
            if c.bytes_out <= cap and (c.bytes_out != conn.fed or c.egress_ended != conn.eof):
                return False
    return True


async def drive(params: WardenParams, windows: int, arrivals: dict[int, list[int]], answers: list[list]):
    """Connect `arrivals[t]` (connection ids) just before tick t, let the server answer what is due and the relay read
    it, then tick, for `windows` windows."""
    tau = params.bucket_ms / 1000
    clock = VirtualClock(-tau / 2)
    server = Server(answers)
    warden = Warden(params, server.connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep,
                    run_nonce=bytes)
    running = asyncio.create_task(warden.run())
    clients: list[Client] = []
    for t in range(windows * params.window):
        for i in arrivals.get(t, ()):
            clients.append(Client(warden, cells(request(i), params.cell_bytes)))
            n = len(clients)
            await warden.until(lambda w, n=n: w.ordinals + w.refused == n)
        server.release(t)
        await warden.until(lambda w: settled(w, server))
        clock.advance(t * tau)
        await warden.until(lambda w, t=t: w.ticks > t)
    await running
    await asyncio.gather(*(c.task for c in clients))
    return warden, clients


def decisions(warden: Warden, clients: list[Client]) -> list:
    """Each client's decision, in arrival order: refused, cut queued, or its admission tick and its lane's cells."""
    log = {e["ordinal"]: e for e in warden.log}
    out, ordinal = [], 0
    for c in clients:
        if c.writes and read_cell(c.writes[0][:warden.params.cell_bytes], warden.params.cell_bytes)[0] == REFUSED:
            out.append("refused")
            continue
        ordinal += 1
        e = log[ordinal]
        out.append("cut" if e["admitted"] is None else [e["admitted"], e["egress_cells"] + (e["failed"] is not None)])
    return out


def last_of(params: WardenParams, answer: list) -> int:
    """The index of the lane's `END` or `FAIL` on `answer`, in a run of one connection with no cap and no cut."""
    sent = sum(len(d) for _, d in answer if d is not EOF)
    span = params.delay + max((b for b, _ in answer), default=0) + sent // params.payload + 8
    free = dataclasses.replace(params, lanes=1, queue=1, egress_cells=sent // params.payload + span)
    warden, _ = asyncio.run(drive(free, -(-span // params.window), {0: [0]}, [answer]))
    (e,) = warden.log
    if e["cut"] or e["admitted"] != 0:
        raise AssertionError(f"the lane on {answer} didn't end in its own run: {e}")
    return e["failed_at"] if e["failed"] is not None else e["egress_cells"] - 1


def answer_of(rng: random.Random, params: WardenParams) -> tuple[str, list]:
    """A server's answer, of one of `SHAPES`: whole at once, in chunks just in time, its first byte late, open (no end
    of stream), or long (past most caps)."""
    p, r, D = params.payload, params.slots, params.delay
    shape = rng.choice(SHAPES)
    body = rng.randbytes
    if shape == "whole":
        return shape, [(0, body(rng.randint(0, 6 * p))), (0, EOF)]
    if shape == "chunks":
        out, at = [], 0
        for _ in range(rng.randint(1, 5)):
            at += rng.randint(0, 2)
            out.append((at, body(rng.randint(1, 2 * r * p))))
        return shape, out + [(at + rng.randint(0, 2), EOF)]
    if shape == "late":
        return shape, [(D + rng.randint(1, 4), body(rng.randint(1, 3 * p))), (D + 6, EOF)]
    if shape == "open":
        return shape, [(0, body(rng.randint(0, 4 * p)))]
    return shape, [(0, body(rng.randint(20, 60) * p)), (rng.randint(0, 3), EOF)]


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    params = WardenParams(cell_bytes=64, slots=rng.randint(1, 4), bucket_ms=100, delay=rng.randint(1, 5),
                          lanes=rng.choice([1, 1, 2, 2, 3]), queue=rng.choice([1, 2, 3, 8]), window=rng.randint(6, 16),
                          ingress_cells=8, ingress_buckets=64, egress_cells=rng.choice([1, 2, 3, 5, 8, 12, 40]))
    windows = rng.randint(1, 3)
    horizon = windows * params.window
    ready: list[int] = []
    for _ in range(rng.randint(1, 4)):
        at = rng.choice([0, rng.randrange(horizon), horizon - 1 - rng.randrange(min(horizon, params.delay + 2))])
        ready += [at] * rng.randint(1, 4)
    ready.sort()
    shapes, answers = zip(*(answer_of(rng, params) for _ in ready))
    conns = [[t, last_of(params, a)] for t, a in zip(ready, answers)]
    return {"seed": seed, "schedule": dataclasses.asdict(params), "windows": windows, "conns": conns,
            "answers": [[shape, [[b, None if d is EOF else len(d)] for b, d in a]] for shape, a in zip(shapes, answers)],
            "_answers": [[[b, None if d is EOF else d.hex()] for b, d in a] for a in answers]}


def python_results(inst: dict) -> dict:
    params = WardenParams(**inst["schedule"])
    answers = [[(b, EOF if d is None else bytes.fromhex(d)) for b, d in a] for a in inst["_answers"]]
    arrivals: dict[int, list[int]] = {}
    for i, (t, _) in enumerate(inst["conns"]):
        arrivals.setdefault(t, []).append(i)
    warden, clients = asyncio.run(drive(params, inst["windows"], arrivals, answers))
    return {"decisions": decisions(warden, clients)}


def line(obj: dict) -> str:
    return json.dumps({k: v for k, v in obj.items() if not k.startswith("_")}, separators=(",", ":"),
                      sort_keys=True) + "\n"


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """Both files for instances 0 … count − 1 under ``out/relay``."""
    d = out / "relay"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"relay/instances.jsonl": d / "instances.jsonl", "relay/python.jsonl": d / "python.jsonl"}
    with open(paths["relay/instances.jsonl"], "w") as fi, open(paths["relay/python.jsonl"], "w") as fp:
        for seed in range(count):
            inst = instance(seed)
            fi.write(line(inst))
            fp.write(line(python_results(inst)))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.core.protocols.certifier is tested")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.core.protocols.certifier.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.core.protocols.certifier comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
