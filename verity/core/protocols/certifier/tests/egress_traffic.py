"""A fake upstream's egress served through the warden, and the slot state each egress bucket reads: the egress tests'
traffic (`test_network_certifier_egress.py`) and `benchmarks/network_traces/egress_zk.py`'s. It imports no test
framework, so a node's bare venv runs it."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from verity.core.protocols.certifier import egress as EG, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.grid import Certifier, BlockGrid

#: the token ids' range (GLM's vocabulary), and the bits an id takes
V = 151_329
IB = (V - 1).bit_length()


@dataclass(frozen=True)
class Served:
    """A link served through the warden's own egress (`warden.Warden._egress`): its schedule, the egress records its
    grid closed (one a link-window), each connection's request (ordinal q = k + 1, its admission bucket, its n
    outputs) and response ids, and the warden's `fails` frames."""

    params: W.WardenParams
    records: tuple[BlockGrid, ...]
    requests: tuple[EG.Request, ...]
    responses: tuple[tuple[int, ...], ...]
    fails: tuple[bytes, ...]


def serve(params: W.WardenParams, arrivals: Sequence[int], responses: Sequence[Sequence[int]], windows: int) -> Served:
    """Connection k, ordinal k + 1, arrives at bucket arrivals[k] with response responses[k] (its ids, each one id
    event of P bytes, then `TRAILER`, all there by their deadlines), and the warden serves the link for `windows`
    link-windows: admitted at the first bucket with a free lane in arrival order."""
    P = params.payload
    fails: list[bytes] = []
    warden = SimpleNamespace(params=params, fails=SimpleNamespace(enqueue=fails.append))
    conns = []
    for k, (a, xs) in enumerate(zip(arrivals, responses, strict=True)):
        c = W._Connection(ordinal=k + 1, reader=None, writer=None, done=None, arrived=a)
        c.egress = bytearray(b"".join(IT.id_event(x, P) for x in xs) + IT.TRAILER)
        c.egress_ended = True
        conns.append(c)
    queue, active, grid, records = deque(conns), [], Certifier("responses", params.grid), []
    for t in range(windows * params.window):
        while queue and queue[0].arrived <= t and len(active) < params.lanes:
            c = queue.popleft()
            c.admitted = t
            active.append(c)
        for c in active:
            for frame in W.Warden._egress(warden, c, t):
                grid.enqueue(frame)
        active = [c for c in active if c.ended is None]
        grid.tick()
        if (t + 1) % params.window == 0:
            records.append(grid.close_window())
    if queue or active:
        raise ValueError("every connection ends within the link's windows")
    requests = tuple(EG.Request(c.ordinal, c.admitted, len(xs)) for c, xs in zip(conns, responses))
    return Served(params, tuple(records), requests, tuple(tuple(xs) for xs in responses), tuple(fails))


def busy(seed: int, params: W.WardenParams, count: int, longest: int, windows: int) -> Served:
    """`count` connections arriving over the link's first half, responses of 0 … `longest` ids."""
    rng = random.Random(seed)
    arrivals = sorted(rng.randrange(max(1, windows * params.window // 2)) for _ in range(count))
    responses = [[rng.randrange(V) for _ in range(rng.randrange(longest + 1))] for _ in range(count)]
    return serve(params, arrivals, responses, windows)


def shape(params: W.WardenParams, NR: int, K: int) -> EG.Shape:
    return EG.Shape(params.cell_bytes, params.grid.allocation, params.slots, NR, K, IB, params.delay)


def states(served: Served, NR: int, K: int) -> list[EG.State]:
    """The state each bucket's egress reads (S_t, the scan's ingress step t's), over the link's buckets: each request
    in the lowest request slot free at its admission, held from its admission to its last egress bucket (admitted +
    D + ⌊n / r⌋) and free after it, with its n and its ids (the anchored Programs' outputs)."""
    p = served.params
    r, D = p.slots, p.delay
    end = [s.admitted + D + s.n // r for s in served.requests]
    held: list[int | None] = [None] * NR
    out = []
    for t in range(len(served.records) * p.window):
        held = [k if k is not None and end[k] >= t else None for k in held]
        for k, s in enumerate(served.requests):
            if s.admitted == t:
                if None not in held:
                    raise ValueError(f"bucket {t} holds more than NR = {NR} requests")
                if s.n > K:
                    raise ValueError(f"request {k} has {s.n} outputs, more than K = {K}")
                held[held.index(None)] = k
        slots = tuple(EG.FREE if k is None else EG.Slot(1, served.requests[k].q, served.requests[k].admitted,
                                                        served.requests[k].n, served.responses[k]) for k in held)
        out.append(EG.State(t, slots))
    return out


def rows(served: Served) -> list[bytes]:
    """The warden's registered egress rows of the link, bucket by bucket (`warden_rows.window_rows`)."""
    return [x for record in served.records for x in WR.window_rows(record, served.params)]
