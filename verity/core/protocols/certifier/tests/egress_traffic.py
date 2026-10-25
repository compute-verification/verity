"""A fake upstream's egress served through the warden, and each link-window's encoder inputs: the egress tests'
traffic (`test_network_certifier_egress.py`) and `benchmarks/network_traces/egress_zk.py`'s. It imports no test
framework, so a node's bare venv runs it."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from verity.core.protocols.certifier import egress as EG, io_tie as IT, warden as W
from verity.core.protocols.certifier.grid import Certifier, Record

#: the token ids' range (GLM's vocabulary)
V = 151_329


@dataclass(frozen=True)
class Served:
    """A link served through the warden's own egress (`warden.Warden._egress`): its schedule, the egress records its
    grid closed (one a link-window), each connection's request (ordinal q = k + 1, its admission bucket, its n
    outputs) and response ids, and the warden's `fails` frames."""

    params: W.WardenParams
    records: tuple[Record, ...]
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


def window(served: Served, w: int, NR: int, K: int) -> SimpleNamespace:
    """Link-window w's encoder inputs: its request slots (the requests that hold a slot in one of its buckets, by
    ordinal, j their place), its NR·K ids (entry j·K + k request slot j's output k, zero past its outputs), each
    bucket's records and ready bits, the prover's switches and each bucket's routed entries; with the window's shape and
    each bucket's overflow bit."""
    p = served.params
    T, r, D = p.window, p.slots, p.delay
    first, last = w * T, w * T + T - 1
    held = [k for k, s in enumerate(served.requests) if s.admitted <= last and first <= s.admitted + D + s.n // r]
    if len(held) > NR or any(served.requests[k].n >= K for k in held):
        raise ValueError(f"window {w} holds {len(held)} requests, more than NR = {NR}, or one of K = {K} outputs or "
                         "more")
    shape = EG.Shape(p.cell_bytes, T, p.grid.allocation, NR, K)
    slots = [served.requests[k] for k in held] + [None] * (NR - len(held))
    ids = [0] * (NR * K)
    for j, k in enumerate(held):
        ids[j * K:j * K + len(served.responses[k])] = served.responses[k]
    maps, overflow = zip(*(EG.bucket_map(first + t, slots, shape.R, r, D) for t in range(T)))
    bits = EG.switches(shape, maps)
    routed = EG.routed(shape, ids, bits)
    return SimpleNamespace(shape=shape, slots=slots, ids=ids, maps=maps, overflow=overflow,
                           ready=[EG.honest_ready(first + t, slots, NR) for t in range(T)], switches=bits,
                           routed=[routed[t * shape.R:(t + 1) * shape.R] for t in range(T)])


def row(win: SimpleNamespace, t: int) -> bytes:
    """Bucket t's row as the encoder renders it (`egress.bucket_row`)."""
    sh = win.shape
    return EG.bucket_row(sh.B, sh.NR, sh.K, win.maps[t], win.routed[t], win.ready[t])
