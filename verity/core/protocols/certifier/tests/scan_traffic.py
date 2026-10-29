"""Warden-form id requests and their responses served through the warden's own ingress and egress, both links' rows and
the scan of the ingress rows: the joint scan's traffic (`test_network_certifier_ingress_scan.py`'s cross-check of the
warden, and `benchmarks/network_traces/scan_zk.py`'s). It imports no test framework, so a node's bare venv runs it."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from verity.core.protocols.certifier import egress as EG, ingress_scan as IS, io_tie as IT, warden as W
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.grid import Certifier, BlockGrid


@dataclass(frozen=True)
class Served:
    """Both links served through the warden: the ingress and egress records its grids closed (one a link-window),
    each connection's admission bucket and response ids (ordinal q = k + 1), and the warden's `fails` frames and the
    `fails` records its grid closed."""

    params: W.WardenParams
    ingress: tuple[BlockGrid, ...]
    egress: tuple[BlockGrid, ...]
    admitted: tuple[int, ...]
    responses: tuple[tuple[int, ...], ...]
    fails: tuple[bytes, ...]
    failed: tuple[BlockGrid, ...] = ()

    @property
    def outputs(self) -> dict[int, tuple[int, ...]]:
        return {k + 1: xs for k, xs in enumerate(self.responses)}

    def rows(self) -> tuple[list[bytes], list[bytes]]:
        """The warden's registered ingress and egress rows, bucket by bucket (`warden_rows.window_rows`)."""
        return tuple([x for rec in recs for x in WR.window_rows(rec, self.params)]  # type: ignore[return-value]
                     for recs in (self.ingress, self.egress))


def serve(params: W.WardenParams, requests: Sequence[bytes], arrivals: Sequence[int], rate: Sequence[int],
          responses: Sequence[Sequence[int]], windows: int, late: Sequence[int] = ()) -> Served:
    """Connection k, ordinal k + 1, arrives at bucket arrivals[k] with request requests[k], its client's bytes arriving
    rate[k] a bucket from its admission, and response responses[k] (its ids, each one id event, then `TRAILER`, all
    there by its egress; a connection of `late` gets no byte from its server, so the warden fails it late at its first
    egress cell), and the warden ticks both links and `fails` for `windows` link-windows as `Warden._tick` does:
    admitted in arrival order at the first bucket with a free lane, `_ingress` until its `END`, `_egress` until its own
    `END` or `FAIL`, its lane free the bucket after."""
    B, P = params.cell_bytes, params.payload
    fails: list[bytes] = []
    failed = Certifier("fails", params.fails_grid)

    def fail(frame: bytes) -> None:
        fails.append(frame)
        failed.enqueue(frame)

    warden = SimpleNamespace(params=params, requests_released=0, run_id=None, fails=SimpleNamespace(enqueue=fail))
    conns = []
    for k, (a, xs) in enumerate(zip(arrivals, responses, strict=True)):
        c = W._Connection(ordinal=k + 1, reader=None, writer=None, done=None, arrived=a)
        if k not in late:
            c.egress = bytearray(b"".join(IT.id_event(x, P) for x in xs) + IT.TRAILER)
            c.egress_ended = True
        conns.append(c)
    queue, active = deque(conns), []
    grids = (Certifier("requests", params.grid), Certifier("responses", params.grid), failed)
    records: tuple[list[BlockGrid], list[BlockGrid], list[BlockGrid]] = ([], [], [])
    for t in range(windows * params.window):
        while queue and queue[0].arrived <= t and len(active) < params.lanes:
            c = queue.popleft()
            c.admitted = t
            active.append(c)
        for c in active:
            k = c.ordinal - 1
            sent = min(len(requests[k]), rate[k] * (t - c.admitted))
            c.ingress += requests[k][c.bytes_in:sent]
            c.bytes_in, c.ingress_ended = sent, sent == len(requests[k])
            if c.ingress_end is None and c.failed is None:
                for frame in W.Warden._ingress(warden, c, t):
                    grids[0].enqueue(frame)
            for frame in W.Warden._egress(warden, c, t):
                grids[1].enqueue(frame)
        active = [c for c in active if c.ended is None]
        for grid in grids:
            grid.tick()
        if (t + 1) % params.window == 0:
            for grid, recs in zip(grids, records):
                recs.append(grid.close_window())
    if queue or active:
        raise ValueError("every connection ends within the link's windows")
    return Served(params, tuple(records[0]), tuple(records[1]), tuple(c.admitted for c in conns),
                  tuple(tuple(xs) for xs in responses), tuple(fails), tuple(records[2]))


@dataclass(frozen=True)
class Chain:
    """The scan of a run's ingress rows (`ingress_scan.scan`, the Programs' rows its outputs): the states S_{−1} (the
    empty state) … S_{T−1} as bit rows, the Programs' rows and the egress views; unit t reads (ingress[t], states[t],
    progs[t]), the first unit no state, and computes (states[t + 1], views[t])."""

    cls: IS.Class
    ingress: list[bytes]
    egress: list[bytes]
    scan: list[IS.State]
    states: list[list[int]]
    progs: list[list[int]]
    views: list[list[int]]

    def unit(self, t: int) -> tuple[bytes, list[int], list[int]] | tuple[bytes, list[int]]:
        """Unit t's arguments: (row, prog) for t = 0 (`IngressScanFirst`), (row, sin, prog) after it
        (`IngressScanBucket`)."""
        return (self.ingress[0], self.progs[0]) if t == 0 else (self.ingress[t], self.states[t], self.progs[t])

    def outputs(self, t: int) -> tuple[list[int], list[int]]:
        """Unit t's o1 and o2: S_t and its egress view."""
        return self.states[t + 1], self.views[t]

    def view(self, t: int) -> EG.State:
        return IS.view(self.cls, self.scan[t + 1], t)


def chain(cls: IS.Class, ingress: Sequence[bytes], egress: Sequence[bytes],
          outputs: dict[int, Sequence[int]]) -> Chain:
    """The scan of the ingress rows, each request's outputs `outputs[q]`, beside the egress rows."""
    ingress, egress = list(ingress), list(egress)
    if len(ingress) != len(egress):
        raise ValueError("one ingress row and one egress row a bucket")
    got = IS.scan(cls, ingress, outputs)
    if got is None:
        raise ValueError("the reference refuses the warden's own ingress rows")
    states, progs = got
    scan = [IS.empty(cls)] + states
    return Chain(cls, ingress, egress, scan, [IS.encode(cls, s) for s in scan],
                 [IS.prog_encode(cls, p) for p in progs], [IS.view_encode(cls, s, t) for t, s in enumerate(states)])
