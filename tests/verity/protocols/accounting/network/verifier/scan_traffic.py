"""Network-attester-form id requests and their responses served through the network attester's own ingress and egress,
both links' rows and the scan of the ingress rows (`test_network_attester_ingress_scan.py`'s cross-check of the network
attester), the classes' synthetic runs (`SETUPS`) and such a run published as the network attester publishes one
(`published`). It imports no test framework, so a node's bare venv runs it."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Sequence

from verity.commitments.crypto import lms

from verity.protocols.accounting.network.prover import egress as EG, ingress_scan as IS
from verity.protocols.accounting.network.verifier import io_tie as IT, network_attester as W
from verity.protocols.accounting.network.verifier import network_attester_rows as WR
from verity.protocols.accounting.network.verifier.certificate import Issuer, LmsSigner
from verity.protocols.accounting.network.verifier.grid import Certifier, BlockGrid
from verity.protocols.accounting.network.verifier.id_form import id_request


@dataclass(frozen=True)
class Served:
    """Both links served through the network attester: the ingress and egress records its grids closed (one an
    execution phase), each connection's admission bucket and response ids (ordinal q = k + 1), and the network
    attester's `fails` frames and the `fails` records its grid closed."""

    params: W.NetworkAttesterParams
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
        """The network attester's registered ingress and egress rows, bucket by bucket (`network_attester_rows.window_rows`)."""
        return tuple([x for rec in recs for x in WR.window_rows(rec, self.params)]  # type: ignore[return-value]
                     for recs in (self.ingress, self.egress))


def serve(params: W.NetworkAttesterParams, requests: Sequence[bytes], arrivals: Sequence[int], rate: Sequence[int],
          responses: Sequence[Sequence[int]], windows: int, late: Sequence[int] = ()) -> Served:
    """Connection k, ordinal k + 1, arrives at bucket arrivals[k] with request requests[k], its client's bytes arriving
    rate[k] a bucket from its admission, and response responses[k] (its ids, each one id event, then `TRAILER`, all
    there by its egress; a connection of `late` gets no byte from its server, so the network attester fails it late at
    its first egress cell), and the network attester ticks both links and `fails` for `windows` execution phases as
    `NetworkAttester._tick` does: admitted in arrival order at the first bucket with a free lane, `_ingress` until its `END`,
    `_egress` until its own `END` or `FAIL`, its lane free the bucket after."""
    B, P = params.cell_bytes, params.payload
    fails: list[bytes] = []
    failed = Certifier("fails", params.fails_grid)

    def fail(frame: bytes) -> None:
        fails.append(frame)
        failed.enqueue(frame)

    network_attester = SimpleNamespace(params=params, requests_released=0, run_id=None, fails=SimpleNamespace(enqueue=fail))
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
                for frame in W.NetworkAttester._ingress(network_attester, c, t):
                    grids[0].enqueue(frame)
            for frame in W.NetworkAttester._egress(network_attester, c, t):
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
    empty state) … S_{T−1} as bit rows, the Programs' rows and the egress views; proof block t reads (ingress[t],
    states[t], progs[t]), states[0] the empty state's row, and computes (states[t + 1], views[t])."""

    cls: IS.Class
    ingress: list[bytes]
    egress: list[bytes]
    scan: list[IS.State]
    states: list[list[int]]
    progs: list[list[int]]
    views: list[list[int]]

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
        raise ValueError("the reference refuses the network attester's own ingress rows")
    states, progs = got
    scan = [IS.empty(cls)] + states
    return Chain(cls, ingress, egress, scan, [IS.encode(cls, s) for s in scan],
                 [IS.prog_encode(cls, p) for p in progs], [IS.view_encode(cls, s, t) for t, s in enumerate(states)])


class Setup(NamedTuple):
    """A class, the network attester's schedule it serves and the synthetic run's traffic: request i's prompt
    ``prompts[i]`` (max_tokens ``max_tokens`` + i, seed ``seed``·i) arrives at bucket ``arrivals[i]`` and is answered
    ``responses[i]``, and the network attester serves ``windows`` execution phases."""

    name: str
    cls: IS.Class
    params: W.NetworkAttesterParams
    prompts: tuple[tuple[int, ...], ...]
    arrivals: tuple[int, ...]
    responses: tuple[tuple[int, ...], ...]
    windows: int
    A: int
    max_tokens: int
    seed: int


W63 = Setup(
    "w63", IS.W63, W.W63,
    (tuple((7 * k + 11) % 128_256 for k in range(1024)), tuple(100_000 + k for k in range(600)),
     tuple(131_071 - 3 * k for k in range(40)), tuple((k * k) % 128_256 for k in range(300)),
     tuple(5_000 + k for k in range(100))),
    (0, 2, 2, 12, 15),
    tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 40, 64, 8, 5), 1)),
    windows=2, A=9, max_tokens=16, seed=7)
W63K4 = Setup(
    "w63k4", replace(IS.W63, KO=4, KI=32), W.W63K4,
    (tuple((7 * k + 11) % 128_256 for k in range(32)), tuple(100_000 + k for k in range(20)),
     tuple(131_071 - 3 * k for k in range(5)), tuple((k * k) % 128_256 for k in range(32)),
     tuple(5_000 + k for k in range(9))),
    W63.arrivals,
    tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 4, 1, 4, 2), 1)),
    windows=2, A=9, max_tokens=4, seed=7)
TEST = Setup(
    "test", IS.Class(B=512, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100),
    W.NetworkAttesterParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                            ingress_buckets=3, egress_cells=9, every=2),
    ((1000, 1007, 1014), tuple(2000 + 3 * k for k in range(60)), tuple(131071 - k for k in range(100))),
    (0, 1, 1), ((11, 22, 33), (131071, 0, 5, 6, 7, 8, 9, 10), (40, 41, 42, 43)), windows=1, A=6, max_tokens=5, seed=11)
SETUPS = {s.name: s for s in (W63, W63K4, TEST)}
SESSION = bytes(range(32))


def schedule(params: W.NetworkAttesterParams) -> list[str]:
    """``params`` as network_attester_serve.py's schedule flags (serve, rehearse)."""
    return " ".join(f"--{k.replace('_', '-')} {getattr(params, k)}" for k in W.INTEGER_FIELDS).split()


def request(setup: Setup, i: int, prompt) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=list(prompt),
                      max_tokens=setup.max_tokens + i, seed=setup.seed * i, temperature=0.7, top_p=0.95,
                      frame_bytes=setup.cls.B, offset=0)


def published(out: Path, setup: Setup = W63, seed: bytes = b"warden-run", windows: int | None = None,
              late: tuple[int, ...] = (), change=None) -> Path:
    """The synthetic run's records over ``windows`` execution phases (the setup's by default) published as
    network_attester_serve's `Publisher` publishes a served run's, under a network attester key of its own, beside its
    `deployment.json`; the connections of ``late`` failed late. ``change`` (a record to the record kept) stands in for a
    network attester that recorded the served cells otherwise."""
    from verity.protocols.accounting.network.attester import network_attester_serve as WS

    windows = windows or setup.windows
    reqs = [request(setup, i, p) for i, p in enumerate(setup.prompts)]
    served = serve(setup.params, reqs, setup.arrivals, [1 << 20] * len(reqs), setup.responses, windows, late)
    out.mkdir(parents=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(seed).digest()),
                              out.parent / f"{out.name}-signer.json")
    issuer = Issuer(signer, SESSION)
    start = WS.start_json(WS.certify_start(issuer, 0, 0, setup.params))
    by = {(r.link, r.window): r for r in (*served.ingress, *served.egress)}
    pub = WS.Publisher(out, issuer, setup.params, lambda link, d, r, *a: WR.Cells.of(link, d, by[r.link, r.window], *a))
    for w in range(windows):
        for rec, grid in ((served.ingress[w], setup.params.grid), (served.egress[w], setup.params.grid),
                          (served.failed[w], setup.params.fails_grid)):
            pub.keep(change(rec) if change else rec, grid)
    dep = WS.deployment_record(setup.params, signer.public_key(), SESSION, origin_unix=0.0, windows=windows,
                               slack=0.025, run_id=bytes(32), start=start)
    (out / "deployment.json").write_text(json.dumps(dep))
    return out
