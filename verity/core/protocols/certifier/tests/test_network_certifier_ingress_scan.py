"""The scan (`ingress_scan.IngressScanBucket`) and the egress buckets on the views it writes (`egress.EgressBucket`), on
the warden's own records: the warden, driven tick by tick (`test_network_certifier_warden`'s harness), carries three
warden-form id requests on two lanes, the third queued until a lane frees, and their responses of id events. The
reference's scan of its ingress rows from the empty state, the Programs' rows its responses, holds each request's prompt
ids at their positions, its head's fields, its admission bucket and its anchor until the anchor and its outputs from
then until its last egress bucket, and reuses a slot only past it; the view ingress bucket t writes is the state egress
bucket t accepts the warden's egress row t on, every bucket; a view from another bucket, or a forged egress row, is
refused. The ingress circuit is the reference on every bucket and on each named negative (two blocks swapped, a data
cell after a pad, an `END` that isn't the request's last cell, a new ordinal not above `last`, a slot reused before its
last egress bucket, a digit, an ordinal or a pad byte changed, a state bit off in the head, a slot's header, its ids or
its outputs, the state rows of two buckets swapped, a Programs' entry off its anchor or off its outputs, a view bit off)
and on random flips, and the egress circuit is its reference on the views and rows; `scan_traffic.serve`, the zk run's
driver, ticks the same rows as the warden; and the w63 class's state is the layout NEG sized
(note:network-accounting/20261025T0830Z-report-net68-scan-router) with the outputs N2's view reads."""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import scan_traffic as ST
from test_network_certifier_warden import EOF, Server, drive

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.definitions.certifier import egress as EG
from verity.ml.definitions.certifier import ingress_scan as IS
from verity.ml.definitions.certifier import io_tie as IT
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import HEADER, id_request
from verity.core.protocols.certifier.warden import DATA, END, PAD, WardenParams, cells

B = 512
PARAMS = WardenParams(cell_bytes=B, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                      ingress_buckets=3, egress_cells=9)
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=3, WITHIN=3, D=5, KO=8)
UNIT = bind(IS.IngressScanBucket, B=B, r=2, lanes=2, C=4, IDB=17, EVERY=3, WITHIN=3, D=5, KO=8)
SHAPE = CLS.egress
EGRESS = bind(EG.EgressBucket, B=B, R=SHAPE.R, r=SHAPE.r, NR=SHAPE.NR, K=SHAPE.K, IB=SHAPE.IB, D=SHAPE.D)
PROMPTS = ([1000, 1007, 1014], [2000 + 3 * k for k in range(60)], [131071 - k for k in range(100)])
#: request 1's three outputs end its egress at 0 + D + 1, so request 3 is admitted at D + 2; request 2's are KO
RESPONSES = ((11, 22, 33), (131071, 0, 5, 6, 7, 8, 9, 10), (40, 41, 42, 43))


def _request(i: int, prompt: list[int]) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=prompt, max_tokens=5 + i, seed=11 * i,
                      temperature=0.7, top_p=0.95, frame_bytes=B, offset=0)


def _response(ids) -> bytes:
    return b"".join(IT.id_event(x, PARAMS.payload) for x in ids) + IT.TRAILER


@pytest.fixture(scope="module")
def run():
    reqs = [_request(i, p) for i, p in enumerate(PROMPTS)]
    arrivals = {0: [cells(reqs[0], B)], 1: [cells(reqs[1], B), cells(reqs[2], B)]}
    CLS.serves(PARAMS)
    server = Server([[(1, _response(xs)), (2, EOF)] for xs in RESPONSES])
    warden, _ = asyncio.run(drive(server, arrivals, params=PARAMS))
    rec = {r.link: r for r in warden.records}
    rows = [WR.window_rows(rec[link], PARAMS) for link in ("requests", "responses")]
    ch = ST.chain(CLS, *rows, {q: xs for q, xs in enumerate(RESPONSES, 1)})
    admitted = {}
    for st in ch.scan:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(warden=warden, reqs=reqs, chain=ch, rows=ch.ingress, egress=ch.egress, states=ch.scan,
                           bits=ch.states, admitted=admitted)


def _blocks(row: bytes) -> list[list[bytes]]:
    r = CLS.r
    return [[row[(p * r + i) * B:(p * r + i + 1) * B] for i in range(r)] for p in range(CLS.lanes)]


def _join(blocks) -> bytes:
    return b"".join(c for blk in blocks for c in blk)


def _last(s0: int, n: int) -> int:
    return s0 + CLS.D + n // CLS.r


def test_the_scan_holds_each_request_through_its_egress_and_reuses_a_slot_only_past_it(run):
    assert run.admitted == {1: 0, 2: 1, 3: _last(0, 3) + 1} == {1: 0, 2: 1, 3: PARAMS.delay + 2}
    for q, (prompt, out) in enumerate(zip(PROMPTS, RESPONSES), 1):
        s0 = run.admitted[q]
        anchor = CLS.anchor(s0, s0 % CLS.EVERY)
        before = run.states[anchor]
        slot = next(s for s in before.slots if s.live and s.q == q)
        assert (slot.done, slot.left, slot.fields, slot.admitted, slot.anchor) == (1, 0, len(prompt), s0, anchor)
        assert (slot.max_tokens, slot.seed, slot.temperature, slot.top_p) == (4 + q, 11 * (q - 1), 700000, 950000)
        assert [slot.ids[IS.position(CLS, p)] for p in range(len(prompt))] == prompt
        assert (slot.n, slot.out) == (0, [0] * CLS.KO)
        # S_t is run.states[t + 1]: the request's slot is live in S_{s0} … S_{last}, its outputs from S_{anchor} on
        for t in range(s0, _last(s0, len(out)) + 1):
            mine = [s for s in run.states[t + 1].slots if s.live and s.q == q]
            assert len(mine) == 1
            assert (mine[0].n, mine[0].out[:len(out)]) == ((len(out), list(out)) if t >= anchor else (0, [0] * len(out)))
    assert [CLS.anchor(s, s % CLS.EVERY) for s in run.admitted.values()] == [3, 6, 12]
    # request 1's slot is live in S_6, its last egress bucket, and request 3 takes it at 7, the lowest free one
    assert run.states[7].slots[0].q == 1 and run.states[8].slots[0].q == 3
    assert run.states[8].slots[0].n == 0 and [s.q for s in run.states[PARAMS.delay + 3].slots] == [3, 2]


def test_each_egress_bucket_accepts_the_view_its_ingress_bucket_wrote_and_refuses_another(run):
    ch = run.chain
    assert len(ch.egress) == len(ch.ingress) == PARAMS.window
    assert sum(row[s * B] == DATA for row in ch.egress for s in range(SHAPE.R)) == sum(map(len, RESPONSES))
    for t in range(PARAMS.window):
        assert EG.verdict(SHAPE, ch.view(t), ch.egress[t]) == EG.ACCEPT, t
        assert EG.state_bits(SHAPE, ch.view(t)) == ch.views[t]
    moved = [t for t in range(PARAMS.window - 1) if ch.egress[t] != ch.egress[t + 1]]
    assert moved and all(EG.verdict(SHAPE, ch.view(t + 1), ch.egress[t]) == 0 for t in moved)
    t, s = next((t, s) for t, row in enumerate(ch.egress) for s in range(SHAPE.R) if row[s * B] == DATA)
    forged = bytearray(ch.egress[t])
    forged[s * B + IT.ID_AT + IT.ID_DIGITS - 1] ^= 1
    assert EG.verdict(SHAPE, ch.view(t), bytes(forged)) == 0


def test_scan_traffic_ticks_the_warden_s_rows(run):
    served = ST.serve(PARAMS, run.reqs, [0, 1, 1], [1 << 20] * 3, RESPONSES, 1)
    assert served.rows() == (run.rows, run.egress) and served.admitted == (0, 1, PARAMS.delay + 2)


def test_the_state_is_the_layout_neg_sized():
    w63 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=10, WITHIN=12)
    assert (w63.K0, w63.K, w63.D, w63.KO, IS.state_bits(w63)) == (12, 45, 21, 64, 149_520)
    assert (IS.prog_bits(w63), IS.view_bits(w63), w63.egress) == (4480, 4832, EG.Shape(512, 16, 4, 4, 64, 17, 21))
    lay = IS.layout(w63)
    at = 0
    for name, (o, w) in lay.items():
        assert o == at, name
        at += w
    assert at == IS.state_bits(w63) and lay["pad"] == (149_516, 4)
    assert lay["3.ids"] == (149_516 - 47 * 45 * 17, 47 * 45 * 17) and lay["3.out"][1] == 64 * 17
    assert [IS.position(w63, p) for p in (0, 11, 12, 56, 57)] == [0, 11, 45, 89, 90]
    assert [w63.anchor(t, t % 10) for t in (0, 7, 8, 9, 18)] == [20, 20, 20, 30, 30]
    for bad in ({"B": 256}, {"C": 1 << 16}, {"IDB": 33}, {"EVERY": 0}, {"D": 20}, {"KO": 1 << 16}):
        with pytest.raises(ValueError):
            IS.Class(**{"B": 512, "r": 4, "lanes": 4, "C": 47, **bad})
    served = WardenParams(cell_bytes=512, slots=4, lanes=4, delay=21, ingress_cells=47, ingress_buckets=12,
                          egress_cells=65)
    w63.serves(served)
    for bad, fix in (({"delay": 22}, "= D = 21"), ({"ingress_buckets": 13}, "<= WITHIN = 12"),
                     ({"ingress_cells": 48}, "<= C = 47"), ({"lanes": 5}, "= lanes = 4"),
                     ({"egress_cells": 66}, "<= KO + 1 = 65")):
        with pytest.raises(ValueError, match=re.escape(fix)):
            w63.serves(replace(served, **bad))


Case = tuple[bytes, list[int], list[int], list[int], list[int]]


def _negatives(run) -> dict[str, Case]:
    ch, rows, bits, states = run.chain, run.rows, run.bits, run.states
    honest = lambda t, **kw: tuple(kw.get(k, v) for k, v in zip(("row", "sin", "prog", "sout", "view"), ch.unit(t)))  # noqa: E731
    out = {}
    blk = _blocks(rows[1])
    assert [[c[0] for c in b] for b in blk] == [[END, PAD], [PAD, PAD]]
    out["swapped"] = honest(1, row=_join([blk[1], blk[0]]))
    # request 2's first data cell, a bucket early, after one of its pads
    out["after-pad"] = honest(1, row=_join([blk[0], [blk[1][0], _blocks(rows[2])[0][0]]]))
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == DATA)
    blk = _blocks(rows[t])
    blk[p][i] = bytes([END]) + blk[p][i][1:]
    out["early-end"] = honest(t, row=_join(blk))
    t3 = run.admitted[3]
    out["stale-ordinal"] = honest(t3, sin=IS.encode(CLS, replace(states[t3], last=3)))
    # request 1's slot held a bucket longer: its n one lane's worth more, so request 3 finds no free slot
    slots = list(states[t3].slots)
    slots[0] = replace(slots[0], n=slots[0].n + CLS.r)
    out["held-slot"] = honest(t3, sin=IS.encode(CLS, replace(states[t3], slots=slots)))
    cell = bytearray(_blocks(rows[t])[p][i])
    at = cell.index(b",", HEADER.size + 400) + 10
    cell[at] = ord("0") + (cell[at] - ord("0") + 1) % 10
    blk = _blocks(rows[t])
    blk[p][i] = bytes(cell)
    out["digit"] = honest(t, row=_join(blk))
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:4] + bytes([blk[p][i][4] ^ 1]) + blk[p][i][5:]
    out["ordinal"] = honest(t, row=_join(blk))
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == PAD and HEADER.unpack_from(c)[1])
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:-1] + b"\x01"
    out["pad-byte"] = honest(t, row=_join(blk))
    lay = IS.layout(CLS)
    for name, field, bit, t in (("now", "now", 0, 2), ("phase", "phase", 1, 2), ("last", "last", 0, 2),
                                ("slot-q", "1.q", 0, 2), ("slot-fields", "1.fields", 2, 2),
                                ("slot-anchor", "1.anchor", 0, 2), ("slot-ids", "1.ids", 5, 2), ("slot-n", "1.n", 0, 7),
                                ("slot-out", "1.out", 3, 7)):
        sout = list(bits[t + 1])
        sout[lay[field][0] + bit] ^= 1
        out[f"state-{name}"] = honest(t, sout=sout)
    # bucket 4 given bucket 5's state rows, the chain's states swapped between buckets, its own row, prog and view
    out["chain"] = honest(4, sin=bits[5], sout=bits[6])
    entries = IS.prog_decode(CLS, ch.progs[3])
    assert entries == [(3, RESPONSES[0] + (0,) * 5), (0, (0,) * CLS.KO)]
    out["prog-off-anchor"] = honest(3, prog=IS.prog_encode(CLS, [entries[0], (1, (7,))]))
    out["prog-outputs"] = honest(3, prog=IS.prog_encode(CLS, [(2, RESPONSES[0][:2]), IS.NONE]))
    for name, t, at in (("view-b", 4, 0), ("view-live-high", 4, SHAPE.slot_at(0) + 1),
                        ("view-out", 7, SHAPE.slot_at(1) + EG.IDS + 2)):
        view = list(ch.views[t])
        view[at] ^= 1
        out[name] = honest(t, view=view)
    return out


def _circuit(fn, args) -> list[int]:
    return [int(x) for x in np.asarray(evaluate_bits(fn, *[np.asarray(a, dtype=np.uint8) for a in args])).reshape(-1)]


def test_each_negative_is_refused_by_the_reference(run):
    for name, case in _negatives(run).items():
        assert IS.reference(CLS, *case) == 0, name


@pytest.mark.slow
def test_the_circuits_are_the_references_on_every_bucket_each_negative_and_random_flips(run):
    """Slow (about 13 s: 224,436 ANDs traced, then evaluated on about 110 cases, and the egress unit on 47)."""
    ch = run.chain
    honest = [ch.unit(t) for t in range(len(run.rows))]
    negs = _negatives(run)
    rng = random.Random(7)
    flips = []
    for _ in range(60):
        case = [bytearray(x) if k == 0 else list(x) for k, x in enumerate(honest[rng.randrange(len(honest))])]
        k = rng.randrange(5)
        if k == 0:
            case[0][rng.randrange(len(case[0]))] ^= 1 << rng.randrange(8)
        else:
            case[k][rng.randrange(len(case[k]))] ^= 1
        flips.append((bytes(case[0]), *case[1:]))
    cases = honest + list(negs.values()) + flips
    got = _circuit(UNIT, IS.arguments(CLS, cases))
    want = [IS.reference(CLS, *c) for c in cases]
    assert got == want
    assert got[:len(honest)] == [1] * len(honest) and got[len(honest):len(honest) + len(negs)] == [0] * len(negs)
    T = len(run.rows)
    pairs = [(ch.view(t), ch.egress[t]) for t in range(T)] + [(ch.view(t + 1), ch.egress[t]) for t in range(T - 1)]
    pairs += [(ch.view(t), ch.egress[(t + 3) % T]) for t in range(T)]
    got = _circuit(EGRESS, [list(col) for col in zip(*(EG.bucket_arguments(SHAPE, s, r) for s, r in pairs))])
    words = np.asarray(got).reshape(len(pairs), 16)
    assert [int(w[0]) for w in words] == [EG.verdict(SHAPE, s, r) for s, r in pairs] and not words[:, 1:].any()
    assert [int(w[0]) for w in words[:T]] == [1] * T
