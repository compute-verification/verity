"""The scan's ingress side (`ingress_scan.IngressScanBucket`) on the warden's own ingress records: the warden, driven tick by
tick (`test_network_certifier_warden`'s harness), carries three warden-form id requests on two lanes, the third queued
until a lane frees. The reference's scan of its ingress rows from the empty state holds each request's prompt ids at
their positions, its head's fields, its admission bucket and its anchor until the anchor, and reuses a slot only past
it; the circuit is the reference on every bucket and on each named negative (two blocks swapped, a data cell after a
pad, an `END` that isn't the request's last cell, a new ordinal not above `last`, a slot reused before its anchor, a
digit, an ordinal or a pad byte changed, a state bit off in the head, a slot's header or its ids) and on random flips;
and the w63 class's state is the layout NEG sized (note:network-accounting/20261025T0830Z-report-net68-scan-router)."""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_network_certifier_warden import EOF, Server, drive

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import HEADER, id_request
from verity.core.protocols.certifier.warden import DATA, END, PAD, WardenParams, cells

B = 512
PARAMS = WardenParams(cell_bytes=B, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                      ingress_buckets=3, egress_cells=12)
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=3, WITHIN=3)
UNIT = bind(IS.IngressScanBucket, B=B, r=2, lanes=2, C=4, IDB=17, EVERY=3, WITHIN=3)
PROMPTS = ([1000, 1007, 1014], [2000 + 3 * k for k in range(60)], [131071 - k for k in range(100)])


def _request(i: int, prompt: list[int]) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=prompt, max_tokens=5 + i, seed=11 * i,
                      temperature=0.7, top_p=0.95, frame_bytes=B, lead=0)


@pytest.fixture(scope="module")
def run():
    reqs = [_request(i, p) for i, p in enumerate(PROMPTS)]
    arrivals = {0: [cells(reqs[0], B)], 1: [cells(reqs[1], B), cells(reqs[2], B)]}
    CLS.serves(PARAMS)
    warden, _ = asyncio.run(drive(Server([[(1, b"a" * 20), (2, EOF)] for _ in reqs]), arrivals, params=PARAMS))
    rows = WR.window_rows(next(r for r in warden.records if r.link == "requests"), PARAMS)
    states = IS.scan(CLS, rows)
    assert states is not None
    bits = [IS.encode(CLS, IS.empty(CLS))] + [IS.encode(CLS, s) for s in states]
    admitted = {}
    for st in states:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(warden=warden, rows=rows, states=[IS.empty(CLS)] + states, bits=bits, admitted=admitted)


def _blocks(row: bytes) -> list[list[bytes]]:
    r = CLS.r
    return [[row[(p * r + i) * B:(p * r + i + 1) * B] for i in range(r)] for p in range(CLS.lanes)]


def _join(blocks) -> bytes:
    return b"".join(c for blk in blocks for c in blk)


def test_the_scan_holds_each_request_until_its_anchor_and_reuses_a_slot_only_past_it(run):
    # request 3 waits in the warden's queue until request 1's egress ends at 0 + D, so it is admitted at D + 1
    assert run.admitted == {1: 0, 2: 1, 3: PARAMS.delay + 1}
    for q, prompt in enumerate(PROMPTS, 1):
        s0 = run.admitted[q]
        anchor = CLS.anchor(s0, s0 % CLS.EVERY)
        before = run.states[anchor]
        slot = next(s for s in before.slots if s.live and s.q == q)
        assert (slot.done, slot.left, slot.fields, slot.admitted, slot.anchor) == (1, 0, len(prompt), s0, anchor)
        assert (slot.max_tokens, slot.seed, slot.temperature, slot.top_p) == (4 + q, 11 * (q - 1), 700000, 950000)
        assert [slot.ids[IS.position(CLS, p)] for p in range(len(prompt))] == prompt
        for st in run.states[s0 + 1:anchor + 1]:
            assert sum(s.live and s.q == q for s in st.slots) == 1
    assert [CLS.anchor(s, s % CLS.EVERY) for s in run.admitted.values()] == [3, 6, 9]
    # request 1's slot is free from its anchor on, and request 3 takes it, the lowest free one
    assert run.states[3].slots[0].live == 1 and run.states[4].slots[0].live == 0
    assert [s.q for s in run.states[PARAMS.delay + 2].slots] == [3, 2]


def test_the_state_is_the_layout_neg_sized():
    w63 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=10, WITHIN=12)
    assert (w63.K0, w63.K, IS.state_bits(w63)) == (12, 45, 145_040)
    lay = IS.layout(w63)
    at = 0
    for name, (o, w) in lay.items():
        assert o == at, name
        at += w
    assert at == IS.state_bits(w63) and lay["pad"] == (145_036, 4)
    assert lay["3.ids"] == (145_036 - 47 * 45 * 17, 47 * 45 * 17)
    assert [IS.position(w63, p) for p in (0, 11, 12, 56, 57)] == [0, 11, 45, 89, 90]
    assert [w63.anchor(t, t % 10) for t in (0, 7, 8, 9, 18)] == [20, 20, 20, 30, 30]
    for bad in ({"B": 256}, {"C": 1 << 16}, {"IDB": 33}, {"EVERY": 0}):
        with pytest.raises(ValueError):
            IS.Class(**{"B": 512, "r": 4, "lanes": 4, "C": 47, **bad})
    served = WardenParams(cell_bytes=512, slots=4, lanes=4, delay=21, ingress_cells=47, ingress_buckets=12)
    w63.serves(served)
    for bad, fix in (({"delay": 20}, "delay >= WITHIN + EVERY - 1 = 21"), ({"ingress_buckets": 13}, "<= WITHIN = 12"),
                     ({"ingress_cells": 48}, "<= C = 47"), ({"lanes": 5}, "= lanes = 4")):
        with pytest.raises(ValueError, match=re.escape(fix)):
            w63.serves(replace(served, **bad))


def _negatives(run) -> dict[str, tuple[bytes, list[int], list[int]]]:
    rows, bits, states = run.rows, run.bits, run.states
    out = {}
    blk = _blocks(rows[1])
    assert [[c[0] for c in b] for b in blk] == [[END, PAD], [PAD, PAD]]
    out["swapped"] = (_join([blk[1], blk[0]]), bits[1], bits[2])
    # request 2's first data cell, a bucket early, after one of its pads
    out["after-pad"] = (_join([blk[0], [blk[1][0], _blocks(rows[2])[0][0]]]), bits[1], bits[2])
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == DATA)
    blk = _blocks(rows[t])
    blk[p][i] = bytes([END]) + blk[p][i][1:]
    out["early-end"] = (_join(blk), bits[t], bits[t + 1])
    t3 = run.admitted[3]
    stale = replace(states[t3], last=3)
    out["stale-ordinal"] = (rows[t3], IS.encode(CLS, stale), bits[t3 + 1])
    slots = list(states[t3].slots)
    slots[0] = replace(slots[0], live=1, anchor=t3 + 1)
    out["held-slot"] = (rows[t3], IS.encode(CLS, replace(states[t3], slots=slots)), bits[t3 + 1])
    blk = _blocks(rows[t])
    cell = bytearray(blk[p][i])
    at = cell.index(b",", HEADER.size + 400) + 10
    cell[at] = ord("0") + (cell[at] - ord("0") + 1) % 10
    blk[p][i] = bytes(cell)
    out["digit"] = (_join(blk), bits[t], bits[t + 1])
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:4] + bytes([blk[p][i][4] ^ 1]) + blk[p][i][5:]
    out["ordinal"] = (_join(blk), bits[t], bits[t + 1])
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == PAD and HEADER.unpack_from(c)[1])
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:-1] + b"\x01"
    out["pad-byte"] = (_join(blk), bits[t], bits[t + 1])
    lay = IS.layout(CLS)
    for name, field, bit in (("now", "now", 0), ("phase", "phase", 1), ("last", "last", 0), ("slot-q", "1.q", 0),
                             ("slot-fields", "1.fields", 2), ("slot-anchor", "1.anchor", 0), ("slot-ids", "1.ids", 5)):
        sout = list(bits[3])
        sout[lay[field][0] + bit] ^= 1
        out[f"state-{name}"] = (rows[2], bits[2], sout)
    return out


def _circuit(cases) -> list[int]:
    args = IS.arguments(CLS, cases)
    return [int(x) for x in np.asarray(evaluate_bits(UNIT, *[np.asarray(a, dtype=np.uint8) for a in args])).reshape(-1)]


def test_each_negative_is_refused_by_the_reference(run):
    for name, (row, sin, sout) in _negatives(run).items():
        assert IS.reference(CLS, row, sin, sout) == 0, name


@pytest.mark.slow
def test_the_circuit_is_the_reference_on_every_bucket_each_negative_and_random_flips(run):
    """Slow (about 15 s: 222,334 ANDs traced, then evaluated on about 100 cases)."""
    honest = [(run.rows[t], run.bits[t], run.bits[t + 1]) for t in range(len(run.rows))]
    negs = _negatives(run)
    rng = random.Random(7)
    flips = []
    for _ in range(60):
        t = rng.randrange(len(run.rows))
        row, sin, sout = bytearray(run.rows[t]), list(run.bits[t]), list(run.bits[t + 1])
        where = rng.randrange(3)
        if where == 0:
            k = rng.randrange(len(row))
            row[k] ^= 1 << rng.randrange(8)
        else:
            s = sin if where == 1 else sout
            s[rng.randrange(len(s))] ^= 1
        flips.append((bytes(row), sin, sout))
    cases = honest + list(negs.values()) + flips
    got = _circuit(cases)
    want = [IS.reference(CLS, *c) for c in cases]
    assert got == want
    assert got[:len(honest)] == [1] * len(honest) and got[len(honest):len(honest) + len(negs)] == [0] * len(negs)
