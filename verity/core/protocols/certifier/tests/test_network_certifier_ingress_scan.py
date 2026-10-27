"""The scan (`ingress_scan.IngressScanBucket`, `IngressScanFirst`) and the egress buckets on the views it writes
(`egress.EgressBucket`), on the warden's own records: the warden, driven tick by tick (`test_network_certifier_warden`'s
harness), carries three warden-form id requests on two lanes, the third queued until a lane frees, and their responses
of id events. The reference's scan of its ingress rows from the empty state, the Programs' rows its responses, holds
each request's prompt ids at their positions, its head's fields, its admission bucket and its anchor until the anchor
(two requests anchored at one boundary in their own slots) and its outputs from then until its last egress bucket,
gives each slot's occupancy word 1 in the state its Program reads and 0 in every other (the token-id row, `token_row`:
the occupancy word, the prompt's length and its prompt in order, each where `slot_row` puts it in the state row), and
reuses a slot only past it; egress bucket t computes the
warden's egress row t from the view ingress bucket t writes, every bucket, and another row (never a forged one) from a
view of another bucket. The ingress circuit computes the reference's outputs on every bucket and on random flips (S_t
and its view, bit for bit, where ok is 1), refuses each named negative (two blocks swapped, a data cell after a pad, an
`END` that isn't the request's last cell, a new ordinal not above `last`, a slot reused before its last egress
bucket, an ordinal or a pad byte changed, a phase of EVERY, a Programs' entry off its anchor, an entry for a slot whose
occupancy word is 0, a Programs' count over KO, a prompt over KI ids), and computes another
S_t than the scan's, which the state row its o1 opens refuses, from a digit changed or a Programs' entry off its
outputs; the same traffic at a KI one below request 3's prompt is refused at the cell that takes it past; a window
circuit's state rows (`window_state`) are the scan's from t0 − 1, the empty state's first; the first bucket's circuit
is the ingress circuit at the empty state, bit for bit; the egress circuit is its reference on the views;
`scan_traffic.serve`, the zk run's driver, ticks the same rows as the warden; and the w63 class's state is the scan's
sized layout, with the outputs N2's view reads."""
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
from verity.core.protocols.certifier import egress as EG
from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import HEADER, id_request
from verity.core.protocols.certifier.warden import DATA, END, PAD, WardenParams, cells

B = 512
PARAMS = WardenParams(cell_bytes=B, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                      ingress_buckets=3, egress_cells=9, every=2)
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
UNIT = bind(IS.IngressScanBucket, B=B, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
FIRST = bind(IS.IngressScanFirst, B=B, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
SHAPE = CLS.egress
EGRESS = bind(EG.EgressBucket, B=B, R=SHAPE.R, r=SHAPE.r, NR=SHAPE.NR, K=SHAPE.K, IB=SHAPE.IB, D=SHAPE.D)
#: request 3's prompt is KI ids
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
        assert slot.occ == 1 and (1, len(prompt), tuple(prompt)) in IS.token_row(CLS, before)
        assert (slot.n, slot.out) == (0, [0] * CLS.KO)
        # S_t is run.states[t + 1]: the request's slot is live in S_{s0} … S_{last}, its outputs from S_{anchor} on
        for t in range(s0, _last(s0, len(out)) + 1):
            mine = [s for s in run.states[t + 1].slots if s.live and s.q == q]
            assert len(mine) == 1
            assert (mine[0].n, mine[0].out[:len(out)]) == ((len(out), list(out)) if t >= anchor else (0, [0] * len(out)))
    assert [CLS.anchor(s, s % CLS.EVERY) for s in run.admitted.values()] == [4, 4, 10]
    # a slot is occupied in the state its Program reads, S_{A−1}, and in no other: once a request
    occupied = [(t, s.q) for t, st in enumerate(run.states) for s in st.slots if s.occ]
    assert occupied == [(4, 1), (4, 2), (10, 3)]
    assert all(IS.token_row(CLS, st)[j] == (0, 0, ()) for t, st in enumerate(run.states) for j, s in enumerate(st.slots)
               if (t, s.q) not in occupied or not s.live)
    # request 1's slot is live in S_6, its last egress bucket, and request 3 takes it at 7, the lowest free one
    assert run.states[7].slots[0].q == 1 and run.states[8].slots[0].q == 3
    assert run.states[8].slots[0].n == 0 and [s.q for s in run.states[PARAMS.delay + 3].slots] == [3, 2]


def test_a_reader_of_the_state_row_finds_each_slot_s_token_row_at_slot_row_s_bits(run):
    NP = CLS.C * CLS.K - CLS.K + CLS.K0
    assert NP >= max(map(len, PROMPTS))
    seen = 0
    for st in run.states:
        bits = IS.encode(CLS, st)
        for j, (occ, lp, ids) in enumerate(IS.token_row(CLS, st)):
            got = [sum(bits[o + i] << i for i in range(w)) for o, w in IS.slot_row(CLS, j, NP)]
            assert got[0] == occ
            if occ:
                assert (got[1], got[2:2 + lp]) == (lp, list(ids))
                seen += 1
    assert seen == len(PROMPTS)


def test_a_prompt_over_KI_ids_is_refused_at_the_cell_that_takes_it_past(run):
    outputs = {q: xs for q, xs in enumerate(RESPONSES, 1)}
    assert max(map(len, PROMPTS)) == CLS.KI and IS.scan(CLS, run.rows, outputs) is not None
    short = replace(CLS, KI=CLS.KI - 1)
    assert IS.scan(short, run.rows, outputs) is None
    st, refused = IS.empty(short), None
    for t, row in enumerate(run.rows):
        nxt = IS.step(short, row, st, IS.programs(short, st, outputs))
        if nxt is None:
            refused = t
            break
        st = nxt
    end = next(t for t, row in enumerate(run.rows) for blk in _blocks(row) for c in blk
               if c[0] == END and HEADER.unpack_from(c)[1] == 3)
    assert refused == end and run.states[end + 1].slots[0].fields == CLS.KI


def test_a_window_circuit_reads_the_scan_s_state_rows_from_t0_minus_one(run):
    ch, T = run.chain, PARAMS.window
    registered = ch.states[1:]
    assert len(registered) == T and IS.window_state(CLS, registered, 0, T) == ch.states[:T]
    assert IS.window_state(CLS, registered, 0, T)[0] == IS.encode(CLS, IS.empty(CLS))
    for t0, n in ((4, 4), (8, 8), (2, T - 2)):
        rows = IS.window_state(CLS, registered, t0, n)
        assert rows == ch.states[t0:t0 + n]
        # boundary a's Programs read row a: S_{t0 + a − 1}, the state bucket t0 + a's unit reads
        assert [IS.token_row(CLS, IS.decode(CLS, x)) for x in rows] == \
            [IS.token_row(CLS, run.states[t0 + a]) for a in range(n)]
    for t0, n, fix in ((1, 4, "multiple of EVERY = 2, not t0 = 1"), (0, 3, "not t0 = 0, buckets = 3"),
                       (-2, 4, "t0 >= 0"), (0, 0, "buckets >= 1"), (T - 2, 4, f"register {T + 1} rows, not {T}")):
        with pytest.raises(ValueError, match=re.escape(fix)):
            IS.window_state(CLS, registered, t0, n)
    with pytest.raises(ValueError, match=f"a state row is {IS.state_bits(CLS)} bits"):
        IS.window_state(CLS, [x[:-16] for x in registered], 0, 2)


def test_each_egress_bucket_computes_the_wardens_row_from_the_view_its_ingress_bucket_wrote(run):
    ch = run.chain
    assert len(ch.egress) == len(ch.ingress) == PARAMS.window
    assert sum(row[s * B] == DATA for row in ch.egress for s in range(SHAPE.R)) == sum(map(len, RESPONSES))
    for t in range(PARAMS.window):
        assert EG.outputs(SHAPE, ch.view(t)) == (EG.ACCEPT, ch.egress[t]), t
        assert EG.state_bits(SHAPE, ch.view(t)) == ch.views[t]
    moved = [t for t in range(PARAMS.window - 1) if ch.egress[t] != ch.egress[t + 1]]
    assert moved and all(EG.outputs(SHAPE, ch.view(t + 1))[1] != ch.egress[t] for t in moved)
    t, s = next((t, s) for t, row in enumerate(ch.egress) for s in range(SHAPE.R) if row[s * B] == DATA)
    forged = bytearray(ch.egress[t])
    forged[s * B + IT.ID_AT + IT.ID_DIGITS - 1] ^= 1
    assert EG.outputs(SHAPE, ch.view(t))[1] != bytes(forged)


def test_scan_traffic_ticks_the_warden_s_rows(run):
    served = ST.serve(PARAMS, run.reqs, [0, 1, 1], [1 << 20] * 3, RESPONSES, 1)
    assert served.rows() == (run.rows, run.egress) and served.admitted == (0, 1, PARAMS.delay + 2)


def test_the_state_is_the_layout_neg_sized():
    w63 = IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12)
    assert (w63.K0, w63.K, w63.D, w63.KO, w63.SLOTS, IS.state_bits(w63)) == (12, 45, 27, 64, 4, 149_648)
    assert (w63.KI, w63.prompt_room) == (1024, 2082) and replace(w63, KI=2082).KI == 2082
    assert (IS.prog_bits(w63), IS.view_bits(w63), w63.egress) == (4480, 4832, EG.Shape(512, 16, 4, 4, 64, 17, 27))
    lay = IS.layout(w63)
    at = 0
    for name, (o, w) in lay.items():
        assert o == at, name
        at += w
    assert at == IS.state_bits(w63) and lay["pad"] == (149_644, 4)
    assert lay["3.ids"] == (149_644 - 47 * 45 * 17, 47 * 45 * 17) and lay["3.out"][1] == 64 * 17
    assert lay["3.occ"][1] == 32 and lay["3.n"][0] == lay["3.occ"][0] + 32
    assert [IS.position(w63, p) for p in (0, 11, 12, 56, 57)] == [0, 11, 45, 89, 90]
    row = IS.slot_row(w63, 3, 1024)
    assert len(row) == 1026 and row[:2] == [lay["3.occ"], lay["3.fields"]]
    assert row[2 + 11] == (lay["3.ids"][0] + 11 * 17, 17) and row[2 + 12] == (lay["3.ids"][0] + 45 * 17, 17)
    assert IS.slot_row(w63, 0, 2082)[-1] == (lay["0.ids"][0] + (47 * 45 - 1) * 17, 17)
    for j, NP in ((4, 1024), (0, 0), (0, 2083)):
        with pytest.raises(ValueError):
            IS.slot_row(w63, j, NP)
    assert [w63.anchor(t, t % 16) for t in (0, 4, 5, 20, 21)] == [16, 16, 32, 32, 48]
    for bad in ({"B": 256}, {"C": 1 << 16}, {"IDB": 33}, {"EVERY": 0}, {"D": 26}, {"KO": 1 << 16}, {"KI": 0}):
        with pytest.raises(ValueError):
            IS.Class(**{"B": 512, "r": 4, "lanes": 4, "C": 47, **bad})
    with pytest.raises(ValueError, match=re.escape("KI <= 2082, the prompt ids a slot of C = 47 cells holds")):
        IS.Class(B=512, r=4, lanes=4, C=47, KI=2083)
    with pytest.raises(ValueError, match=re.escape("KI <= 147")):
        IS.Class(B=512, r=2, lanes=2, C=4, EVERY=2, WITHIN=3, D=5, KO=8)
    served = WardenParams(cell_bytes=512, slots=4, lanes=4, delay=27, ingress_cells=47, ingress_buckets=12,
                          egress_cells=65, window=32, every=16)
    w63.serves(served)
    assert (served.program_slots, served.window_programs) == (w63.SLOTS, 8)
    for bad, fix in (({"delay": 28}, "= D = 27"), ({"ingress_buckets": 13}, "<= WITHIN = 12"),
                     ({"ingress_cells": 48}, "<= C = 47"), ({"lanes": 5}, "= lanes = 4"),
                     ({"egress_cells": 66}, "<= KO + 1 = 65"), ({"every": 8}, "= EVERY = 16")):
        with pytest.raises(ValueError, match=re.escape(fix)):
            w63.serves(replace(served, **bad))


#: (row, sin, prog), the first bucket's sin the empty state's
Case = tuple[bytes, list[int], list[int]]


def _negatives(run) -> tuple[dict[str, Case], dict[str, tuple[int, Case]]]:
    """The cases a check refuses, and those of bucket t whose S_t is another than the scan's (the state row t, which
    the unit's o1 must open, refuses them)."""
    ch, rows, states = run.chain, run.rows, run.states

    def honest(t, **kw) -> Case:
        return tuple(kw.get(k, v) for k, v in zip(("row", "sin", "prog"), (ch.ingress[t], ch.states[t], ch.progs[t])))

    out, other = {}, {}
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
    other["digit"] = t, honest(t, row=_join(blk))
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:4] + bytes([blk[p][i][4] ^ 1]) + blk[p][i][5:]
    out["ordinal"] = honest(t, row=_join(blk))
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == PAD and HEADER.unpack_from(c)[1])
    blk = _blocks(rows[t])
    blk[p][i] = blk[p][i][:-1] + b"\x01"
    out["pad-byte"] = honest(t, row=_join(blk))
    out["phase-every"] = honest(2, sin=IS.encode(CLS, replace(states[2], phase=CLS.EVERY)))
    entries = IS.prog_decode(CLS, ch.progs[4])
    assert entries == [(3, RESPONSES[0] + (0,) * 5), (8, RESPONSES[1])]
    # request 3 is anchored at 10 in slot 0; slot 1 still holds request 2, live and not anchored there
    late = IS.prog_decode(CLS, ch.progs[10])
    assert late == [(4, RESPONSES[2] + (0,) * 4), (0, (0,) * CLS.KO)] and states[10].slots[1].live
    out["prog-off-anchor"] = honest(10, prog=IS.prog_encode(CLS, [late[0], (1, (7,))]))
    # slot 1 due at 4 with its occupancy word 0: its entry is masked, the honest one refused
    out["prog-empty-slot"] = honest(4, sin=IS.encode(CLS, replace(states[4], slots=[
        states[4].slots[0], replace(states[4].slots[1], occ=0)])))
    # slot 1's Program gives KO + 1 outputs, a request its class can't hold (`SlotEntry_v1`'s n = KO + 1, no ids)
    out["prog-over-KO"] = honest(4, prog=IS.prog_encode(CLS, [entries[0], (CLS.KO + 1, ())]))
    # request 3's slot one prompt id further on before its last cell: the cell takes its prompt to KI + 1 ids
    t, j = next((t, j) for t, st in enumerate(states) for j, s in enumerate(st.slots)
                if s.live and s.q == 3 and s.fields and not s.done)
    slots = list(states[t].slots)
    slots[j] = replace(slots[j], fields=slots[j].fields + 1)
    out["prompt-over-KI"] = honest(t, sin=IS.encode(CLS, replace(states[t], slots=slots)))
    return out, other


def _circuit(fn, args) -> list[list[int]]:
    """Each case's output bits."""
    out = np.asarray(evaluate_bits(fn, *[np.asarray(a, dtype=np.uint8) for a in args]))
    return [[int(x) for x in o] for o in out.reshape(len(args[0]), -1)]


def _scan_outputs(fn, cases) -> list[tuple[int, list[int], list[int]]]:
    """Each case's (o0, o1, o2): o0's value, S_t's bits and the view's."""
    NS = IS.state_bits(CLS)
    return [(sum(b << i for i, b in enumerate(o[:16])), o[16:16 + NS], o[16 + NS:])
            for o in _circuit(fn, IS.arguments(CLS, cases))]


def test_each_negative_is_refused_by_the_reference_or_computes_another_state(run):
    refused, other = _negatives(run)
    for name, case in refused.items():
        assert IS.reference(CLS, *case) == (0, None, None), name
    for name, (t, case) in other.items():
        ok, st, _ = IS.reference(CLS, *case)
        assert ok == 1 and st != run.chain.outputs(t)[0], name


@pytest.mark.slow
def test_the_circuits_are_the_references_on_every_bucket_each_negative_and_random_flips(run):
    """Slow (about 15 s: 217,065 and 137,969 ANDs traced, the ingress unit evaluated on about 90 cases, the first
    bucket's on about 30, and the egress unit on 31)."""
    ch = run.chain
    T = len(run.rows)
    honest = [(ch.ingress[t], ch.states[t], ch.progs[t]) for t in range(T)]
    refused, other = _negatives(run)
    negs = list(refused.values()) + [c for _, c in other.values()]
    rng = random.Random(7)
    flips, at = [], []
    for _ in range(60):
        at.append(rng.randrange(T))
        case = [bytearray(x) if k == 0 else list(x) for k, x in enumerate(honest[at[-1]])]
        k = rng.randrange(3)
        if k == 0:
            case[0][rng.randrange(len(case[0]))] ^= 1 << rng.randrange(8)
        else:
            case[k][rng.randrange(len(case[k]))] ^= 1
        flips.append((bytes(case[0]), *case[1:]))
    cases = honest + negs + flips
    got = _scan_outputs(UNIT, cases)
    for c, (o0, o1, o2) in zip(cases, got):
        ok, st, view = IS.reference(CLS, *c)
        assert o0 == ok and (not ok or (o1, o2) == (st, view))
    assert got[:T] == [(1, *ch.outputs(t)) for t in range(T)]
    assert [o0 for o0, _, _ in got[T:T + len(refused)]] == [0] * len(refused)
    assert [(o0, o1 != ch.outputs(t)[0]) for (t, _), (o0, o1, _) in zip(other.values(), got[T + len(refused):])] \
        == [(1, True)] * len(other)
    moved = [o1 != ch.outputs(t)[0] for t, (o0, o1, _) in zip(at, got[T + len(negs):]) if o0]
    assert sum(moved) >= 5, moved
    # the first bucket's unit is the ingress unit at the empty state, its outputs bit for bit where a check fails too
    starts = [(c[0], c[2]) for c in cases if c[1] == ch.states[0]]
    for _ in range(20):
        row, prog = bytearray(ch.ingress[0]), list(ch.progs[0])
        if rng.randrange(2):
            row[rng.randrange(len(row))] ^= 1 << rng.randrange(8)
        else:
            prog[rng.randrange(len(prog))] ^= 1
        starts.append((bytes(row), prog))
    first = _scan_outputs(FIRST, starts)
    assert first == _scan_outputs(UNIT, [(row, ch.states[0], prog) for row, prog in starts])
    assert first[0] == (1, *ch.outputs(0)) and 0 in [o0 for o0, _, _ in first]
    assert [o0 for o0, _, _ in first] == [IS.reference(CLS, row, None, prog)[0] for row, prog in starts]
    views = [ch.view(t) for t in range(T)] + [ch.view(t + 1) for t in range(T - 1)]
    got = _circuit(EGRESS, [list(col) for col in zip(*(EG.bucket_arguments(SHAPE, s) for s in views))])
    assert got == [EG.output_bits(SHAPE, s) for s in views]
    assert [EG.outputs(SHAPE, ch.view(t)) for t in range(T)] == [(EG.ACCEPT, row) for row in ch.egress]
