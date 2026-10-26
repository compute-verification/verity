"""N2's egress side of the scan (`egress`) on the warden's own egress: links of variable-length responses served
through the warden's schedule (`egress_traffic`, `warden.Warden._egress`), and the slot state each bucket's egress
reads. The reference renders every bucket row the warden registered from that state; the Boolean Definition is its
reference's verdict on honest and adversarial states and rows, accepts the warden's rows and refuses every other; its
one output is a 16-bit word; and the verifier's registered read of the row port holds the rows to the warden's."""

import random

import numpy as np
import pytest

from egress_traffic import IB, busy, rows, serve, shape, states
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import egress as EG
from verity.core.protocols.certifier import head_check as HC, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, PAD
from verity.core.protocols.certifier.certificate import Direction

#: the warden's schedule the tests serve: T = 6 buckets of 4 lanes of r = 2 slots, D = 2
PARAMS = W.WardenParams(cell_bytes=256, slots=2, delay=2, lanes=4, window=6)
SESSION = bytes(range(32))
#: the request slots and outputs a slot the served links' states hold
NR, K = 4, 16


def _verdicts(fn, sh, cases) -> list[list[int]]:
    """The circuit's output bits on each (state, row)."""
    args = [EG.bucket_arguments(sh, s, r) for s, r in cases]
    out = np.asarray(evaluate_bits(fn, *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    return [[int(x) for x in o] for o in out.reshape(len(cases), -1)]


def _fn(sh):
    return bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D)


def test_the_shape_and_the_statics_are_checked():
    """The state's layout at the scan note's class (4 request slots of 64 outputs, 17-bit ids) and its checks."""
    sh = EG.Shape(256, 16, 4, 4, 64, 17, 22)
    assert (sh.lanes, sh.slot_bits, sh.state_bits, sh.state_words, sh.slot_at(1)) == (4, 1200, 4832, 302, 1232)
    assert EG.Shape(148, 4, 2, 3, 5, 7, 0).slot_bits == 160
    refused = ((dict(B=255), "even"), (dict(B=146), "at least"), (dict(R=0), "lanes of r"), (dict(r=3), "lanes of r"),
               (dict(NR=0), "NR < 2"), (dict(K=IT.END_ENTRY), "K < 2"), (dict(IB=0), "IB <= 32"),
               (dict(IB=33), "IB <= 32"), (dict(D=-1), "delay"))
    for kw, match in refused:
        statics = dict(B=256, R=4, r=2, NR=2, K=4, IB=8, D=1) | kw
        with pytest.raises(ValueError, match=match):
            EG.Shape(**statics)
        with pytest.raises(ValueError, match=match):
            bind(EG.EgressBucket, **statics).signature()


def test_the_state_row_is_its_layout():
    """b, then each request slot's live, q, admitted, n and ids at their offsets, zero-padded; refused past its
    capacities."""
    sh = EG.Shape(148, 4, 2, 3, 5, 7, 1)
    st = EG.State(9, (EG.Slot(3, 7, 2, 4, (1, 127)), EG.FREE, EG.Slot(1, (1 << 32) - 1, 5, 0, ())))
    bits = EG.state_bits(sh, st)
    assert len(bits) == sh.state_bits == 16 * len(EG.state_words(sh, st))

    def field(at, w):
        return sum(b << i for i, b in enumerate(bits[at:at + w]))

    assert field(0, 32) == 9
    for j, s in enumerate((st.slots[0], EG.FREE, st.slots[2])):
        at = sh.slot_at(j)
        assert (field(at + EG.LIVE, 16), field(at + EG.Q, 32), field(at + EG.ADMITTED, 32), field(at + EG.N, 32)) \
            == s[:4]
        assert [field(at + EG.IDS + k * 7, 7) for k in range(5)] == list(s.ids) + [0] * (5 - len(s.ids))
        assert not any(bits[at + EG.IDS + 35:at + sh.slot_bits])
    words = EG.state_words(sh, st)
    assert words[0] == 9 and words[1] == 0
    for bad, match in ((st._replace(slots=st.slots + (EG.FREE,)), "NR = 3"),
                       (EG.State(0, (EG.Slot(1, 1, 0, 0, (128,)),)), "IB = 7"),
                       (EG.State(0, (EG.Slot(1, 1, 0, 0, (0,) * 6),)), "K = 5"),
                       (EG.State(0, (EG.Slot(1 << 16, 1, 0, 0, ()),)), "u16"),
                       (EG.State(1 << 32, ()), "u32")):
        with pytest.raises(ValueError, match=match):
            EG.state_bits(sh, bad)


def test_the_map_is_the_wardens_slots():
    """Two lanes of r = 2, D = 1: request slot 0 (stream 1, admitted at 0, outputs 3), slot 1 (stream 2, admitted at 0,
    no output) and slot 2 (stream 3, admitted at 2, one output). Bucket 0 is the delay's pads; at bucket 1 slot 0's
    outputs 0 and 1 and slot 1's END; at bucket 2 slot 0's output 2 and END, and slot 2's pads, its lane freed by slot
    1; a third live request overflows."""
    E, P = IT.END_ENTRY, IT.EMPTY
    slots = [EG.Request(1, 0, 3), EG.Request(2, 0, 0), EG.Request(3, 2, 1), None]
    maps = [EG.bucket_map(b, slots, 4, 2, 1) for b in range(5)]
    R = EG.Record
    assert maps[0] == ((R(1, 0, P), R(1, 0, P), R(2, 1, P), R(2, 1, P)), False)
    assert maps[1] == ((R(1, 0, 0), R(1, 0, 1), R(2, 1, E), R(2, 1, P)), False)
    assert maps[2] == ((R(1, 0, 2), R(1, 0, E), R(3, 2, P), R(3, 2, P)), False)
    assert maps[3] == ((R(3, 2, 0), R(3, 2, E), EG.UNHELD, EG.UNHELD), False)
    assert maps[4] == ((EG.UNHELD,) * 4, False)
    over = EG.bucket_map(2, [EG.Request(1, 0, 3), EG.Request(2, 2, 0), EG.Request(3, 2, 1)], 4, 2, 1)
    assert over == ((R(1, 0, 2), R(1, 0, E), R(2, 1, P), R(2, 1, P)), True)
    for args in ((0, slots, 4, 3, 1), (0, slots, 4, 2, -1)):
        with pytest.raises(ValueError, match="lanes of r"):
            EG.bucket_map(*args)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_state_renders_the_wardens_registered_rows(seed):
    """Every bucket row of a busy link, idle ones too, is the warden's registered row (`warden_rows.window_rows` of
    the record its grid closed), rendered from the state the scan's ingress step wrote at that bucket (each request
    in the lowest free slot from its admission to its last egress bucket)."""
    served = busy(seed, PARAMS, 14, 15, 6)
    assert not served.fails
    sh, kinds = shape(PARAMS, NR, K), set()
    for t, (st, want) in enumerate(zip(states(served, NR, K), rows(served))):
        assert EG.rendered(sh, st) == want and EG.verdict(sh, st, want) == EG.ACCEPT, t
        kinds |= {want[s * sh.B] for s in range(sh.R)}
    assert kinds == {PAD, DATA, W.END}


def _states(rng, sh, n):
    """Random states through every case of the map: slots not live (live's high bits set either way), admitted after
    b, in the delay, at their outputs, at their END and past it; ties in q; n past K; u32 extremes (admitted + D past
    2^32, b and n 2^32 − 1); ids 0 and 2^IB − 1. A third have every slot live at b, to overflow when NR > R / r."""
    top, out = (1 << 32) - 1, []
    for c in range(n):
        b = rng.choice([rng.randrange(40)] * 5 + [top, rng.randrange(1 << 32)])
        slots = []
        for _ in range(sh.NR):
            q = rng.choice([1, 2, 3, top, rng.randrange(1 << 32)])
            ids = tuple(rng.choice([0, (1 << sh.IB) - 1, rng.randrange(1 << sh.IB)])
                        for _ in range(rng.randrange(sh.K + 1)))
            if c % 3 == 0:
                slots.append(EG.Slot(1, q, b - rng.randrange(min(b, sh.D + 3) + 1), rng.randrange(2 * sh.K), ids))
                continue
            a = rng.choice([b - rng.randrange(min(b, sh.D + 6) + 1)] * 4
                           + [min(b + 1, top), top, rng.randrange(1 << 32)])
            n_ = rng.choice([rng.randrange(sh.K + 1)] * 4 + [sh.K + 1, top, rng.randrange(1 << 32)])
            live = rng.choice([1, 1, 1, 0, 2, 3, (1 << 16) - 2, (1 << 16) - 1])
            slots.append(EG.Slot(live, q, a, n_, ids))
        out.append(EG.State(b, tuple(slots)))
    return out


@pytest.mark.parametrize("B, R, r, NR, K, IB_, D", [(148, 4, 2, 3, 5, 7, 1), (150, 6, 3, 2, 4, 5, 0),
                                                    (148, 2, 1, 3, 1, 3, 2), (148, 4, 1, 2, 8, 32, 3)])
def test_the_bucket_circuit_is_its_reference(B, R, r, NR, K, IB_, D):
    """`EgressBucket`'s output against `verdict` on random states (r a power of two and not, K one and a power of two,
    NR above the lanes and below, D = 0, 32-bit ids), each with the row it renders, that row with one bit flipped, the
    row another state renders, and random bytes: every verdict both ways, states that render no row among them, and
    the other 15 bits 0."""
    sh = EG.Shape(B, R, r, NR, K, IB_, D)
    rng = random.Random(B * R + NR * K + D)
    sts = _states(rng, sh, 150)
    cases, none = [], 0
    for i, st in enumerate(sts):
        want = EG.rendered(sh, st)
        none += want is None
        if want is not None:
            cases.append((st, want))
            k = rng.randrange(len(want) * 8)
            cases.append((st, bytes(x ^ (1 << k % 8 if j == k // 8 else 0) for j, x in enumerate(want))))
        other = EG.rendered(sh, sts[i - 1])
        cases.append((st, other if other is not None else bytes(B * R)))
        cases.append((st, bytes(rng.randrange(256) for _ in range(B * R))))
    got = _verdicts(_fn(sh), sh, cases)
    for (st, row), out in zip(cases, got):
        assert out == [EG.verdict(sh, st, row)] + [0] * 15
    accepted = sum(o[0] for o in got)
    assert accepted >= 40 and len(cases) - accepted >= 40 and none >= 1, (accepted, len(cases), none)


def test_the_bucket_circuit_accepts_the_wardens_rows_and_refuses_any_other():
    """On every bucket of a busy link, the circuit accepts the warden's registered row under the state the ingress
    step wrote, and refuses that row with one bit flipped, the next bucket's row, and the row under a state one of
    whose rendered output ids is another (what links the state's layout to the circuit)."""
    served = busy(4, PARAMS, 14, 15, 6)
    sh = shape(PARAMS, NR, K)
    sts, rs = states(served, NR, K), rows(served)
    rng = random.Random(4)
    cases, want = [], []
    for t, (st, row) in enumerate(zip(sts, rs)):
        k = rng.randrange(len(row) * 8)
        flipped = bytes(x ^ (1 << k % 8 if j == k // 8 else 0) for j, x in enumerate(row))
        cases += [(st, row), (st, flipped)]
        want += [1, 0]
        if t + 1 < len(rs) and rs[t + 1] != row:
            cases.append((st, rs[t + 1]))
            want.append(0)
        records, _ = EG.bucket_map(st.b, [EG.Request(s.q, s.admitted, s.n) if s.live else None for s in st.slots],
                                   sh.R, sh.r, sh.D)
        data = [m for m in records if m.e < K]
        if data:
            m = data[0]
            ids = list(st.slots[m.j].ids)
            ids[m.e] = (ids[m.e] + 1) % (1 << IB)
            slots = list(st.slots)
            slots[m.j] = slots[m.j]._replace(ids=tuple(ids))
            cases.append((st._replace(slots=tuple(slots)), row))
            want.append(0)
    assert len(cases) > 3 * len(rs) and sum(want) == len(rs)
    assert [o[0] for o in _verdicts(_fn(sh), sh, cases)] == want


def test_the_unit_reads_two_rows_and_outputs_one_word():
    """A proof unit's outputs are at most 16 bits: the unit's one output is a u16, and its only inputs are the
    state's row and the warden's row (every switch computed, none witness)."""
    for sh in (shape(PARAMS, NR, K), EG.Shape(256, 16, 4, 4, 64, 17, 22)):
        params, out = _fn(sh).signature()
        assert out.leaves == 16 and out.width_set() == frozenset({1})
        assert [(name, t.leaves) for name, t in params] == [("state", sh.state_words * 16), ("row", sh.R * sh.B * 8)]


def _committed(served, w):
    cells = WR.Cells.of("responses", Direction.EGRESS, served.records[w], PARAMS, SESSION)
    return cells, WR.bucket_port(cells.head, PARAMS, reads_from=3)


def test_the_registered_read_holds_the_row_port_to_the_wardens_rows():
    """The verifier's registered read of the unit's ``row`` port (`warden_rows.refusal` on `bucket_port`: unit 3 + t
    reads row t) accepts a window's rows committed under the warden's salts, and refuses a row with one bit flipped and
    a row moved to another bucket, at its own position and at the other's."""
    served = busy(6, PARAMS, 14, 15, 6)
    T = PARAMS.window
    w = max(range(len(served.records)), key=lambda w: len(set(WR.window_rows(served.records[w], PARAMS))))
    cells, port = _committed(served, w)
    honest = list(WR.window_rows(served.records[w], PARAMS))
    _words, salts, entry = cells.statement(range(T))

    def refusal(rows_, positions=range(T)):
        e = entry | {"positions": list(positions), "paths": [cells.path(p).hex() for p in positions]}
        return WR.refusal(port, cells.domain, e, [WR.bc(r, s) for r, s in zip(rows_, salts)])

    assert refusal(honest) is None
    flipped = list(honest)
    flipped[2] = bytes([honest[2][0] ^ 1]) + honest[2][1:]
    assert "does not open" in refusal(flipped)
    t = next(t for t in range(T - 1) if honest[t] != honest[t + 1])
    moved = list(range(T))
    moved[t], moved[t + 1] = t + 1, t
    swapped = [honest[p] for p in moved]
    assert "does not open" in refusal(swapped)
    salts[t], salts[t + 1] = salts[t + 1], salts[t]
    assert "reads position" in refusal(swapped, moved)


def test_a_request_not_yet_in_the_state_is_refused():
    """The row of a request's admission bucket holds its lane (a `PAD` on its q), so a state that omits a request the
    warden admitted, or admits it a bucket late, renders another row: the verdict is 0."""
    served = serve(PARAMS, [0, 0], [[5, 6, 7], [8]], 2)
    sh = shape(PARAMS, 2, 4)
    sts, rs = states(served, 2, 4), rows(served)
    assert EG.verdict(sh, sts[0], rs[0]) == EG.ACCEPT
    omitted = sts[0]._replace(slots=(EG.FREE, sts[0].slots[1]))
    late = sts[0]._replace(slots=(sts[0].slots[0]._replace(admitted=1), sts[0].slots[1]))
    for st in (omitted, late):
        assert EG.verdict(sh, st, rs[0]) == 0
    assert [o[0] for o in _verdicts(_fn(sh), sh, [(sts[0], rs[0]), (omitted, rs[0]), (late, rs[0])])] == [1, 0, 0]


def test_the_units_ands():
    """The unit's ANDs at the ZK driver's shape and at the scan note's class (16 cells of 256 bytes in 4 lanes, 4
    request slots of 64 outputs, 17-bit ids, D = 22): what circuit-check pins and the status extrapolates."""
    memo = {}
    assert HC.ands(_fn(shape(PARAMS, NR, K)), memo) == ANDS_ZK
    assert HC.ands(_fn(EG.Shape(256, 16, 4, 4, 64, 17, 22)), memo) == ANDS_CLASS


#: `EgressBucket_v1{B=256,R=8,r=2,NR=4,K=16,IB=18,D=2}`'s, egress_zk.py's unit
ANDS_ZK = 22_179
#: `EgressBucket_v1{B=256,R=16,r=4,NR=4,K=64,IB=17,D=22}`'s
ANDS_CLASS = 44_323
