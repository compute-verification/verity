"""N2's egress side of the scan (`egress`) on the warden's own egress: links of variable-length responses served
through the warden's schedule (`egress_traffic`, `warden.Warden._egress`), and the slot state each bucket's egress
reads. The reference renders every bucket row the warden registered from that state; the Boolean Definition computes
its reference's outputs, bit for bit, on honest and adversarial states (the row of a state that renders none too); it
reads only the state; and the verifier's registered read of its row output holds the computed rows to the warden's."""

import random

import numpy as np
import pytest

from egress_traffic import IB, busy, rows, serve, shape, states
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.protocols.certifier import egress as EG
from verity.core.protocols.certifier import head_check as HC, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, PAD
from verity.core.protocols.certifier.certificate import Direction

#: the warden's schedule the tests serve: T = 6 buckets of 4 lanes of r = 2 slots, D = 2
PARAMS = W.WardenParams(cell_bytes=256, slots=2, delay=2, lanes=4, window=6)
SESSION = bytes(range(32))
#: the request slots and outputs a slot the served links' states hold
NR, K = 4, 16


def _outputs(fn, sh, sts) -> list[list[int]]:
    """The circuit's output bits on each state."""
    args = [EG.bucket_arguments(sh, s) for s in sts]
    out = np.asarray(evaluate_bits(fn, *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    return [[int(x) for x in o] for o in out.reshape(len(sts), -1)]


def _row(sh, out: list[int]) -> bytes:
    """o1 of the circuit's output bits, as bytes."""
    bits = out[16:]
    return bytes(sum(b << i for i, b in enumerate(bits[8 * k:8 * k + 8])) for k in range(sh.R * sh.B))


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
        assert EG.rendered(sh, st) == want and EG.outputs(sh, st) == (EG.ACCEPT, want), t
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
    """`EgressBucket`'s outputs against `output_bits` on random states (r a power of two and not, K one and a power of
    two, NR above the lanes and below, D = 0, 32-bit ids), bit for bit: o0 both ways, its other 15 bits 0, and the row
    of a state that renders none (one that overflows, where NR is above the lanes, and one with a `DATA` cell of an
    output k ≥ K) among them."""
    sh = EG.Shape(B, R, r, NR, K, IB_, D)
    rng = random.Random(B * R + NR * K + D)
    sts = _states(rng, sh, 150)
    got = _outputs(_fn(sh), sh, sts)
    over = past = 0
    for st, out in zip(sts, got):
        assert out == EG.output_bits(sh, st)
        live = [EG.Request(s.q, s.admitted, s.n) if s.live & 1 else None for s in st.slots]
        o = EG.bucket_map(st.b, live, sh.R, sh.r, sh.D)[1]
        over += o
        past += not o and EG.rendered(sh, st) is None
    rendering = sum(o[0] for o in got)
    assert rendering >= 40 and past >= 1, (rendering, over, past)
    assert over >= 1 or NR <= sh.lanes


def test_the_bucket_circuit_computes_the_wardens_rows():
    """On every bucket of a busy link, the circuit computes the warden's registered row from the state the ingress
    step wrote, with o0 `ACCEPT`; under a state one of whose rendered output ids is another (what links the state's
    layout to the circuit) it computes another row, which no registered read of the warden's opens."""
    served = busy(4, PARAMS, 14, 15, 6)
    sh = shape(PARAMS, NR, K)
    sts, rs = states(served, NR, K), rows(served)
    changed = []
    for st, row in zip(sts, rs):
        records, _ = EG.bucket_map(st.b, [EG.Request(s.q, s.admitted, s.n) if s.live else None for s in st.slots],
                                   sh.R, sh.r, sh.D)
        data = [m for m in records if m.e < K]
        if data:
            m = data[0]
            ids = list(st.slots[m.j].ids)
            ids[m.e] = (ids[m.e] + 1) % (1 << IB)
            slots = list(st.slots)
            slots[m.j] = slots[m.j]._replace(ids=tuple(ids))
            changed.append((st._replace(slots=tuple(slots)), row))
    got = _outputs(_fn(sh), sh, sts + [st for st, _ in changed])
    assert [(o[0], _row(sh, o)) for o in got[:len(sts)]] == [(EG.ACCEPT, row) for row in rs]
    assert len(changed) > len(rs) // 3
    for (_, row), out in zip(changed, got[len(sts):]):
        assert out[0] == EG.ACCEPT and _row(sh, out) != row


def test_the_unit_reads_the_state_and_computes_the_row():
    """The unit's one input is the state's row (every switch computed, none witness); its outputs are o0, a u16, and
    o1, the row of R B-byte cells, gates of the circuit and never an input."""
    for sh in (shape(PARAMS, NR, K), EG.Shape(256, 16, 4, 4, 64, 17, 22)):
        params, out = _fn(sh).signature()
        assert [(name, t.leaves) for name, t in params] == [("state", sh.state_words * 16)]
        assert [t.leaves for t in out.fields] == [16, sh.R * sh.B * 8] and out.width_set() == frozenset({1})


def _committed(served, w):
    cells = WR.Cells.of("responses", Direction.EGRESS, served.records[w], PARAMS, SESSION)
    return cells, WR.bucket_port(cells.head, PARAMS, reads_from=3)


def test_the_registered_read_holds_the_row_output_to_the_wardens_rows():
    """The verifier's registered read of the unit's row output o1 (`warden_rows.refusal` on `bucket_port`: unit 3 + t
    reads row t) accepts the rows a window's units compute from its states, committed under the warden's salts, and
    refuses a row with one bit flipped and a row moved to another bucket, at its own position and at the other's."""
    served = busy(6, PARAMS, 14, 15, 6)
    T, sh = PARAMS.window, shape(PARAMS, NR, K)
    w = max(range(len(served.records)), key=lambda w: len(set(WR.window_rows(served.records[w], PARAMS))))
    cells, port = _committed(served, w)
    sts = states(served, NR, K)[w * T:(w + 1) * T]
    honest = [_row(sh, o) for o in _outputs(_fn(sh), sh, sts)]
    assert honest == list(WR.window_rows(served.records[w], PARAMS))
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


def test_a_request_not_yet_in_the_state_computes_another_row():
    """The row of a request's admission bucket holds its lane (a `PAD` on its q), so a state that omits a request the
    warden admitted, or admits it a bucket late, renders another row than the warden's, which its registered read
    refuses."""
    served = serve(PARAMS, [0, 0], [[5, 6, 7], [8]], 2)
    sh = shape(PARAMS, 2, 4)
    sts, rs = states(served, 2, 4), rows(served)
    omitted = sts[0]._replace(slots=(EG.FREE, sts[0].slots[1]))
    late = sts[0]._replace(slots=(sts[0].slots[0]._replace(admitted=1), sts[0].slots[1]))
    got = _outputs(_fn(sh), sh, [sts[0], omitted, late])
    assert [o[0] for o in got] == [EG.ACCEPT] * 3
    assert [_row(sh, o) == rs[0] for o in got] == [True, False, False]


def test_the_units_ands():
    """The unit's ANDs at the ZK driver's shape and at the scan note's class (16 cells of 256 bytes in 4 lanes, 4
    request slots of 64 outputs, 17-bit ids, D = 22): what circuit-check pins and the status extrapolates."""
    memo = {}
    assert HC.ands(_fn(shape(PARAMS, NR, K)), memo) == ANDS_ZK
    assert HC.ands(_fn(EG.Shape(256, 16, 4, 4, 64, 17, 22)), memo) == ANDS_CLASS


#: `EgressBucket_v1{B=256,R=8,r=2,NR=4,K=16,IB=18,D=2}`'s, egress_zk.py's unit
ANDS_ZK = 5_795
#: `EgressBucket_v1{B=256,R=16,r=4,NR=4,K=64,IB=17,D=22}`'s
ANDS_CLASS = 11_555
