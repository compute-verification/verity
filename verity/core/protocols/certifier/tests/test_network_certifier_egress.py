"""N2's egress encoder (`egress`) on the warden's own egress: links of variable-length responses served through the
warden's schedule (`egress_traffic`, `warden.Warden._egress`), each link-window's map, ready bits and routing made by
the prover's side. The reference renders every bucket row the warden registered, the Boolean Definitions are their
references bit for bit on honest and adversarial inputs, a row reads only its ready data slots' ids, and the
verifier's registered read accepts the rendered rows and refuses each way they can differ."""

import random

import numpy as np
import pytest

from egress_traffic import busy, row, serve, window
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import egress as EG, head_check as HC, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, PAD
from verity.core.protocols.certifier.certificate import Direction

#: the warden's schedule the tests serve: T = 6 buckets of 4 lanes of r = 2 slots, D = 2
PARAMS = W.WardenParams(cell_bytes=256, slots=2, delay=2, lanes=4, window=6)
SESSION = bytes(range(32))


def _bytes(bits) -> bytes:
    bits = [int(b) for b in bits]
    return bytes(sum(bits[8 * k + i] << i for i in range(8)) for k in range(len(bits) // 8))


def _rows(fn, cases, B: int, R: int) -> list[tuple[bytes, list[int]]]:
    """The circuit's (o0 as bytes, o1's bits) on each case's (records, routed, ready)."""
    args = [EG.bucket_arguments(fn.bindings["NR"], *c) for c in cases]
    out = np.asarray(evaluate_bits(fn, *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    return [(_bytes(o[:8 * R * B]), [int(x) for x in o[8 * R * B:]]) for o in out.reshape(len(cases), -1)]


def _windows(served, NR, K):
    return [window(served, w, NR, K) for w in range(len(served.records))]


def test_the_shapes_and_the_statics_are_checked():
    sh = EG.Shape(256, 600, 64, 16, 4096)
    assert (sh.PB, sh.W, sh.D, sh.M, sh.switches) == (17, 49, 16, 65_536, 1_015_808)
    assert EG.entry_bits(5, 6) == 5 and EG.ready_words(16) == 1 and EG.ready_words(17) == 2
    refused = ((dict(B=255), "even"), (dict(B=146), "at least"), (dict(R=0), "R >= 1"), (dict(NR=0), "NR >= 1"),
               (dict(K=IT.END_ENTRY), "K < 2"), (dict(NR=1 << 16, K=1 << 16), "NR·K < 2"), (dict(T=0), "T >= 1"))
    for kw, match in refused:
        with pytest.raises(ValueError, match=match):
            EG.Shape(**(dict(B=256, T=2, R=4, NR=2, K=4) | kw))
        if "T" not in kw:
            statics = dict(B=256, R=4, NR=2, K=4) | kw
            with pytest.raises(ValueError, match=match):
                bind(EG.EgressBucket, **statics).signature()


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
    assert EG.honest_ready(1, slots, 6) == [1, 1, 0, 0, 0, 0]
    for args in ((0, slots, 4, 3, 1), (0, slots, 4, 2, -1)):
        with pytest.raises(ValueError, match="lanes of r"):
            EG.bucket_map(*args)


@pytest.mark.parametrize("seed, NR, K", [(1, 12, 16), (2, 12, 20), (3, 16, 18)])
def test_the_encoder_renders_the_wardens_registered_rows(seed, NR, K):
    """Every bucket row of every link-window of a busy link, idle ones too, is the warden's registered row
    (`warden_rows.window_rows` of the record its grid closed), from the window's ids, an honest map and the prover's
    switches; each DATA slot's routed entry is its record's (j·K + e, id)."""
    served = busy(seed, PARAMS, 14, 15, 6)
    assert not served.fails
    kinds, data = set(), 0
    for w, (record, win) in enumerate(zip(served.records, _windows(served, NR, K))):
        rows = WR.window_rows(record, PARAMS)
        assert not any(win.overflow)
        for t in range(PARAMS.window):
            assert row(win, t) == rows[t], (w, t)
            kinds |= {rows[t][s * PARAMS.cell_bytes] for s in range(win.shape.R)}
            for m, (p, x) in zip(win.maps[t], win.routed[t]):
                if m.e < K:
                    assert (p, x) == (m.j * K + m.e, win.ids[p])
                    data += 1
    assert kinds == {PAD, DATA, W.END} and data >= 60


def _edge_cases(rng, R, NR, K, n):
    """Random slots that reach every case of `cell`: q = 0 and 2^32 − 1, j at and past NR, e at and past K and both
    markers, routed indices right, with high bits set, wrong by one bit and the filler, ids 0, 2^32 − 1 and around
    powers of ten, and not-ready DATA and END."""
    PB = EG.entry_bits(NR, K)
    cases = []
    for _ in range(n):
        ready = [rng.randrange(2) for _ in range(NR)]
        records, routed = [], []
        for _ in range(R):
            q = rng.choice([0, 1, 7, (1 << 32) - 1, rng.randrange(1 << 32)])
            j = rng.choice([NR, NR + 1, (1 << 32) - 1, rng.randrange(1 << 32)] + [rng.randrange(NR)] * 6)
            e = rng.choice([IT.EMPTY, IT.END_ENTRY, K, K + 1, rng.randrange(1 << 32)] + [rng.randrange(K)] * 8)
            x = rng.choice([0, (1 << 32) - 1, 9, 10, 99, 100, 999_999_999, 1_000_000_000, rng.randrange(1 << 32),
                            rng.randrange(1000)])
            at = (j * K + e) % (1 << PB)
            p = rng.choice([at] * 4 + [at | rng.randrange(1, 1 << (32 - PB)) << PB, at ^ 1 << rng.randrange(PB), NR * K,
                                       rng.randrange(1 << 32)])
            records.append((q, j, e))
            routed.append((p, x))
        cases.append((records, routed, ready))
    return cases


@pytest.mark.parametrize("B, R, NR, K", [(256, 8, 5, 6), (148, 8, 17, 4), (150, 4, 1, 1), (256, 4, 3, 7)])
def test_the_bucket_circuit_is_its_reference(B, R, NR, K):
    """`EgressBucket` against `bucket_row` bit for bit, and its presence, on random slots through every case (K a
    power of two and not, NR across one ready word and two); every cell but the poison is one `parse_cell` reads."""
    rng = random.Random(B * R + NR * K)
    cases = _edge_cases(rng, R, NR, K, 480 // R)
    fn = bind(EG.EgressBucket, B=B, R=R, NR=NR, K=K)
    kinds = {}
    for c, (o0, o1) in zip(cases, _rows(fn, cases, B, R)):
        want = EG.bucket_row(B, NR, K, *c)
        assert o0 == want and o1 == [1] + [0] * 15
        for s in range(R):
            cell = want[s * B:(s + 1) * B]
            kinds[cell[0]] = kinds.get(cell[0], 0) + 1
            parsed = IT.parse_cell(cell)
            assert (parsed is None) == (cell == bytes([EG.POISON]) * B)
            if parsed is not None:
                q, j, e = c[0][s]
                assert parsed.stream == q and (parsed.id == c[1][s][1] if parsed.data else True)
    assert set(kinds) == {PAD, DATA, W.END, EG.POISON} and min(kinds.values()) >= 4


def test_the_bucket_circuit_renders_the_wardens_rows():
    """`EgressBucket` on every bucket of a busy link's busiest windows is the warden's registered row: what the
    reference's test can't reach is the circuit's wiring of the map, the routed entries and the ready words."""
    served = busy(4, PARAMS, 14, 15, 6)
    NR, K, B, R = 12, 16, PARAMS.cell_bytes, PARAMS.grid.allocation
    wins = _windows(served, NR, K)
    cases, want = [], []
    for w in sorted(range(len(wins)), key=lambda w: -sum(m.e < K for ms in wins[w].maps for m in ms))[:2]:
        rows = WR.window_rows(served.records[w], PARAMS)
        for t in range(PARAMS.window):
            cases.append((wins[w].maps[t], wins[w].routed[t], wins[w].ready[t]))
            want.append(rows[t])
    assert [o0 for o0, _ in _rows(bind(EG.EgressBucket, B=B, R=R, NR=NR, K=K), cases, B, R)] == want


def _maps(fn, cases, R):
    """`EgressMap`'s (records, overflow, o1's other bits) on each (b, requests)."""
    args = [EG.map_arguments(b, reqs) for b, reqs in cases]
    out = np.asarray(evaluate_bits(fn, *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    got = []
    for o in out.reshape(len(cases), -1):
        words = [sum(int(x) << i for i, x in enumerate(o[32 * k:32 * k + 32])) for k in range(3 * R)]
        got.append((tuple(EG.Record(*words[3 * s:3 * s + 3]) for s in range(R)), bool(o[96 * R]),
                    [int(x) for x in o[96 * R + 1:]]))
    return got


def _map_cases(rng, NR, D, n):
    """Buckets and request slots through every case of `bucket_map`: admitted after b, in the delay, at its outputs, at
    its END and past it; not live, with live's high bits set either way; ties in q; and u32 extremes (admitted + D
    past 2^32, n and b 2^32 − 1). A third of the buckets have every slot live, to overflow."""
    top = (1 << 32) - 1
    cases = []
    for c in range(n):
        b = rng.choice([rng.randrange(64)] * 4 + [top, rng.randrange(1 << 32)])
        reqs = []
        for _ in range(NR):
            q = rng.choice([1, 2, 3, top, rng.randrange(1 << 32)])
            if c % 3 == 0:
                reqs.append((q, b - rng.randrange(min(b, D) + 1), rng.choice([rng.randrange(12), top]), 1))
                continue
            a = rng.choice([b - rng.randrange(min(b, D + 6) + 1)] * 4 + [min(b + 1, top), top, rng.randrange(1 << 32)])
            n_ = rng.choice([rng.randrange(12)] * 3 + [0, top, rng.randrange(1 << 32)])
            live = rng.choice([1, 1, 1, 0, 2, 3, top - 1, top])
            reqs.append((q, a, n_, live))
        cases.append((b, reqs))
    return cases


@pytest.mark.parametrize("R, r, NR, D", [(8, 2, 5, 2), (6, 3, 4, 0), (4, 1, 7, 1), (12, 2, 6, 3), (16, 4, 3, 5)])
def test_the_map_circuit_is_its_reference(R, r, NR, D):
    """`EgressMap` against `map_records` on random buckets through every case (r a power of two and not, D = 0,
    more live slots than lanes and fewer): every kind of record, ties in q among the live slots, and overflow."""
    rng = random.Random(R * r + NR * D)
    cases = _map_cases(rng, NR, D, 300)
    got = _maps(bind(EG.EgressMap, R=R, r=r, NR=NR, D=D), cases, R)
    seen = {"data": 0, "end": 0, "pad": 0, "unheld": 0, "over": 0, "tie": 0}
    for (b, reqs), (records, over, rest) in zip(cases, got):
        want = EG.map_records(R, r, NR, D, b, reqs)
        assert (records, over) == want and rest == [0] * 15
        live = [q for q, a, n, lv in reqs if lv & 1 and a <= b <= a + D + n // r]
        seen["tie"] += len(set(live)) < len(live)
        seen["over"] += over
        for m in records:
            held = "unheld" if m == EG.UNHELD else "pad"
            seen["data" if m.e < IT.END_ENTRY else "end" if m.e == IT.END_ENTRY else held] += 1
    assert min(v for k, v in seen.items() if k != "over") >= 4 and (seen["over"] >= 4) == (R // r < NR), seen
    for bad, match in (((8, 3, 5, 2), "lanes of r"), ((8, 2, 0, 2), "NR"), ((8, 2, 5, -1), "delay")):
        with pytest.raises(ValueError, match=match):
            EG.map_records(*bad, 0, [(0, 0, 0, 0)] * max(bad[2], 0))
    with pytest.raises(ValueError, match="request slots"):
        EG.map_records(8, 2, 5, 2, 0, [(0, 0, 0, 0)] * 4)


def test_the_map_circuit_is_the_wardens_map():
    """`EgressMap` on every bucket of a busy link, its request slots the window's (`map_requests`), is the map the
    encoder's test renders the warden's rows from, overflow 0: what links the map stage to the encoder."""
    served = busy(4, PARAMS, 14, 15, 6)
    NR, K, T, r = 12, 16, PARAMS.window, PARAMS.slots
    cases, want = [], []
    for w, win in enumerate(_windows(served, NR, K)):
        reqs = EG.map_requests(win.slots, NR)
        for t in range(T):
            cases.append((w * T + t, reqs))
            want.append((win.maps[t], win.overflow[t]))
            assert EG.map_records(PARAMS.grid.allocation, r, NR, PARAMS.delay, w * T + t, reqs) == want[-1]
    assert not any(over for _, over in want) and sum(m.e < K for ms, _ in want for m in ms) > 40
    fn = bind(EG.EgressMap, R=PARAMS.grid.allocation, r=r, NR=NR, D=PARAMS.delay)
    assert [(m, o) for m, o, _ in _maps(fn, cases, PARAMS.grid.allocation)] == want


def test_a_row_reads_no_id_but_its_ready_data_slots():
    """The local lemma (the spec's §7): a bucket's row is a function of its map, its ready bits and the ids of its
    ready DATA slots whose routed index is theirs. Every other slot's routed id, and every routed index of a slot
    whose record isn't DATA, changed at random, leaves the circuit's row as it was."""
    B, R, NR, K = 256, 8, 5, 6
    rng = random.Random(9)
    cases = _edge_cases(rng, R, NR, K, 30)
    moved = []
    for records, routed, ready in cases:
        out = []
        for (q, j, e), (p, x) in zip(records, routed):
            cell = EG.cell(B, NR, K, (q, j, e), (p, x), ready)
            if cell[0] != DATA:
                x = rng.randrange(1 << 32)
                p = rng.randrange(1 << 32) if e >= K else p
            out.append((p, x))
        moved.append((records, out, ready))
    fn = bind(EG.EgressBucket, B=B, R=R, NR=NR, K=K)
    assert _rows(fn, moved, B, R) == _rows(fn, cases, B, R)
    assert any(r != m for c, d in zip(cases, moved) for r, m in zip(c[1], d[1]))


def test_the_router_places_each_entry_at_its_records_slot():
    """The prover's switches put entry j·K + e at the slot whose record is DATA (q, j, e), in every window of a busy
    link; any setting of the switches is a permutation of the entries. `EgressRoute` is `routed` on the window's ids,
    under the prover's switches and random ones."""
    served = busy(5, W.WardenParams(cell_bytes=256, slots=2, delay=1, lanes=2, window=3), 6, 4, 6)
    NR, K = 4, 5
    wins = _windows(served, NR, K)
    sh = wins[0].shape
    assert (sh.T * sh.R, NR * K, sh.D, sh.PB) == (12, 20, 5, 5)
    rng = random.Random(11)
    cases = []
    for win in wins:
        for t, (ms, rs) in enumerate(zip(win.maps, win.routed)):
            for m, (p, x) in zip(ms, rs):
                if m.e < K:
                    assert (p, x) == (m.j * K + m.e, win.ids[m.j * K + m.e])
        cases.append((win.ids, win.switches))
        noise = [rng.randrange(2) for _ in range(sh.switches)]
        cases.append((win.ids, noise))
        full = IT.benes_apply(EG.entries(sh, win.ids), noise)
        assert sorted(full) == sorted(EG.entries(sh, win.ids))
    fn = bind(EG.EgressRoute, NR=NR, K=K, T=sh.T, R=sh.R)
    args = [EG.route_arguments(*c) for c in cases]
    out = np.asarray(evaluate_bits(fn, *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    for (ids, bits), o in zip(cases, out.reshape(len(cases), sh.T * sh.R, 2, 32)):
        got = [(sum(int(b) << i for i, b in enumerate(p)), sum(int(b) << i for i, b in enumerate(x))) for p, x in o]
        assert got == EG.routed(sh, ids, bits)


def _committed(served, w):
    cells = WR.Cells.of("responses", Direction.EGRESS, served.records[w], PARAMS, SESSION)
    return cells, WR.bucket_port(cells.head, PARAMS, reads_from=3)


def test_the_registered_read_holds_the_rendered_rows():
    """The verifier's registered read (`warden_rows.refusal` on `bucket_port`: unit 3 + t reads row t) accepts a
    window's rendered rows committed under the warden's salts, and refuses a row with one bit flipped, a row moved to
    another bucket, and a row with one slot poisoned (a DATA record naming another output)."""
    served = busy(6, PARAMS, 14, 15, 6)
    NR, K, T = 12, 16, PARAMS.window
    wins = _windows(served, NR, K)
    w = max(range(len(wins)), key=lambda w: sum(m.e < K for ms in wins[w].maps for m in ms))
    win, (cells, port) = wins[w], _committed(served, w)
    rendered = [row(win, t) for t in range(T)]
    _words, salts, entry = cells.statement(range(T))

    def refusal(rows, positions=range(T)):
        e = entry | {"positions": list(positions), "paths": [cells.path(p).hex() for p in positions]}
        return WR.refusal(port, cells.domain, e, [WR.bc(r, s) for r, s in zip(rows, salts)])

    assert refusal(rendered) is None
    flipped = list(rendered)
    flipped[2] = bytes([rendered[2][0] ^ 1]) + rendered[2][1:]
    assert "does not open" in refusal(flipped)
    t = next(t for t in range(T - 1) if rendered[t] != rendered[t + 1])
    moved = list(range(T))
    moved[t], moved[t + 1] = t + 1, t
    swapped = [rendered[p] for p in moved]
    assert "does not open" in refusal(swapped)
    salts[t], salts[t + 1] = salts[t + 1], salts[t]
    assert "reads position" in refusal(swapped, moved)
    salts[t], salts[t + 1] = salts[t + 1], salts[t]
    t, s = next((t, s) for t in range(T) for s, m in enumerate(win.maps[t]) if m.e < K)
    m = win.maps[t][s]
    maps = list(win.maps[t])
    maps[s] = m._replace(e=(m.e + 1) % K)
    poisoned = EG.bucket_row(PARAMS.cell_bytes, NR, K, maps, win.routed[t], win.ready[t])
    assert poisoned[s * PARAMS.cell_bytes:(s + 1) * PARAMS.cell_bytes] == bytes([EG.POISON]) * PARAMS.cell_bytes
    assert "does not open" in refusal([poisoned if i == t else r for i, r in enumerate(rendered)])


def test_a_late_ingress_poisons_its_slots():
    """A DATA or END record of a request slot whose ready bit is 0 renders the poison, which no registered row is;
    the warden's row needs the slot ready."""
    served = serve(PARAMS, [0, 0], [[5, 6, 7], [8]], 2)
    win = window(served, 0, 2, 4)
    t = next(t for t in range(PARAMS.window) if any(m.e != IT.EMPTY for m in win.maps[t]))
    late = list(win.ready[t])
    late[0] = 0
    rendered = EG.bucket_row(PARAMS.cell_bytes, 2, 4, win.maps[t], win.routed[t], late)
    assert rendered != row(win, t) == WR.window_rows(served.records[0], PARAMS)[t]
    assert all(rendered[s * 256] == EG.POISON for s, m in enumerate(win.maps[t]) if m.j == 0 and m.e != IT.EMPTY)


def test_the_parts_ands():
    """The router is at most its Beneš network, W ANDs a switch (the constant index bits fold and the positions past
    T·R are pruned); the encoder is R slots of the same ANDs (at the warden's defaults, NR = 16 and K = 4,096,
    `ANDS_SLOT` a slot): what the circuit-check pins and the status extrapolate."""
    memo = {}
    for NR, K, T, R in ((3, 5, 3, 4), (4, 16, 6, 8)):
        sh = EG.Shape(256, T, R, NR, K)
        assert 0 < HC.ands(bind(EG.EgressRoute, NR=NR, K=K, T=T, R=R), memo) <= sh.W * sh.switches
    one = HC.ands(bind(EG.EgressBucket, B=256, R=1, NR=16, K=4096), memo)
    assert one == ANDS_SLOT and HC.ands(bind(EG.EgressBucket, B=256, R=8, NR=16, K=4096), memo) == 8 * one
    assert HC.ands(bind(EG.EgressMap, R=64, r=4, NR=16, D=10), memo) == ANDS_MAP


ANDS_SLOT = 1_004
#: `EgressMap`'s at the warden's defaults (R = 64 slots, lanes of r = 4, D = 10) and NR = 16
ANDS_MAP = 37_696
