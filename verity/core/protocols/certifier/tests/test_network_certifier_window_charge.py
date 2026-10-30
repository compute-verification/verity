"""window_charge: `ChargeCount_v2` charges advice v1's whole window, its hits and order entries with its decisions.
Held to v1's own charges (each fixture stream's windows at their own sizes, the hits included), to `code_v1`'s on
windows with hits and order entries seeded and on their cut into slots, to `ChargeCount_v1` on windows of decisions
alone, and the circuits to their reference."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.ml.protocols.certifier import window_charge as WC
from verity.core.protocols.certifier.head_check import ands
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.ml.commitments import hm96

FIXTURE = Path(__file__).parent / "fixtures" / "arith_encode_vectors.json"
PROGRAM, REGISTRATION = "window-charge-test", "ab" * 64


def _ints(s):
    return [int(c) for c in s]


def _streams():
    return [s for s in json.loads(FIXTURE.read_text())["streams"] if "window" in s]


def _whole(book, rows, lens):
    """Each of `rows` (decisions, hits, positions, ranks) at its own sizes, every slot enabled, the carry passed
    along: (charge, its length) each."""
    carry, out = WC.first_carry(book), []
    for (bits, hits, p, r, *_x), length in zip(rows, lens, strict=True):
        N, H, O = len(bits), len(hits), len(p)
        ok, got, carry = WC.reference_window(N, H, O, WC.window_book(book, N, H, O)[0], carry, [1] * N, _ints(bits),
                                             [1] * H, hits, [1] * O, p, r)
        assert ok
        out.append((got, length))
    return out


def _cut(stream):
    win = stream["window"]
    slots = [(_ints(en), _ints(bits), _ints(hen), short, _ints(oen), p, r) for en, bits, hen, short, oen, p, r, _l
             in win["cut"]]
    return win["N"], win["H"], win["O"], stream["decisions"]["book"], slots, [w[-1] for w in win["cut"]]


def test_each_window_at_its_own_sizes_is_v1s_charge():
    """Every fixture stream's own windows, decisions, hits and order entries each in as many slots as it sends, are
    charged v1's charges over the whole stream: async-prefix-cache's first window sends 27 hits, which
    `ChargeCount_v1` leaves out. The same windows with 12 hits and 10 order entries seeded in are charged
    `code_v1`'s, as is their cut into (4, 2, 2) slots with gaps."""
    windows = hits = seeded = cut = 0
    for s in _streams():
        win = s["window"]
        got = _whole(s["decisions"]["book"], win["own"], s["charge"])
        assert [g for g, _l in got] == s["charge"], s["source"]
        got = _whole(s["decisions"]["book"], win["seeded"], [w[-1] for w in win["seeded"]])
        assert all(g == length for g, length in got), s["source"]
        N, H, O, book, slots, lens = _cut(s)
        assert [x[0] for x in WC.session(N, H, O, book, slots)] == lens, s["source"]
        windows += len(win["own"])
        hits += sum(len(w[1]) for w in win["own"])
        seeded += sum(len(w[1]) + len(w[2]) for w in win["seeded"])
        cut += len(slots)
    assert (windows, hits, seeded, cut) == (15, 27, 115, 188)


def _random_book(rng, N, H, O):
    tables = []
    for _name, bins in WC._bins(N, H, O):
        freqs = [rng.choice([1, rng.randrange(1, 64), rng.randrange(1, 1 << 11)]) for _ in range(bins)]
        total = min(sum(freqs) + rng.choice([0, 0, rng.randrange(1 << 12)]), AE.MAX_TOTAL)
        if rng.random() < 0.02:
            freqs[rng.randrange(bins)] = rng.choice([0, 1 << 16, (1 << 17) - 1])
        tables.append((tuple(freqs), total))
    return WC.WindowBook(tuple(tables))


def test_a_window_of_decisions_alone_is_v1s():
    """With no hit or order slots the charge, the counts and the run are `ChargeCount_v1`'s, on the fixture's cut
    windows and on random ones, and pos and rank pass through."""
    rng = random.Random(0)
    for s in json.loads(FIXTURE.read_text())["streams"]:
        if "decisions" not in s:
            continue
        dec = s["decisions"]
        N = dec["cut"]["N"]
        (v1book, carry), wb = AE.charge_book(dec["book"], N), WC.window_book(dec["book"], N, 0, 0)[0]
        c2 = WC.WindowCarry(carry.counts, carry.run, 7, 9)
        for en, bits, _l in dec["cut"]["windows"]:
            want = AE.reference_charge(N, v1book, carry, _ints(en), _ints(bits))
            got = WC.reference_window(N, 0, 0, wb, c2, _ints(en), _ints(bits))
            assert got[:2] == want[:2] and got[2] == WC.WindowCarry(want[2].counts, want[2].run, 7, 9)
            carry, c2 = want[2], got[2]
    for _ in range(300):
        N = rng.choice([1, 2, 5, 14])
        book = _random_book(rng, N, 0, 0)
        v1book = AE.ChargeBook(book["n_arrived"][0], book["n_arrived"][1],
                               tuple((book[n][0][0], book[n][1]) for n in AE.ZERO_COUNTS))
        counts = tuple((rng.randrange(1 << 14), rng.randrange(1 << 14)) for _ in range(AE.RUNS))
        run = rng.randrange(4)
        en, bits = [rng.randrange(2) for _ in range(N)], [rng.randrange(2) for _ in range(N)]
        want = AE.reference_charge(N, v1book, AE.Carry(counts, run), en, bits)
        got = WC.reference_window(N, 0, 0, book, WC.WindowCarry(counts, run), en, bits)
        assert got[0] == want[0] and (not want[0] or got[1:] == (want[1], WC.WindowCarry(want[2].counts, want[2].run)))


def test_the_coding_is_code_pys():
    """`binned_symbols` is `code.Binned.put`'s coding as `code.py` writes it: a bin from the table, then the offset
    uniform, one symbol up to 2^16 values and two past it, nothing for a bin of one value."""
    freqs = list(range(1, 28))
    total = sum(freqs)
    assert WC.binned_symbols(0, freqs, total, WC.GAP_CAP) == [(0, 1, total)]
    assert WC.binned_symbols(1, freqs, total, WC.GAP_CAP) == [(1, 2, total)]
    assert WC.binned_symbols(5, freqs, total, WC.GAP_CAP) == [(6, 4, total), (1, 1, 4)]
    assert WC.binned_symbols((1 << 16) + 3, freqs, total, WC.GAP_CAP) == [(sum(freqs[:17]), 18, total), (3, 1, 1 << 16)]
    v = (1 << 23) + (5 << 15) + 7
    assert WC.binned_symbols(v, freqs, total, WC.GAP_CAP) == [(sum(freqs[:24]), 25, total), (5, 1, 1 << 8),
                                                              (7, 1, 1 << 15)]
    assert WC.binned_symbols(WC.GAP_CAP, freqs, total, WC.GAP_CAP) == [(sum(freqs[:25]), 26, total)]
    assert WC.binned_symbols(WC.ZIG_CAP, freqs, total, WC.ZIG_CAP) == [(sum(freqs[:26]), 27, total), (1, 1, 2)]
    assert WC.binned_symbols(WC.HIT_CAP, freqs, total, WC.HIT_CAP) == [(sum(freqs[:17]), 18, total)]
    assert [WC.zigzag(d) for d in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]


def test_refusals():
    """A carry-in's run past 2 or rank past 2^24 + 1, a hit past hit_units, an order entry before the last one's
    position, a gap past 2^24, a rank past 2^24 or a position of 2^32 − 1; and the books `window_book` refuses."""
    book = _random_book(random.Random(1), 1, 1, 1)
    good = WC.WindowCarry(rank=(1 << 24) + 1, pos=10)
    one = ([1], [0])

    def ok(carry=good, short=(0,), p=(10,), r=(0,)):
        return WC.reference_window(1, 1, 1, book, carry, *one, [1], list(short), [1], list(p), list(r))[0]

    assert ok() == 1 and ok(short=(1 << 16,), p=(10 + (1 << 24),), r=(1 << 24,)) == 1
    for bad in (dict(carry=WC.WindowCarry(run=3)), dict(carry=WC.WindowCarry(rank=(1 << 24) + 2)),
                dict(short=((1 << 16) + 1,)), dict(p=(9,)), dict(p=(11 + (1 << 24),)), dict(r=((1 << 24) + 1,)),
                dict(carry=WC.WindowCarry(pos=(1 << 32) - 1), p=((1 << 32) - 1,))):
        assert ok(**bad) == 0, bad
    assert WC.reference_window(1, 1, 1, book, good, *one, [0], [(1 << 17) - 1], [0], [9], [(1 << 25) - 1])[0] == 1
    raw = json.loads(FIXTURE.read_text())["streams"][0]["decisions"]["book"]
    assert WC.window_book(raw, 4, 2, 2)[1] == WC.first_carry(raw)
    for caps, N, H, O in (({"gap": 1 << 20}, 1, 1, 1), ({"order_len": 2}, 1, 1, 1), ({"hits": 2}, 1, 4, 1),
                          ({"order": 0}, 1, 1, 1), ({}, 1, 1 << 16, 1)):
        with pytest.raises(ValueError):
            WC.window_book({**raw, "caps": {**raw["caps"], **caps}}, N, H, O)
    with pytest.raises(ValueError):
        WC.window_book({**raw, "models": {**raw["models"], "order_gap": {"binned": {"lo": 0, "freqs": [1] * 25,
                                                                                    "escape": 1}}}}, 1, 1, 1)


def _random_slots(rng, N, H, O, carry):
    """Slots at a random density, the values at their caps' edges now and then and past them, the order entries'
    positions after the carry's or before it."""
    en, bits = [int(rng.random() < 0.6) for _ in range(N)], [rng.randrange(2) for _ in range(N)]
    def past(good, bad):
        return rng.choice(bad) if rng.random() < 0.05 else rng.choice(good)

    hen = [int(rng.random() < 0.6) for _ in range(H)]
    short = [past([0, 1, rng.randrange(1 << 16), 1 << 15, 1 << 16], [(1 << 16) + 1, rng.randrange(1 << 17)])
             for _ in range(H)]
    oen, p, r = [int(rng.random() < 0.6) for _ in range(O)], [], []
    pos, rank = carry.pos, carry.rank
    for _ in range(O):
        g = past([0, 1, rng.randrange(1 << 16), (1 << 16) + rng.randrange(1 << 10), rng.randrange(1 << 24), 1 << 24],
                 [(1 << 24) + 1, -1])
        pp = pos + g if 0 <= pos + g < 1 << 32 else rng.randrange(1 << 32)
        rr = past([0, rank, max(rank - 1, 0), min(rank + rng.randrange(64), 1 << 24), rng.randrange(1 << 24), 1 << 24],
                  [(1 << 24) + 1])
        p.append(pp)
        r.append(rr)
        pos, rank = (pp + 1, rr + 1) if pp < (1 << 32) - 1 and rr < 1 << 25 else (pos, rank)
    return en, bits, hen, short, oen, p, r


def _random_carry(rng):
    counts = []
    for _ in range(AE.RUNS):
        s = rng.choice([(1 << 15) - 1, rng.randrange(1 << 15), rng.randrange(40)])
        a = rng.randrange(s + 1)
        counts.append((a, s - a))
    return WC.WindowCarry(tuple(counts), rng.choice([0, 1, 2, 3]) if rng.random() < 0.05 else rng.randrange(3),
                          rng.choice([0, rng.randrange(1 << 31), (1 << 32) - 2]),
                          (1 << 24) + 2 if rng.random() < 0.05 else rng.choice([0, rng.randrange(1 << 24), (1 << 24) + 1]))


def _random_link(rng, N, H, O):
    book, carry = _random_book(rng, N, H, O), _random_carry(rng)
    slots = _random_slots(rng, N, H, O, carry)
    row_in = list(WC.encode(carry))
    if rng.random() < 0.1:
        row_in[rng.randrange(149, WC.ROW_BITS)] = 1
    ok, _length, nxt = WC.reference_window(N, H, O, book, carry, *slots)
    row_out = list(WC.encode(nxt) if ok else WC.encode(carry))
    if rng.random() < 0.2:
        row_out[rng.randrange(WC.ROW_BITS)] ^= 1
    return (book, row_in, row_out, *slots)


@pytest.mark.slow
def test_the_chain_on_the_cut_is_code_v1s():
    """Each fixture stream's cut windows chained under hiding commitments: the verifier's check passes, and
    `ChargeLink_v2{4, 2, 2}` on each window's committed rows is ok at `code_v1`'s charge. A carry-in other than the last
    carry-out is refused by the chain (CH-carry), and the session must start at the book's first carry (CH-first)."""
    for s in _streams():
        N, H, O, book, slots, lens = _cut(s)
        ses = WC.session(N, H, O, book, slots)
        values = []
        for _length, _c_in, c_out in ses:
            values.append(WC.commit_window(PROGRAM, book, WC.carry_of(c_out), REGISTRATION,
                                           values[-1] if values else None))
        ops = [list(v.bcs) for v in values]
        assert WC.check_chain(ops, PROGRAM, book) is None
        assert WC.check_chain(ops, "another-program", book).startswith("CH-first")
        if len(ops) > 2:
            assert WC.check_chain([ops[0], ops[2]], PROGRAM, book).startswith("CH-carry")
        wb = WC.window_book(book, N, H, O)[0]
        cases = [(wb, WC.CC.unrow(v.rows[0]), WC.CC.unrow(v.rows[1]), *sl) for v, sl in zip(values, slots)]
        assert WC.evaluate_links(N, H, O, cases) == [(1, length) for length in lens], s["source"]
    v = values[0]
    assert v.salts[0] == WC.CC.start_salt(PROGRAM, WC.VALUE) and v.bcs[0] == WC.start_opening(PROGRAM, book)


@pytest.mark.slow
def test_the_circuits_are_the_reference():
    """`ChargeLink_v2{4, 2, 2}` against `reference_link` on random cases (books with skewed bins and now and then a
    bin the coder refuses, carries at their edges, values at and past their caps, carry-outs one bit off), and
    `ChargeCount_v2{1, 1, 1}`'s outputs, the carry' included, against `reference_window`."""
    rng = random.Random(422)
    cases = [_random_link(rng, 4, 2, 2) for _ in range(400)]
    want = [WC.reference_link(4, 2, 2, *c) for c in cases]
    assert sum(w[0] for w in want) > 100 and sum(not w[0] for w in want) > 100
    for c, g, w in zip(cases, WC.evaluate_links(4, 2, 2, cases), want):
        assert (g[0] == 0) if w[0] == 0 else g == w, (c, g, w)
    cases = []
    for _ in range(300):
        book, carry = _random_book(rng, 1, 1, 1), _random_carry(rng)
        cases.append((book, carry, *_random_slots(rng, 1, 1, 1, carry)))
    want = [WC.reference_window(1, 1, 1, *c) for c in cases]
    assert sum(w[0] for w in want) > 100 and sum(not w[0] for w in want) > 50
    for c, g, w in zip(cases, WC.evaluate_windows(1, 1, 1, cases), want):
        assert (g[0] == 0) if w[0] == 0 else g == w, (c, g, w)


@pytest.mark.slow
def test_build_link_inline_and_the_bound():
    """`build_link`, the call a caller's circuit makes, is the unit, and a caller holding its charge to c_w (as the
    compile slice does) accepts c_w = the charge and refuses one bit less."""
    from verity.ml.circuits.boolean.forms import Circuit, sub
    N, H, O, rng = 1, 1, 1, random.Random(3)
    cases = [_random_link(rng, N, H, O) for _ in range(40)]
    C = Circuit(cse=True)
    fields = WC.book_fields(N, H, O)
    pub = C.input("pub", sum(n * w for _f, n, w in fields))
    row_in, row_out = C.input("in", WC.ROW_BITS), C.input("out", WC.ROW_BITS)
    en, bits, hen, oen = C.input("en", N), C.input("bits", N), C.input("hen", H), C.input("oen", O)
    short, p, r, c = C.input("short", 17), C.input("p", 32), C.input("r", 25), C.input("c", AE.CW)
    elems, at = [], 0
    for _f, n, w in fields:
        elems += [pub[at + k * w:at + (k + 1) * w] for k in range(n)]
        at += n * w
    ok, count = WC.build_link(C, N, H, O, elems, row_in, row_out, en, bits, hen, [short], oen, [p], [r])
    C.output("res", [ok, sub(C, c, count, AE.CW)[1].inv()] + count)

    def sliced(rows):
        return [sum(int(row[i]) << t for t, row in enumerate(rows)) for i in range(len(rows[0]))]

    def ints(xs, w):
        return [[(x >> j) & 1 for j in range(w)] for x in xs]

    for drop in (0, 1):
        pubs = [[b for f in AE._args(fields, WC.book_values(cs[0])) for b in f] for cs in cases]
        want = [WC.reference_link(N, H, O, *cs) for cs in cases]
        bound = [max(w[1] - drop, 0) for w in want]
        outs, _fail = C.evaluate({
            "pub": sliced(pubs), "in": sliced([cs[1] for cs in cases]), "out": sliced([cs[2] for cs in cases]),
            "en": sliced([cs[3] for cs in cases]), "bits": sliced([cs[4] for cs in cases]),
            "hen": sliced([cs[5] for cs in cases]), "short": sliced(ints([cs[6][0] for cs in cases], 17)),
            "oen": sliced([cs[7] for cs in cases]), "p": sliced(ints([cs[8][0] for cs in cases], 32)),
            "r": sliced(ints([cs[9][0] for cs in cases], 25)), "c": sliced(ints(bound, AE.CW))}, len(cases))
        for t, (cs, w) in enumerate(zip(cases, want)):
            res = [(v >> t) & 1 for v in outs["res"]]
            g = (res[0], sum(b << j for j, b in enumerate(res[2:])))
            assert (g[0] == 0) if w[0] == 0 else g == w, (cs, g, w)
            if w[0]:
                assert res[1] == int(not drop or w[1] == 0)


def test_rows():
    """The 160-bit row: `ChargeCount_v1`'s 92 bits, then pos and rank, then zeros; its commitment under this module's
    own value, the session's first row the book's first carry under the value's public salt."""
    carry = WC.WindowCarry(((1, (1 << 15) - 1), (7, 0), (0, 3)), 2, (1 << 32) - 1, (1 << 24) + 1)
    bits = WC.encode(carry)
    assert len(bits) == WC.ROW_BITS == 160 and not any(bits[149:]) and WC.carry_of(bits) == carry
    assert bits[92:124] == (1,) * 32 and bits[124:149] == tuple(((1 << 24) + 1 >> j) & 1 for j in range(25))
    assert WC.carry_of(bits[:149] + (1,) * 11) == carry
    for bad in (WC.WindowCarry(pos=1 << 32), WC.WindowCarry(rank=1 << 25), WC.WindowCarry(((0, 0),) * 2)):
        with pytest.raises(ValueError):
            WC.encode(bad)
    book = _streams()[0]["decisions"]["book"]
    v0 = WC.commit_window(PROGRAM, book, carry, REGISTRATION)
    v1 = WC.commit_window(PROGRAM, book, carry, REGISTRATION, v0)
    assert v0.rows[0] == WC.CC.row(WC.encode(WC.first_carry(book))) and v0.bcs[0] == WC.start_opening(PROGRAM, book)
    assert v0.salts[0] != WC.CC.start_salt(PROGRAM) and v0.bcs[0] != WC.CC.start_opening(PROGRAM, book)
    assert (v1.rows[0], v1.salts[0], v1.bcs[0]) == (v0.rows[1], v0.salts[1], v0.bcs[1])
    assert WC.reads() == {WC.VALUE: {"carry_in": 0, "carry_out": 1}}
    forged = produced.commit(PROGRAM, WC.VALUE, [WC.CC.row(WC.encode(WC.WindowCarry())), v0.rows[1]], REGISTRATION,
                             [hm96.SHA512.fresh_salt(), hm96.SHA512.fresh_salt()], WC.CC.WORD_BITS, WC.CC.SCHEMA)
    assert WC.check_chain([list(forged.bcs)], PROGRAM, book).startswith("CH-first")


@pytest.mark.slow
def test_size_counts_the_gates():
    assert WC.size("charge", 2, 0, 0)["ands"] == AE.size("charge", N=2)["ands"] + 24 == 19560
    assert WC.size("charge", 1, 1, 1)["ands"] == ands(WC.charge_unit(1, 1, 1)) == 53287


@pytest.mark.slow
def test_no_and_reads_a_form_its_earlier_slots_grew():
    """The carried words are committed after each hit and each order entry's value, as after each decision, so the
    widest AND operand stays put as slots are added (the count's popcount moves it a few terms) and the build is linear
    in H and O. Without the copies each hit widens it by about 1,200 terms and each order entry by about 3,500."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.boolean.forms import Circuit

    def widest(N, H, O):
        S = SimpleNamespace(N=N, H=H, O=O)
        C = Circuit(cse=True)
        params, _ = WC._sig_window(S)
        ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(BL.elements(t))]
        C.output("out", [b for e in WC._build_window(C, S, *ins) for b in e])
        return max(len(C.data[v][k].s) for v in C.live() if C.kind[v] == "and" for k in (0, 1))

    assert widest(0, 4, 0) - widest(0, 2, 0) < 100 and widest(0, 0, 3) - widest(0, 0, 2) < 100
