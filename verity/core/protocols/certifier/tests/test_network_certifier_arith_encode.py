"""arith_encode: advice v1's arithmetic encoder as Boolean Definitions, held to the advice coders (`code.py`, `code_v1`)
on what v1 actually codes (the fixture's streams, as `benchmarks/network_e2e/encodings/arith_vectors.py --steps` cut
them: advice v1's windows, the replay's decisions, the arrival order's deviations and the hits sent past the pool
bound, on the replay's step traces with v1's charges, and the derived encoding's still-sent fields on the trace corpus
fixture, each symbol tagged a bit or an integer's, with code.py's state after every symbol and its segment), to the
references on random runs, and refusing every other segment and every symbol code.py refuses."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier.head_check import ands

FIXTURE = Path(__file__).parent / "fixtures" / "arith_encode_vectors.json"
K = AE.K_MAX


def _streams():
    data = json.loads(FIXTURE.read_text())
    assert data["K"] == K
    return data["streams"]


def _seg(win):
    return [int(c) for c in win["segment"]]


def _look(seg, pos, k=K):
    return (seg[pos:pos + k] + [0] * k)[:k]


def _general(sym):
    """A tagged symbol as code.py's (cum, freq, total): a bit's is `encode(f0 if bit else 0, f1 if bit else f0, f0 + f1)`."""
    if sym[0]:
        _, b, f0, f1 = sym
        return [f0 if b else 0, f1 if b else f0, f0 + f1]
    return list(sym[1:])


def _runs(stream):
    """Each window's reference run along its segment: [(BIN, state, symbol, lookahead, (next, n))], and the finish."""
    for win in stream["windows"]:
        seg, st, pos, steps = _seg(win), AE.fresh(), 0, []
        for BIN, *sym in win["symbols"]:
            look = _look(seg, pos)
            nxt = AE.reference_step(K, BIN, st, 1, sym, look)
            steps.append((BIN, st, sym, look, nxt))
            st, pos = nxt[0], pos + nxt[1]
        yield win, steps, (st, _look(seg, pos, 2), pos)


def test_the_reference_is_code_py():
    seen = {0: 0, 1: 0}
    for stream in _streams():
        for win, steps, (st, tail, pos) in _runs(stream):
            for k, ((BIN, _s, _sym, _l, (nxt, _n)), (low, high, pending, used)) in enumerate(
                    zip(steps, win["states"], strict=True)):
                assert (nxt.ok, nxt.low, nxt.high, nxt.open, nxt.used) == (1, low, high, int(pending > 0), used), \
                    f"{stream['source']} symbol {k}"
                seen[BIN] += 1
            ok, n = AE.reference_finish(st, tail)
            assert ok and pos + n == len(win["segment"]), stream["source"]
    assert seen[1] >= 300 and seen[0] >= 200


def _model_cases(stream):
    """Each bit's (C, counts, en, ctx, bit) under the book's model, from its priors, and its estimate in the stream."""
    tables = {name: [tuple(p) for p in ps] for name, ps in stream["priors"].items()}
    freqs = [tuple(sym[2:]) for win in stream["windows"] for sym in win["symbols"] if sym[0]]
    assert len(freqs) == len(stream["kt"]), stream["source"]
    for (name, ctx, bit), f in zip(stream["kt"], freqs):
        c = (len(tables[name]), tuple(tables[name]), 1, ctx, bit)
        tables[name] = AE.reference_book_kt(*c)[3]
        yield c, f, name


def test_the_model_is_the_books():
    """Each bit's estimate is the book's (`code_v1.KT`), and each decision's context min(run, 2), the decisions since the
    last arrival, carried across windows: a function of the bits alone."""
    seen = 0
    for stream in _streams():
        run = 0
        for c, f, name in _model_cases(stream):
            ok, f0, f1, _ = AE.reference_book_kt(*c)
            assert ok and (f0, f1) == f, stream["source"]
            if name == "arrived":
                assert c[3] == min(run, 2), stream["source"]
                run = 0 if c[4] else run + 1
                seen += 1
    assert seen >= 300


def test_every_other_segment_is_refused():
    """Each fixture window's segment is accepted, and every single-bit flip of it, one bit more and one less refused."""
    for stream in _streams():
        for win in stream["windows"]:
            seg, syms = _seg(win), [_general(s) for s in win["symbols"]]
            assert AE.check_window(K, 0, syms, seg)
            assert not AE.check_window(K, 0, syms, seg + [0])
            assert not AE.check_window(K, 0, syms, seg + [1])
            if seg:
                assert not AE.check_window(K, 0, syms, seg[:-1])
            for k in range(len(seg)):
                assert not AE.check_window(K, 0, syms, seg[:k] + [1 - seg[k]] + seg[k + 1:])


def _same(K_, BIN, cases):
    got = AE.evaluate_steps(K_, BIN, cases)
    for c, g in zip(cases, got):
        want = AE.reference_step(K_, BIN, *c)
        assert (g[0].ok == 0) if want[0].ok == 0 else g == want, (c, g, want)
    return got


def test_the_circuits_are_the_references_on_the_fixture():
    rng = random.Random(0)
    cases, fins, models = {0: [], 1: []}, [], {}
    for stream in _streams():
        for _win, steps, (st, tail, _pos) in _runs(stream):
            for BIN, s, sym, look, _ in steps:
                flip = list(look)
                flip[rng.randrange(K)] ^= 1
                cases[BIN] += [(s, 1, sym, look), (s, 0, sym, look), (s, 1, sym, flip)]
            fins.append((st, tail))
        for (nc, *c), _f, _name in _model_cases(stream):
            models.setdefault(nc, []).append(tuple(c))
    for BIN in (0, 1):
        _same(K, BIN, cases[BIN])
    assert AE.evaluate_finishes(fins) == [AE.reference_finish(*f) for f in fins]
    for nc, cs in models.items():
        assert AE.evaluate_book_kt(nc, cs) == [AE.reference_book_kt(nc, *c) for c in cs]


def _same_book_kt(C, cases):
    got = AE.evaluate_book_kt(C, cases)
    for c, g in zip(cases, got):
        want = AE.reference_book_kt(C, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)
    return got


@pytest.mark.parametrize("C", [1, 3])
def test_the_models_are_their_references_on_random_counts(C):
    """`BookKT_v1` and `KTContext_v1` on random counts, the book's skewed to the halving edge, where the bit's count
    reaching n0 + n1 = 2^15 halves both."""
    rng = random.Random(C)
    book, kt = [], []
    for _ in range(300):
        counts = []
        for _c in range(C):
            s = rng.choice([(1 << 15) - 1, (1 << 15) - 2, rng.randrange(1 << 15), rng.randrange(40)])
            a = rng.randrange(s + 1)
            counts.append((a, s - a))
        ctx = rng.randrange(1 << AE._bitlen(C - 1))
        book.append((tuple(counts), rng.choice([1, 1, 1, 0]), ctx, rng.randrange(2)))
        kt.append((tuple((rng.randrange(1 << 15), rng.randrange(1 << 15)) for _c in range(C)), rng.choice([1, 1, 0]),
                   ctx, rng.randrange(2)))
    want = [AE.reference_book_kt(C, *c) for c in book]
    halved = [w for w, c in zip(want, book) if w[0] and c[1] and sum(w[3][c[2]]) < sum(c[0][c[2]])]
    assert len(halved) > 10 and sum(not w[0] for w in want) > (10 if C == 3 else 0)
    _same_book_kt(C, book)
    got = AE.evaluate_kt(C, 16, kt)
    for c, g in zip(kt, got):
        w = AE.reference_kt(C, 16, *c)
        assert (g[0] == 0) if w[0] == 0 else g == w, (c, g, w)


class _WNC:
    """A plain WNC encoder at 32 bits, to give the random runs segments to read (the fixture holds the references to
    code.py's own)."""

    def __init__(self):
        self.low, self.high, self.pending, self.used, self.bits = 0, AE.TOP, 0, False, []

    def emit(self, b):
        self.bits += [b] + [1 - b] * self.pending
        self.pending = 0

    def encode(self, cum, freq, total):
        if freq == total:
            return
        self.used, r = True, self.high - self.low + 1
        self.high, self.low = self.low + r * (cum + freq) // total - 1, self.low + r * cum // total
        while True:
            if self.high < AE.HALF:
                self.emit(0)
            elif self.low >= AE.HALF:
                self.emit(1)
                self.low, self.high = self.low - AE.HALF, self.high - AE.HALF
            elif self.low >= AE.Q1 and self.high < 3 * AE.Q1:
                self.pending += 1
                self.low, self.high = self.low - AE.Q1, self.high - AE.Q1
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        if self.used:
            self.pending += 1
            self.emit(0 if self.low < AE.Q1 else 1)
        return self.bits


def _symbol(rng, BIN):
    """A random symbol, skewed to the extremes (rare symbols scale up to 17 times), now and then one code.py refuses."""
    if BIN:
        t = rng.choice([2, 3, 17, 1000, AE.MAX_TOTAL, rng.randrange(2, AE.MAX_TOTAL + 1)])
        f0 = rng.choice([1, t - 1, rng.randrange(1, t)])
        sym = (rng.randrange(2), f0, t - f0)
        if rng.random() < 0.03:
            sym = rng.choice([(1, 0, 5), (0, 7, 0), (1, 40000, 30000)])
        return sym
    t = rng.choice([1, 2, 3, 17, 1000, AE.MAX_TOTAL, rng.randrange(1, AE.MAX_TOTAL + 1)])
    cum = rng.choice([0, t - 1, rng.randrange(0, t)])
    sym = (cum, rng.choice([1, t - cum, rng.randrange(1, t - cum + 1)]), t)
    if rng.random() < 0.03:
        sym = rng.choice([(0, 0, 5), (3, 4, 6), (0, 1, AE.MAX_TOTAL + 1)])
    return sym


def _encode(BIN, syms):
    e = _WNC()
    for s in syms:
        e.encode(*((s[1] if s[0] else 0, s[2] if s[0] else s[1], s[1] + s[2]) if BIN else s))
    return e.finish()


@pytest.mark.parametrize("BIN", [1, 0])
def test_the_circuit_is_the_reference_on_random_runs(BIN):
    rng = random.Random(BIN)
    cases = []
    for _ in range(60):
        syms = [_symbol(rng, BIN) for _ in range(rng.randrange(1, 30))]
        valid = [s for s in syms if AE._symbol(BIN, s)[0]]
        seg, st, pos = _encode(BIN, valid), AE.fresh(), 0
        assert AE.check_window(K, BIN, valid, seg)
        for s in syms:
            look = _look(seg, pos)
            flip = list(look)
            flip[rng.randrange(K)] ^= 1
            cases += [(st, 1, s, look), (st, 0, s, look), (st, 1, s, flip)]
            if AE._symbol(BIN, s)[0]:
                st, n = AE.reference_step(K, BIN, st, 1, s, look)
                pos += n
    want = [AE.reference_step(K, BIN, *c) for c in cases]
    assert any(w[1] >= 15 for w in want) and sum(not w[0].ok for w in want) > 50
    _same(K, BIN, cases)


def test_refusals():
    st = AE.fresh()
    for BIN, sym in [(1, (1, 0, 5)), (1, (0, 5, 0)), (1, (1, 40000, 30000)), (0, (0, 0, 5)), (0, (3, 4, 6)),
                     (0, (0, 1, AE.MAX_TOTAL + 1))]:
        (got,) = _same(K, BIN, [(st, 1, sym, [0] * K)])
        assert not got[0].ok
        (got,) = _same(K, BIN, [(st, 0, sym, [0] * K)])
        assert got == (st, 0)
    # a symbol of probability 2^-16 needs 16 scalings, which K = 3 slots can't hold
    sym = (1, AE.MAX_TOTAL - 1, 1)
    seg = _encode(1, [sym])
    assert AE.reference_step(K, 1, st, 1, sym, _look(seg, 0))[1] == 16
    (got,) = _same(3, 1, [(st, 1, sym, seg[:3])])
    assert not got[0].ok
    # code.py's early return: the only possible symbol leaves the window unused, and its finish consumes nothing
    (got,) = _same(K, 0, [(st, 1, (0, 9, 9), [1] * K)])
    assert got == (st, 0)
    assert AE.evaluate_finishes([(st, [1, 1])]) == [AE.reference_finish(st, [1, 1])] == [(1, 0)]
    # the model: a context past C, a count past NW bits, and a disabled decision that changes nothing
    full = ((0, 0), ((1 << 15) - 1, 0), (0, 0))
    cases = [(full, 1, 3, 0), (full, 1, 1, 0), (full, 1, 1, 1), (full, 0, 3, 1)]
    want = [AE.reference_kt(3, 15, *c) for c in cases]
    assert [w[0] for w in want] == [0, 0, 1, 1] and want[3] == (1, 0, 0, full)
    got = AE.evaluate_kt(3, 15, cases)
    assert [g[0] for g in got[:2]] == [0, 0] and got[2:] == want[2:]


def _same_counts(K_, BIN, CW, cases):
    got = AE.evaluate_counts(K_, BIN, CW, cases)
    for c, g in zip(cases, got):
        want = AE.reference_count_step(K_, BIN, CW, *c)
        assert (g.ok == 0) if want.ok == 0 else g == want, (c, g, want)
    return got


def _same_count_finishes(CW, cases):
    got = AE.evaluate_count_finishes(CW, cases)
    for c, g in zip(cases, got):
        want = AE.reference_count_finish(CW, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)
    return got


def _tallies(stream):
    """Each window's counting run: [(BIN, state, symbol)] and its last tally."""
    for win in stream["windows"]:
        st, run = AE.Tally(), []
        for BIN, *sym in win["symbols"]:
            run.append((BIN, st, sym))
            st = AE.reference_count_step(K, BIN, AE.CW, st, 1, sym)
        yield win, run, st


def test_the_count_is_code_pys_segment_length():
    """On every fixture window the counting reference keeps code.py's interval and `used` after each symbol, its count
    is the segment code.py wrote, and its finish gives v1's charge: accepted at that charge, refused one bit below it."""
    seen = 0
    for stream in _streams():
        for w, (win, run, st) in enumerate(_tallies(stream)):
            after = [s for _b, s, _sym in run[1:]] + [st]
            for k, (s, (low, high, _pending, used)) in enumerate(zip(after, win["states"], strict=True)):
                assert (s.ok, s.low, s.high, s.used) == (1, low, high, used), f"{stream['source']} symbol {k}"
            n, c = len(win["segment"]), stream["charge"][w]
            assert st.bits + 2 * st.used == n == c
            assert AE.reference_count_finish(AE.CW, st, c) == (1, c)
            if c:
                assert AE.reference_count_finish(AE.CW, st, c - 1)[0] == 0
            seen += 1
    assert seen >= 16


def test_the_counting_circuits_are_the_references():
    """On the fixture's symbols and on random runs, each with its step disabled too, and each window's finish at its
    charge and one bit below it."""
    rng = random.Random(2)
    fixture, ends = {0: [], 1: []}, []
    for stream in _streams():
        for w, (_win, run, st) in enumerate(_tallies(stream)):
            for BIN, s, sym in run:
                fixture[BIN] += [(s, 1, sym), (s, 0, sym)]
            c = stream["charge"][w]
            ends += [(st, c), (st, max(c - 1, 0))]
    _same_count_finishes(AE.CW, ends)
    for BIN in (1, 0):
        cases, fins = list(fixture[BIN]), []
        runs = [[_symbol(rng, BIN) for _ in range(rng.randrange(1, 30))] for _ in range(60)]
        for syms in runs:
            st = AE.Tally()
            for s in syms:
                cases += [(st, 1, s), (st, 0, s)]
                if AE._symbol(BIN, s)[0]:
                    st = AE.reference_count_step(K, BIN, AE.CW, st, 1, s)
            valid = [s for s in syms if AE._symbol(BIN, s)[0]]
            n = len(_encode(BIN, valid))
            assert AE.count_window(K, BIN, valid) == n
            fins += [(st, n), (st, max(n - 1, 0))]
        want = [AE.reference_count_step(K, BIN, AE.CW, *c) for c in cases]
        assert any(w.bits - c[0].bits >= 15 for w, c in zip(want, cases) if w.ok) and sum(not w.ok for w in want) > 20
        _same_counts(K, BIN, AE.CW, cases)
        _same_count_finishes(AE.CW, fins)


def test_counting_refusals():
    st = AE.Tally()
    for BIN, sym in [(1, (1, 0, 5)), (1, (0, 5, 0)), (1, (1, 40000, 30000)), (0, (0, 0, 5)), (0, (3, 4, 6)),
                     (0, (0, 1, AE.MAX_TOTAL + 1))]:
        (got,) = _same_counts(K, BIN, AE.CW, [(st, 1, sym)])
        assert not got.ok
        assert _same_counts(K, BIN, AE.CW, [(st, 0, sym)]) == [st]
    # a symbol of probability 2^-16 takes 16 scalings: refused at K = 3, and by a 4-bit count
    sym = (1, AE.MAX_TOTAL - 1, 1)
    assert AE.reference_count_step(K, 1, AE.CW, st, 1, sym).bits == 16
    assert not _same_counts(3, 1, 4, [(st, 1, sym)])[0].ok
    assert not _same_counts(K, 1, 5, [(AE.Tally(bits=16), 1, sym)])[0].ok
    two = AE.reference_count_step(K, 1, 5, st, 1, (1, 3, 1))
    assert two.ok and two.bits == 2 and _same_counts(K, 1, 5, [(st, 1, (1, 3, 1))]) == [two]
    # code.py's early return: the only possible symbol leaves the window unused, and its finish counts nothing
    (got,) = _same_counts(K, 0, AE.CW, [(st, 1, (0, 9, 9))])
    assert got == st
    # the finish: two bits once used, held to c, and a count that overflows CW bits refused
    used, full = AE.Tally(used=1, bits=13), AE.Tally(used=1, bits=14)
    got = _same_count_finishes(4, [(st, 0), (used, 15), (used, 14), (full, 15)])
    assert got[:2] == [(1, 0), (1, 15)] and [g[0] for g in got[2:]] == [0, 0]
    with pytest.raises(ValueError):
        AE.size("count_finish", CW=1)


def test_book_kt_refusals():
    """A context past C, counts the book can't hold (n0 + n1 ≥ 2^15), and a disabled bit that changes nothing."""
    full = ((0, 0), ((1 << 15) - 1, 1), (3, 4))
    cases = [(full, 1, 3, 0), (full, 1, 1, 0), (full, 1, 2, 1), (full, 0, 3, 1)]
    want = [AE.reference_book_kt(3, *c) for c in cases]
    assert [w[0] for w in want] == [0, 0, 1, 1] and want[2] == (1, 7, 9, ((0, 0), ((1 << 15) - 1, 1), (3, 5)))
    assert want[3] == (1, 0, 0, full)
    got = _same_book_kt(3, cases)
    assert [g[0] for g in got[:2]] == [0, 0]
    # the halving: n0 + n1 reaching 2^15 halves both, rounding up
    edge = (((1 << 14) - 1, 1 << 14),)
    assert _same_book_kt(1, [(edge, 1, 0, 0)])[0] == (1, (1 << 15) - 1, (1 << 15) + 1, ((1 << 13, 1 << 13),))


def _ints(s):
    return [int(c) for c in s]


def _charge_streams():
    return [s for s in _streams() if "decisions" in s]


def _cut_cases(stream):
    """The stream's cut windows as `ChargeCount_v1`'s cases at the cut's N, each carry the reference's, with
    `code_v1`'s charge for each."""
    dec = stream["decisions"]
    N = dec["cut"]["N"]
    book, carry = AE.charge_book(dec["book"], N)
    for en, bits, length in dec["cut"]["windows"]:
        case = (book, carry, _ints(en), _ints(bits))
        want = AE.reference_charge(N, *case)
        yield N, case, want, length
        carry = want[2]


def test_the_charge_is_code_v1s():
    """`reference_charge` is `code_v1`'s charge: on each fixture window's decisions alone, the KT counts and the run
    carried from window to window, and so v1's own charge where the window sends only decisions; and on the decisions
    cut into windows of the compile slice's P·A = 14 slots with disabled slots between."""
    seen = only = cut = 0
    for stream in _charge_streams():
        dec = stream["decisions"]
        carry = AE.charge_book(dec["book"], 1)[1]
        for w, (bits, length, alone) in enumerate(dec["windows"]):
            n = max(len(bits), 1)
            pad = [0] * (n - len(bits))
            ok, got, carry = AE.reference_charge(n, AE.charge_book(dec["book"], n)[0], carry, [1] * len(bits) + pad,
                                                 _ints(bits) + pad)
            assert ok and got == length, f"{stream['source']} window {w}"
            if alone:
                assert got == stream["charge"][w], f"{stream['source']} window {w}"
                only += 1
            seen += len(bits)
        for N, _case, (ok, got, _carry), length in _cut_cases(stream):
            assert N == 14 and ok and got == length, stream["source"]
            cut += 1
    assert seen >= 300 and only >= 10 and cut >= 30


def _random_charge(rng, N):
    """A random `ChargeCount_v1{N}` case: a book with skewed bins, now and then one the coder refuses; a carry at the
    halving edge, now and then run 3; slots enabled at a random density, all or none."""
    nb = N.bit_length() + 1
    arrived = [rng.choice([1, rng.randrange(1, 64), rng.randrange(1, 1 << 12)]) for _ in range(nb)]
    total = min(sum(arrived) + rng.choice([0, 0, rng.randrange(1 << 12)]), AE.MAX_TOTAL)
    zeros = []
    for _ in AE.ZERO_COUNTS:
        t = rng.choice([2, 1000, AE.MAX_TOTAL, rng.randrange(1, AE.MAX_TOTAL + 1)])
        zeros.append((rng.choice([1, t, rng.randrange(1, t + 1)]), t))
    if rng.random() < 0.05:
        arrived[rng.randrange(nb)] = rng.choice([0, 1 << 16, (1 << 17) - 1])
    if rng.random() < 0.05:
        zeros[rng.randrange(len(zeros))] = rng.choice([(0, 5), (6, 5), (1, AE.MAX_TOTAL + 1)])
    counts = []
    for _ in range(AE.RUNS):
        s = rng.choice([(1 << 15) - 1, (1 << 15) - 2, rng.randrange(1 << 15), rng.randrange(40)])
        a = rng.randrange(s + 1)
        counts.append((a, s - a))
    run = rng.choice([0, 1, 2, 2, 3]) if rng.random() < 0.1 else rng.randrange(3)
    p = rng.choice([0.0, 0.08, 0.3, 0.8, 1.0])
    book = AE.ChargeBook(tuple(arrived), total, tuple(zeros))
    return (book, AE.Carry(tuple(counts), run), [int(rng.random() < p) for _ in range(N)],
            [int(rng.random() < 0.3) for _ in range(N)])


def _same_charges(N, cases):
    got = AE.evaluate_charges(N, cases)
    for c, g in zip(cases, got):
        want = AE.reference_charge(N, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)
    return got


def test_the_charge_circuit_is_the_reference():
    """`ChargeCount_v1{14}`, the compile slice's class, against the reference: on the fixture's cut windows (so against
    `code_v1`'s charges too, its carry passed along) and on random cases, which reach every bin of n_arrived, the KT
    counts' halving and the refusals."""
    rng = random.Random(14)
    fixture = [(case, length) for s in _charge_streams() for N, case, _w, length in _cut_cases(s)]
    got = _same_charges(14, [c for c, _l in fixture])
    assert [(g[0], g[1]) for g in got] == [(1, length) for _c, length in fixture]
    cases = [_random_charge(rng, 14) for _ in range(400)]
    want = [AE.reference_charge(14, *c) for c in cases]
    assert {sum(c[2]).bit_length() for c in cases} == set(range(5))
    assert sum(not w[0] for w in want) > 20 and sum(w[0] for w in want) > 200
    _same_charges(14, cases)


def test_build_charge_inline():
    """`build_charge`, the call a caller's circuit makes, is the unit: its public inputs split by `charge_fields`."""
    from verity.ml.circuits.boolean.forms import Circuit
    N, rng = 3, random.Random(3)
    cases = [_random_charge(rng, N) for _ in range(40)]
    C = Circuit(cse=True)
    fields = AE.charge_fields(N)
    pub, en, bits = C.input("pub", sum(n * w for _f, n, w in fields)), C.input("en", N), C.input("bits", N)
    elems, at = [], 0
    for _f, n, w in fields:
        elems += [pub[at + k * w:at + (k + 1) * w] for k in range(n)]
        at += n * w
    ok, count, carry = AE.build_charge(C, N, elems, en, bits)
    C.output("out", [ok] + count + [b for e in carry for b in e])

    def sliced(rows):
        return [sum(r[i] << t for t, r in enumerate(rows)) for i in range(len(rows[0]))]

    pubs = [[b for name, n, w in fields for v in AE.charge_values(bk, cy)[name] for b in AE._bits(v, w)]
            for bk, cy, _e, _b in cases]
    outs, _fail = C.evaluate({"pub": sliced(pubs), "en": sliced([c[2] for c in cases]),
                              "bits": sliced([c[3] for c in cases])}, len(cases))
    for t, c in enumerate(cases):
        row = [(v >> t) & 1 for v in outs["out"]]
        f = AE._fields(AE._charge_out(SimpleNamespace(N=N)), row)
        g = (f["ok"][0], f["bits"][0], AE.Carry(tuple(zip(f["n0"], f["n1"])), f["run"][0]))
        want = AE.reference_charge(N, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)


def test_charge_refusals():
    """A carry's run past 2 with no decision to read it, an n_arrived bin whose cumulative frequency passes 2^16, and
    the books `charge_book` refuses: a top bin the cap cuts short, a model that isn't Exp-Golomb, a table short of its
    cap's bin."""
    book = AE.ChargeBook((5, 5, 5, 5, 5), 25, ((3, 4),) * 4)
    zero = [0] * 14
    cases = [(book, AE.Carry(run=3), zero, zero), (book, AE.Carry(run=2), zero, zero),
             (AE.ChargeBook((40000, 40000, 1, 1, 1), 65535, ((3, 4),) * 4), AE.Carry(), [1, 1] + zero[2:], zero)]
    want = [AE.reference_charge(14, *c) for c in cases]
    assert [w[0] for w in want] == [0, 1, 0]
    _same_charges(14, cases)
    good = _charge_streams()[0]["decisions"]["book"]
    caps, short = {**good["caps"], "arrived": 15}, {"lo": 0, "freqs": [1, 2], "escape": 1}
    freqs = good["models"]["n_arrived"]["binned"]["freqs"]
    assert AE.charge_book({**good, "caps": caps}, 15)[0].arrived == tuple(freqs[:5])
    for bad, N in [({**good, "caps": caps}, 16), ({**good, "models": {**good["models"], "n_hits": short}}, 4),
                   ({**good, "models": {**good["models"], "n_lag": {"binned": short}}}, 4)]:
        with pytest.raises(ValueError):
            AE.charge_book(bad, N)


def test_size_counts_the_gates():
    assert AE.size("step", K=K, BIN=1)["ands"] == ands(AE.step_unit(K, 1))
    assert AE.size("step", K=K, BIN=0)["ands"] == ands(AE.step_unit(K, 0))
    assert AE.size("finish")["ands"] == ands(AE.finish_unit())
    assert AE.size("kt", C=3, NW=24)["ands"] == ands(AE.kt_unit(3, 24))
    for BIN in (1, 0):
        assert AE.size("count", K=K, BIN=BIN, CW=AE.CW)["ands"] == ands(AE.count_unit(K, BIN, AE.CW))
    assert AE.size("count_finish", CW=AE.CW)["ands"] == ands(AE.count_finish_unit(AE.CW))
    for C in (1, 3):
        assert AE.size("book_kt", C=C)["ands"] == ands(AE.book_kt_unit(C))
    assert AE.size("charge", N=2)["ands"] == ands(AE.charge_unit(2))


@pytest.mark.parametrize("part, key", [(AE.ArithEncodeFinish, ()), (AE.ArithEncodeStep, (3, 1))])
def test_a_parts_build_is_called_as_the_lowering_calls_it(part, key):
    """Flock's lowering (`ir_lower._boolean_activation`) passes the statics to ``build`` only when the part has some, so a
    part without any, ArithEncodeFinish_v1, takes the elements alone."""
    from verity.ml.circuits.boolean.trace import trace

    S = SimpleNamespace(**dict(zip(part.statics, key)))
    build = (lambda C, *e: part.build(C, S, *e)) if part.statics else part.build
    params, ret = part.signature(S)
    got, want = trace(build, [t for _, t in params], ret), part.traced(*key)
    assert (got.gates, got.outputs) == (want.gates, want.outputs)


@pytest.mark.parametrize("traced", [False, True])
def test_no_and_of_the_charge_reads_a_form_its_earlier_decisions_grew(traced):
    """The carried words are committed once a decision, as rows of a lowering and in the gate trace alike (where
    `C.commit` is the word itself), so the widest AND operand is the same at every N and the build linear in N. Without
    the copies it grows with each decision, 2,983 terms at N = 2 and 3,006 at N = 3."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.boolean.forms import Circuit
    from verity.ml.circuits.boolean.trace import TracingCircuit

    def widest(N):
        C = (TracingCircuit if traced else Circuit)(cse=True)
        params, _ = AE._sig_charge(SimpleNamespace(N=N))
        ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(BL.elements(t))]
        C.output("out", [b for e in AE._build_charge(C, SimpleNamespace(N=N), *ins) for b in e])
        return max(len(C.data[v][k].s) for v in C.live() if C.kind[v] == "and" for k in (0, 1))

    assert widest(2) == widest(3)
