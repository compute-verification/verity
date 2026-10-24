"""arith_encode: advice v1's arithmetic encoder as Boolean Definitions, held to ENC-CODE's coders (`code.py`, `code_v1`)
on what v1 actually codes (the fixture's streams, as `benchmarks/network_e2e/encodings/arith_vectors.py --steps` cut
them: advice v1's windows, the replay's decisions, the arrival order's deviations and the hits sent past the pool
bound, on ENC-REPLAY's step traces with v1's charges, and ENC-DERIVED's still-sent fields on the TRACE-CORPUS
fixture, each symbol tagged a bit or an integer's, with code.py's state after every symbol and its segment), to the
references on random runs, and refusing every other segment and every symbol code.py refuses."""
from __future__ import annotations

import json
import random
from pathlib import Path

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


def _same_count_finishes(CW, cases, NB=0):
    got = AE.evaluate_count_finishes(CW, cases, NB)
    for c, g in zip(cases, got):
        want = AE.reference_count_finish(CW, c[0], c[1], NB, *c[2:])
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
    # NB ≥ 1, a pool header advice v1 no longer sends: Elias-gamma of the pool (1, 3 and 5 bits for pools 1, 2–3 and
    # 4–7), a pool of 0 refused, and the header held to c with the segment
    got = _same_count_finishes(AE.CW, [(st, 1, 1), (st, 3, 3), (used, 20, 7), (used, 19, 7), (st, 9, 0)], NB=3)
    assert got[:3] == [(1, 1), (1, 3), (1, 20)] and [g[0] for g in got[3:]] == [0, 0]
    assert _same_count_finishes(5, [(AE.Tally(bits=29), 31, 2)], NB=2)[0][0] == 0
    with pytest.raises(ValueError):
        AE.size("count_finish", CW=4, NB=4)


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


def test_size_counts_the_gates():
    assert AE.size("step", K=K, BIN=1)["ands"] == ands(AE.step_unit(K, 1))
    assert AE.size("step", K=K, BIN=0)["ands"] == ands(AE.step_unit(K, 0))
    assert AE.size("finish")["ands"] == ands(AE.finish_unit())
    assert AE.size("kt", C=3, NW=24)["ands"] == ands(AE.kt_unit(3, 24))
    for BIN in (1, 0):
        assert AE.size("count", K=K, BIN=BIN, CW=AE.CW)["ands"] == ands(AE.count_unit(K, BIN, AE.CW))
    for NB in (0, 21):
        assert AE.size("count_finish", CW=AE.CW, NB=NB)["ands"] == ands(AE.count_finish_unit(AE.CW, NB))
    for C in (1, 3):
        assert AE.size("book_kt", C=C)["ands"] == ands(AE.book_kt_unit(C))
