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


def test_the_contexts_are_the_bits():
    """Each decision's context is min(run, 2), the decisions since the last arrival, carried across windows: the
    replay's position in the bits, never witness. Every bit symbol is one the model gave an estimate."""
    seen = 0
    for stream in _streams():
        bits = [sym for win in stream["windows"] for sym in win["symbols"] if sym[0]]
        assert len(bits) == len(stream["kt"]), stream["source"]
        run = 0
        for (name, ctx, bit), sym in zip(stream["kt"], bits):
            assert sym[1] == bit, stream["source"]
            if name == "arrived":
                assert ctx == min(run, 2), stream["source"]
                run = 0 if bit else run + 1
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
    cases, fins = {0: [], 1: []}, []
    for stream in _streams():
        for _win, steps, (st, tail, _pos) in _runs(stream):
            for BIN, s, sym, look, _ in steps:
                flip = list(look)
                flip[rng.randrange(K)] ^= 1
                cases[BIN] += [(s, 1, sym, look), (s, 0, sym, look), (s, 1, sym, flip)]
            fins.append((st, tail))
    for BIN in (0, 1):
        _same(K, BIN, cases[BIN])
    assert AE.evaluate_finishes(fins) == [AE.reference_finish(*f) for f in fins]


@pytest.mark.parametrize("C", [1, 3])
def test_the_model_is_its_reference_on_random_counts(C):
    rng = random.Random(C)
    kt = []
    for _ in range(300):
        ctx = rng.randrange(1 << AE._bitlen(C - 1))
        kt.append((tuple((rng.randrange(1 << 15), rng.randrange(1 << 15)) for _c in range(C)), rng.choice([1, 1, 0]),
                   ctx, rng.randrange(2)))
    for c, g in zip(kt, AE.evaluate_kt(C, 16, kt)):
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


def test_size_counts_the_gates():
    assert AE.size("step", K=K, BIN=1)["ands"] == ands(AE.step_unit(K, 1))
    assert AE.size("step", K=K, BIN=0)["ands"] == ands(AE.step_unit(K, 0))
    assert AE.size("finish")["ands"] == ands(AE.finish_unit())
    assert AE.size("kt", C=3, NW=24)["ands"] == ands(AE.kt_unit(3, 24))
