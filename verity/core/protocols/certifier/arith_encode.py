"""Advice v1's arithmetic encoder as Boolean Definitions, run forward inside the compile unit alongside the replay scan:
the bits it would write are the window's charge.

    ArithEncodeStep_v1{K, BIN}   (coder, en, symbol, seg[K]) -> (coder', n)
    ArithEncodeFinish_v1         (coder, seg[2]) -> (ok, n)
    KTContext_v1{C, NW}          (counts[C], en, ctx, bit) -> (ok, f0, f1, counts')
    coder = (low[32], high[32], open, x, used, ok)

Under advice v1 (the certifier PROTOCOL's "Advice v1") each certifier window is one terminated segment of ENC-CODE's
32-bit Witten–Neal–Cleary arithmetic coder (`benchmarks/network_e2e/encodings/code.py`, `ArithEncoder`), as `code_v1`
codes it under the registered book: the window's counts, then its values (the replay's decisions, each a bit at the
book's estimate for its context, the arrival order's deviations and the hits sent past the pool bound). The committed
advice is the values themselves, which the replay scan reads, and the charge is their coded length: the encoder runs
over the same symbols, and the window's segment length, its steps' n and its finish's, must be at most the declared
charge. No codeword is committed or compared. A symbol's scalings, and so n, depend only on the interval, so these
Definitions, which check a segment, count the same bits as a counter with no segment. The compile unit runs the encoder
forward, since which value is read next, and so its window, is the replay's position.

**What the circuit computes, never witness.** A decision's context is the replay's position (under `code_v1`
min(run, 2), with run the decisions since the last arrival), carried in the circuit. Its (f0, f1) are the model's, from
counts that start at the registered book's public priors and that each coded decision raises in the circuit. An
integer's (cum, freq, total) are read from the book's public tables at the value's bin. A context or a frequency the
prover supplied would pick the code per value and charge any decisions next to nothing.

**The coder's state.** `low` and `high` are code.py's, 32 bits each. code.py also counts its pending bits (the
underflow scalings not yet resolved); the circuit doesn't, because it never writes the output. A pending run's bits are
the complement of the bit that resolves it, so the circuit checks the segment in the decoder's order, one segment bit per
scaling: `open` says a pending run is open, and `x` is the segment's bit at the run's first position. `used` is
code.py's (a symbol of frequency below its total was coded), and `ok` is 0 once anything was refused. A window starts at
`fresh()`: (0, 2^32 − 1, 0, 0, 0, 1).

**The step** codes one symbol when `en` is 1 (when 0, it passes the state through and consumes nothing):
- *The symbol.* With BIN = 1 it is a decision bit with its model's frequencies f0, f1 ≥ 1, f0 + f1 ≤ 2^16, coded as
  code.py's `encode(f0 if bit else 0, f1 if bit else f0, f0 + f1)` (`code_v1.Encoder.bit`). One product and one
  division serve both ends: with r = high − low + 1 and q = ⌊r·f0 / (f0 + f1)⌋, a 1 sets low to low + q and a 0 sets
  high to low + q − 1. With BIN = 0 it is code.py's general (cum, freq, total), with two products and two divisions;
  freq = total leaves the interval and `used` as they are, as code.py's early return does.
- *The scalings*, K slots of code.py's loop: E1 (high < 2^31) and E2 (low ≥ 2^31) resolve a bit (low's top bit), E3
  (2^30 ≤ low, high < 3·2^30) adds a pending bit, and each shifts the interval. A slot that doesn't scale leaves the
  state as it is, so every later slot doesn't either. After the update the range is at least ⌊(2^30 + 2)/2^16⌋ = 2^14,
  and each scaling doubles it until it exceeds 2^31, so a symbol takes at most 18 scalings and K = 18 never refuses
  one. A symbol that would need more than K is refused.
- *The segment check.* `seg` is the window's next K segment bits, at its position (the bits the decoder would read,
  one per scaling). The encoder writes position t when a run resolves: the run's first position gets the resolving bit
  b, and every other position of the run gets ¬b. So a resolution with no open run needs seg = b there. A resolution
  that closes an open run needs seg = ¬b there, and x = b. An E3 that opens a run sets x to its segment bit, and an E3
  inside an open run needs its segment bit to be ¬x. Every segment position is checked once, when it is consumed, and a
  run's first position once more, when the run resolves.
- *Outputs*: the next state and n, the scalings, which is the segment bits consumed. The window's reader moves by n.

**The finish** is code.py's `finish`: when `used`, one more pending bit, then b_f = 0 if low < 2^30 else 1 resolves the
run. In positions, that is the run's first position (the open run's, or the first new one) gets b_f, and the two new
positions get the rest. So it needs seg[0] = b_f ⊕ open and seg[1] = ¬b_f, and, if a run was open, x = b_f. It
consumes n = 2 bits, or none when nothing was used. Its window's segment must end there: n positions after the reader's.

**The model** (`KTContext_v1`) is ENC-REPLAY's estimate, which advice v1 coded with before `code_v1`. C contexts each
count their 0s and 1s (`replay._KT`). Context ctx gives f = 2n + 1 for each count, halved with rounding up while
f0 + f1 > 2^16 (`scaled`), and the decision then adds to its count. After h ≥ 1 halvings f is ⌊n / 2^(h−1)⌋ + 1, so the
circuit takes the first h whose two candidates fit. With `en` = 0 it passes the counts through. It refuses a context
≥ C and a count that would overflow NW bits. In the compile unit the counts start public, at the registered priors,
and pass from model to model, and the context index is the replay's (plain: min(passes, 2), C = 3): neither is
witness. `code_v1.KT` halves its counts in place instead.

**Statics** (public): K, the scaling slots (18 covers every symbol); BIN, binary (1) or general (0) symbols; C, the
contexts; and NW, a count's width. Fields are little-endian.

**Cost** (`size`, ANDs): ArithEncodeStep_v1{K=18, BIN=1} is 3,244: the interval (one 33 × 16-bit product, one
non-restoring division with 32 quotient bits, and the two ends) 1,855, the 18 scaling slots 1,188 and the segment check
201. {K=18, BIN=0} is 4,964. ArithEncodeFinish_v1 is 6. KTContext_v1{C=3, NW=24} is 1,202. One decision under plain
contexts is the step and the model: 4,446 ANDs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, multiply, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.head_check import BIT, _part

PREC = 32
TOP = (1 << PREC) - 1
HALF = 1 << (PREC - 1)
Q1 = 1 << (PREC - 2)
Q3 = 3 * Q1
MAX_TOTAL = 1 << 16
#: the most scalings one symbol needs: its interval keeps at least 2^14 values, and each scaling doubles it past 2^31
K_MAX = 18


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


# --- the specification -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Coder:
    """One window's encoder state (the module's docstring)."""

    low: int = 0
    high: int = TOP
    open: int = 0
    x: int = 0
    used: int = 0
    ok: int = 1


def fresh() -> Coder:
    return Coder()


def _symbol(BIN: int, sym: Sequence[int]) -> tuple[bool, int, int, int]:
    """(valid, cum, freq, total) of a symbol: (bit, f0, f1) with BIN, else (cum, freq, total)."""
    if BIN:
        bit, f0, f1 = sym
        t = f0 + f1
        return f0 >= 1 and f1 >= 1 and t <= MAX_TOTAL, (f0 if bit else 0), (f1 if bit else f0), t
    cum, freq, total = sym
    return freq >= 1 and cum + freq <= total <= MAX_TOTAL, cum, freq, total


def reference_step(K: int, BIN: int, st: Coder, en: int, sym: Sequence[int], seg: Sequence[int]) -> tuple[Coder, int]:
    """One symbol through code.py's encoder, its output checked in the decoder's order against `seg`, the window's next
    K bits (the module's docstring). Returns the next state and the bits consumed; a refusal sets `ok` to 0, and then
    only `ok` is specified."""
    low, high, op, x, used, ok = st.low, st.high, st.open, st.x, st.used, st.ok
    if en:
        valid, cum, freq, total = _symbol(BIN, sym)
        if not valid:
            return replace(st, ok=0), 0
        r = high - low + 1
        high = low + (r * (cum + freq)) // total - 1
        low = low + (r * cum) // total
        used |= int(freq != total)
    n = 0
    while True:
        if high < HALF or low >= HALF:
            b = int(low >= HALF)
            if n >= K:
                return replace(st, ok=0), 0
            s = seg[n]
            if op:
                ok &= int(s != b and x == b)
            else:
                ok &= int(s == b)
            op = 0
            low, high = low & (HALF - 1), high & (HALF - 1)
        elif low >= Q1 and high < Q3:
            if n >= K:
                return replace(st, ok=0), 0
            s = seg[n]
            if op:
                ok &= int(s != x)
            else:
                x, op = s, 1
            low, high = low - Q1, high - Q1
        else:
            break
        low, high = 2 * low, 2 * high + 1
        n += 1
    return Coder(low, high, op, x, used, ok), n


def reference_finish(st: Coder, seg: Sequence[int]) -> tuple[int, int]:
    """code.py's `finish` checked against the window's last two bits: (ok, bits consumed)."""
    if not st.used:
        return st.ok, 0
    b = int(st.low >= Q1)
    ok = st.ok & int(seg[0] == b ^ st.open and seg[1] != b)
    if st.open:
        ok &= int(st.x == b)
    return ok, 2


def scaled(f0: int, f1: int) -> tuple[int, int]:
    """ENC-REPLAY's estimate (`replay._KT`) halved, rounding up, into the coder's total."""
    while f0 + f1 > MAX_TOTAL:
        f0, f1 = (f0 + 1) >> 1, (f1 + 1) >> 1
    return f0, f1


def reference_kt(C: int, NW: int, counts: Sequence[tuple[int, int]], en: int, ctx: int,
                 bit: int) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    """(ok, f0, f1, counts') of one decision under ENC-REPLAY's KT counts per context, scaled as advice v1 does. A
    refusal returns ok 0, and then only ok is specified."""
    counts = tuple(tuple(c) for c in counts)
    if not en:
        return 1, 0, 0, counts
    if ctx >= C:
        return 0, 0, 0, counts
    n0, n1 = counts[ctx]
    f0, f1 = scaled(2 * n0 + 1, 2 * n1 + 1)
    m0, m1 = n0 + (not bit), n1 + bool(bit)
    if max(m0, m1) >> NW:
        return 0, 0, 0, counts
    return 1, f0, f1, counts[:ctx] + ((m0, m1),) + counts[ctx + 1:]


def check_window(K: int, BIN: int, symbols: Sequence[Sequence[int]], segment: Sequence[int]) -> bool:
    """The window's segment against its symbols (en = 1 each), by the reference: every step and the finish accept, and
    the segment ends where the finish does."""
    st, pos = fresh(), 0
    seg = list(segment)
    for s in symbols:
        st, n = reference_step(K, BIN, st, 1, s, (seg[pos:pos + K] + [0] * K)[:K])
        pos += n
        if not st.ok or pos > len(seg):
            return False
    ok, n = reference_finish(st, (seg[pos:pos + 2] + [0, 0])[:2])
    return bool(ok) and pos + n == len(seg)


# --- the circuits ----------------------------------------------------------------------------------------------------

_CODER = [("low", 1, PREC), ("high", 1, PREC), ("open", 1, 1), ("x", 1, 1), ("used", 1, 1), ("ok", 1, 1)]


def _step_in(S) -> list[tuple[str, int, int]]:
    sym = [("bit", 1, 1), ("f0", 1, 16), ("f1", 1, 16)] if S.BIN else [("cum", 1, 16), ("freq", 1, 17), ("total", 1, 17)]
    return _CODER + [("en", 1, 1)] + sym + [("seg", 1, S.K)]


def _step_out(S) -> list[tuple[str, int, int]]:
    return _CODER + [("n", 1, _bitlen(S.K))]


_FIN_IN = _CODER + [("seg", 1, 2)]
_FIN_OUT = [("ok", 1, 1), ("n", 1, 2)]


def _kt_in(S) -> list[tuple[str, int, int]]:
    return [("n0", S.C, S.NW), ("n1", S.C, S.NW), ("en", 1, 1), ("ctx", 1, _bitlen(S.C - 1)), ("bit", 1, 1)]


def _kt_out(S) -> list[tuple[str, int, int]]:
    return [("ok", 1, 1), ("f0", 1, 16), ("f1", 1, 16), ("n0", S.C, S.NW), ("n1", S.C, S.NW)]


def _type(fields) -> Tuple:
    return Tuple(*[Array(n, Array(w, BIT)) for _, n, w in fields])


def _split(fields, elements) -> dict:
    out, at = {}, 0
    for name, n, _w in fields:
        out[name] = [list(e) for e in elements[at:at + n]]
        at += n
    return out


def _flat(fields, res: dict) -> list:
    return [e for name, _n, _w in fields for e in res[name]]


def _or(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _nonzero(C, x):
    return or_reduce(C, x)


def _divide(C, num, den, qbits):
    """⌊num / den⌋ in `qbits` bits for 1 ≤ den and num < den · 2^qbits, by non-restoring division: the partial remainder
    R stays in [−den, den), each step adds den when R < 0 and subtracts it otherwise, and quotient bit i is R ≥ 0 after
    step i, the restoring quotient's bit."""
    dw = len(den)
    W = dw + 2
    num = _pad(num, qbits + dw)
    rem, neg = _pad(num[qbits:], W), ZERO
    d = _pad(den, W)
    q = [ZERO] * qbits
    for i in reversed(range(qbits)):
        t = [num[i]] + rem[:W - 1]
        rem = add(C, t, [b ^ neg.inv() for b in d], cin=neg.inv(), width=W)
        neg = rem[W - 1]
        q[i] = neg.inv()
    return q


def _le_pow2(C, x, k):
    """x ≤ 2^k for a (k+1)-bit x."""
    return C.AND(x[k], _nonzero(C, x[:k])).inv()


def _build_step(C, S, *elements):
    f = _split(_step_in(S), elements)
    low, high = f["low"][0], f["high"][0]
    op, x, used, ok = f["open"][0][0], f["x"][0][0], f["used"][0][0], f["ok"][0][0]
    en, seg = f["en"][0][0], f["seg"][0]

    C.phase = "interval"
    r = add(C, sub(C, high, low, PREC)[0], [ONE], width=PREC + 1)
    if S.BIN:
        bit, f0, f1 = f["bit"][0][0], f["f0"][0], f["f1"][0]
        total = add(C, f0, f1, width=17)
        valid = and_reduce(C, [_nonzero(C, f0), _nonzero(C, f1), _le_pow2(C, total, 16)])
        q = _divide(C, multiply(C, r, f0), total, PREC)
        up = add(C, low, q, width=PREC)
        e1, e0 = C.AND(en, bit), C.AND(en, bit.inv())
        low = mux(C, e1, up, low)
        high = mux(C, e0, sub(C, up, [ONE], PREC)[0], high)
        used = _or(C, used, en)
    else:
        cum, freq, total = f["cum"][0], f["freq"][0], f["total"][0]
        cf = add(C, cum, freq, width=18)
        valid = and_reduce(C, [_nonzero(C, freq), sub(C, total + [ZERO], cf, 18)[1].inv(), _le_pow2(C, total, 16)])
        teff = mux(C, en, total, const_bits(1, 17))
        q_lo = _divide(C, multiply(C, r, [C.AND(en, b) for b in cum]), teff, PREC)
        q_hi = _divide(C, multiply(C, r, mux(C, en, cf[:17], const_bits(1, 17))), teff, PREC + 1)
        low_ = add(C, low, q_lo, width=PREC)
        high = sub(C, add(C, low, q_hi, width=PREC + 1), [ONE], PREC + 1)[0][:PREC]
        low = low_
        diff = or_reduce(C, [a ^ b for a, b in zip(freq, total)])
        used = _or(C, used, C.AND(en, diff))
    ok = C.AND(ok, C.AND(en, valid.inv()).inv())

    therm, bad = [], []
    for j in range(S.K):
        C.phase = "scale"
        l31, l30, h31, h30 = low[31], low[30], high[31], high[30]
        e12 = (l31 ^ h31).inv()
        e3 = C.AND(C.AND(e12.inv(), l30), h30.inv())
        s = e12 ^ e3
        C.phase = "check"
        sj = seg[j]
        bad.append(C.AND(e12, _or(C, sj ^ l31 ^ op, C.AND(op, x ^ l31))))
        bad.append(C.AND(C.AND(e3, op), (sj ^ x).inv()))
        start = C.AND(e3, op.inv())
        x = x ^ C.AND(start, sj ^ x)
        op = _or(C, C.AND(op, e12.inv()), e3)
        C.phase = "scale"
        low = mux(C, s, [ZERO] + low[:30] + [l30 ^ e3], low)
        high = mux(C, s, [ONE] + high[:30] + [h30 ^ e3], high)
        therm.append(s)
    C.phase = "check"
    more = (low[31] ^ high[31]).inv() ^ C.AND(C.AND((low[31] ^ high[31]), low[30]), high[30].inv())
    ok = C.AND(ok, _or(C, or_reduce(C, bad), more).inv())
    n = [ZERO] * _bitlen(S.K)
    for j, t in enumerate(therm):
        for k in range(len(n)):
            if ((j >> k) ^ ((j + 1) >> k)) & 1:
                n[k] = n[k] ^ t
    res = {"low": [low], "high": [high], "open": [[op]], "x": [[x]], "used": [[used]], "ok": [[ok]], "n": [n]}
    return _flat(_step_out(S), res)


def _build_finish(C, S, *elements):
    f = _split(_FIN_IN, elements)
    low, op, x, used, ok, seg = f["low"][0], f["open"][0][0], f["x"][0][0], f["used"][0][0], f["ok"][0][0], f["seg"][0]
    b = _or(C, low[31], low[30])
    bad = or_reduce(C, [seg[0] ^ b ^ op, (seg[1] ^ b).inv(), C.AND(op, x ^ b)])
    ok = C.AND(ok, C.AND(used, bad).inv())
    return _flat(_FIN_OUT, {"ok": [[ok]], "n": [[ZERO, used]]})


def _build_kt(C, S, *elements):
    f = _split(_kt_in(S), elements)
    n0, n1, en, ctx, bit = f["n0"], f["n1"], f["en"][0][0], f["ctx"][0], f["bit"][0][0]
    C.phase = "select"
    hot = []
    for c in range(S.C):
        hot.append(and_reduce(C, [b if (c >> k) & 1 else b.inv() for k, b in enumerate(ctx)]))
    in_range = or_reduce(C, hot)
    a0 = [ZERO] * S.NW
    a1 = [ZERO] * S.NW
    for c in range(S.C):
        a0 = [u ^ C.AND(hot[c], v) for u, v in zip(a0, n0[c])]
        a1 = [u ^ C.AND(hot[c], v) for u, v in zip(a1, n1[c])]
    C.phase = "scale"
    H = max(0, S.NW - 13)
    cands, fits = [], []
    for h in range(H + 1):
        if h == 0:
            s = add(C, a0, a1, width=S.NW + 1)
            fit = sub(C, s, const_bits((1 << 15), S.NW + 1), S.NW + 1)[1]
            cand = ([ZERO] + a0, [ZERO] + a1)
        else:
            u0, u1 = a0[h - 1:], a1[h - 1:]
            w = len(u0) + 1
            s = add(C, u0, u1, width=w)
            fit = sub(C, s, const_bits(MAX_TOTAL - 1, max(w, 17)), max(w, 17))[1]
            cand = (u0, u1)
        fits.append(fit)
        cands.append((_pad(cand[0], 16), _pad(cand[1], 16)))
    g0, g1 = [ZERO] * 16, [ZERO] * 16
    prev = ZERO
    for h in range(H + 1):
        first = C.AND(fits[h], prev.inv())
        g0 = [u ^ C.AND(first, v) for u, v in zip(g0, cands[h][0])]
        g1 = [u ^ C.AND(first, v) for u, v in zip(g1, cands[h][1])]
        prev = _or(C, prev, fits[h])
    f0 = add(C, g0, [ONE], width=16)
    f1 = add(C, g1, [ONE], width=16)
    C.phase = "update"
    m0 = add(C, a0, [bit.inv()], width=S.NW + 1)
    m1 = add(C, a1, [bit], width=S.NW + 1)
    over = _or(C, m0[S.NW], m1[S.NW])
    ok = C.AND(en, _or(C, in_range.inv(), _or(C, over, prev.inv()))).inv()
    out0, out1 = [], []
    for c in range(S.C):
        w = C.AND(en, hot[c])
        out0.append(mux(C, w, m0[:S.NW], n0[c]))
        out1.append(mux(C, w, m1[:S.NW], n1[c]))
    res = {"ok": [[ok]], "f0": [[C.AND(en, b) for b in f0]], "f1": [[C.AND(en, b) for b in f1]], "n0": out0, "n1": out1}
    return _flat(_kt_out(S), res)


def _sig_step(S):
    if S.K < 1 or S.BIN not in (0, 1):
        raise ValueError(f"statics {dict(S.__dict__)}: K ≥ 1 and BIN a bit")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _step_in(S)), _type(_step_out(S))


def _sig_finish(S):
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _FIN_IN), _type(_FIN_OUT)


def _sig_kt(S):
    if S.C < 1 or S.NW < 15:
        raise ValueError(f"statics {dict(S.__dict__)}: C ≥ 1 and NW ≥ 15")
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _kt_in(S)), _type(_kt_out(S))


ArithEncodeStep = _part(
    "ArithEncodeStep", ("K", "BIN"), _sig_step, _build_step,
    "one symbol through ENC-CODE's 32-bit WNC arithmetic encoder, its output bits checked in the decoder's order against "
    "the window's next K segment bits; the next coder state and the bits consumed, the symbol's charge "
    "(`arith_encode.reference_step`)")

ArithEncodeFinish = _part(
    "ArithEncodeFinish", (), _sig_finish, _build_finish,
    "ENC-CODE's arithmetic encoder's termination checked against the window's last two segment bits "
    "(`arith_encode.reference_finish`)")

KTContext = _part(
    "KTContext", ("C", "NW"), _sig_kt, _build_kt,
    "ENC-REPLAY's adaptive KT estimate for one decision's context, scaled into the coder's total as advice v1 did "
    "before code_v1, and its counts updated (`arith_encode.reference_kt`)")


def step_unit(K: int = K_MAX, BIN: int = 1):
    return bind(ArithEncodeStep, K=K, BIN=BIN)


def finish_unit():
    return bind(ArithEncodeFinish)


def kt_unit(C: int = 3, NW: int = 24):
    return bind(KTContext, C=C, NW=NW)


# --- arguments and evaluation ----------------------------------------------------------------------------------------


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def _args(fields, vals: dict) -> list[list[int]]:
    return [[b for v in (vals[name] if isinstance(vals[name], (list, tuple)) else [vals[name]]) for b in _bits(v, w)]
            for name, _n, w in fields]


def _fields(fields, flat: Sequence[int]) -> dict:
    at, out = 0, {}
    for name, n, w in fields:
        out[name] = [sum(int(flat[at + e * w + j]) << j for j in range(w)) for e in range(n)]
        at += n * w
    return out


def _coder_vals(st: Coder) -> dict:
    return {"low": st.low, "high": st.high, "open": st.open, "x": st.x, "used": st.used, "ok": st.ok}


def step_arguments(K: int, BIN: int, st: Coder, en: int, sym: Sequence[int], seg: Sequence[int]) -> list[list[int]]:
    S = SimpleNamespace(K=K, BIN=BIN)
    names = ("bit", "f0", "f1") if BIN else ("cum", "freq", "total")
    vals = {**_coder_vals(st), "en": en, **dict(zip(names, sym)), "seg": sum(int(b) << j for j, b in enumerate(seg))}
    return _args(_step_in(S), vals)


def evaluate_steps(K: int, BIN: int, cases: Sequence[tuple[Coder, int, Sequence[int], Sequence[int]]]) -> list[tuple[Coder, int]]:
    """The step at (K, BIN) on each (state, en, symbol, seg), in one evaluation."""
    S = SimpleNamespace(K=K, BIN=BIN)
    args = [list(rows) for rows in zip(*(step_arguments(K, BIN, *c) for c in cases))]
    out = evaluate_bits(step_unit(K, BIN), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_step_out(S), row)
        res.append((Coder(f["low"][0], f["high"][0], f["open"][0], f["x"][0], f["used"][0], f["ok"][0]), f["n"][0]))
    return res


def evaluate_finishes(cases: Sequence[tuple[Coder, Sequence[int]]]) -> list[tuple[int, int]]:
    args = [list(rows) for rows in zip(*(_args(_FIN_IN, {**_coder_vals(st), "seg": seg[0] | seg[1] << 1})
                                         for st, seg in cases))]
    out = evaluate_bits(finish_unit(), *args)
    return [(f["ok"][0], f["n"][0]) for f in (_fields(_FIN_OUT, row) for row in out.reshape(len(cases), -1).tolist())]


def evaluate_kt(C: int, NW: int, cases: Sequence[tuple[Sequence[tuple[int, int]], int, int, int]]
                ) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    S = SimpleNamespace(C=C, NW=NW)

    def one(counts, en, ctx, bit):
        return _args(_kt_in(S), {"n0": [c[0] for c in counts], "n1": [c[1] for c in counts], "en": en, "ctx": ctx,
                                 "bit": bit})

    args = [list(rows) for rows in zip(*(one(*c) for c in cases))]
    out = evaluate_bits(kt_unit(C, NW), *args)
    res = []
    for row in out.reshape(len(cases), -1).tolist():
        f = _fields(_kt_out(S), row)
        res.append((f["ok"][0], f["f0"][0], f["f1"][0], tuple(zip(f["n0"], f["n1"]))))
    return res


# --- cost ------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _staged(kind: str, key: tuple) -> dict[str, int]:
    sig, build, statics = {"step": (_sig_step, _build_step, ("K", "BIN")), "finish": (_sig_finish, _build_finish, ()),
                           "kt": (_sig_kt, _build_kt, ("C", "NW"))}[kind]
    S = SimpleNamespace(**dict(zip(statics, key)))
    C = Circuit(cse=True)
    params, _ = sig(S)
    ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate([t.elem.n] * t.n)]
    C.output("out", [b for e in build(C, S, *ins) for b in e])
    return C.phase_counts()


def size(kind: str = "step", **st) -> dict[str, int]:
    """ANDs by stage of the step (K, BIN), the finish or the model (C, NW), traced on a counting circuit."""
    keys = {"step": ("K", "BIN"), "finish": (), "kt": ("C", "NW")}[kind]
    out = dict(_staged(kind, tuple(st[k] for k in keys)))
    out["ands"] = sum(out.values())
    return out


__all__ = ["ArithEncodeFinish", "ArithEncodeStep", "Coder", "HALF", "K_MAX", "KTContext", "MAX_TOTAL", "Q1", "TOP",
           "check_window", "evaluate_finishes", "evaluate_kt", "evaluate_steps", "finish_unit", "fresh", "kt_unit",
           "reference_finish", "reference_kt", "reference_step", "scaled", "size", "step_arguments", "step_unit"]
