"""The stacking cheater's cheapest exact squaring mod p1024 under randomized arithmetic, built and measured in 6b's gate
model (`benchmarks/property_redteam/squaring.py`, a port of `Security/Definitions/Accounting/Gates/Model.lean`).

**What the randomization does to a gate.** property/v1 lets a budgeted gate compute any function of at most 128 operand
bits. Randomized rounding can't bite in that model: a cheater picks a gate whose exact pre-rounding value is already an
integer, and stochastic rounding (SR) or a dither of an integer returns it unchanged. So the randomized protocol has to
restrict the gates to real primitive semantics, and this module models them that way:

* a **rounding** gate (`nfloor`) computes an exact linear form `y` of its operands and returns `⌊y / 2^k⌋` perturbed by
  the mechanism: SR (`⌊(y + U) / 2^k⌋`, `U` a fresh uniform `k`-bit coin, so `⌊y/2^k⌋` or one more), dither (the same
  law, since an offset drawn per op is a fresh uniform per gate), or bounded noise (round-to-nearest, then a uniform
  integer in `[−N, N]` added, even on integer inputs). Under `det` it is the exact floor.
* the **compare_select** gates stay deterministic, as no randomization applies to them: `clamp(x, lo, hi)` (S32ClampS8,
  a saturating cast of an integer), and `cmp`, the bit `[y < 0]` of a linear form (a comparison).
* **any-function** gates (`anyfn`, property/v1's own budgeted gate) exist only where nothing is randomized (`det`): they
  are the baseline model, in which `seam_square` costs 160.
* **noisy dots** (`noisy_mma`): an s8-family dot whose int32 output carries a fresh uniform integer in
  `[−2^(g−1), 2^(g−1))` (the accumulator-noise mechanism). The cheater buys `G ≥ g` guard bits, by repeating each product
  `2^rep` times in the dot (free in units, `2^rep` times the MACs) or by scaling `b`-bit digits up by `2^(8−b)` in the u8
  operands (`2(8 − b)` bits), and cleans each column with `⌊(y + 2^(G−1)) / 2^G⌋`.

The coins are never inputs the cheater's gates read; predicting them means computing the verifier's PRF in the circuit
(`prf_units`). Every construction is run on random inputs and fresh coins and checked exact mod p, `Proven` and within
`OperandBits 128`; its units are counted by class, with its MACs (products summed by its dots).

**The seam.** Each construction folds its top carry `c_top` back into a low window of `w` digits (`2^(bn) ≡ δ`), which
it computed before `c_top` existed, so it is wrong on inputs where `c_top·δ` moves that window's carry out.
`window_fail_log2` bounds the share of such inputs from the gates' derived intervals, with the window's own value uniform
over its `R^w` values (modeled); a randomized law's window is centered, so it can be exact. The reference, `seam_square`,
is wrong on 2^−12.4 of inputs (measured in `squaring.py`), and every construction takes the narrowest window whose bound,
computed the same way, is at most the reference's (`reference_fail_log2`): compared like with like.

**The constructions** (`CONSTRUCTIONS`), each one squaring of a 1024-bit word:

* `anyfn_seam`: property/v1's cheapest, `squaring.seam_square` (160 units), the reference.
* `anyfn_fixup(b, G, rep)`: one greedy any-function carry pass over `b`-bit digits, its top carry folded into a fixup group;
  with `G > 0` the group gates read noisy columns and clean them inside (accumulator noise, deterministic roundings).
* `restricted(mech, b)`: rounding gates are `nfloor`s and narrowing is a `clamp`: a ripple a digit (`nfloor` carry +
  `clamp` digit), with an exact precomputed seam carry. `det` (exact floors) is the restricted model's own baseline,
  and what the format's own exact byte primitives give at the same count whatever the coins do (`S32ByteCarry_v2`
  `(y − sext8(y))/256` the carry, `S32LowByte` the digit, `verity/ml/circuits/ops/boolean/s8.py`); `sr` needs signed
  radix-128 digits (`d = x − 128·SR(x/128)` always lies in `[−128, 127]`); `noise1` radix 64.
* `restricted_acc(mech, g, rep, b)`: both mechanisms: each unfolded noisy column cleaned exactly (one `nfloor` under
  `det`, as a byte carry does; `nfloor` + `cmp` under SR), then `restricted`'s ripple.
* `hadamard_anchor_b{b}`: a fresh random-sign block-Hadamard rotation inside every free dot, quantized with one scale a
  row as `honest.py`'s `had` does. One nonzero a block is exact at 128x the MACs; cheaper, an anchor coordinate pins
  the row's scale, so blocks of `⌊127/2^(b−1)⌋` balanced digits rotate to exact integer codes (`hadamard_anchor`,
  checked in binary32), and `anyfn_fixup`'s pass reads the `128 ·` columns. A per-block scale would close the anchor,
  but each block's float rescale is then a budgeted rounding a block, which grows the honest budget instead."""

from __future__ import annotations

import functools
import math
import random
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "property_redteam"))

import squaring as Q  # noqa: E402

P = Q.P
NBITS = 1024
S8, U8 = Q.S8, Q.U8
COIN_SLOT0 = 1 << 20


@dataclass(frozen=True)
class Mech:
    """A rounding gate's law: `det` (exact floor), `sr` (stochastic rounding, ≡ dither), or `noise` (RNE then ±n)."""

    kind: str = "det"
    n: int = 0

    @property
    def name(self) -> str:
        return self.kind if self.kind != "noise" else f"noise{self.n}"


DET, SR = Mech("det"), Mech("sr")


class RBuilder(Q.Builder):
    """`squaring.Builder` with randomized gates. Coins come from `self.rng` as each gate evaluates (rounding noise) or
    are coin inputs (accumulator noise) that only the noisy dot's own affine reads."""

    def __init__(self, mech: Mech = DET) -> None:
        super().__init__()
        self.mech = mech
        self.rng = random.Random(0)
        self.macs = 0
        self.coins: list[tuple[int, int, int]] = []  # (slot, lo, hi) of each accumulator-noise coin
        self.seam_fail_log2 = -math.inf

    def mma(self, head, a, b, fa, fb):  # noqa: D102
        self.macs += len(a)
        return super().mma(head, a, b, fa, fb)

    def nfloor(self, terms: list[tuple[int, int]], k: int, c: int = 0) -> int:
        """`⌊(c + Σ coeff·v) / 2^k⌋` under the mechanism, one rounding unit (operands: the terms' gates)."""
        args = [g for _, g in terms]
        w = [cf for cf, _ in terms]
        lo = c + sum(min(cf * self.gates[g].lo, cf * self.gates[g].hi) for cf, g in terms)
        hi = c + sum(max(cf * self.gates[g].lo, cf * self.gates[g].hi) for cf, g in terms)
        m, rng = self.mech, self
        if m.kind == "det":
            def f(v):
                return (c + sum(x * y for x, y in zip(w, v))) >> k
            dlo, dhi = lo >> k, hi >> k
        elif m.kind == "sr":
            def f(v):
                return (c + sum(x * y for x, y in zip(w, v)) + rng.rng.randrange(1 << k)) >> k
            dlo, dhi = lo >> k, (hi >> k) + 1
        elif m.kind == "noise":
            half = 1 << (k - 1)

            def f(v):
                return ((c + sum(x * y for x, y in zip(w, v)) + half) >> k) + rng.rng.randint(-m.n, m.n)
            dlo, dhi = ((lo + half) >> k) - m.n, ((hi + half) >> k) + m.n
        else:
            raise ValueError(m.kind)
        return self.budgeted(f, args, dlo, dhi, "rounding")

    def clamp(self, x: int, lo: int, hi: int) -> int:
        return self.budgeted(lambda v: min(max(v[0], lo), hi), [x], lo, hi, "compare_select")

    def cmp(self, terms: list[tuple[int, int]], c: int = 0) -> int:
        """`[c + Σ coeff·v < 0]`, one compare_select unit."""
        w = [cf for cf, _ in terms]
        return self.budgeted(lambda v: int(c + sum(x * y for x, y in zip(w, v)) < 0), [g for _, g in terms], 0, 1,
                             "compare_select")

    def anyfn(self, f: Callable[[list[int]], int], args: list[int], lo: int, hi: int) -> int:
        assert self.mech.kind == "det", "any-function gates exist only where nothing is randomized"
        return self.budgeted(f, args, lo, hi, "rounding")

    def noisy_mma(self, a: list[int], b: list[int], fa, fb, g: int, rep: int = 0) -> int:
        """`2^rep · Σ a_i b_i + η` (each product repeated `2^rep` times in the dot), `η` uniform in `[−2^(g−1), 2^(g−1))`;
        with `g = 0` no noise. The exact sum is internal: callers only get the noisy one."""
        exact = self.mma(self.const(0), a, b, fa, fb)
        self.macs += len(a) * ((1 << rep) - 1)
        if g <= 0:
            return self.affine(0, [(1 << rep, exact)]) if rep else exact
        slot = COIN_SLOT0 + len(self.coins)
        lo, hi = -(1 << (g - 1)), (1 << (g - 1)) - 1
        self.coins.append((slot, lo, hi))
        eta = self.inp(slot, lo, hi, "coin")
        return self.affine(0, [(1 << rep, exact), (1, eta)])

    def draw_coins(self) -> dict[int, int]:
        return {s: self.rng.randint(lo, hi) for s, lo, hi in self.coins}

    def units_by_class(self) -> dict[str, int]:
        c: Counter = Counter()
        for g, u in zip(self.gates, self._fixed):
            if u:
                c[g.op.budget or "input"] += u
        return dict(c)


# ---- digits ---------------------------------------------------------------------------------------------------------------


def n_digits(b: int) -> int:
    return -(-NBITS // b)


def fold_split(b: int) -> list[int]:
    """`R^n mod p` for `R = 2^b`, `n = ⌈1024/b⌉`, as radix-`R` digits: `2^(bn) ≡ 105·2^(bn − 1024)`."""
    delta = Q.DELTA << (b * n_digits(b) - NBITS)
    out = []
    while delta:
        out.append(delta % (1 << b))
        delta >>= b
    # a third digit would carry the top column's fold past position n − 1, which needs a second fold
    assert len(out) <= 2, f"δ has {len(out)} radix-2^{b} digits; the one-pass fold needs at most 2"
    return out


def window_fail_log2(a: tuple[int, int], d: tuple[int, int], low: tuple[int, int], ok: tuple[int, int],
                     width: int) -> float:
    """log2 of the share of inputs on which a seam is wrong: the window's value `A` (interval `a`, uniform over its
    `width` values) plus the fold's `D` less the lower digits' `L` must stay in `ok`, at `D` and `L` extreme."""
    over = max(0, a[1] + d[1] - low[0] - ok[1]) + max(0, ok[0] - (a[0] + d[0] - low[1]))
    return math.log2(over / width) if over else -math.inf


def fold_term(c_top: tuple[int, int], split: list[int], b: int) -> tuple[int, int]:
    """`D = c_top·δ`'s interval, `δ`'s digits at their window positions."""
    dw = value(split, b)
    return min(c_top[0] * dw, c_top[1] * dw), max(c_top[0] * dw, c_top[1] * dw)


def group_fail_log2(c_top: tuple[int, int], split: list[int], b: int, w: int) -> float:
    """A window of `w` digits recomputed as one group (`seam_square`'s limbs 0..3, `anyfn_fixup`'s fixup digits): wrong
    when `A + D` leaves `[0, R^w)`."""
    width = 1 << (b * w)
    return window_fail_log2((0, width - 1), fold_term(c_top, split, b), (0, 0), (0, width - 1), width)


def ripple_fail_log2(c_top: tuple[int, int], split: list[int], b: int, s: int, digits: tuple[int, int],
                     centered: bool) -> float:
    """`restricted_ripple`'s low window: its last digit is `(A + D − L) / R^(s−1)`, `A` the window's value less `R^s`
    times the precomputed carry, `L` the lower `s − 1` digits. Under `det` the digits are canonical, so the last digit
    is `⌊(A + D) / R^(s−1)⌋` and the window is a group's. Under a randomized law the precomputed carry rounds to
    nearest (`A ∈ [−R^s/2, R^s/2)`) and `L` is any value its redundant digits reach."""
    if not centered:
        return group_fail_log2(c_top, split, b, s)
    R = 1 << b
    Rs, Rs1 = R ** s, R ** (s - 1)
    S = (Rs1 - 1) // (R - 1)
    return window_fail_log2((-(Rs // 2), Rs // 2 - 1), fold_term(c_top, split, b), (digits[0] * S, digits[1] * S),
                            (digits[0] * Rs1, digits[1] * Rs1 + Rs1 - 1), Rs)


def to_digits(x: int, b: int, signed: bool) -> list[int]:
    """`x`'s `n` radix-`2^b` digits, canonical in `[0, R)` or balanced in `[−R/2, R/2)`."""
    R, n = 1 << b, n_digits(b)
    out = []
    for _ in range(n):
        d = x % R
        if signed and d >= R // 2:
            d -= R
        out.append(d)
        x = (x - d) // R
    out[-1] += x * R  # a balanced top digit absorbs what is left (0 for x < 2^1024 at b | 1024; small otherwise)
    return out


def value(ds: list[int], b: int) -> int:
    return sum(d << (b * i) for i, d in enumerate(ds))


def unfolded_columns(bld: RBuilder, a: list[int], fmt, g: int = 0, rep: int = 0) -> list[int]:
    """The `2n − 1` columns `Σ_{i+j=m} a_i a_j`, each one dot (noisy with `g > 0`, products repeated `2^rep` times)."""
    n = len(a)
    cols = []
    for m in range(2 * n - 1):
        idx = list(range(max(0, m - n + 1), min(m, n - 1) + 1))
        cols.append(bld.noisy_mma([a[i] for i in idx], [a[m - i] for i in idx], fmt, fmt, g, rep))
    return cols


def fold(bld: RBuilder, cols: list[int], b: int) -> list[int]:
    """Fold the top columns down by `R^n ≡ Σ_t δ_t R^t`: position `j` gets `col_j + Σ_t δ_t col_{j+n−t}`."""
    n = n_digits(b)
    split = fold_split(b)
    out = []
    for j in range(n):
        terms = [(1, cols[j])]
        for t, dt in enumerate(split):
            m = j + n - t
            if dt and n <= m < len(cols):
                terms.append((dt, cols[m]))
        out.append(bld.affine(0, terms))
    return out


# ---- the restricted model: nfloor carries and clamp digits -----------------------------------------------------------


def digit_range(mech: Mech, b: int) -> tuple[int, int]:
    """Where `x − R·carry` always lands for the mechanism's carry `nfloor(x / R)`."""
    R = 1 << b
    if mech.kind == "det":
        return 0, R - 1
    if mech.kind == "sr":
        return -R, R - 1
    return -R // 2 - mech.n * R, R // 2 + mech.n * R - 1


def exact_floor(bld: RBuilder, x: int, k: int, c: int = 0) -> int:
    """`⌊(c + x) / 2^k⌋` exactly: one `nfloor`, then as many `cmp` corrections as its law needs."""
    q = bld.nfloor([(1, x)], k, c)
    m = bld.mech
    if m.kind == "det":
        return q
    R = 1 << k
    if m.kind == "sr":  # q ∈ {⌊⌋, ⌊⌋ + 1}: down one when c + x − R·q < 0
        return bld.affine(0, [(1, q), (-1, bld.cmp([(1, x), (-R, q)], c))])
    # noise n: q = RNE + e ∈ ⌊⌋ + [−n, n + 1], so r = c + x − R·q ∈ [−(n+1)R, (n+1)R) and ⌊⌋ = q + ⌊r/R⌋, with
    # ⌊r/R⌋ = −Σ_{t=1}^{n+1} [r < −(t−1)R] + Σ_{t=1}^{n} (1 − [r < tR]): 2n + 1 comparisons
    terms = [(1, q)]
    for t in range(1, m.n + 2):
        terms.append((-1, bld.cmp([(1, x), (-R, q)], c + (t - 1) * R)))
    for t in range(1, m.n + 1):
        terms.append((-1, bld.cmp([(1, x), (-R, q)], c - t * R)))
    return bld.affine(m.n, terms)


def restricted_ripple(bld: RBuilder, f: list[int], b: int, s: int, signed_out: bool) -> list[int]:
    """The ripple over folded positions `f` (each an s32 affine), `R = 2^b`: an exact carry into position `s` first
    (rounded to nearest under a randomized law, so the last low digit centers), then `nfloor` carries and `clamp` digits
    over `s … n−1`, the top carry folded into positions `0 …` by `fold_split`, and the low window `0 … s−1` last, closed
    by the precomputed carry. Returns the `n` digit gates."""
    n, R = len(f), 1 << b
    mech = bld.mech
    lo, hi = digit_range(mech, b)
    if not signed_out:
        assert mech.kind == "det"
    split = fold_split(b)
    assert len(split) <= s, "the fold's δ must land inside the low window"
    c = bld.const(0)
    for j in range(s):
        x = bld.affine(0, [(1, f[j]), (1, c)])
        off = R // 2 if (j == s - 1 and mech.kind != "det") else 0
        c = exact_floor(bld, x, b, off)
    cs = c
    out: list[int] = [0] * n
    for j in range(s, n):
        x = bld.affine(0, [(1, f[j]), (1, c)])
        c = bld.nfloor([(1, x)], b)
        out[j] = bld.clamp(bld.affine(0, [(1, x), (-R, c)]), lo, hi)
    c_top = c
    ct = bld.gates[c_top]
    bld.seam_fail_log2 = ripple_fail_log2((ct.lo, ct.hi), split, b, s, (lo, hi), mech.kind != "det")
    c = bld.const(0)
    for j in range(s):
        terms = [(1, f[j]), (1, c)]
        if j < len(split) and split[j]:
            terms.append((split[j], c_top))
        x = bld.affine(0, terms)
        if j < s - 1:
            c = bld.nfloor([(1, x)], b)
            out[j] = bld.clamp(bld.affine(0, [(1, x), (-R, c)]), lo, hi)
        else:
            out[j] = bld.clamp(bld.affine(0, [(1, x), (-R, cs)]), lo, hi)
    return out


def restricted(bld: RBuilder, a: list[int], b: int, s: int) -> list[int]:
    """One squaring in the restricted model: honest columns, the fold, `restricted_ripple`."""
    signed = bld.mech.kind != "det"
    f = fold(bld, unfolded_columns(bld, a, S8 if signed else U8), b)
    return restricted_ripple(bld, f, b, s, signed)


def clean_column(bld: RBuilder, y: int, G: int) -> int:
    """The exact column `c` from `y = 2^G c + η`, `|η| < 2^(G−1)`: `⌊(y + 2^(G−1)) / 2^G⌋` exactly."""
    return exact_floor(bld, y, G, 1 << (G - 1))


def restricted_acc(bld: RBuilder, a: list[int], b: int, g: int, rep: int, s: int) -> list[int]:
    """Both mechanisms: noisy dots (`g`) with `rep` repetition bits and `b`-bit digits scaled into s8 (guard
    `2(7 − b) + rep ≥ g`... the clean needs `G ≥ g`), every unfolded column cleaned exactly, then the ripple."""
    signed = bld.mech.kind != "det"
    scale_bits = (7 if signed else 8) - b
    G = 2 * scale_bits + rep
    assert G >= g, f"guard {G} < noise {g}"
    sa = [bld.affine(0, [(1 << scale_bits, x)]) for x in a] if scale_bits else a
    fmt = S8 if signed else U8
    ys = unfolded_columns(bld, sa, fmt, g, rep)
    cols = [clean_column(bld, y, G) if G else y for y in ys]
    f = fold(bld, cols, b)
    return restricted_ripple(bld, f, b, s, signed)


# ---- the any-function model with noisy dots (deterministic roundings) ---------------------------------------------------


def anyfn_fixup(bld: RBuilder, a: list[int], b: int, g: int = 0, rep: int = 0, fixup: int | None = None,
                cap: int = 128, rot: bool = False) -> list[int]:
    """One greedy carry pass of any-function gates over `b`-bit canonical digits (scaled into u8 by `2^(8−b)`, so the
    dots carry `2(8 − b) + rep` guard bits), each group gate reading the carry in and its positions' noisy columns and
    cleaning them inside; the top carry folded into the low digits and one fixup group of `fixup` digit gates (no carry,
    wrong when `δ·c_top` moves the group's carry out: `group_fail_log2`). With `rot` the columns are `anchor_columns`'
    (unscaled digits, `HAD_BLOCK ·` each column, no noise)."""
    assert bld.mech.kind == "det"
    n, R = n_digits(b), 1 << b
    if rot:
        assert g == 0 and rep == 0
        G = HAD_BLOCK.bit_length() - 1
        ys = anchor_columns(bld, a, b)
    else:
        scale_bits = 8 - b
        G = 2 * scale_bits + rep
        assert g <= G, f"guard {G} < noise {g}"
        sa = [bld.affine(0, [(1 << scale_bits, x)]) for x in a] if scale_bits else a
        ys = unfolded_columns(bld, sa, U8, g, rep)
    split = fold_split(b)
    half = (1 << (G - 1)) if G else 0

    def clean(v: int) -> int:
        return (v + half) >> G

    # position j reads y_j and y_{j+n−t} for each nonzero δ_t; noiseless columns fold first, free
    if G == 0:
        reads = [[(1, f)] for f in fold(bld, ys, b)]
    else:
        reads = []
        for j in range(n):
            r = [(1, ys[j])]
            for t, dt in enumerate(split):
                m = j + n - t
                if dt and n <= m < len(ys):
                    r.append((dt, ys[m]))
            reads.append(r)

    def bits(gate: int) -> int:
        return Q.clog2(bld.gates[gate].hi + 1 - bld.gates[gate].lo)

    def pos_max(j: int) -> int:
        return sum(w * clean(bld.gates[y].hi) for w, y in reads[j])

    out: list[int] = [0] * n
    carry, carry_hi, base = bld.const(0), 0, 0
    while base < n:
        ops, w, used = [carry], 0, bits(carry)
        while base + w < n:
            need = sum(bits(y) for _, y in reads[base + w])
            if w and used + need > cap:
                break
            ops += [y for _, y in reads[base + w]]
            used += need
            w += 1
        layout = [(base + i, [wt for wt, _ in reads[base + i]]) for i in range(w)]

        def total(v: list[int], layout=layout) -> int:
            acc, k = v[0], 1
            for pos, wts in layout:
                for wt in wts:
                    acc += (wt * clean(v[k])) << (b * (pos - layout[0][0]))
                    k += 1
            return acc

        vmax = carry_hi + sum(pos_max(p) << (b * (p - base)) for p, _ in layout)
        for i in range(w):
            out[base + i] = bld.anyfn(lambda v, i=i, total=total: (total(v) >> (b * i)) % R, ops, 0, R - 1)
        carry_hi = vmax >> (b * w)
        carry = bld.anyfn(lambda v, total=total, w=w: total(v) >> (b * w), ops, 0, carry_hi)
        base += w
    if fixup is None:
        fixup = -(-48 // b)
    assert len(split) <= fixup, "the fold's δ must land inside the fixup group"
    bld.seam_fail_log2 = group_fail_log2((0, carry_hi), split, b, fixup)
    t0 = bld.affine(0, [(1, out[0])] + ([(split[0], carry)] if split[0] else []))
    low = [t0] + [bld.affine(0, [(1, out[i])] + ([(split[i], carry)] if i < len(split) and split[i] else []))
                  for i in range(1, fixup)]
    for i in range(fixup):
        out[i] = bld.anyfn(lambda v, i=i: (sum(x << (b * j) for j, x in enumerate(v)) >> (b * i)) % R, low, 0, R - 1)
    return out


# ---- measuring ------------------------------------------------------------------------------------------------------------


def seam_square(bld: RBuilder, a: list[int]) -> list[int]:
    """`squaring.seam_square`, its seam's bound recorded."""
    out, c_top = Q.seam_square(bld, a)
    ct = bld.gates[c_top]
    bld.seam_fail_log2 = group_fail_log2((ct.lo, ct.hi), [Q.DELTA], 8, Q.SEAM_W)
    return out


@functools.cache
def reference_fail_log2() -> float:
    """`seam_square`'s bound, the one every construction's window is held to."""
    bld = RBuilder()
    seam_square(bld, [bld.inp(i, *U8) for i in range(Q.N_LIMBS)])
    return bld.seam_fail_log2


@dataclass
class Spec:
    name: str
    mech: Mech
    b: int
    build: Callable[[RBuilder, list[int], int], list[int]]  # (builder, input digits, seam window) → output digits
    digits: tuple[int, int]  # the input digits' range, which is the output's (squarings chain)
    noise_g: int = 0
    rep: int = 0
    window: bool = True  # whether the construction takes a seam window to size (`seam_square`'s is fixed)
    rot: bool = False  # its dots run under the rotation (`hadamard_anchor`)


def spec(name: str) -> Spec:
    """A construction by name: `anyfn_seam`, `anyfn_fixup_b{b}[_g{g}_r{rep}]`, `restricted_{det|sr|noise1}_b{b}`,
    `restricted_acc_{det|sr}_b{b}_g{g}_r{rep}`, `hadamard_anchor_b{b}`."""
    parts = name.split("_")
    kw = {p[0]: int(p[1:]) for p in parts if len(p) > 1 and p[0] in "bgr" and p[1:].isdigit()}
    b, g, rep = kw.get("b", 8), kw.get("g", 0), kw.get("r", 0)
    if name == "anyfn_seam":
        return Spec(name, DET, 8, lambda bld, a, w: seam_square(bld, a), U8, window=False)
    if name.startswith("anyfn_fixup"):
        return Spec(name, DET, b, lambda bld, a, w: anyfn_fixup(bld, a, b, g, rep, w), (0, (1 << b) - 1), g, rep)
    if name.startswith("hadamard_anchor"):
        return Spec(name, DET, b, lambda bld, a, w: anyfn_fixup(bld, a, b, fixup=w, rot=True), (0, (1 << b) - 1),
                    rot=True)
    acc = name.startswith("restricted_acc")
    mech = {"det": DET, "sr": SR, "noise1": Mech("noise", 1)}[parts[2] if acc else parts[1]]
    if acc:
        return Spec(name, mech, b, lambda bld, a, w: restricted_acc(bld, a, b, g, rep, w), digit_range(mech, b), g,
                    rep)
    return Spec(name, mech, b, lambda bld, a, w: restricted(bld, a, b, w), digit_range(mech, b))


def build(sp: Spec) -> tuple[RBuilder, list[int], int | None]:
    """The construction with the narrowest seam window whose bound is at most the reference's."""
    n = n_digits(sp.b)
    for w in range(len(fold_split(sp.b)), n) if sp.window else [None]:
        bld = RBuilder(sp.mech)
        out = sp.build(bld, [bld.inp(i, *sp.digits, "coin") for i in range(n)], w)
        if not sp.window or bld.seam_fail_log2 <= reference_fail_log2():
            return bld, out, w
    raise AssertionError("no seam window keeps the construction within the reference's failure bound")


def measure(name: str, trials: int = 32, seed: int = 20261026) -> dict:
    """Build the construction once; run it on `trials` random digit vectors over the whole input range (so redundant
    and out-of-[0, p) values too), each with fresh coins; check the output ≡ input² (mod p); count units and MACs."""
    sp = spec(name)
    bld, out, seam_window = build(sp)
    n = n_digits(sp.b)
    lo, hi = sp.digits
    rng = random.Random(seed)
    right = window = hold = 0
    for t in range(trials):
        if t % 2:
            ds = [rng.randint(lo, hi) for _ in range(n)]
        else:
            ds = to_digits(rng.randrange(P), sp.b, lo < 0)
        x = value(ds, sp.b)
        bld.rng = random.Random(rng.getrandbits(64))
        xin = dict(enumerate(ds))
        xin.update(bld.draw_coins())
        vals, w, h = bld.evaluate(xin)
        o = [vals[g] for g in out]
        right += all(lo <= v <= hi for v in o) and value(o, sp.b) % P == x * x % P
        window += w
        hold += h
    budgeted = [g for g in bld.gates if g.op.kind == "budgeted"]
    units = bld.fixed_units
    rot = {}
    if sp.rot:
        # the longest column's dot, through the rotation's binary32 arithmetic; the first at the ℓ1 bound
        blocks, rt = 1 + -(-n // anchor_q(sp.b)), max(2, trials // 4)
        rot = {"rotation_trials": rt,
               "rotation_right": sum(hadamard_anchor(rng, sp.b, blocks, extreme=t == 0) for t in range(rt))}
    return {**rot, "name": name, "mech": sp.mech.name, "b": sp.b, "digits": n, "noise_g": sp.noise_g, "rep": sp.rep,
            "units": units, "units_by_class": bld.units_by_class(), "macs": bld.macs, "gates": len(bld.gates),
            "proven": Q.proven(bld.gates) == [], "operand_bits_128": Q.operand_bits(128, bld.gates),
            "max_operand_bits": max((sum(bld.gates[j].bits for j in g.args) for g in budgeted), default=0),
            "seam_window": seam_window,
            "seam_fail_log2": None if bld.seam_fail_log2 == -math.inf else round(bld.seam_fail_log2, 2),
            "trials": trials, "right": right, "in_window": window, "declared_hold": hold}


# ---- the PRF a cheater would compute to predict the coins -------------------------------------------------------------

#: SHA-512's compression, in property/v1 units (modeled, `prf_units`)
SHA512_ROUNDS = 80


def prf_units() -> dict:
    """What predicting one coin byte costs a cheater who computes the verifier's stream (`verity.ml.randomness`, SHA-512
    over the key and the op's index) inside its circuit. Modeled two ways, both per 64-byte block:

    * a floor from fan-in: each round's new `e` reads `d, e, f, g, h, W_t` (384 bits) and its new `a` reads also `a, b, c`
      (576 bits); a gate reads at most 128 operand bits, so a 64-bit output over `r` input bits needs at least
      `⌈(r − 64) / 64⌉` gates of 64-bit outputs chained: 5 for `e`, 8 for `a`; the schedule's `W_t` reads 256 bits, 3.
      That is 16 a round, 1,280 a compression less the 16 schedule-free rounds' 48: **1,232**.
    * an estimate with 16-bit limbs (s32 affine maps can't hold 64-bit sums): per round `T1` (Σ1, Ch, h, K, W and the
      carries) 8 gates, `T2` (Σ0, Maj) 8, `a' = T1 + T2` and `e' = d + T1` 4 each, and the schedule 8: **32 a round,
      2,560 a compression**.

    So one coin byte costs at least 1,232 / 64 ≈ 19 units, about 40 by the estimate, against the 1 unit (`cmp`) that
    corrects an SR'd carry without knowing its coin."""
    floor = SHA512_ROUNDS * (5 + 8 + 3) - 16 * 3
    est = SHA512_ROUNDS * 32
    return {"compression_floor": floor, "compression_estimate": est, "per_coin_byte_floor": floor / 64,
            "per_coin_byte_estimate": est / 64}


# ---- a fresh block-Hadamard rotation inside every free dot ------------------------------------------------------------

#: `honest.py`'s `ROT_BLOCK`
HAD_BLOCK = 128


def f32(x: float) -> float:
    """`x` rounded to binary32, to nearest even."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def rotated_codes(x: list[float], sign: list[int], blk: int = HAD_BLOCK) -> list[int]:
    """One row of `honest.py`'s `QLinear` under `had`: the signs, the block Hadamard in binary32 (Sylvester's
    `H[j][i] = (−1)^popcount(i & j)` times `1/√blk`, one rounding a multiply-add, as an FMA chain), one scale
    `amax / 127` for the whole row, and round-to-nearest-even into `[−128, 127]`."""
    r = f32(1 / math.sqrt(blk))
    v = []
    for b0 in range(0, len(x), blk):
        xs = [(j, f32(x[b0 + j] * sign[b0 + j])) for j in range(blk) if x[b0 + j]]
        for i in range(blk):
            acc = 0.0
            for j, xj in xs:
                acc = f32(acc + xj * (-r if (i & j).bit_count() & 1 else r))
            v.append(acc)
    s = f32(max(map(abs, v)) / 127)
    return [max(-128, min(127, round(f32(t / s)))) for t in v]


def anchor_q(b: int) -> int:
    """Balanced radix-`2^b` digits (`[−2^(b−1), 2^(b−1))`) a block can hold with its ℓ1 norm at most 127."""
    return 127 >> (b - 1)


def hadamard_anchor(rng: random.Random, b: int, blocks: int, extreme: bool = False, q: int | None = None,
                    anchor: bool = True, blk: int = HAD_BLOCK) -> bool:
    """One dot of two rows of balanced radix-`2^b` digits under fresh signs, through `rotated_codes`, against
    `blk · Σ x_j w_j`. Block 0 holds the anchors: 127 at coordinate 0 of `x` and at coordinate 1 of `w`. Rows 0 and 1 of
    `H` are orthogonal, so the anchors add nothing to the dot, and each pins its row's amax at `127/√blk` whatever the
    signs. Every other block holds `q = anchor_q(b)` digits of each row, so each rotated entry there is an integer of
    magnitude at most 127 over `√blk`, and its code is that integer. `extreme` puts every digit at `−2^(b−1)` and picks
    the signs that reach the ℓ1 bound at entry 0."""
    q, h = anchor_q(b) if q is None else q, 1 << (b - 1)
    n = blocks * blk
    x, w = [0.0] * n, [0.0] * n
    if anchor:
        x[0] = w[1] = 127.0
    sign = [rng.choice((-1, 1)) for _ in range(n)]
    want = 0
    for b0 in range(blk, n, blk):
        for j in rng.sample(range(b0, b0 + blk), q):
            dx, dw = (-h, -h) if extreme else (rng.randrange(-h, h), rng.randrange(-h, h))
            x[j], w[j] = float(dx), float(dw)
            if extreme:
                sign[j] = 1
            want += dx * dw
    got = sum(p * t for p, t in zip(rotated_codes(x, sign, blk), rotated_codes(w, sign, blk)))
    return got == blk * want


def anchor_columns(bld: RBuilder, a: list[int], b: int) -> list[int]:
    """The columns `blk · Σ_{i+j=m} a_i a_j` of canonical digits as `hadamard_anchor`'s dots give them: each column a dot
    of the balanced digits `a_i − 2^(b−1)` (`blk ·` the sum, exact), recentred by a free affine map. Its MACs are the
    rotated blocks it spans, the anchor's and `⌈|column| / anchor_q(b)⌉` of digits, `blk` each."""
    n, h, q = len(a), 1 << (b - 1), anchor_q(b)
    assert q >= 1, f"a radix-2^{b} digit alone is past ℓ1 127"
    bal = [bld.affine(-h, [(1, x)]) for x in a]
    macs0, macs = bld.macs, 0
    cols = []
    for m in range(2 * n - 1):
        idx = list(range(max(0, m - n + 1), min(m, n - 1) + 1))
        d = bld.noisy_mma([bal[i] for i in idx], [bal[m - i] for i in idx], S8, S8, 0, HAD_BLOCK.bit_length() - 1)
        corr = [(HAD_BLOCK * h, a[i]) for i in idx] + [(HAD_BLOCK * h, a[m - i]) for i in idx]
        cols.append(bld.affine(-HAD_BLOCK * h * h * len(idx), [(1, d)] + corr))
        macs += HAD_BLOCK * (1 + -(-len(idx) // q))
    bld.macs = macs0 + macs
    return cols


BASE = ["anyfn_seam", "anyfn_fixup_b8", "restricted_det_b8", "restricted_sr_b7", "restricted_noise1_b6"]
NOISE_BITS = (4, 6, 8, 10, 12)
#: the anchor construction at each digit width the fold allows (b ≥ 4) and a block holds (b ≤ 7)
ROTATED = [f"hadamard_anchor_b{b}" for b in range(4, 8)]


def grid() -> list[str]:
    """Every accumulator-noise construction the cheater could pick at each noise `g`: each digit width `b ≥ 4` (below
    that `δ` has three radix digits, `fold_split`), with just enough repetition bits to reach `g` guard bits (digits
    scaled into the operand give `2(8 − b)`, or `2(7 − b)` signed)."""
    names = []
    for g in NOISE_BITS:
        names += [f"anyfn_fixup_b{b}_g{g}_r{max(0, g - 2 * (8 - b))}" for b in range(8, 3, -1)]
        names += [f"restricted_acc_det_b{b}_g{g}_r{max(0, g - 2 * (8 - b))}" for b in range(8, 3, -1)]
        names += [f"restricted_acc_sr_b{b}_g{g}_r{max(0, g - 2 * (7 - b))}" for b in range(7, 3, -1)]
    return names


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=32, help="random inputs (with fresh coins) a base construction runs")
    ap.add_argument("--grid-trials", type=int, default=8, help="the same for each accumulator-noise construction")
    ap.add_argument("names", nargs="*", help="constructions (default: BASE and the accumulator-noise grid)")
    a = ap.parse_args()
    rows = {}
    for nm in a.names or BASE + grid() + ROTATED:
        try:
            rows[nm] = measure(nm, a.trials if nm in BASE + ROTATED else a.grid_trials)
        except AssertionError as e:  # an s32 or guard bound: the construction doesn't exist at these parameters
            rows[nm] = {"name": nm, "infeasible": str(e)}
        print(json.dumps(rows[nm]), flush=True)
    res = {"model": "squaring.py's port of Model.lean, extended with randomized gates (this module's docstring)",
           "squarings_per_word": Q.ROUNDS * Q.SQUARINGS, "constructions": rows, "prf": prf_units()}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
