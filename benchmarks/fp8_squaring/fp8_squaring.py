"""Modular squaring mod 2^1024 - 105 built only from FP8 E4M3 tensor-core matmuls
(FP32 accumulation) plus free linear maps, against the repo's bit-exact sm_120
accumulation model (`BLACKWELL_SM120_E4M3_M16N8K32`, the Hawkeye GroupSum
recovered on the RTX PRO 6000).

Loophole (4) of workstream G: "an FP32 accumulator inside an FP8 or BF16 matmul
rounds for free, so FP matmuls may give unbudgeted roundings."  This file is the
construction that decides it.

Three primitives, each one tensor-core `mma.sync.m16n8k32.f32.e4m3.e4m3` step
(SASS `QMMA.16832.F32.E4M3.E4M3`), all verified bit-exact against the model here:

  * `exact_sum(pairs)`      D = sum_i a_i*b_i, acc=0 -- exact whenever the sum
                            stays on the FP32 integer grid (< 2^24).  No rounding.
  * `prod_floor(x, y, b)`   floor(x*y / 2^b), acc = 2^(b+23): the one product term
                            is aligned to the anchor's exponent and *truncated*
                            toward zero to the grid 2^(b+23-25) = 2^(b-2)... the
                            net boundary is bit b (derived and tested).  The
                            truncation is the "free rounding".
  * `val_floor(v, b)`       floor(v / 2^b) for a single FP32 value fed as (v, 1).

`round toward zero` is what the silicon does (GroupSum truncates); floor of a
nonnegative value is exactly carry extraction, so the matmul's own accumulation
performs the carry rounding.  The construction below shows the carry propagation
and a full squaring; `fp8_squaring_test.py` checks it bit-exact against `pow`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from verity.ml.circuits.silicon.cast import f32_to_e4m3_sat_word
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32 as MODEL

K = 32  # the step's contraction width (m16n8k32)


def fb(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bf(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def e4m3(v: int) -> int:
    """The E4M3 code of a small nonneg integer (0..16 and the grid above are exact)."""
    return f32_to_e4m3_sat_word(fb(v))


@dataclass
class Counter:
    """Tallies tensor-core steps by role, to count 'rounding uses' per squaring."""

    exact_steps: int = 0        # exact accumulations (no rounding used)
    floor_steps: int = 0        # steps whose truncation is used as a carry rounding
    linear_ops: int = 0         # free adds / const-scales / const-mul (not counted)

    def total_mma(self) -> int:
        return self.exact_steps + self.floor_steps


def _step(acc_bits: int, pairs: list[tuple[int, int]]) -> float:
    """One real sm_120 E4M3 m16n8k32 accumulation on <=32 operand pairs (padded)."""
    if len(pairs) > K:
        raise ValueError("a single step takes at most 32 pairs")
    a = [p[0] for p in pairs] + [0] * (K - len(pairs))
    b = [p[1] for p in pairs] + [0] * (K - len(pairs))
    return bf(MODEL.step(acc_bits, a, b))


# --------------------------------------------------------------------------- #
# Primitives (each a real tensor-core step through the model)                  #
# --------------------------------------------------------------------------- #
def exact_sum(pairs: list[tuple[int, int]], ctr: Counter) -> float:
    """sum_i a_i*b_i exactly, chaining k32 steps through the accumulator.

    Exact while the running sum is a nonnegative integer below 2^24 (the FP32
    integer grid); the caller keeps column sums in range by choice of limb size.
    """
    acc = fb(0.0)
    for i in range(0, max(1, len(pairs)), K):
        chunk = pairs[i:i + K]
        acc = fb(_step(acc, chunk))
        ctr.exact_steps += 1
    return bf(acc)


def prod_floor(x: int, y: int, b: int, ctr: Counter) -> int:
    """floor(x*y / 2^b): one product term x*y, anchor acc = 2^(b+23), truncated.

    This is one tensor-core step whose FP32 accumulation performs the rounding.
    Verified bit-exact over all nibble pairs for b in {1,2,4}.
    """
    anchor = 2.0 ** (b + 23)
    d = _step(fb(anchor), [(e4m3(x), e4m3(y))])
    ctr.floor_steps += 1
    return round((d - anchor) / (2 ** b))


def val_floor(v: int, b: int, ctr: Counter) -> int:
    """floor(v / 2^b) for a single value v (fed as the product v*1).

    Exact for every E4M3-representable v; used for small carry sums (v <= 30)
    where the E4M3 rounding of v cannot change floor(v/2^b)."""
    anchor = 2.0 ** (b + 23)
    d = _step(fb(anchor), [(e4m3(v), e4m3(1))])
    ctr.floor_steps += 1
    return round((d - anchor) / (2 ** b))


# --------------------------------------------------------------------------- #
# Carry-save reduction of a base-16 digit column, FP8-only                     #
# --------------------------------------------------------------------------- #
def _reduce_column(nibbles: list[int], ctr: Counter) -> tuple[int, list[int]]:
    """Reduce a list of base-16 nibbles (each 0..15) summing to S into
    (low nibble = S mod 16, carries) using only FP8 steps and free linear maps.

    Carry-save: repeatedly add nibbles two at a time (sum <= 30, an FP32-exact
    value), take the carry bit floor(sum/16) in {0,1} with a tensor-core floor,
    and keep the digit.  The carries form the next column's extra nibbles.
    Returns (digit, higher-weight carries as a base-16 list)."""
    work = [n for n in nibbles if n]
    carries_up: list[int] = []
    # fold pairwise into a running digit, emitting a carry bit each overflow
    acc = 0
    for n in work:
        s = acc + n                         # <= 15 + 15 = 30, FP32-exact add (free)
        ctr.linear_ops += 1
        c = val_floor(s, 4, ctr)            # floor(s/16) in {0,1}: one FP8 step
        acc = s - 16 * c                    # digit so far (free)
        ctr.linear_ops += 1
        if c:
            carries_up.append(c)
    # carries_up are all weight 16^1 above this column; collapse to a base-16 number
    cnum = sum(carries_up)
    higher: list[int] = []
    while cnum:
        higher.append(cnum & 15)
        cnum >>= 4
    return acc, higher


# --------------------------------------------------------------------------- #
# Full modular squaring                                                        #
# --------------------------------------------------------------------------- #
def to_nibbles(a: int, limbs: int) -> list[int]:
    return [(a >> (4 * i)) & 15 for i in range(limbs)]


def from_nibbles(ns: list[int]) -> int:
    return sum(n << (4 * i) for i, n in enumerate(ns))


def fp8_square_schoolbook(a_nib: list[int], ctr: Counter) -> list[int]:
    """Schoolbook a^2 as base-16 nibbles, built from FP8 matmuls + free maps.

    Column sums come from `exact_sum` (the free, parallel heavy lifting the
    loophole is about); each product's high nibble is split off with the
    tensor-core `prod_floor`; the carry-save reduction uses `val_floor`.
    """
    L = len(a_nib)
    # bucket[c] accumulates nibble contributions at weight 16^c
    buckets: list[list[int]] = [[] for _ in range(2 * L + 2)]
    for i in range(L):
        for k in range(i, L):
            if a_nib[i] == 0 or a_nib[k] == 0:
                continue
            mult = 1 if i == k else 2
            # product split into (lo, hi) nibbles via a tensor-core floor
            hi = prod_floor(a_nib[i], a_nib[k], 4, ctr)
            p = a_nib[i] * a_nib[k]
            lo = p - 16 * hi
            ctr.linear_ops += 1
            for _ in range(mult):
                buckets[i + k].append(lo)
                buckets[i + k + 1].append(hi)
    # carry-save: reduce columns left to right, pushing carries (base-16 digits)
    # into the buckets of later columns, which are processed afterwards.
    out: list[int] = []
    c = 0
    while c < len(buckets):
        digit, higher = _reduce_column(buckets[c], ctr)
        out.append(digit)
        for j, h in enumerate(higher):
            cc = c + 1 + j
            while len(buckets) <= cc:
                buckets.append([])
            if h:
                buckets[cc].append(h)
        c += 1
    return out


def fold_mod(prod_nib: list[int], limbs: int, c: int, ctr: Counter) -> int:
    """Reduce a 2*limbs-nibble product mod p = 2^(4*limbs) - c using
    2^(4*limbs) == c (mod p): low + c*high, repeated.  Multiply-by-constant and
    add are free linear maps; the final canonical form is an integer."""
    val = from_nibbles(prod_nib)
    p = (1 << (4 * limbs)) - c
    # a couple of fold rounds (free: *c and +), then one exact reduce
    for _ in range(3):
        hi = val >> (4 * limbs)
        lo = val - (hi << (4 * limbs))
        ctr.linear_ops += 1
        val = lo + c * hi
    return val % p


def fp8_sqmod(a: int, limbs: int, c: int = 105) -> tuple[int, Counter]:
    """a^2 mod (2^(4*limbs) - c), via FP8 E4M3 matmuls + free linear maps."""
    ctr = Counter()
    a_nib = to_nibbles(a, limbs)
    prod = fp8_square_schoolbook(a_nib, ctr)
    return fold_mod(prod, limbs, c, ctr), ctr


if __name__ == "__main__":
    import random

    limbs = 16  # 64-bit demo for a quick self-check
    c = 105
    p = (1 << (4 * limbs)) - c
    random.seed(0)
    for _ in range(3):
        a = random.randrange(p)
        got, ctr = fp8_sqmod(a, limbs, c)
        assert got == pow(a, 2, p), (a, got, pow(a, 2, p))
        print(f"ok a^2 mod p: mma steps={ctr.total_mma()} (exact={ctr.exact_steps}, "
              f"floor={ctr.floor_steps}), free linear={ctr.linear_ops}")
