"""ACCEL ALGO-COST for SEG-COINS: the cheapest price c_bit of O = b c^T for 0/1 operands, per card.

On the RTX PRO 6000 the FP4 MMA (E2M1, block scale 1) runs at 2x INT8; Strassen's depth-r forms of 0/1 blocks are signed
sums of at most 2^r entries, and E2M1 holds every integer in [-4, 4] exactly (0, 1, 2, 3, 4 and their negatives), so
depth 2 is exact in FP4 (FP32 accumulation is exact below 2^24). This checks it exactly on random bits: every form
value at depth r, whether all lie in E2M1's integers, and the product. On the H200 the b1 MMA (AND.POPC, rated 8x INT8
by RT-UNITS) takes 0/1 operands only, and a Strassen form of bits leaves {0, 1}, so c_bit = 1/8 there.
"""

from __future__ import annotations

import numpy as np

from word_width import OUT, PAIRS, quads

E2M1_INTS = {0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6}


def strassen(a, b, depth, seen):
    if depth == 0:
        seen.update(np.unique(a).tolist())
        seen.update(np.unique(b).tolist())
        return a @ b
    A, B = quads(a), quads(b)
    prods = [strassen(sum(c * q for c, q in zip(ca, A, strict=True)), sum(c * q for c, q in zip(cb, B, strict=True)),
                      depth - 1, seen) for ca, cb in PAIRS]
    c = [sum(w * p for w, p in zip(row, prods, strict=True)) for row in OUT]
    return np.block([[c[0], c[1]], [c[2], c[3]]])


def main() -> None:
    rng = np.random.default_rng(7)
    b = rng.integers(0, 2, (64, 256))
    c = rng.integers(0, 2, (256, 64))
    for r in (1, 2, 3):
        seen: set[int] = set()
        ok = bool(np.array_equal(strassen(b, c, r, seen), b @ c))
        fits = seen <= E2M1_INTS
        print(f"depth {r}: exact={ok}; leaf operand values {sorted(seen)}; all E2M1 integers: {fits}; "
              f"FP4 tensor price {0.5 * (7 / 8) ** r:.4f} of an INT8 product" + ("" if fits else " (not exact in FP4)"))
    for L in (640, 2688):
        d1, d2 = 1.5, 1.5 + 1.75 * 1.5
        print(f"RTX PRO 6000 with prefix words at L = {L} (2-input FADD deliveries, 8 MAC-times each): "
              f"depth 1 {0.5 * 0.875 + 8 * d1 / L:.4f}, depth 2 {0.5 * 0.875**2 + 8 * d2 / L:.4f}")
    t, alu = 0.5 * 0.875**2, 0.77
    print(f"RTX PRO 6000 with emulated b1 (LOP3 + POPC, ~{alu}) on the ALU alongside FP4 depth 2: "
          f"1 / (1/{t:.3f} + 1/{alu}) = {1 / (1 / t + 1 / alu):.3f}")


if __name__ == "__main__":
    main()
