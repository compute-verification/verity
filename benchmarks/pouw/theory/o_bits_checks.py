"""O-BITS (ACCEL, CMP-112): exact CPU checks of the non-ring attacks on Dither's transcript, each priced against the k32
INT8 tensor core's word at `o_bits_rates.py`'s ratios.

Dither's transcript word at cell (i, j) and block l is the exact prefix sum `T = sum_{t < (l+1)L} Ã_it B̃_tj mod 2^32`,
L = 32, `Ã = A + R` (R bits, A ≤ 126) and `B̃ = B + S` (S_tj = U_{j, t mod L}, U in ±1..±64, |B| ≤ 63). The adversary
picks A and B, and A = B = 0 leaves the narrowest operands: R bits against U (7 bits and a sign). Every check below runs at
A = B = 0 unless it says otherwise.

1. **Packing capacity.** Spaced fields in one IMAD (32-bit) or IMAD.WIDE (64-bit) product read off a dot product exactly
   only while no field carries into the next; the largest exact length for Dither's operand widths.
2. **IMAD bit 31 against Theorem K's measure.** One IMAD of 32 packed R bits by 32 reversed U bits has the full pair-
   difference rank 2L mod 2 at weight ≤ 2 (what `theoremK` counts), yet its bit 31 agrees with the word's parity only
   half the time: it defeats the rank measure, not the exact word.
3. **Bit-sliced words.** A carry-save adder over 32 bit-sliced words, checked exact, with its 3-input operations
   counted a word.
4. **Four Russians.** R's 32 bits a block in chunks of 11, 11, 10: a table of U's subset sums a chunk and column (built
   once: S repeats every L), three lookups and two adds a word; checked exact, priced with the lookups as LDS.128 that
   fetch four columns' entries a lane.
5. **duo's per-word step.** duo's word step is not affine over GF(2) or Z/2^32 in the word, so no linear combination of
   words (Strassen's products) folds in without forming each word.

    uv run python benchmarks/pouw/theory/o_bits_checks.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("o_bits_rates", HERE / "o_bits_rates.py")
R8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R8)

L = 32
M32 = 0xFFFFFFFF
RNG = np.random.default_rng(20261026)


def ratio(card: str, reading: str, prefix: str) -> float:
    """A writer's time a word over the k32 tensor core's (`o_bits_rates.ratios`), by name prefix."""
    return next(x[2] for x in R8.ratios(R8.CARDS[card], reading) if x[0].startswith(prefix))


def pack_capacity(amax: int, bmax: int, bits: int) -> tuple[int, int]:
    """Largest K, and its field width w, such that one product of two 32-bit packed operands, kept mod 2^bits, holds
    sum_i a_i b'_i exactly in its field K-1 (a in [0, amax], b' = b + bmax in [0, 2 bmax]; sum a_i b_i is that minus
    bmax sum a_i). Field K-1 sums K products, every lower field fewer, so none carries into it while each fits in w
    bits; both operands must fit in 32 bits and field K-1 in the product's."""
    best = (0, 0)
    for k in range(1, 64):
        w = (k * amax * 2 * bmax).bit_length()
        if (k - 1) * w + max(amax.bit_length(), (2 * bmax).bit_length()) <= 32 and k * w <= bits:
            best = (k, w)
    return best


def check_packing() -> None:
    print("1. packing capacity (an exact dot product read off one multiply's spaced fields)")
    for name, amax, bmax in (("R·U (1 bit x 7 bits + sign)", 1, 64), ("Ã·B̃ (7 bits x 7 bits + sign)", 127, 127)):
        for bits, op in ((32, "IMAD"), (64, "IMAD.WIDE")):
            k, w = pack_capacity(amax, bmax, bits)
            for _ in range(2000):
                a = [int(x) for x in RNG.integers(0, amax + 1, k)]
                b = [int(x) for x in RNG.integers(-bmax, bmax + 1, k)]
                pa = sum(a[i] << (i * w) for i in range(k))
                pb = sum((b[i] + bmax) << ((k - 1 - i) * w) for i in range(k))
                assert pa < 1 << 32 and pb < 1 << 32
                field = ((pa * pb) % (1 << bits)) >> ((k - 1) * w) & ((1 << w) - 1)
                assert field - bmax * sum(a) == sum(x * y for x, y in zip(a, b))
            print(f"   {name:30s} {op:9s}: K = {k} products exact (field {w} bits); checked on 2,000 draws")
    for card in ("H200", "sm_120"):
        imadw = ratio(card, "sheet", "IMAD.WIDE")
        alu = ratio(card, "sheet", "IADD3" if card == "H200" else "LOP3")
        k = pack_capacity(1, 64, 64)[0]
        ops = -(-L // k)
        cost = ops * (2 * imadw + 2 * alu)
        print(f"   {card}: a word by packed IMAD.WIDE costs >= {cost:.1f} k32 words ({ops} IMAD.WIDE writing 2 words at "
              f"{imadw:.2f} each, then a SHF and an IADD3 at {alu:.2f} each to take the field and add it)")


def check_imad_bit31() -> None:
    """Theorem K's measure (pairDiff mod 2 at weight ≤ 2) against one IMAD of packed bits, and its true agreement."""
    print("2. IMAD bit 31 against Theorem K's pair-difference rank")

    def f(r: np.ndarray, u: np.ndarray) -> np.ndarray:
        a = (r.astype(np.uint64) << np.arange(32, dtype=np.uint64)).sum(-1) & M32
        b = (u.astype(np.uint64) << (31 - np.arange(32, dtype=np.uint64))).sum(-1) & M32
        return ((a * b) & M32) >> 31

    def parity(r: np.ndarray, u: np.ndarray) -> np.ndarray:
        return (r & u).sum(-1) & 1

    def rank2(m: np.ndarray) -> int:
        m = m.copy() % 2
        r = 0
        for c in range(m.shape[1]):
            piv = next((i for i in range(r, m.shape[0]) if m[i, c]), None)
            if piv is None:
                continue
            m[[r, piv]] = m[[piv, r]]
            for i in range(m.shape[0]):
                if i != r and m[i, c]:
                    m[i] ^= m[r]
            r += 1
        return r

    for name, fn in (("IMAD bit 31", f), ("the word's parity", parity)):
        e = np.eye(64, dtype=np.uint8)
        z = np.zeros(64, dtype=np.uint8)

        def ev(x):
            return int(fn(x[None, :32], x[None, 32:])[0])

        f0 = ev(z)
        f1 = [ev(e[a]) for a in range(64)]
        pd = np.zeros((64, 64), dtype=np.uint8)
        for a in range(64):
            for b in range(a + 1, 64):
                pd[a, b] = pd[b, a] = (ev(e[a] ^ e[b]) ^ f1[a] ^ f1[b] ^ f0) & 1
        print(f"   {name:18s} pair-difference rank mod 2 = {rank2(pd)} (2L = {2 * L})")
    n = 1 << 18
    r = RNG.integers(0, 2, (n, 32), dtype=np.uint64)
    u = RNG.integers(0, 2, (n, 32), dtype=np.uint64)
    agree = float((f(r, u) == parity(r, u)).mean())
    print(f"   agreement of IMAD bit 31 with the parity over {n} uniform draws: {agree:.4f}")
    for card in ("H200", "sm_120"):
        lop, popc = ratio(card, "sheet", "IADD3" if card == "H200" else "LOP3"), ratio(card, "sheet", "POPC")
        print(f"   {card}: the exact parity a word costs LOP3 + POPC = {lop + popc:.2f} k32 words lane by lane, or "
              f"{lop:.2f} bit-sliced (one LOP3 a word a block); the exact word is check 3's")


def check_bitsliced() -> None:
    """Bit-sliced carry-save accumulation of 32 words' R·U block sums: exact, and its 3-input operations a word."""
    print("3. bit-sliced exact words (A = B = 0)")
    ops = 0
    words = 32
    R = RNG.integers(0, 2, (words, L), dtype=np.uint32)
    U = RNG.integers(1, 65, (words, L)) * RNG.choice([-1, 1], (words, L))
    Ub = (U + 64).astype(np.uint32)  # 0..128 as 8 unsigned bits; subtract 64 * sum(R) at the end
    # slices: bit b of word w lives in bit w of slice[b]
    width = 13  # sum of 32 terms < 32 * 129 < 2^13
    S = [np.uint32(0)] * width
    C = [np.uint32(0)] * width
    for t in range(L):
        rs = np.uint32(sum(int(R[w, t]) << w for w in range(words)))
        x = []
        for bit in range(8):
            us = np.uint32(sum(int((Ub[w, t] >> bit) & 1) << w for w in range(words)))
            x.append(rs & us)
            ops += 1
        x += [np.uint32(0)] * (width - 8)
        nS, nC = [np.uint32(0)] * width, [np.uint32(0)] * width
        for bit in range(width):
            nS[bit] = S[bit] ^ C[bit] ^ x[bit]
            carry = (S[bit] & C[bit]) | (S[bit] & x[bit]) | (C[bit] & x[bit])
            ops += 2  # two LOP3: sum and majority
            if bit + 1 < width:
                nC[bit + 1] = carry
        S, C = nS, nC
    # resolve S + C (ripple, two LOP3 a bit) and read the words back (a 32 x 32 transpose: 5 stages of 32 SHF/LOP3)
    out = [np.uint32(0)] * width
    cy = np.uint32(0)
    for bit in range(width):
        out[bit] = S[bit] ^ C[bit] ^ cy
        cy = (S[bit] & C[bit]) | (S[bit] & cy) | (C[bit] & cy)
        ops += 2
    ops += 5 * 32
    got = np.array([sum(((int(out[b]) >> w) & 1) << b for b in range(width)) for w in range(words)])
    want = (R.astype(np.int64) * (U + 64)).sum(1)
    assert (got == want).all()
    assert ((got - 64 * R.sum(1)) == (R.astype(np.int64) * U).sum(1)).all()
    per_word = ops / words
    tree = (L * 8 - 2 * width) * 2 + L * 8  # a Wallace tree: one full adder a bit removed, plus the ANDs
    print(f"   exact over {words} words; {ops} 3-input operations a block = {per_word:.1f} a word "
          f"(a Wallace tree: {tree / words:.1f}), before adding the block sum into the 32-bit prefix")
    for card in ("H200", "sm_120"):
        lop = ratio(card, "sheet", "IADD3" if card == "H200" else "LOP3")
        print(f"   {card}: >= {tree / words * lop:.0f} k32 words a word (LOP3 at {lop:.2f} a word written)")


def check_four_russians() -> None:
    """Four Russians at A = B = 0: chunk tables of U's subset sums; lookups and adds a word; exact."""
    print("4. Four Russians (A = B = 0)")
    n, nb = 64, 4
    chunks = [(0, 11), (11, 22), (22, 32)]
    R = RNG.integers(0, 2, (n, nb * L))
    U = RNG.integers(1, 65, (n, L)) * RNG.choice([-1, 1], (n, L))
    S = U[:, np.arange(nb * L) % L].T  # S[t, j] = U[j, t mod L]
    want = np.cumsum((R[:, :, None] * S[None, :, :]).reshape(n, nb, L, n).sum(2), axis=1)
    tables = []
    for lo, hi in chunks:
        g = hi - lo
        v = np.arange(1 << g)
        bits = (v[:, None] >> np.arange(g)) & 1
        tables.append(bits @ U[:, lo:hi].T)  # [value, j]
    got = np.zeros((n, nb, n), dtype=np.int64)
    acc = np.zeros((n, n), dtype=np.int64)
    for blk in range(nb):
        for c, (lo, hi) in enumerate(chunks):
            idx = (R[:, blk * L + lo: blk * L + hi] << np.arange(hi - lo)).sum(1)
            acc = acc + tables[c][idx]
        got[:, blk, :] = acc
    assert (got == want.transpose(0, 1, 2)).all()
    lookups, adds = len(chunks), 2
    print(f"   exact at n = {n}, {nb} blocks; a word = {lookups} lookups (LDS.128 fetching 4 columns' entries a lane) "
          f"and {adds} IADD3; tables 2^11 entries a chunk and column, built once (S repeats every L)")
    for card in ("H200", "sm_120"):
        for reading in ("ds", "sheet"):
            lds = ratio(card, reading, "LDS.128 (copy)")
            add = ratio(card, reading, "IADD3")
            cost = lookups * lds + adds * add
            print(f"   {card} {reading:6s}: {lookups} x {lds:.2f} + {adds} x {add:.2f} = {cost:.2f} k32 words a word")


def check_duo() -> None:
    """duo's word step y' = (y ^ p ^ K) KA + z is affine in p over neither GF(2)^32 nor Z/2^32."""
    print("5. duo's word step is nonlinear in the word")
    ka, kk, y, z = 0xA0000001, 0x5BD1E995, 0x12345678, 0x9ABCDEF0

    def step(p):
        return (((y ^ p ^ kk) * ka) + z) & M32

    ps = RNG.integers(0, 1 << 32, (2000, 2), dtype=np.uint64)
    xor_fail = sum(step(int(a) ^ int(b)) ^ step(int(a)) ^ step(int(b)) ^ step(0) != 0 for a, b in ps)
    add_fail = sum((step((int(a) + int(b)) & M32) - step(int(a)) - step(int(b)) + step(0)) & M32 != 0 for a, b in ps)
    print(f"   affine-over-GF(2) test fails on {xor_fail}/2000 pairs, affine-over-Z/2^32 on {add_fail}/2000")


def main() -> None:
    check_packing()
    check_imad_bit31()
    check_bitsliced()
    check_four_russians()
    check_duo()


if __name__ == "__main__":
    main()
