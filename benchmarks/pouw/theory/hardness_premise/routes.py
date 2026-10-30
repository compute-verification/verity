"""Every direct route for Y = E·Rᵀ, checked exact and priced in W1 (HT-PREMISE, P1).

E is m×K live noise (±1, written E = 2e − 1 with e ∈ {0, 1}, or wider), R is n×K uniform int8, public and
preprocessed. Each route below is run on random instances in exact integer arithmetic and compared with the int64
product; its price is W1's (CORE.md §2a; the H200's measured table, CORE-H32): an instruction costs 32 units per 32-bit
register written, so an MMA costs 32/k units a MAC at depth k (s8 and e4m3 k32: 1; sparse s8 k64: 0.5; b1 k256:
0.125; f16 k16: 2), an ALU add 32 a register, a load 128 a word through L1 (472 from L2).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

W1 = {"s8": 1.0, "e4m3": 1.0, "sp_s8": 0.5, "b1": 0.125, "f16": 2.0, "alu_reg": 32.0, "ld_l1_word": 128.0}


def ref(E, R):
    return E.astype(np.int64) @ R.astype(np.int64).T


def route_b1_planes(e, R):
    """E = 2e − 1; R's eight two's-complement bit planes, each an AND-popcount product (b1 MMA)."""
    out = np.zeros((e.shape[0], R.shape[0]), dtype=np.int64)
    Ru = R.astype(np.int64) & 0xFF
    for p in range(8):
        plane = (Ru >> p) & 1
        w = -(1 << p) if p == 7 else (1 << p)
        popc_and = e.astype(np.int64) @ plane.T  # what b1 .and.popc accumulates
        rowsum = plane.sum(axis=1)  # precomputed from R
        out += w * (2 * popc_and - rowsum[None, :])
    return out


def route_b1_planes_mod(e, R, w):
    """Y mod 2^w for E = 2e − 1 odd: plane p ≥ w vanishes, and plane w − 1 contributes 2^(w−1)·(Σ_k plane_k mod 2),
    which needs no E (E·b ≡ Σ b mod 2 for odd E). Returns (Y mod 2^w, planes that needed an MMA)."""
    out = np.zeros((e.shape[0], R.shape[0]), dtype=np.int64)
    Ru = R.astype(np.int64) & 0xFF
    used = 0
    for p in range(min(8, w)):
        plane = (Ru >> p) & 1
        wt = -(1 << p) if p == 7 else (1 << p)
        if p == w - 1:
            out += wt * (plane.sum(axis=1) % 2)[None, :]
            continue
        out += wt * (2 * (e.astype(np.int64) @ plane.T) - plane.sum(axis=1)[None, :])
        used += 1
    return np.mod(out, 1 << w), used


def route_sparse_halves(E, R):
    """R as the sum of two 2:4-sparse matrices (entries 0, 1 and 2, 3 of each group of four)."""
    K = R.shape[1]
    mask = (np.arange(K) % 4) < 2
    R1 = np.where(mask[None, :], R, 0)
    R2 = np.where(~mask[None, :], R, 0)
    assert ((R1 != 0).reshape(R.shape[0], -1, 4).sum(-1) <= 2).all()
    assert ((R2 != 0).reshape(R.shape[0], -1, 4).sum(-1) <= 2).all()
    return ref(E, R1) + ref(E, R2)


def route_fp8_digits(E, R):
    """R = 16·h + l with l in [0, 15] and h in [−8, 7], both exact in e4m3; two e4m3 products, f32 accumulate."""
    Ri = R.astype(np.int64)
    lo = Ri & 15
    hi = (Ri - lo) // 16
    assert lo.min() >= 0 and lo.max() <= 15 and hi.min() >= -8 and hi.max() <= 7
    f = lambda x: x.astype(np.float32)
    y = f(E) @ f(lo).T + np.float32(16) * (f(E) @ f(hi).T)  # every partial sum is an integer below 2^24
    return y.astype(np.int64)


STRASSEN_U = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
STRASSEN_V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
STRASSEN_W = [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]


def _blocks(M):
    h, w = M.shape[0] // 2, M.shape[1] // 2
    return [M[:h, :w], M[:h, w:], M[h:, :w], M[h:, w:]]


def route_peel_strassen(E, R):
    """R = 2·R_hi + R_lo with R_hi in [−64, 63]: one Strassen level on E·R_hiᵀ (every combination fits s8) and
    E·R_loᵀ on bit planes (b1). Seven s8 products of a quarter size plus one b1 plane: 7/8 + 1/8 = 1 in MMA units."""
    Ri = R.astype(np.int64)
    lo = Ri & 1
    hi = (Ri - lo) // 2
    assert hi.min() >= -64 and hi.max() <= 63
    A = _blocks(E.astype(np.int64))  # m×K in 2×2 blocks
    B = _blocks(hi.T)  # K×n in 2×2 blocks
    M = []
    for u, v in zip(STRASSEN_U, STRASSEN_V):
        a = sum(c * A[p] for p, c in enumerate(u) if c)
        b = sum(c * B[q] for q, c in enumerate(v) if c)
        assert np.abs(a).max() <= 127 and b.min() >= -128 and b.max() <= 127  # s8 operands
        M.append(a @ b)
    C = [sum(c * M[t] for t, c in enumerate(w) if c) for w in STRASSEN_W]
    Yhi = np.block([[C[0], C[1]], [C[2], C[3]]])
    return 2 * Yhi + ref(E, lo)


R_SINGLETON = [t for t, v in enumerate(STRASSEN_V) if sum(map(abs, v)) == 1]  # (A21+A22)·B11 and (A11+A12)·B22
USES = [sum(1 for w in STRASSEN_W if w[t]) for t in range(7)]


class Ops:
    """Counted work: MMA MACs by format, int32 output-side register writes, packed E-side writes (4 int8 a register)."""

    def __init__(self):
        self.s8 = self.b1 = self.c_writes = self.e_elems = 0

    def w1(self) -> float:
        return self.s8 * W1["s8"] + self.b1 * W1["b1"] + (self.c_writes + self.e_elems / 4) * W1["alu_reg"]


def strassen_selective(A, B, depth, ops, shape=None):
    """A·B with A the noise side (small integers) and B the R_hi side (7-bit), Strassen at the root and again, down to
    `depth` levels, in the two products whose B form is a single block, so B's combinations stay within s8.

    With `shape = (m, K, n)` and A, B None, only counts (the same recursion, no arithmetic). Output side: a product
    used in u outputs costs u − 1 int32 writes an element (the first use chains on the MMA accumulator); E side: one
    packed write per combined element."""
    m, K, n = shape if shape else (A.shape[0], A.shape[1], B.shape[1])
    if depth == 0:
        if A is not None:
            assert np.abs(A).max() <= 127 and B.min() >= -128 and B.max() <= 127
        ops.s8 += m * K * n
        return None if A is None else A @ B
    assert m % 2 == 0 and K % 2 == 0 and n % 2 == 0
    sub = (m // 2, K // 2, n // 2)
    Ab = _blocks(A) if A is not None else None
    Bb = _blocks(B) if B is not None else None
    M = []
    for t, (u, v) in enumerate(zip(STRASSEN_U, STRASSEN_V)):
        ops.e_elems += (sum(map(abs, u)) - 1) * sub[0] * sub[1]
        a = sum(c * Ab[p] for p, c in enumerate(u) if c) if Ab else None
        b = sum(c * Bb[q] for q, c in enumerate(v) if c) if Bb else None  # R side: preprocessed, free
        M.append(strassen_selective(a, b, depth - 1 if t in R_SINGLETON else 0, ops, None if Ab else sub))
    ops.c_writes += sum(u - 1 for u in USES) * sub[0] * sub[2]
    if Ab is None:
        return None
    C = [sum(c * M[t] for t, c in enumerate(w) if c) for w in STRASSEN_W]
    return np.block([[C[0], C[1]], [C[2], C[3]]])


def _wrap8(x):
    return ((x + 128) % 256) - 128


def strassen_mod256(A, B, depth, ops):
    """A·B mod 2^8 by full Strassen over Z/256: every combination of either operand wraps back into s8, so nothing
    widens, at any depth. Counts s8 MACs and output additions (Σ uses − 4 − single-use, as in `strassen_selective`)."""
    m, K, n = A.shape[0], A.shape[1], B.shape[1]
    if depth == 0:
        A, B = _wrap8(A), _wrap8(B)
        assert A.min() >= -128 and A.max() <= 127 and B.min() >= -128 and B.max() <= 127
        ops.s8 += m * K * n
        return A @ B
    Ab, Bb = _blocks(A), _blocks(B)
    M = []
    for u, v in zip(STRASSEN_U, STRASSEN_V):
        ops.e_elems += (sum(map(abs, u)) - 1) * (m // 2) * (K // 2)
        a = _wrap8(sum(c * Ab[p] for p, c in enumerate(u) if c))
        b = _wrap8(sum(c * Bb[q] for q, c in enumerate(v) if c))
        M.append(strassen_mod256(a, b, depth - 1, ops))
    ops.c_writes += (sum(USES) - 4 - USES.count(1)) * (m // 2) * (n // 2)
    C = [sum(c * M[t] for t, c in enumerate(w) if c) for w in STRASSEN_W]
    return np.block([[C[0], C[1]], [C[2], C[3]]])


def strassen_mod65536(A, B, depth, inner, ops):
    """A·B mod 2^16: exact Strassen to `depth` (the noise side stays within s8), then at each leaf the widened R form
    X = lo + 256·hi with lo its low byte: A·lo exactly in s8 (lo shifted by 128, one correction), and A·hi only mod
    2^8, by `strassen_mod256` to `inner` levels."""
    m, K, n = A.shape[0], A.shape[1], B.shape[1]
    if depth == 0:
        assert np.abs(A).max() <= 127
        lo = np.mod(B, 256)
        hi = (B - lo) // 256
        ops.s8 += m * K * n
        ops.c_writes += 2 * m * n  # the lo shift's correction and the 256·hi combine
        ops.e_elems += m * K  # row sums of the noise leaf
        Ylo = A @ (lo - 128) + 128 * A.sum(axis=1)[:, None]
        Yhi = strassen_mod256(A, hi, inner, ops)
        return Ylo + 256 * Yhi
    Ab, Bb = _blocks(A), _blocks(B)
    M = []
    for u, v in zip(STRASSEN_U, STRASSEN_V):
        ops.e_elems += (sum(map(abs, u)) - 1) * (m // 2) * (K // 2)
        a = sum(c * Ab[p] for p, c in enumerate(u) if c)
        b = sum(c * Bb[q] for q, c in enumerate(v) if c)
        M.append(strassen_mod65536(a, b, depth - 1, inner, ops))
    ops.c_writes += (sum(USES) - 4 - USES.count(1)) * (m // 2) * (n // 2)
    C = [sum(c * M[t] for t, c in enumerate(w) if c) for w in STRASSEN_W]
    return np.block([[C[0], C[1]], [C[2], C[3]]])


def route_peel_strassen_selective(e, R, depth, shape=None):
    """E = 2e − 1, R = 2·R_hi + R_lo: E·R_hiᵀ by `strassen_selective` in s8, e·R_loᵀ by one b1 AND-popcount chained on
    the same accumulator, then Y = 2·acc − colsum(R_lo) (one write an output). Returns (Y or None, Ops)."""
    ops = Ops()
    if shape is None:
        Ri = R.astype(np.int64)
        lo = Ri & 1
        hi = (Ri - lo) // 2
        E = 2 * e.astype(np.int64) - 1
        acc = strassen_selective(E, hi.T, depth, ops)
        # 2·(E·R_hiᵀ) + E·R_loᵀ = 2·(E·R_hiᵀ + e·R_loᵀ) − colsum(R_lo)
        acc = acc + e.astype(np.int64) @ lo.T
        Y = 2 * acc - lo.sum(axis=1)[None, :]
        m, K, n = e.shape[0], e.shape[1], R.shape[0]
    else:
        m, K, n = shape
        strassen_selective(None, None, depth, ops, shape)
        Y = None
    ops.b1 += m * K * n
    ops.c_writes += m * n
    return Y, ops


def route_four_russians(E, R, c):
    """Chunks of c noise signs index a table of ±-sums of c columns of R (preprocessed); one add per chunk."""
    m, K = E.shape
    n = R.shape[0]
    assert K % c == 0
    Ri = R.astype(np.int64)
    pats = np.array([[1 if (p >> i) & 1 else -1 for i in range(c)] for p in range(1 << c)], dtype=np.int64)
    tables = [pats @ Ri[:, j * c:(j + 1) * c].T for j in range(K // c)]  # (2^c) × n each
    Y = np.zeros((m, n), dtype=np.int64)
    for j in range(K // c):
        bits = (E[:, j * c:(j + 1) * c] > 0).astype(np.int64)
        idx = (bits << np.arange(c)).sum(axis=1)
        Y += tables[j][idx]
    return Y


def route_crt(E, R, moduli):
    """Residues of Y mod coprime m, each from R mod m, recombined by CRT (Theorem 2's form)."""
    Y = ref(E, R)
    lo, hi = -128 * E.shape[1], 128 * E.shape[1]
    Mprod = math.prod(moduli)
    assert Mprod > hi - lo
    res = [(ref(E, np.mod(R.astype(np.int64), m)) % m) for m in moduli]
    out = np.zeros_like(Y)
    for m, r in zip(moduli, res):
        Mi = Mprod // m
        out = (out + r * Mi * pow(Mi, -1, m)) % Mprod
    out = np.where(out > Mprod // 2, out - Mprod, out)
    return out


def prices(K: int, n: int, m: int, c_list=(4, 8, 12, 16)) -> dict:
    """W1 units a MAC for each route (honest s8 = 1), and Four Russians' table bytes for one n×K matrix."""
    p = {
        "s8": 1.0,
        "b1_planes": 8 * W1["b1"] + 8 * W1["alu_reg"] / K,  # 8 plane products, 7 shift-adds + 1 row-sum correction
        "sparse_halves": 2 * W1["sp_s8"],  # the second half chains on the first's accumulator
        "fp8_digits": 2 * W1["e4m3"] + W1["alu_reg"] / K,
        "f16": W1["f16"],
        # Strassen on R_hi, b1 for R_lo; 8 output adds per 2×2 block level and 5 E-combinations, int8x4-packed
        "peel_strassen": 7 / 8 * W1["s8"] + W1["b1"] + 8 * W1["alu_reg"] / (4 * K) + 5 * W1["alu_reg"] / (4 * 4 * n),
    }
    for c in c_list:
        # per chunk and output: one int16x2 add (16 a value) and its table load (128 a word, two values a word)
        p[f"four_russians_c{c}_adds_only"] = 16 / c
        p[f"four_russians_c{c}_with_loads"] = (16 + 64) / c
        p[f"four_russians_c{c}_table_GB"] = (K // c) * (2**c) * n * 2 / 1e9
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    checks = {}
    ok = True
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        m, n, K = 16, 24, 64
        e = rng.integers(0, 2, (m, K), dtype=np.int64)
        E = 2 * e - 1
        R = rng.integers(-128, 128, (n, K), dtype=np.int64)
        Y = ref(E, R)
        got = {
            "b1_planes": route_b1_planes(e, R),
            "sparse_halves": route_sparse_halves(E, R),
            "fp8_digits": route_fp8_digits(E, R),
            "peel_strassen": route_peel_strassen(E, R),
            "four_russians_c4": route_four_russians(E, R, 4),
            "four_russians_c8": route_four_russians(E, R, 8),
            "crt_2_3_5_7_11_13_17_19": route_crt(E, R, [2, 3, 5, 7, 11, 13, 17, 19]),
        }
        for k, v in got.items():
            eq = bool(np.array_equal(v, Y))
            checks.setdefault(k, []).append(eq)
            ok &= eq
    # The selective peel: exact at depths 1–4 (every leaf operand asserted within s8), and its counted W1 a MAC.
    peel = {}
    for depth in (1, 2, 3, 4):
        for s in range(args.seeds):
            rng = np.random.default_rng(100 + s)
            m, n, K = 32, 48, 64
            e = rng.integers(0, 2, (m, K), dtype=np.int64)
            R = rng.integers(-128, 128, (n, K), dtype=np.int64)
            Y, ops = route_peel_strassen_selective(e, R, depth)
            eq = bool(np.array_equal(Y, ref(2 * e - 1, R)))
            checks.setdefault(f"peel_selective_d{depth}", []).append(eq)
            ok &= eq
        _, dry = route_peel_strassen_selective(None, None, depth, shape=(m, K, n))
        ok &= (dry.s8, dry.b1, dry.c_writes, dry.e_elems) == (ops.s8, ops.b1, ops.c_writes, ops.e_elems)
        tensor = (ops.s8 * 8 + ops.b1) / (m * K * n)  # eighths a MAC, naive 8
        peel[f"d{depth}"] = {"tensor_eighths": tensor}
        for (mm, KK, nn) in ((4096, 2048, 4096), (4096, 4096, 4096), (8192, 8192, 8192), (8192, 28672, 8192),
                             (16384, 53248, 16384)):
            _, o = route_peel_strassen_selective(None, None, depth, shape=(mm, KK, nn))
            peel[f"d{depth}"][f"m{mm}_K{KK}_n{nn}"] = {
                "w1_per_mac": o.w1() / (mm * KK * nn),
                "tensor_only": (o.s8 + o.b1 * W1["b1"]) / (mm * KK * nn),
                "alu_writes_per_mac": (o.c_writes + o.e_elems / 4) / (mm * KK * nn),
            }
    # Committed width w: odd noise makes plane w − 1 free, so b1 slicing costs (w − 1)/8 for w ≤ 8.
    for w in (4, 8, 16):
        for s in range(args.seeds):
            rng = np.random.default_rng(200 + s)
            e = rng.integers(0, 2, (16, 64), dtype=np.int64)
            R = rng.integers(-128, 128, (24, 64), dtype=np.int64)
            got, used = route_b1_planes_mod(e, R, w)
            eq = bool(np.array_equal(got, np.mod(ref(2 * e - 1, R), 1 << w)))
            checks.setdefault(f"b1_planes_mod_w{w}", []).append(eq)
            ok &= eq
        peel[f"b1_planes_mod_w{w}_mma_planes"] = used
    # Committed width 8: Strassen over Z/256, any depth, s8 only; exact mod 2^8 for ±1 and 2-bit noise.
    for depth in (1, 2, 3, 4):
        for a in (2, 4):
            rng = np.random.default_rng(300 + depth + a)
            side = 2**depth * 4
            E = rng.integers(0, a, (side, side), dtype=np.int64) * (2 if a == 2 else 1) - (1 if a == 2 else 2)
            R = rng.integers(-128, 128, (side, side), dtype=np.int64)
            ops = Ops()
            Y = strassen_mod256(E, R.T, depth, ops)
            eq = bool(np.array_equal(np.mod(Y, 256), np.mod(ref(E, R), 256)))
            checks.setdefault(f"strassen_mod256_d{depth}", []).append(eq)
            ok &= eq
            peel[f"strassen_mod256_d{depth}_tensor"] = ops.s8 / side**3
            # c_writes scale as m·n·Σ_l (7/4)^(l−1)·6/4 for every shape; per MAC, divide by K
            peel[f"strassen_mod256_d{depth}_out_writes_per_output"] = ops.c_writes / side**2
    # Committed width 16: exact outer Strassen, mod-2^8 inner Strassen on the high parts.
    for depth, inner in ((1, 1), (2, 2), (3, 3), (2, 4)):
        for a in (2, 4):
            rng = np.random.default_rng(400 + depth * 10 + inner + a)
            side = 2 ** (depth + inner) * 2
            E = rng.integers(0, a, (side, side), dtype=np.int64) * (2 if a == 2 else 1) - (1 if a == 2 else 2)
            R = rng.integers(-128, 128, (side, side), dtype=np.int64)
            ops = Ops()
            Y = strassen_mod65536(E, R.T, depth, inner, ops)
            eq = bool(np.array_equal(np.mod(Y, 1 << 16), np.mod(ref(E, R), 1 << 16)))
            checks.setdefault(f"strassen_mod65536_d{depth}_i{inner}", []).append(eq)
            ok &= eq
            peel[f"strassen_mod65536_d{depth}_i{inner}_tensor"] = ops.s8 / side**3
            peel[f"strassen_mod65536_d{depth}_i{inner}_out_writes_per_output"] = ops.c_writes / side**2
    # the same mutation for the selective peel
    rng = np.random.default_rng(98)
    e = rng.integers(0, 2, (16, 32), dtype=np.int64)
    R = rng.integers(-128, 128, (16, 32), dtype=np.int64)
    R2 = R.copy()
    R2[2, 7] ^= 1
    mut2 = not np.array_equal(route_peel_strassen_selective(e, R2, 2)[0], ref(2 * e - 1, R))
    checks["mutation_caught_peel"] = mut2
    ok &= mut2

    # a mutation: flipping one bit of R must break each route's equality with the unflipped reference
    rng = np.random.default_rng(99)
    e = rng.integers(0, 2, (8, 32), dtype=np.int64)
    E = 2 * e - 1
    R = rng.integers(-128, 128, (8, 32), dtype=np.int64)
    R2 = R.copy()
    R2[3, 5] ^= 1
    mut = not np.array_equal(route_b1_planes(e, R2), ref(E, R))
    checks["mutation_caught"] = mut
    ok &= mut

    # 2:4: the chance that a uniform int8 tile has a free half (every group of four with two zeros)
    p0 = 1 / 256
    p_group = sum(math.comb(4, z) * p0**z * (1 - p0) ** (4 - z) for z in range(2, 5))
    tiles = {"p_group_two_zeros": p_group, "log10_p_tile_m64_k64": 64 * 16 * math.log10(p_group)}
    # rank: uniform int8 is full rank over Q and over small primes
    ranks = {}
    rng = np.random.default_rng(7)
    for nn in (64, 256):
        Rm = rng.integers(-128, 128, (nn, nn)).astype(np.float64)
        ranks[f"rank_Q_{nn}"] = int(np.linalg.matrix_rank(Rm))
    pr = {f"K{K}": prices(K, K, K) for K in (2048, 4096, 8192)}
    out = {"ok": bool(ok), "exact": {k: all(v) if isinstance(v, list) else v for k, v in checks.items()},
           "sparse_tile": tiles, "ranks": ranks, "w1_prices": pr, "peel_selective": peel}
    text = json.dumps(out, indent=1)
    if args.json:
        open(args.json, "w").write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
