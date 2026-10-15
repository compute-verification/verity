"""The 2:4-sparse route to the product of noise matrices, checked exactly on the CPU, with its cost from the census's MMA rates.

sm_120's sparse MMAs (census §1) run 2:4-sparse A operands at twice the dense rate: FP8/int8 2,048 and NVFP4 4,096 logical
MACs/SM/clk, against dense 1,024 and 2,048. A dense noise matrix isn't 2:4 sparse, but a group constant makes it so:
in each group of 4 consecutive k of a row, subtract the group's most frequent value c,

    A = M + C ⊗ 1_4 + R,      A·B = M·B  +  C·Bsum  +  R·B,      Bsum[g, j] = Σ_{k in group g} B[k, j],

where M keeps at most 2 nonzeros a group (2:4 sparse), C is n × K/4 (a dense product with a quarter of the k), and R holds
the third nonzero of the groups whose 4 values are all distinct (none for a two-valued alphabet). The script builds the
split, checks A·B = M·B + C·Bsum + R·B exactly (int64), finds the cheapest format for each part, and prices it in units of
one dense int8 GEMM (1,024 MACs/SM/clk). R·B is priced at the cheaper of a second 2:4 product and FP32 FFMA on the CUDA
cores (124 lanes/SM/clk; FFMA and MMA don't overlap, census §3).

    python benchmarks/d4_attacks/sparse.py [--n 256] [--k 4096] [--json OUT]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from represent import ALPHABETS, FORMATS, scales_holding

FP4 = FORMATS["E2M1 (FP4)"]
E4M3 = FORMATS["E4M3 (FP8)"]
DENSE = {"fp4": 0.5, "8bit": 1.0}
SPARSE = {"fp4": 0.25, "8bit": 0.5}
FFMA_PER_MAC = 1024 / 124


def fmt(values) -> str:
    """The narrowest format holding every value (with one scale): fp4, else 8-bit (E4M3, or int8 when it fits)."""
    vals = sorted({int(v) for v in np.unique(values)} - {0}) or [1]
    if scales_holding(vals, FP4):
        return "fp4"
    if scales_holding(vals, E4M3) or max(abs(v) for v in vals) <= 127:
        return "8bit"
    return "wide"


def split(A: np.ndarray):
    n, K = A.shape
    G = A.reshape(n, K // 4, 4)
    C = np.zeros((n, K // 4), dtype=np.int64)
    for i in range(n):
        for g in range(K // 4):
            vals, cnt = np.unique(G[i, g], return_counts=True)
            C[i, g] = vals[np.argmax(cnt)]
    D = G - C[:, :, None]
    M = np.zeros_like(D)
    R = np.zeros_like(D)
    nz_rank = np.cumsum(D != 0, axis=2)
    keep = (D != 0) & (nz_rank <= 2)
    M[keep] = D[keep]
    R[(D != 0) & ~keep] = D[(D != 0) & ~keep]
    return M.reshape(n, K), C, R.reshape(n, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {}
    print("| alphabet | 2:4 groups w/o residual | residual density | M·B | C·Bsum (k/4) | R·B | total / int8 | dense best / int8 | exact |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, alph in ALPHABETS.items():
        alph = np.array(alph, dtype=np.int64)
        A = rng.choice(alph, size=(a.n, a.k))
        B = rng.choice(alph, size=(a.k, a.n))
        M, C, R = split(A)
        Bsum = B.reshape(a.k // 4, 4, a.n).sum(axis=1)
        exact = bool(np.array_equal(A @ B, M @ B + C @ Bsum + R @ B))
        assert (np.count_nonzero(M.reshape(a.n, -1, 4), axis=2) <= 2).all()
        rho = np.count_nonzero(R) / R.size
        good = 1 - np.count_nonzero(np.count_nonzero(R.reshape(a.n, -1, 4), axis=2)) / (a.n * a.k // 4)
        fB = fmt(B)
        fM = "fp4" if fmt(M) == "fp4" and fB == "fp4" else "8bit"
        fC = "fp4" if fmt(C) == "fp4" and fmt(Bsum) == "fp4" else "8bit"
        cM = SPARSE[fM]
        cC = DENSE[fC] / 4
        fR = "fp4" if fmt(R) == "fp4" and fB == "fp4" else "8bit"
        cR_sparse = SPARSE[fR] if rho else 0.0
        cR_ffma = rho * FFMA_PER_MAC
        cR = min(cR_sparse, cR_ffma)
        rway = "none" if not rho else ("2:4 " + fR if cR_sparse <= cR_ffma else "FFMA")
        total = cM + cC + cR
        dense = 0.5 if fB == "fp4" and scales_holding(sorted(set(alph.tolist())), FP4) else 1.0
        out[name] = {"exact": exact, "residual_density": rho, "groups_without_residual": good, "M": [fM, cM],
                     "C_Bsum": [fC, cC], "R": [rway, cR], "total_int8_units": total, "dense_best_int8_units": dense,
                     "n": a.n, "k": a.k}
        print(f"| {name} | {good:.3f} | {rho:.4f} | {fM} 2:4: {cM:.3f} | {fC}: {cC:.3f} | {rway}: {cR:.3f} | "
              f"**{total:.3f}** | {dense:.2f} | {exact} |")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
