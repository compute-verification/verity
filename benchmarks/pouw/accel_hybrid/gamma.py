"""HYB's γ on the H200 from data-sheet rates and `wrap_real.py`'s measured wrap densities.

    python gamma.py WRAP.json [--uX 64] [--k 8192] [--m 4096] [--n 4096] [--sw 32]

Tensor time 1 + d·L/k (d peel products of inner size L: 7 for HYB, 3 for H1), in units of the plain INT8 GEMM at
989.5 T MAC/s. ALU time c_op·(f/32 + p/2 + 1/m + 1/n) + 0.79/S_w: f the fold's lane-ops a 32-MAC word (li 0.54, duo 1.5),
p = p_X + p_W wrap adds a MAC (an IADD3 adds two), one lane-op an operand entry formed per call, SHA-512 at 4.79 G
compressions/s for one 8-byte warp checkpoint per 1,024 cells per S_w segments; c_op = 32 MAC-times a lane-op at 128
lane-ops per SM per clock (issue-priced) or 64 at the ALU pipe's 64. Or the wraps run on the 2:4 sparse INT8 MMA at
twice the dense rate: tensor + 0.5, p out of the ALU term. γ = max(tensor, ALU), the cheaper route.
"""

from __future__ import annotations

import argparse
import json

L = 32
FOLDS = {"li": 0.54, "duo": 1.5}
READINGS = {"issue": 32, "pipe": 64}
HASH_PER_SEGMENT = 8 / 128 * 2 / (1024 * L) * 989.5e12 / 4.79e9


def gamma(p: float, peels: int, f: float, c_op: int, k: int, m: int, n: int, sw: int) -> tuple[float, str]:
    tensor = 1 + peels * L / k
    rest = c_op * (f / 32 + 1 / m + 1 / n) + HASH_PER_SEGMENT / sw
    alu = max(tensor, rest + c_op * p / 2)
    sparse = max(tensor + 0.5, rest) if p > 0 else alu
    return (alu, "ALU") if alu <= sparse else (sparse, "2:4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrap")
    ap.add_argument("--uX", type=int, default=64)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--sw", type=int, default=32)
    a = ap.parse_args()
    t = json.load(open(a.wrap))["total"]
    pW, pX = {int(u): v for u, v in t["p_W"].items()}, {int(u): v for u, v in t["p_X"].items()}
    cols = [(fn, rn) for fn in FOLDS for rn in READINGS]
    print("design        p       " + "  ".join(f"{fn}/{rn:5s}" for fn, rn in cols))
    rows = [("H1 U64", pW[64], 3)] + [(f"HYB U{u} V{a.uX}", pW[u] + pX[a.uX], 7) for u in sorted(pW) if u > 64]
    for name, p, peels in rows:
        cells = [gamma(p, peels, FOLDS[fn], READINGS[rn], a.k, a.m, a.n, a.sw) for fn, rn in cols]
        print(f"{name:13s} {p:.4f}  " + "  ".join(f"{g:5.2f} {r:3s}  " for g, r in cells))


if __name__ == "__main__":
    main()
