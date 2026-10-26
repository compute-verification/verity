"""ACCEL PROTO-DESIGN's scorer (CMP-118): each candidate protocol's honest overhead γ on the H200 with its SHA-512
hashing, and the share of a step's static INT8 MACs it can credit at γ ≤ Γ.

γ is the protocol's time over plain INT8 serving's on the roofline: plain is max(MACs at η of the data-sheet dense INT8
rate, bytes at η of HBM); the protocol's compute is γ_c of the plain tensor time (the machine model's per-MMA ratio, a
measured run, or an estimate, `Design.src` says which) plus its SHA-512 compressions at the H200's measured 4.79 G/s
(note:hash/20261018T0636Z-draft-structure), added because a compression is 2,837 integer-ALU instructions and every
γ_c > 1 here is ALU-bound; compute overlaps the weight stream. Crediting a fraction f of each call's output columns
scales its extra time by f (row noise is then overcharged, conservative).

Hashing per call is noise + fold keys + receipt commitments. The H1 family's counts are `h1tile`'s
(`unit_compressions`, J10-R's closed form of `j10r_compressions`, held to it in `check_j10r`); a receipt of b bits costs
2·⌈b/1024⌉ (its leaves and as many tree nodes). Per-call designs derive their fresh bits once a call in counter mode,
512 bits a compression, and commit CP-W's two words a 32-wide segment of each 32 × 32 tile.

    uv run python benchmarks/pouw/accel_score/score.py            # the tables
    uv run python benchmarks/pouw/accel_score/score.py --json     # every number
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from verity.core.protocols.pouw.circuit import h1tile
from verity.core.protocols.pouw.circuit.h1tile import Layout

MAC_RATE = 1978.9e12 / 2   # dense INT8 MAC/s, the census's H100 SXM5 row, which the H200 (GH100) shares
HBM = 4.8e12               # H200 data sheet, B/s
ETA = 0.85                 # plain GEMM's share of both, CMP-107's convention
SHA = 4.79e9               # SHA-512 compressions/s on the H200, measured (commit_gpu)
L = 32

# (name, K, N) a layer; layers; head
SHAPES = {
    "llama-8b": ([("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096)], 32),
    "llama-70b": ([("qkv", 8192, 10240), ("o", 8192, 8192), ("gate_up", 8192, 57344), ("down", 28672, 8192)], 80),
    "frozen": ([("qkv", 16384, 16512), ("o", 16384, 16384), ("up", 16384, 65536), ("down", 65536, 16384)], 52),
}


def ceil_to(x: int, q: int) -> int:
    return -(-x // q) * q


def commit(bits: int) -> int:
    return 2 * math.ceil(bits / 1024)


def h1_unit(tile: tuple[int, int], k: int) -> Callable[[int, int, int], float]:
    """Today's per-unit noise: `unit_compressions` plus the unit's receipt, a unit an r × c × k tile."""
    r, c = tile

    def h(m: int, n: int, kk: int) -> float:
        lay = Layout(r, c, k)
        units = (ceil_to(m, r) // r) * (n // c) * max(1, kk // k)
        return units * (h1tile.unit_compressions(k, tile) + commit(64 * lay.sets))
    return h


def j10r(tile: tuple[int, int], call_keys: bool = False) -> Callable[[int, int, int], float]:
    """J10-R (`PouwH1Tile_v5`/`v6`): each row and column block once a call, seed_c's two, and per unit σ_R's two and a
    key block a lane; with ``call_keys`` the keys are derived once a call instead (a protocol change, not in v5/v6)."""
    r, c = tile

    def h(m: int, n: int, k: int) -> float:
        lay, mm = Layout(r, c, k), ceil_to(m, r)
        units = (mm // r) * (n // c)
        keys = (2 + lay.lanes) if call_keys else units * (2 + lay.lanes)
        return mm * lay.rb + n * lay.cb + 2 + keys + units * commit(64 * lay.sets)
    return h


def check_j10r() -> None:
    for m, n, k, t in [(8, 16, 64, (4, 8)), (64, 64, 4096, (32, 32)), (4, 256, 256, (4, 8))]:
        lay = Layout(*t, k)
        units = (m // t[0]) * (n // t[1])
        closed = Fraction(m * lay.rb + n * lay.cb + 2, units) + 2 + lay.lanes
        assert closed == h1tile.j10r_compressions(m, n, k, t), (m, n, k, t)


def per_call(a_bits: Callable[[int, int, int], float], b_bits: Callable[[int, int, int], float],
             extra: Callable[[int, int, int], float] = lambda m, n, k: 0.0,
             seg: int = L) -> Callable[[int, int, int], float]:
    """Fresh bits once a call (A side, B side), 512 a compression, plus CP-W's receipts (two words a segment of
    ``seg``) of each 32 × 32 tile."""
    def h(m: int, n: int, k: int) -> float:
        tiles = (ceil_to(m, 32) // 32) * (n // 32)
        return (a_bits(m, n, k) + b_bits(m, n, k)) / 512 + extra(m, n, k) + tiles * commit(64 * (k // seg)) + 4
    return h


def l16_receipts(h: Callable[[int, int, int], float]) -> Callable[[int, int, int], float]:
    """``h`` with each 32 × 32 tile's receipts doubled: two words a 16-wide segment instead of a 32-wide one."""
    def g(m: int, n: int, k: int) -> float:
        return h(m, n, k) + (ceil_to(m, 32) // 32) * (n // 32) * commit(64 * (k // L))
    return g


@dataclass(frozen=True)
class Design:
    name: str
    gamma_c: float
    src: str
    hash: Callable[[int, int, int], float]


DESIGNS = [
    Design("h1-v1 (4x8, K=64, per-unit noise)", 4.61, "model: duo at L=32 (fx1's IMAD on its own pipe)",
           h1_unit((4, 8), 64)),
    Design("h1u (32x32, whole K, per-unit noise)", 4.61, "model: duo at L=32", lambda m, n, k: h1_unit((32, 32), k)(m, n, k)),
    Design("J10-R v5/v6 (4x8, whole K)", 4.61, "model: duo at L=32", j10r((4, 8))),
    Design("J10-R v5/v6 (32x32, whole K)", 4.61, "model: duo at L=32", j10r((32, 32))),
    Design("J10-R + call keys (32x32)", 4.61, "model: duo at L=32", j10r((32, 32), call_keys=True)),
    Design("J10-R v5/v6 as built (32x32, duo)", 16.53 * ETA,
           "BET-COVERAGE's projection: 16.53 peak MAC-times of lane work a MAC (h1umjc_sm, utilization 0.448)",
           j10r((32, 32))),
    Design("H1/J10-R + LI fold + call keys (32x32)", 1.558, "model DC-90: dither-c-li-l32 (binds JI90_0)",
           j10r((32, 32), call_keys=True)),
    Design("H1/J10-R + L=16 k16 tree3 fold (32x32)", 1.44,
           "BET-BANDWIDTH sheet: mma.sync k16, L=16, fan-in-3 tree, issue-bound",
           l16_receipts(j10r((32, 32), call_keys=True))),
    Design("Dither-2S LI (fresh bits both sides)", 1.558, "model DC-90: dither-2s-li-l32 (binds JI90_0)",
           per_call(lambda m, n, k: m * k, lambda m, n, k: n * k)),
    Design("Dither-C LI (A bits + U)", 1.558, "model DC-90: dither-c-li-l32",
           per_call(lambda m, n, k: m * k, lambda m, n, k: 7 * n * L)),
    Design("Dither-2S LI-CP-W redux (fresh bits both sides)", 2.344, "model DC-90, honest/credited 1.407",
           per_call(lambda m, n, k: m * k, lambda m, n, k: n * k)),
    Design("WRAP-HYBRID (A mod 2^8 full-width, B + U)", 2.6, "estimate: 2 MMAs (wrap correction) beside LI-CP-W",
           per_call(lambda m, n, k: m * k + 8 * 16 * m, lambda m, n, k: 7 * n * L)),
    Design("Pearl-C (FP8, as measured)", 4.52, "measured r20261025-145823-df25 (top 25 Oct), its own hashing inside",
           lambda m, n, k: 0.0),
    Design("BET-PEARL-I8 (int8 forming, low-rank N)", 4.52, "estimate: Pearl-C's forming and peel, integer",
           per_call(lambda m, n, k: 8 * 64 * m, lambda m, n, k: 8 * 64 * k)),
    Design("ROM per word (SHA-512 of every word)", 1.558, "LI plus a compression every 32 words",
           per_call(lambda m, n, k: m * k, lambda m, n, k: 7 * n * L, lambda m, n, k: m * n * k / (L * 32))),
]


def call_times(d: Design, m: int, k: int, n: int) -> tuple[float, float, float]:
    """(plain, plain tensor time, protocol compute) for one call, seconds."""
    tc = m * n * k / (ETA * MAC_RATE)
    mem = (n * k + m * k + 2 * m * n) / (ETA * HBM)
    return max(tc, mem), tc, d.gamma_c * tc + d.hash(m, n, k) / SHA


def step(d: Design, shape: str, m: int, f: float) -> tuple[float, float]:
    """(plain step time, protocol step time) over the shape's static linears, crediting f of every call's columns."""
    calls, layers = SHAPES[shape]
    t0 = t = 0.0
    for _, k, n in calls:
        plain, tc, prot = call_times(d, m, k, n)
        mem = (n * k + m * k + 2 * m * n) / (ETA * HBM)
        t0 += layers * plain
        t += layers * max(mem, tc + f * (prot - tc))
    return t0, t


def coverage(d: Design, shape: str, m: int, cap: float) -> float:
    t0, t1 = step(d, shape, m, 1.0)
    if t1 <= cap * t0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t0, t = step(d, shape, m, mid)
        lo, hi = (mid, hi) if t <= cap * t0 else (lo, mid)
    return lo


def per_mac(d: Design, shape: str, m: int) -> float:
    calls, _ = SHAPES[shape]
    return sum(d.hash(m, n, k) for _, k, n in calls) / sum(m * n * k for _, k, n in calls)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cap", type=float, default=10.0)
    a = ap.parse_args()
    check_j10r()
    ms = [4, 64, 1024, 20480]
    rows = []
    for shape in SHAPES:
        for d in DESIGNS:
            for m in ms:
                t0, t1 = step(d, shape, m, 1.0)
                rows.append({"shape": shape, "design": d.name, "src": d.src, "m": m, "gamma_c": d.gamma_c,
                             "sha_per_mac": per_mac(d, shape, m), "gamma": t1 / t0,
                             "coverage": coverage(d, shape, m, a.cap)})
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    for shape in SHAPES:
        print(f"\n{shape}: γ (whole step credited) and coverage at γ ≤ {a.cap:g}, by rows a call m")
        print(f"{'design':52s} " + " ".join(f"{'m=' + str(m):>22s}" for m in ms))
        for d in DESIGNS:
            cells = []
            for m in ms:
                r = next(x for x in rows if x["shape"] == shape and x["design"] == d.name and x["m"] == m)
                cells.append(f"{r['sha_per_mac']:7.1e} {r['gamma']:6.2f} {100 * r['coverage']:6.1f}%")
            print(f"{d.name:52s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\ncells: SHA-512 compressions a credited MAC, γ, coverage. γ_c sources:")
    for d in DESIGNS:
        print(f"  {d.name}: γ_c {d.gamma_c:.4g} ({d.src})")


if __name__ == "__main__":
    main()
