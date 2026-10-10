"""Front F6's prover column: the inner C-Flock prover's wall time and peak memory per session at each bucket's nonzero
bound, fitted from recorded prover runs, and where a session stops fitting on one node.

    python -m verity_numerical.bench.unit_costs_prover --inventory unit-costs.json --out DIR

**Source runs** (:data:`SOURCE`): every ``flock-circuit prove --zk`` record of ``benchmarks/pouw/hidden_zk`` on the Nebius
nodes (CPU prover, Rust, Flock b684b12 with C-Flock's patches; Intel Xeon 6776P, 192 CPUs, 1,716 GiB): the session's
instances, its block ``2^k_log`` bits per instance and ``D = 2^dense_m`` bits in all, the circuit text's bytes (one unit
netlist shared by the instances, plus the SHA-512 and hm96 glue), the prove's median time, the prover process's wall time
(circuit load included) and peak RSS (GiB), and the prover's threads.

**The law** (:func:`fit`): each of RSS, prove time and process wall time is affine in (D / 2^30, circuit bytes / GiB), by
least squares: RSS over every record, unweighted; the times over the 32-thread records, weighted by 1 / y² (relative
residuals). The fitted coefficients and every residual are in the output.

**A session** (:func:`session`) of B slots at bound Z, each slot one unit of at most Z nonzeros with ρ nonzeros per row:
rows = Z / ρ; ``k_log`` = max(22, ceil(log2(rows + 1)) + 2), the slot rounded up to a power of two and at most doubled again
by the SHA-512, hm96 and lookup ranges (every recorded block is at least 2^22); D = B 2^k_log; circuit bytes =
:data:`GLUE_BYTES` + B :data:`BYTES_PER_NNZ` Z, each slot its own netlist (a session's slots may hold different unit types).
ρ is the least nonzeros per committed row among the unit types in the bucket (the inventory's ``nnz / committed``).

**Per workload** (:func:`audit`): the padding table's body in its dp4 buckets at B = 16 and its tail at B = 1
(``unit_costs_padding``); each shape's sessions (mean and the public 1 − 2⁻⁴⁰ count), its session's RSS and prove time, and
the prover time per audit, Σ sessions x prove time, on one prover. **The cap** is the largest power of two Z at which a
B = 16 session at the workload's least ρ fits one node's memory (:data:`NODE_GIB`; :data:`GPU_GIB` for one GPU's, the CPU
law taken as a bound on the CUDA prover's). A unit over the cap is split below the subcircuit into ceil(z / cap) pieces,
its pieces in a bucket of their own at the cap (sessions ceil(p X / B) for X of its units drawn); every other unit keeps its
dp4 bucket.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from . import unit_costs_padding as P

#: vy-nebius-1 and vy-nebius-2: ``free -g`` total (GiB)
NODE_GIB = 1716
#: one NVIDIA RTX PRO 6000 Blackwell Server Edition: 97,887 MiB
GPU_GIB = 97_887 / 1024
SLOTS = 16
#: flock-ir-unit/v2 netlist text per nonzero: the largest of four registered units' (``ir_lower.netlist``: GEMM tc-step 4.93,
#: RoPE pair 4.73, SiLU element 4.33, RMSNorm-Triton N=128 6.04)
BYTES_PER_NNZ = 6.04
#: the circuit text that is not a unit's netlist (SHA-512 compressions, hm96 rows, META): the smallest recorded circuit's,
#: Pc4RowATail_v1{K=256,KS=64} of 5,280 ANDs
GLUE_BYTES = 55_911_585
K_MIN_LOG = 22
POW2 = tuple(range(16, 33))
CAP_SEARCH = tuple(range(16, 41))
#: ρ for the power-of-two table: RoPE's and RMSNorm's registered layouts (GEMM's is 27.9, attention's 27.9-29.6)
RHO_TABLE = 26.5
MENU_K = 4

#: (run, definition, instances, k_log, dense_m, circuit bytes, prove s (median), prover process wall s, prover peak RSS GiB,
#: threads): ``benchmarks/pouw/hidden_zk`` summaries on vy-nebius-1/2
SOURCE: tuple[tuple[str, str, int, int, int, int, float, float, float, int], ...] = (
    ('r20261003-060202-b967', 'Pc4Tile_v1{K=1024,TM=1,TN=1}', 16, 22, 26, 100136943, 1.0987, 6.53, 1.39, 32),
    ('r20261003-060202-b967', 'Pc4Tile_v1{K=256,TM=1,TN=1}', 16, 22, 25, 70617623, 0.938, 5.7, 1.04, 32),
    ('r20261003-060227-1efd', 'Pc4RowA_v1{K=1024}', 16, 25, 29, 5760857510, 7.5869, 116.56, 20.16, 32),
    ('r20261003-060227-1efd', 'Pc4RowA_v1{K=256}', 16, 23, 27, 1384534519, 1.4038, 26.79, 5.52, 32),
    ('r20261003-060232-34df', 'Pc4RowB_v1{K=1024}', 16, 25, 29, 6581669534, 8.0521, 134.38, 21.78, 32),
    ('r20261003-060232-34df', 'Pc4RowB_v1{K=256}', 16, 24, 28, 2115332939, 6.0679, 58.93, 8.96, 32),
    ('r20261003-064840-b83a', 'Pc4RowA_v1{K=4096}', 16, 27, 31, 26151124996, 32.3803, 574.82, 87.05, 32),
    ('r20261003-064856-dc42', 'Pc4RowB_v1{K=4096}', 16, 27, 31, 26971971824, 31.7834, 600.02, 84.18, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=1024,TM=1,TN=1}', 16, 22, 26, 108727949, 1.3135, 7.81, 1.42, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=256,TM=1,TN=1}', 16, 22, 25, 78692059, 1.1261, 6.95, 1.07, 32),
    ('r20261003-064901-8d6e', 'Pc4TileHidden_v1{K=4096,TM=1,TN=1}', 16, 23, 27, 237376407, 2.1184, 13.49, 2.93, 32),
    ('r20261003-064901-8d6e', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 16, 23, 27, 228784891, 2.3568, 14.43, 2.9, 32),
    ('r20261003-071851-a2f3', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 256, 23, 31, 228784891, 11.2523, 42.06, 11.07, 32),
    ('r20261003-073948-0e7d', 'Pc4Tile_v1{K=4096,TM=1,TN=1}', 4096, 23, 35, 228784891, 283.247, 949.49, 129.92, 32),
    ('r20261003-102619-cab2', 'Pc4Tile_v1{K=4096,TM=4,TN=4}', 256, 25, 33, 3192819893, 30.0277, 150.22, 46.11, 32),
    ('r20261003-105542-b162', 'Pc4Tile_v1{K=4096,TM=8,TN=8}', 64, 27, 33, 13762384409, 44.209, 372.6, 78.18, 32),
    ('r20261003-215938-bdf1', "Pc8TileCap_v1{K=8192,TM=1,TN=1,N=1,RHO=1000,DEV='sm120'}", 8, 25, 28, 3905011930, 5.2177, 79.61, 15.38, 32),
    ('r20261003-215938-bdf1', "Pc8CapWord_v1{K=8192,DEV='sm120'}", 8, 25, 28, 3682754201, 5.3155, 81.85, 15.03, 32),
    ('r20261003-215938-bdf1', "Pc8Tile_v1{K=8192,TM=1,TN=1,DEV='sm120'}", 8, 25, 28, 934944359, 11.0517, 76.97, 8.75, 32),
    ('r20261003-215938-bdf1', "Pc8Tile_v1{K=8192,TM=2,TN=2,DEV='sm120'}", 8, 26, 29, 3718279248, 10.2693, 104.97, 21.62, 32),
    ('r20261004-003758-fc56', 'Pc4RowA_v1{K=4096}', 16, 27, 31, 23357804541, 26.9612, 459.41, 80.74, 32),
    ('r20261004-012846-3deb', 'Pc4RowB_v1{K=4096}', 16, 27, 31, 24178651369, 28.6005, 585.85, 79.28, 32),
    ('r20261004-032911-9751', 'Pc8CapRowDigest_v1{TN=64}', 8, 26, 29, 73409170, 14.7667, 73.66, 13.02, 32),
    ('r20261004-032911-9751', 'Pc8CapRowTally_v1{K=8192,TN=64}', 8, 26, 29, 174229833, 13.9842, 71.5, 13.27, 32),
    ('r20261004-032911-9751', "Pc8CapStrip_v1{K=8192,TM=64,TN=64,N=16384,RHO=1000,DEV='sm120'}", 8, 26, 29, 65255064, 16.5827, 81.29, 13.0, 32),
    ('r20261004-032911-9751', "Pc8CapTally_v1{K=8192,TM=64,TN=64,N=16384,RHO=1000,DEV='sm120'}", 8, 26, 29, 72451183, 12.9208, 64.39, 13.04, 32),
    ('r20261004-032911-9751', 'Pc8CapTile_v1{TM=64,TN=64}', 8, 26, 29, 186715558, 14.6531, 77.54, 13.35, 32),
    ('r20261004-032911-9751', "Pc8CapWordUnit_v1{K=8192,DEV='sm120'}", 8, 25, 28, 3615080965, 5.7912, 82.72, 14.68, 32),
    ('r20261004-065458-8301', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 0.8181, 11.48, 1.66, 8),
    ('r20261004-065458-8301', 'Pc4RowAForm_v1{K=256,KS=64,J=1}', 2, 22, 25, 311011608, 4.45, 23.4, 1.68, 8),
    ('r20261004-065458-8301', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.8439, 5.96, 1.07, 8),
    ('r20261004-065458-8301', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 1.0986, 7.87, 0.98, 8),
    ('r20261004-065940-d834', 'Pc4RowAForm_v1{K=256,KS=64,J=2}', 2, 22, 25, 311011608, 0.8469, 11.87, 1.65, 8),
    ('r20261004-065940-d834', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.8698, 14.55, 1.06, 8),
    ('r20261004-065940-d834', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 0.9024, 13.3, 0.98, 8),
    ('r20261004-142622-188e', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 40.653, 342.19, 38.04, 16),
    ('r20261004-142637-5819', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 30.7314, 336.08, 37.77, 16),
    ('r20261004-161659-ccf7', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 2.3429, 17.95, 1.65, 8),
    ('r20261004-161659-ccf7', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290781, 1.1397, 7.46, 1.07, 8),
    ('r20261004-163914-c6e3', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739417, 10.8955, 585.18, 65.85, 16),
    ('r20261004-212718-0d67', 'Pc4RowAFirst_v1{K=256,KS=64}', 2, 22, 25, 306126257, 0.9273, 10.19, 1.66, 8),
    ('r20261004-212718-0d67', 'Pc4RowAForm_v1{K=256,KS=64,J=1}', 2, 22, 25, 311011608, 0.8636, 9.04, 1.65, 8),
    ('r20261004-212718-0d67', 'Pc4RowAHead_v1{K=256}', 2, 22, 25, 74290889, 0.7688, 5.36, 1.07, 8),
    ('r20261004-212718-0d67', 'Pc4RowATail_v1{K=256,KS=64}', 2, 22, 25, 55911585, 0.8755, 5.73, 1.0, 8),
    ('r20261004-213401-c9f1', 'Pc4RowAForm_v1{K=14336,KS=2048,J=3}', 16, 26, 30, 10328215509, 13.7513, 207.61, 38.08, 18),
    ('r20261004-213401-c9f1', 'Pc4RowAForm_v1{K=14336,KS=2048,J=4}', 16, 26, 30, 10328215509, 13.6487, 227.09, 38.03, 18),
    ('r20261004-213401-c9f1', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739557, 8.9423, 438.11, 65.85, 18),
    ('r20261004-213401-c9f1', 'Pc4RowATail_v1{K=14336,KS=2048}', 16, 25, 29, 58626310, 7.3724, 36.3, 8.62, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAFirst_v1{K=14336,KS=2048}', 16, 26, 30, 10321753902, 14.4287, 220.46, 36.74, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAForm_v1{K=14336,KS=2048,J=1}', 16, 26, 30, 10328215509, 14.9206, 227.28, 38.02, 18),
    ('r20261004-213406-1ca3', 'Pc4RowAForm_v1{K=14336,KS=2048,J=2}', 16, 26, 30, 10328215509, 13.5812, 209.28, 38.13, 18),
    ('r20261004-213411-ee31', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 14.1627, 222.12, 38.11, 16),
    ('r20261004-213416-7b3a', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 13.642, 222.5, 38.22, 16),
    ('r20261004-230445-e566', 'Pc4RowAForm_v1{K=14336,KS=2048,J=3}', 16, 26, 30, 10328215509, 17.5762, 225.93, 38.04, 18),
    ('r20261004-230445-e566', 'Pc4RowAForm_v1{K=14336,KS=2048,J=4}', 16, 26, 30, 10328215509, 12.3578, 194.38, 38.03, 18),
    ('r20261004-230445-e566', 'Pc4RowAHead_v1{K=14336}', 16, 25, 29, 22456739557, 9.1976, 547.96, 65.85, 18),
    ('r20261004-230445-e566', 'Pc4RowATail_v1{K=14336,KS=2048}', 16, 25, 29, 58626310, 7.7063, 38.24, 8.62, 18),
    ('r20261004-230450-e202', 'Pc4RowAFirst_v1{K=14336,KS=2048}', 16, 26, 30, 10321753902, 15.693, 218.52, 36.6, 18),
    ('r20261004-230450-e202', 'Pc4RowAForm_v1{K=14336,KS=2048,J=1}', 16, 26, 30, 10328215509, 17.6504, 300.32, 38.05, 18),
    ('r20261004-230450-e202', 'Pc4RowAForm_v1{K=14336,KS=2048,J=2}', 16, 26, 30, 10328215509, 14.375, 221.08, 38.09, 18),
    ('r20261004-230455-eac8', 'Pc4RowAForm_v1{K=14336,KS=2048,J=5}', 16, 26, 30, 10328215509, 14.938, 222.25, 38.13, 16),
    ('r20261004-230501-ceb4', 'Pc4RowAForm_v1{K=14336,KS=2048,J=6}', 16, 26, 30, 10328215509, 15.1235, 221.66, 37.97, 16),
)


# --- the fit -----------------------------------------------------------------------------------------------------------------

def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def lstsq(xs: Sequence[Sequence[float]], ys: Sequence[float], ws: Sequence[float] | None = None) -> list[float]:
    """Weighted least squares for y = c0 + Σ c_j x_j."""
    ws = ws or [1.0] * len(ys)
    X = [[1.0, *x] for x in xs]
    m = len(X[0])
    A = [[sum(w * r[i] * r[j] for r, w in zip(X, ws)) for j in range(m)] for i in range(m)]
    b = [sum(w * r[i] * y for r, y, w in zip(X, ys, ws)) for i in range(m)]
    return _solve(A, b)


def features(D: float, cb: float) -> tuple[float, float]:
    return D / 2 ** 30, cb / 2 ** 30


def _at(c: Sequence[float], x: Sequence[float]) -> float:
    return c[0] + sum(a * v for a, v in zip(c[1:], x))


def fit(source: Sequence[tuple] = SOURCE) -> dict[str, Any]:
    """The three laws over (D / 2^30, circuit bytes / GiB) and each record's residuals."""
    xs = [features(2.0 ** r[4], r[5]) for r in source]
    t32 = [i for i, r in enumerate(source) if r[9] == 32]
    coef = {"rss_gib": lstsq(xs, [r[8] for r in source]),
            "prove_s": lstsq([xs[i] for i in t32], [source[i][6] for i in t32], [source[i][6] ** -2 for i in t32]),
            "wall_s": lstsq([xs[i] for i in t32], [source[i][7] for i in t32], [source[i][7] ** -2 for i in t32])}
    points = []
    for r, x in zip(source, xs):
        p = {"run": r[0], "definition": r[1], "instances": r[2], "k_log": r[3], "dense_m": r[4], "circuit_bytes": r[5],
             "prove_s": r[6], "wall_s": r[7], "rss_gib": r[8], "threads": r[9]}
        for k, c in coef.items():
            p[f"{k}_fit"] = _at(c, x)
        points.append(p)
    resid = {"rss_gib": max(abs(p["rss_gib"] - p["rss_gib_fit"]) for p in points),
             **{k: max(abs(p[k] / p[f"{k}_fit"] - 1) for p in points if p["threads"] == 32) for k in ("prove_s", "wall_s")}}
    return {"coef": coef, "points": points, "max_residual": resid,
            "law": "y = c0 + c1 D/2^30 + c2 (circuit bytes)/GiB; RSS GiB unweighted over all records, prove and process wall "
                   "seconds weighted by 1/y^2 over the 32-thread records"}


# --- a session ---------------------------------------------------------------------------------------------------------------

def session(Z: float, rho: float, coef: Mapping[str, Sequence[float]], slots: int = SLOTS, bytes_per_nnz: float = BYTES_PER_NNZ,
            glue: float = GLUE_BYTES) -> dict[str, float]:
    """One session of ``slots`` slots at bound Z (the module docstring): its rows per slot, k_log, D, circuit bytes, and the
    laws' RSS (GiB), prove and process wall seconds."""
    rows = Z / rho
    k_log = max(K_MIN_LOG, math.ceil(math.log2(rows + 1)) + 2)
    D = slots * 2.0 ** k_log
    cb = glue + slots * bytes_per_nnz * Z
    x = features(D, cb)
    return {"Z": Z, "slots": slots, "rho": rho, "rows": rows, "k_log": k_log, "D": D, "circuit_bytes": cb,
            **{k: _at(c, x) for k, c in coef.items()}}


def cap(rho: float, coef: Mapping[str, Sequence[float]], gib: float, slots: int = SLOTS, **kw: Any) -> float | None:
    """The largest power of two Z in 2^16..2^40 whose session fits ``gib`` (None when none does)."""
    fits = [2.0 ** j for j in CAP_SEARCH if session(2.0 ** j, rho, coef, slots, **kw)["rss_gib"] <= gib]
    return max(fits) if fits else None


# --- a workload's audit ------------------------------------------------------------------------------------------------------

def rhos(inv: Mapping[str, Any], workload: str) -> dict[str, float]:
    """Each unit type's nonzeros per committed row (its subcircuit's ``nnz / committed``)."""
    out = {}
    for r in inv["rows"]:
        c = r["subcircuit"].get("per_unit") or {}
        if r["workload"] == workload and c.get("committed"):
            out[r["unit_type"]] = c["nnz"] / c["committed"]
    return out


def shape(N: int, K: int, us: Sequence[P.Unit], Z: float, slots: int, pieces: int, rho: Mapping[str, float],
          coef: Mapping[str, Sequence[float]], **kw: Any) -> dict[str, Any]:
    """A bucket of ``us`` at bound Z, each unit ``pieces`` slots: its sessions (mean, public) and their cost."""
    n = sum(u.n for u in us)
    k0, pmf = P.hyper(N, n, K)
    mean = sum(p * -(-pieces * (k0 + j) // slots) for j, p in enumerate(pmf))
    public = -(-pieces * P.upper_q(N, n, K) // slots)
    s = session(Z, min(rho.get(u.label, RHO_TABLE) for u in us), coef, slots, **kw)
    return {"Z": Z, "slots": slots, "pieces": pieces, "types": len(us), "units": n, "labels": [u.label for u in us][:8],
            "sessions_mean": mean, "sessions_public": public, "session": s,
            "prover_s_mean": mean * s["prove_s"], "prover_s_public": public * s["prove_s"]}


def pieces_pmf(N: int, K: int, groups: Mapping[int, int]) -> list[float]:
    """The pmf of Y = Σ_p p X_p, X_p ~ Hypergeometric(N, n_p, K) for ``groups`` {p: n_p}, convolved as independent (the
    draw without replacement correlates them negatively, so this overstates Y's upper tail)."""
    y = [1.0]
    for p, n in sorted(groups.items()):
        k0, pmf = P.hyper(N, n, K)
        nxt = [0.0] * (len(y) + p * (k0 + len(pmf) - 1))
        for a, pa in enumerate(y):
            if pa:
                for j, pj in enumerate(pmf):
                    nxt[a + p * (k0 + j)] += pa * pj
        y = nxt
    return y


def pieces_shape(N: int, K: int, us: Sequence[P.Unit], limit: float, slots: int, rho: Mapping[str, float],
                 coef: Mapping[str, Sequence[float]], delta: float = P.DELTA, **kw: Any) -> dict[str, Any]:
    """One bucket at ``limit`` holding the pieces of ``us`` (each unit ceil(z / limit) of them): sessions ceil(Y / slots)."""
    groups: dict[int, int] = defaultdict(int)
    for u in us:
        groups[math.ceil(u.z / limit)] += u.n
    pmf = pieces_pmf(N, K, groups)
    mean = sum(p * -(-y // slots) for y, p in enumerate(pmf))
    tail, q = 0.0, len(pmf) - 1
    for y in range(len(pmf) - 1, -1, -1):
        if tail + pmf[y] > delta:
            q = y
            break
        tail += pmf[y]
    public = -(-q // slots)
    s = session(limit, min(rho.get(u.label, RHO_TABLE) for u in us), coef, slots, **kw)
    return {"Z": limit, "slots": slots, "pieces": dict(sorted(groups.items())), "types": len(us), "units": sum(groups.values()),
            "labels": [u.label for u in us][:8], "sessions_mean": mean, "sessions_public": public, "session": s,
            "prover_s_mean": mean * s["prove_s"], "prover_s_public": public * s["prove_s"]}


def _buckets(N: int, K: int, groups: Mapping[float, Sequence[P.Unit]], slots: int, limit: float | None,
             rho: Mapping[str, float], coef: Mapping[str, Sequence[float]], **kw: Any) -> list[dict[str, Any]]:
    """Each bucket at its Z; with ``limit``, a bucket's units over it go to one bucket of their pieces at the limit."""
    out = []
    for Z, us in groups.items():
        keep = [u for u in us if limit is None or u.z <= limit]
        if keep:
            out.append(shape(N, K, keep, max(u.z for u in keep) if len(keep) < len(us) else Z, slots, 1, rho, coef, **kw))
        if len(keep) < len(us):
            out.append(pieces_shape(N, K, [u for u in us if u not in keep], limit, slots, rho, coef, **kw))
    return out


def audit(N: int, K: int, units: Sequence[P.Unit], rho: Mapping[str, float], coef: Mapping[str, Sequence[float]],
          limit: float | None = None, slots: int = SLOTS, k: int = MENU_K, **kw: Any) -> dict[str, Any]:
    """The padding table's body in its dp``k`` buckets at ``slots`` and its tail at one slot (each sampler its own bucket,
    the other large units one bucket at their largest Z), units over ``limit`` split (the module docstring)."""
    body, tail, _thr = P.tail_split(units)
    small = [u for u in body if limit is None or u.z <= limit]
    shapes = _buckets(N, K, P.assign(small, P.menu_dp(small, k)) if small else {}, slots, None, rho, coef, **kw)
    shapes += _buckets(N, K, {limit: [u for u in body if u not in small]} if len(small) < len(body) else {}, slots, limit,
                       rho, coef, **kw)
    samplers = [u for u in tail if u.template in P.SAMPLERS]
    big = [u for u in tail if u.template not in P.SAMPLERS]
    tails = _buckets(N, K, P.assign(samplers, P.menu_natural(samplers)) if samplers else {}, 1, limit, rho, coef, **kw)
    tails += _buckets(N, K, P.assign(big, P.menu_max(big)) if big else {}, 1, limit, rho, coef, **kw)
    every = shapes + tails
    tot = lambda key: sum(s[key] for s in every)  # noqa: E731
    return {"limit": limit, "body": shapes, "tail": tails, "sessions_mean": tot("sessions_mean"),
            "sessions_public": tot("sessions_public"), "prover_h_mean": tot("prover_s_mean") / 3600,
            "prover_h_public": tot("prover_s_public") / 3600,
            "max_rss_gib": max((s["session"]["rss_gib"] for s in every), default=0.0),
            "max_prove_s": max((s["session"]["prove_s"] for s in every), default=0.0)}


def workload(inv: Mapping[str, Any], name: str, coef: Mapping[str, Sequence[float]], **kw: Any) -> dict[str, Any]:
    K = int(inv["K"])
    N, units, skipped = P.units_of(inv, name)
    rho = rhos(inv, name)
    body, _tail, _thr = P.tail_split(units)
    r0 = min((rho.get(u.label, RHO_TABLE) for u in body), default=RHO_TABLE)
    unsplit = audit(N, K, units, rho, coef, **kw)
    top = max(s["Z"] for s in unsplit["body"])
    out: dict[str, Any] = {"workload": name, "N": N, "K": K, "skipped": skipped, "rho_min": r0, "dp4_top": top,
                           "dp4": [s["Z"] for s in unsplit["body"]], "top_session": session(top, r0, coef, **kw),
                           "unsplit": unsplit}
    for node, gib in (("cpu", NODE_GIB), ("gpu", GPU_GIB)):
        c = cap(r0, coef, gib, **kw)
        over = [u for u in units if c is not None and u.z > c]
        out[node] = {"gib": gib, "cap": c, "fits_unsplit": unsplit["max_rss_gib"] <= gib,
                     "split": [{"unit_type": u.label, "nnz": u.z, "draw_share": u.n / N, "pieces": math.ceil(u.z / c)} for u in over],
                     "at_cap": audit(N, K, units, rho, coef, limit=c, **kw) if over else unsplit}
    return out


def prover(inv: Mapping[str, Any], law: Mapping[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    law = law or fit()
    coef = law["coef"]
    return {"format": "verity/f6-prover/v1", "K": inv["K"], "slots": SLOTS, "menu": f"dp{MENU_K}", "node_gib": NODE_GIB,
            "gpu_gib": GPU_GIB, "bytes_per_nnz": BYTES_PER_NNZ, "glue_bytes": GLUE_BYTES, "fit": law,
            "pow2": [session(2.0 ** j, RHO_TABLE, coef, **kw) for j in POW2],
            "workloads": [workload(inv, w, coef, **kw) for w in dict.fromkeys(r["workload"] for r in inv["rows"])]}


# --- rendering ---------------------------------------------------------------------------------------------------------------

def _p2(z: float | None) -> str:
    if z is None:
        return "–"
    j = math.log2(z)
    return f"2^{int(j)}" if j == int(j) else P._si(z)


def _t(s: float) -> str:
    return f"{s:.3g} s" if s < 600 else f"{s / 60:.3g} min" if s < 36_000 else f"{s / 3600:.3g} h"


def render(pr: Mapping[str, Any]) -> str:
    f = pr["fit"]
    c = f["coef"]
    law = lambda k: f"{c[k][0]:.4g} + {c[k][1]:.4g}·D/2^30 + {c[k][2]:.4g}·cb/GiB"  # noqa: E731
    L = [f"# F6 prover column: sessions of {pr['slots']} slots, {pr['menu']} buckets, K = {pr['K']}", "",
         "C-Flock's CPU ZK prover (`flock-circuit prove --zk`, 32 threads, Xeon 6776P), fitted on the source runs below. D: "
         "the session's bits (slots x 2^k_log); cb: its circuit text (each slot's netlist at "
         f"{pr['bytes_per_nnz']} B per nonzero, plus {pr['glue_bytes'] / 1e6:.0f} MB of glue).", "",
         f"- peak RSS (GiB) = {law('rss_gib')}; max |residual| {f['max_residual']['rss_gib']:.2f} GiB",
         f"- prove (s) = {law('prove_s')}; max relative residual {f['max_residual']['prove_s']:.2f}",
         f"- prover process wall, circuit load included (s) = {law('wall_s')}; max relative residual "
         f"{f['max_residual']['wall_s']:.2f}",
         f"- fits: one node {pr['node_gib']} GiB (vy-nebius-1/2), one GPU {pr['gpu_gib']:.1f} GiB (RTX PRO 6000; the CPU law as "
         "a bound)", "",
         "## Per workload: dp4 at B = 16, unsplit and at the cap", "",
         "cap: the largest power-of-two bound whose 16-slot session fits; units over it split into ceil(z / cap) pieces. "
         "Prover hours: Σ sessions x prove time on one prover, body and tail, mean / public session counts.", "",
         "| workload | dp4 top Z | session prove | session RSS | fits node / GPU | node cap | sessions unsplit (mean / public) | "
         "prover h unsplit | GPU cap | units split at GPU cap (draw share) | sessions at GPU cap | prover h at GPU cap |",
         "|" + "---|" * 12]
    for w in pr["workloads"]:
        u, g, s = w["unsplit"], w["gpu"], w["top_session"]
        a = g["at_cap"]
        L.append("| " + " | ".join([
            P._short(w["workload"]), P._si(w["dp4_top"]), _t(s["prove_s"]), f"{s['rss_gib']:.0f} GiB",
            f"{'yes' if w['cpu']['fits_unsplit'] else 'no'} / {'yes' if g['fits_unsplit'] else 'no'}", _p2(w["cpu"]["cap"]),
            f"{u['sessions_mean']:.0f} / {u['sessions_public']}", f"{u['prover_h_mean']:.3g} / {u['prover_h_public']:.3g}",
            _p2(g["cap"]), f"{len(g['split'])} ({sum(x['draw_share'] for x in g['split']):.2%})",
            f"{a['sessions_mean']:.0f} / {a['sessions_public']}", f"{a['prover_h_mean']:.3g} / {a['prover_h_public']:.3g}"]) + " |")
    L += ["", f"## Power-of-two bounds (ρ = {RHO_TABLE} nonzeros per row)", "",
          "| Z | rows per slot | k_log | D | circuit text | peak RSS | prove | process wall | fits node | fits GPU |",
          "|" + "---|" * 10]
    for s in pr["pow2"]:
        L.append(f"| {_p2(s['Z'])} | {P._si(s['rows'])} | {s['k_log']} | 2^{int(math.log2(s['D']))} | {s['circuit_bytes'] / 2 ** 30:.3g} GiB | "
                 f"{s['rss_gib']:.3g} GiB | {_t(s['prove_s'])} | {_t(s['wall_s'])} | {'yes' if s['rss_gib'] <= pr['node_gib'] else 'no'} | "
                 f"{'yes' if s['rss_gib'] <= pr['gpu_gib'] else 'no'} |")
    L += ["", "## Shapes", "", "| workload | split at | part | Z | slots | pieces | types | sessions mean / public | session RSS | "
          "session prove | prover h mean |", "|" + "---|" * 11]
    for w in pr["workloads"]:
        for tag, a in (("unsplit", w["unsplit"]), ("GPU cap", w["gpu"]["at_cap"] if w["gpu"]["split"] else None)):
            if a is None:
                continue
            for part in ("body", "tail"):
                for s in a[part]:
                    pc = s["pieces"] if isinstance(s["pieces"], int) else f"{min(s['pieces'])}-{max(s['pieces'])}"
                    L.append(f"| {P._short(w['workload'])} | {tag} | {part} | {P._si(s['Z'])} | {s['slots']} | {pc} | "
                             f"{s['types']} | {s['sessions_mean']:.1f} / {s['sessions_public']} | {s['session']['rss_gib']:.3g} GiB | "
                             f"{_t(s['session']['prove_s'])} | {s['prover_s_mean'] / 3600:.3g} |")
    L += ["", "## Source runs", "", "| run | definition | instances | k_log | D | circuit text | prove | fit | process wall | fit | "
          "peak RSS | fit | threads |", "|" + "---|" * 13]
    for p in f["points"]:
        L.append(f"| {p['run']} | {p['definition']} | {p['instances']} | {p['k_log']} | 2^{p['dense_m']} | "
                 f"{p['circuit_bytes'] / 1e6:.0f} MB | {p['prove_s']:.3g} | {p['prove_s_fit']:.3g} | {p['wall_s']:.3g} | "
                 f"{p['wall_s_fit']:.3g} | {p['rss_gib']:.3g} | {p['rss_gib_fit']:.3g} | {p['threads']} |")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", required=True, help="unit_costs' unit-costs.json")
    ap.add_argument("--out", required=True, help="directory for prover.json and prover.md")
    a = ap.parse_args(argv)
    from pathlib import Path

    from .unit_costs import retree

    inv = json.loads(Path(a.inventory).read_text())
    if "token_select" not in inv:
        retree(inv)
    pr = prover(inv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prover.json").write_text(json.dumps(pr, indent=1) + "\n")
    (out / "prover.md").write_text(render(pr))
    print(f"wrote {out / 'prover.json'} and {out / 'prover.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
