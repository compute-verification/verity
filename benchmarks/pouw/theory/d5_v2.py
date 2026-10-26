"""D5 v2 (CMP-125): the public best on protocol v3's operands at the period's dual prices (P12), in model units.

Protocol v3 (PROTO-DESIGN-3, CMP-118 e0cd4061; `accel_score/nofold.py` and `v3_caps.py` at bbe690dfe5): v2's words, 16-bit
prefix sums every L = t L' inner indices as SHA-512 hiding leaves (8 blocks, 16-ary tree, O the last 32-bit word); HYB
noise U, V ±80 with U's columns permuted afresh in each tile; the peels Z_X U^T and V Y_W^T in nofold's base-128
digits; b streamed by the network certifier, the other coins seed-expanded on the node (P2); t = 2 tiles on the H200
and 3 on the RTX PRO 6000, L' capped per static shape and per PV length n. Credited: the static matmuls and PV with O
bound (R-ACCEL-8).

A round's job is `public_best.tile_job` at A = B = 0, the cheapest registrable instance: the honest kernel (dense s8,
the peels, the hashing at nofold's compressions a MAC) against the hull of the prover's methods, every way to the words
(the plain product with DP4A, FFMA, packed DFMA and O-IO's lookups beside the MMA, Strassen's orbit at depth 1 and 2,
Hopcroft-Kerr and the integral catalogue, the tile route) with every way to O (the peels, the zero-high-digit peel,
PD2-1's O-route). SHA-512 is priced in model units (U13): every schedule the model admits, so c_sha = 1 in-model.

Prices mu are a period's LP duals (`price_methods`): SM-clocks a unit of each resource and facet, summing to 1 over the
capacities. At mu, pub is the hull's cheapest program, H the honest kernel family's (no route beside the MMA, the
peels), OPT_honest the honest words' best (every route beside the MMA, the peels): c = pub / H, leak = 1 - pub /
OPT_honest (the routes), eps_alg = 1 - OPT_honest / H (the deployed kernel's gap), each also for the matmul alone
(pub_mm, H_mm: the job less its hashing, which costs pub_sha at mu on both sides).

- job-only: the round alone, the hull's own duals. Where they sit on the lanes alone (the H200), the matmul has no
  price on either side and c_mm is reported as unpriced.
- served period (the headline, 06:37Z): the round beside tensor-bound work the period always holds (QK^T, decode's
  uncredited share, other jobs), X MAC-times of dense s8 MMA a credited MAC. The duals are piecewise constant in X;
  the headline is the first segment where the tensor pipe has a price (X just past X0; below X0 the period's prices
  are the job's). Further segments are the sensitivity rows (X0 + 0.01 ... 100 round-times): as X grows the lanes
  get cheap, so DP4A or Strassen's assembly on them take the round's MACs off the tensor pipe. In the limit only the
  tensor pipe (and the port) has a price and the dense product runs on idle lanes, which the honest words can do too
  (eps_alg); nofold's "any prices" saving (saved/T) counts only the routes' tensor time, a smaller saving.
  `world_x` gives a world model's own QK^T (and V4's indexer) a credited MAC, the X its rounds bring themselves.
- c_inc beside each served c: the round's makespan increment in the period, hull against the honest kernel family,
  with the capacity rows kept (`increment`). Prices at a given mu have none, so where the duals leave a resource free
  a route may run there past the period's slack; c_inc can't, and it doesn't depend on whose period's duals are used.
  At X = 0 it is the job's makespan ratio; it falls smoothly in X where c steps.
- pub(L') for PROTO-DESIGN-4 (`sweep`, 07:04Z): every static shape and a PV grid over L', each also at a fixed X of
  1 MAC-time a credited MAC, so cells at different L' face the same period.

    ACCEL_SCORE_DIR=<dir holding nofold.py at bbe690dfe5> python benchmarks/pouw/theory/d5_v2.py [--out DIR] [--quick]

`git show bbe690dfe5:benchmarks/pouw/accel_score/nofold.py` (cursor/compute-research-accel-proto-design-ea24) gives it.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_best as pb  # noqa: E402

CARDS = {"H200": pb.H200, "RTX PRO 6000": pb.SM120}
T_V3 = {"H200": 2, "RTX PRO 6000": 3}
SHAPES = {   # nofold.SHAPES: (K, N, layers)
    "frozen": ((16384, 16512, 1), (16384, 16384, 1), (16384, 65536, 1), (65536, 16384, 1)),
    "Llama-3.1-8B": ((4096, 6144, 32), (4096, 4096, 32), (4096, 28672, 32), (14336, 4096, 32), (4096, 128256, 1)),
    "Llama-3.1-70B": ((8192, 10240, 80), (8192, 8192, 80), (8192, 57344, 80), (28672, 8192, 80), (8192, 128256, 1)),
    "DeepSeek-V4-Pro": ((7168, 43008, 61), (3072, 50176, 61), (7168, 1536, 61), (1536, 65536, 61), (7168, 576, 61),
                        (4096, 16384, 61), (16384, 7168, 61), (7168, 129280, 1)),
}
PV_DV = {"Llama-3.1-8B": 128, "Llama-3.1-70B": 128, "DeepSeek-V4-Pro": 512}
PV_GRID = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072)
LPS = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048)   # v3_caps.LPS
PD_RULE = {"H200": (None, 1e-4), "RTX PRO 6000": (256, 0.01)}    # v3_caps.CARDS: L' limit and leak tolerance
V4_SHORT = ((1536, 65536), (7168, 1536), (7168, 576))           # A25's L' 512 shapes on the H200
# CMP-118 (e0cd4061; v3_caps.static_caps at bbe690dfe5, art:1ce8c5c1328d2f6a…): L' per static shape (K, N).
V3_STATIC = {
    "H200": {(K, N): (640 if (K, N) in ((4096, 6144), (4096, 4096), (4096, 28672), (4096, 128256), (3072, 50176),
                                         (1536, 65536), (7168, 576), (4096, 16384)) else 512)
             for shapes in SHAPES.values() for K, N, _ in shapes},
    "RTX PRO 6000": {(K, N): (192 if (K, N) == (1536, 65536) else 256)
                     for shapes in SHAPES.values() for K, N, _ in shapes},
}
DESIGNS = ("CMP-118", "06:37Z", "L' 64")
SENS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
FIXED_X = (1.0,)
X0_STEPS = 24


def nofold():
    d = os.environ.get("ACCEL_SCORE_DIR", "")
    if not (Path(d) / "nofold.py").is_file():
        sys.exit("ACCEL_SCORE_DIR must hold nofold.py: mkdir -p /tmp/v3 && git show "
                 "bbe690dfe5:benchmarks/pouw/accel_score/nofold.py > /tmp/v3/nofold.py && ACCEL_SCORE_DIR=/tmp/v3 ...")
    sys.path.insert(0, d)
    import nofold as nf
    return nf


def _f(x):
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# --------------------------------------------------------------------------------------------------------------------
# A round's job and its prices.

@functools.cache
def fast_schemes(card: str) -> tuple:
    """Each fast method's best scheme on v3's operands (the 2x2 orbit at depth 1 and 2, every integral catalogue
    scheme, Hopcroft-Kerr's <2,k,n> among them, at depth 1), chosen alone at the frozen qkv shape's v3 L with every
    route beside the MMA; `tile_methods` adds each with every way to O."""
    c = CARDS[card]
    eX, eW, _ = pb._tile_operands(T_V3[card], pb.V2_V)
    L = T_V3[card] * V3_STATIC[card][16384, 16512]
    out = []
    for name, (schemes, d) in pb.methods(pb.orbit(), pb.catalogue()).items():
        if name != "naive":
            p, s = pb.best_scheme(c, d, eX, eW, L, 16512, pb.TOKENS, schemes, beside=True, lookups=True)
            out.append((name, (s, d)))
    return tuple(out)


def job(nf, card: str, K: int, N: int, t: int, Lp: int, pv: bool = False, schemes: bool = True,
        model_sha: bool = True, match: str = "comps", sha_scale: float = 1.0, **kw) -> tuple[dict, dict]:
    """nofold's round (16-bit words; PV with the weight side streamed) and D5's job at it. match: 'comps' prices
    nofold's compressions a MAC in D5's units (PROTO-DESIGN-3's v3_caps); 'lane' scales the hashing to nofold's
    honest ALU-over-tensor load instead (ALGO-COST-2's v3_caps, which carries nofold's η). sha_scale multiplies the
    compressions, to price a compression at another figure than D5's schedules at θ."""
    r = nf.v2_round(card, K, N, t * Lp, 16, tiles=t, weight_coins=not pv)
    r = dict(r, comps=r["comps"] * sha_scale)
    c = CARDS[card]
    _, bit, depth = pb.BIT_COST[c.family][0]
    sch = dict(fast_schemes(card)) if schemes else None
    args = dict(digits=r["digits"], model_sha=model_sha, schemes=sch, lookups=schemes, **kw)
    if match == "lane":
        j = pb.tile_job_at(c, t, Lp, K, N, bit, depth, nf.V_WIDTH[card], r["A_nw"] / r["T"], **args)
    else:
        j = pb.tile_job(c, t, Lp, K, N, bit, depth, nf.V_WIDTH[card], comps=r["comps"], **args)
    return r, j


def other_loads(card: pb.Card, X: float) -> dict:
    """X MAC-times of the honest dense s8 MMA (its tile's operand reads, issue and register words)."""
    mma = dict(tensor=1 / card.s8, smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf,
               issue=card.mma_issue)
    return {r: X * v for r, v in mma.items()}


def period(card: pb.Card, j: dict, N: int, X: float, **kw) -> pb.Priced:
    """The period's LP: the round's hull beside X MAC-times of other tensor work a credited MAC."""
    return pb.price_methods(card, j["methods"], j["L"], N, pb.TOKENS, beside=True, items=(j["sha"],),
                            lookups=bool(kw.get("lookups", True)), extra_fixed=other_loads(card, X))


def increment(card: pb.Card, j: dict, N: int, X: float, lookups: bool = True) -> dict:
    """The round's makespan increment in the period (capacity rows kept): T(round + X) - T(X) for the hull and for the
    honest kernel family, and their ratio c_inc. It is the round's cost at the duals averaged over its own addition, so
    it needs no choice of whose period's duals, and no route runs on a resource past the period's slack."""
    o = other_loads(card, X)
    t0 = card.s8 * max(sum(x * o.get(r, 0.0) for r, x in w.items()) / cap for _, w, cap in pb._rows(card))
    pub = period(card, j, N, X, lookups=lookups).c - t0
    hon = pb.price_methods(card, j["methods"][:1], j["L"], N, pb.TOKENS, beside=False, items=(j["sha"],),
                           extra_fixed=o).c - t0
    return dict(c_inc=pub / hon, pub_inc=pub, H_inc=hon)


def tensor_price(card: pb.Card, mu: dict) -> float:
    return mu.get("tensor", 0.0) + sum(mu.get(f"joint{i}", 0.0) * a for i, (a, _) in enumerate(card.facets))


def mm_ratio(pub_mm: float, H_mm: float, H: float) -> float | None:
    """c_mm = pub_mm / H_mm, or None where the matmul has no price (the round's prices sit on the lanes alone, so
    both programs' matmuls cost 0 and the round is all hashing: c_mm is unpriced, not 1 or 0)."""
    return pub_mm / H_mm if H_mm > 1e-9 * H else None


def at_prices(card: pb.Card, j: dict, N: int, mu: dict, lookups: bool = True) -> dict:
    """pub, OPT_honest and H at prices mu (MAC-times a credited MAC), their ratios, and the matmul's share."""
    L, ms, sha = j["L"], j["methods"], j["sha"]
    kw = dict(items=(sha,), mu=mu)
    pub = pb.price_methods(card, ms, L, N, pb.TOKENS, beside=True, lookups=lookups, **kw)
    opt = pb.price_methods(card, ms[:1], L, N, pb.TOKENS, beside=True, lookups=lookups, **kw)
    hon = pb.price_methods(card, ms[:1], L, N, pb.TOKENS, beside=False, **kw)
    ps = min(pb.mu_cost(card, ld, mu) for ld in sha[2].values())
    hsh = sha[1] * ps
    routes = {k: v for k, v in pub.routes.items() if v > 1e-6 and not k.startswith("sha")}
    pub_mm, H_mm = (0.0 if abs(x) < 1e-9 * hon.c else x for x in (pub.c - hsh, hon.c - hsh))
    return dict(c=pub.c / hon.c, leak=1 - pub.c / opt.c, eps_alg=1 - opt.c / hon.c,
                c_mm=mm_ratio(pub_mm, H_mm, hon.c), pub=pub.c, opt=opt.c, H=hon.c, pub_mm=pub_mm,
                H_mm=H_mm, pub_sha=hsh, pub_sha_clocks=ps / card.s8, comps=sha[1],
                mu={k: v * card.s8 for k, v in mu.items() if v > 1e-12}, mix=pub.mix, routes=routes)


def first_tensor_bound(card: pb.Card, j: dict, N: int, **kw) -> float:
    """X0: the least X at which the period's tensor pipe (with its facets) has a price."""
    if tensor_price(card, period(card, j, N, 0.0, **kw).mu) > 1e-12:
        return 0.0
    lo, hi = 0.0, j["route"].c
    while tensor_price(card, period(card, j, N, hi, **kw).mu) <= 1e-12:
        assert hi < 1e6 * j["route"].c, "the tensor pipe never binds"
        lo, hi = hi, 2 * hi
    for _ in range(X0_STEPS):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tensor_price(card, period(card, j, N, mid, **kw).mu) <= 1e-12 else (lo, mid)
    return hi


def analyse(nf, card: str, K: int, N: int, t: int, Lp: int, pv: bool = False, sens=SENS, fixed=FIXED_X,
            **kw) -> dict:
    """One round: nofold's figures, the job-only and served-period prices, the served sensitivity in X (round-times
    past X0), and the period at fixed X (MAC-times of other tensor work a credited MAC, the same at every L')."""
    c = CARDS[card]
    r, j = job(nf, card, K, N, t, Lp, pv, **kw)
    lk = kw.get("schemes", True)
    tc = j["route"].c
    out = dict(K=K, N=N, t=t, Lp=Lp, L=t * Lp, pv=pv, digits=r["digits"], gamma=r["gamma_max"],
               any=r["saved"] / r["T"], lane=j["lane"], minT=dict(c=j["c"], c_opt=j["c_opt"], eps_alg=j["eps_alg"]),
               hull=[(m.label, sorted(k for it in m.items for k in it[2])) for m in j["methods"]],
               job=at_prices(c, j, N, j["route"].mu, lk), job_binding=j["route"].binding)
    out["job"].update(c_inc=j["route"].c / j["honest"].c, pub_inc=j["route"].c, H_inc=j["honest"].c)
    x0 = first_tensor_bound(c, j, N, lookups=lk)
    p = period(c, j, N, x0 + 1e-4 * tc, lookups=lk)
    out["served"] = dict(at_prices(c, j, N, p.mu, lk), **increment(c, j, N, x0 + 1e-4 * tc, lk), X0=x0 / tc,
                         X0_macs=x0, binding=p.binding)
    rows = []
    for f in sens:
        p = period(c, j, N, x0 + f * tc, lookups=lk)
        a = at_prices(c, j, N, p.mu, lk)
        rows.append(dict(X=x0 / tc + f, c=a["c"], c_inc=increment(c, j, N, x0 + f * tc, lk)["c_inc"], c_mm=a["c_mm"],
                         leak=a["leak"], eps_alg=a["eps_alg"], binding=p.binding, mix=a["mix"], routes=a["routes"]))
    out["sens"] = rows
    out["worst"] = min(rows + [dict(out["served"], X=out["served"]["X0"])], key=lambda x: x["c"])
    out["fixed"] = {}
    for X in fixed:
        p = period(c, j, N, X, lookups=lk)
        a = at_prices(c, j, N, p.mu, lk)
        out["fixed"][X] = dict(c=a["c"], **increment(c, j, N, X, lk), binding=p.binding, mix=a["mix"])
    return out


# --------------------------------------------------------------------------------------------------------------------
# The caps: CMP-118's (followed), the coordinator's 06:37Z, and the RTX PRO 6000's L' 64 row.

def leak_of(nf, card: str, K: int, N: int, t: int, Lp: int, pv: bool, pd: bool = True) -> float:
    """The job-only leak 1 - c_opt without the fast schemes; pd: PROTO-DESIGN-3's LP (measured SHA-512 schedules),
    else model units."""
    _, j = job(nf, card, K, N, t, Lp, pv, schemes=False, model_sha=not pd)
    return 1 - j["c_opt"]


def pd_cap(nf, card: str, K: int, N: int, pv: bool) -> tuple[int, int]:
    """v3_caps.cap: the largest L' in LPS (t L' <= K, under the card's limit) whose leak is within the card's
    tolerance, else the least-leak L'; one tile below t·64 inner indices."""
    lmax, tol = PD_RULE[card]
    t = T_V3[card] if T_V3[card] * LPS[0] <= K else 1
    tried = []
    for Lp in sorted((x for x in LPS if t * x <= K and (lmax is None or x <= lmax)), reverse=True) or [K]:
        lk = leak_of(nf, card, K, N, t, Lp, pv)
        tried.append((lk, Lp))
        if lk < tol:
            return t, Lp
    return t, min(tried)[1]


def coord_cap(nf, card: str, K: int, N: int, pv: bool) -> tuple[int, int]:
    """The 06:37Z caps. Static: H200 t 2, L' 256 (512 at A25's V4 shapes); RTX PRO 6000 t 3, L' 256. PV: A25's rule,
    the least-γ (t, L') with zero job leak (under 5e-4, D5's model-unit LP), else the least leak; on the RTX PRO 6000
    the least-γ t <= 3, L' <= 256."""
    if not pv:
        return T_V3[card], (512 if card == "H200" and (K, N) in V4_SHORT else 256)
    if card == "H200":
        cands = [(2, lp) for lp in (64, 128, 256, 512, 1024) if 2 * lp <= K] or [(1, 64)]
        rows = []
        for t, lp in cands:
            r = nf.v2_round(card, K, N, t * lp, 16, tiles=t, weight_coins=False)
            rows.append((r["gamma_max"], leak_of(nf, card, K, N, t, lp, True, pd=False), t, lp))
        clean = [x for x in rows if x[1] < 5e-4]
        g = min(clean) if clean else min(rows, key=lambda x: x[1])
        return g[2], g[3]
    cands = [(t, lp) for t in (3, 2) for lp in (64, 128, 192, 256) if t * lp <= K] or [(1, 64)]
    return min(cands, key=lambda x: nf.v2_round(card, K, N, x[0] * x[1], 16, tiles=x[0],
                                                weight_coins=False)["gamma_max"])


def pick(nf, design: str, card: str, K: int, N: int, pv: bool) -> tuple[int, int] | None:
    if design == "CMP-118":
        return pd_cap(nf, card, K, N, pv) if pv else (T_V3[card], V3_STATIC[card][K, N])
    if design == "06:37Z":
        return coord_cap(nf, card, K, N, pv)
    if card != "RTX PRO 6000":
        return None
    t = 3 if 3 * 64 <= K else 1
    return t, 64


def pv_lengths(nf, quick: bool) -> dict:
    """The attended lengths n (rounded to 64) a world model's PV uses, with their PV MACs, and the grid to 131072."""
    out = {}
    for m in nf.WORLD:
        hist = nf.world_macs(m)[0]
        w = {}
        for n, cnt in hist.items():
            nb = nf.pv_bucket(n)
            w[nb] = w.get(nb, 0.0) + cnt * n
        if quick:
            w = {nb: x for nb, x in w.items() if nb in (128, 1024, 4096) or nb == max(w)}
        out[m["name"]] = w
    return out


def world_x(nf) -> dict:
    """A world model's uncredited tensor work a credited MAC (static and PV): QK^T, and V4's indexer, in MACs (at the
    s8 rate if run in FP8; BF16 takes twice)."""
    out = {}
    for m in nf.WORLD:
        _, static, qk, pv, ix = nf.world_macs(m)
        out[m["name"]] = dict(qk=qk / (static + pv), indexer=ix / (static + pv))
    return out


NF = None    # nofold, set by main for the pool's forked workers


def _pick(a):
    return a, pick(NF, *a)


def _analyse(a):
    card, K, N, t, Lp, pv = a
    return a, analyse(NF, card, K, N, t, Lp, pv, sens=SENS if not pv else SENS[::2])


def pool_map(fn, args: list, jobs: int) -> dict:
    if jobs <= 1:
        return dict(map(fn, args))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(jobs) as p:
        return dict(p.imap_unordered(fn, args, chunksize=1))


def _lite(a):
    return a, analyse(NF, *a, sens=())


def sweep(nf, quick: bool, jobs: int) -> dict:
    """pub(L') for PROTO-DESIGN-4 (07:04Z): every static shape, and PV on a grid of n, at the card's t (1 where t·64 >
    K), over every L' in LPS with t L' <= K."""
    global NF
    NF = nf
    args = []
    for card in CARDS:
        shapes = sorted({(K, N) for sh in SHAPES.values() for K, N, _ in (sh[:1] if quick else sh)})
        pvs = sorted({(n, dv) for n in (PV_GRID[::6] if quick else PV_GRID[::2]) for dv in set(PV_DV.values())})
        for K, N, pv in [(K, N, False) for K, N in shapes] + [(n, dv, True) for n, dv in pvs]:
            t = T_V3[card] if T_V3[card] * LPS[0] <= K else 1
            args += [(card, K, N, t, Lp, pv) for Lp in LPS if t * Lp <= K][:4 if quick else None]
    done = pool_map(_lite, args, jobs)
    out = {}
    for (card, K, N, t, Lp, pv), r in sorted(done.items()):
        out.setdefault(card, {}).setdefault(f"{'PV n ' if pv else ''}{K}x{N}", []).append(r)
    return out


def rounds(nf, quick: bool, jobs: int) -> tuple[dict, dict]:
    """Every design's static and PV rounds: the picks, then each distinct round analysed once (in a pool)."""
    global NF
    NF = nf
    lengths, grid = pv_lengths(nf, quick), (PV_GRID[::6] if quick else PV_GRID)
    want = []
    for design in DESIGNS:
        for card in CARDS:
            if pick(nf, design, card, 16384, 16512, False) is None:
                continue
            for model, shapes in SHAPES.items():
                want += [(design, card, model, K, N, False, K * N * reps) for K, N, reps in shapes[:1 if quick else None]]
                if model in lengths:
                    w = lengths[model]
                    want += [(design, card, model, n, PV_DV[model], True, w.get(n, 0.0))
                             for n in sorted(set(w) | set(grid))]
    fast_schemes.cache_clear()
    for card in CARDS:
        fast_schemes(card)
    picks = pool_map(_pick, sorted({(d, c, K, N, pv) for d, c, _, K, N, pv, _ in want}), jobs)
    done = pool_map(_analyse, sorted({(c, K, N, *picks[d, c, K, N, pv], pv) for d, c, _, K, N, pv, _ in want}), jobs)
    static, pv_ = {}, {}
    for d, c, model, K, N, pv, w in want:
        t, Lp = picks[d, c, K, N, pv]
        row = dict(done[c, K, N, t, Lp, pv], model=model, w=w)
        (pv_ if pv else static).setdefault(d, {}).setdefault(c, {}).setdefault(model, []).append(row)
    return static, pv_


def weighted(rows: list, col: str) -> dict:
    """MAC-weighted over the rows with weight: c = sum w pub / sum w H (and the matmul's), leak and eps_alg alike."""
    rows = [r for r in rows if r["w"] > 0]
    if not rows:
        return {}
    W = sum(r["w"] for r in rows)

    def tot(a):
        return sum(r["w"] * r[col][a] for r in rows)

    def ratio(a, b):
        return tot(a) / tot(b)

    return dict(c=ratio("pub", "H"), c_inc=ratio("pub_inc", "H_inc"),
                c_mm=mm_ratio(tot("pub_mm"), tot("H_mm"), tot("H")), leak=1 - ratio("pub", "opt"),
                eps_alg=1 - ratio("opt", "H"), gamma=sum(r["w"] * r["gamma"] for r in rows) / W,
                worst_c=min(r[col]["c"] for r in rows), any=sum(r["w"] * r["any"] for r in rows) / W)


# --------------------------------------------------------------------------------------------------------------------
# pub_sha in model units (U13), P-HYB3's rows (A21), advice (P14, L28), the named programs (L3, WORKSOUND-INST-2).

THETA_SHA = {"H200": 38.29, "RTX PRO 6000": 38.11}   # θ's sha_ceiling (CALIB #4513), IMAD.WIDE at two lanes (U13)
U13_ONE_LANE = {"H200": 35.4, "RTX PRO 6000": 35.6}  # the model's own figure with IMAD.WIDE one imad lane (RT-UNITS)


def sha_rows() -> dict:
    """SM-clocks a compression: the model's (θ's ceiling and D5's best schedule at θ), the measured builds, the paper
    schedules (the upper sensitivity)."""
    out = {}
    for name, c in CARDS.items():
        fam = c.family
        sch = {s: pb.sha_clocks(c, s) for s in pb.SHA_SCHEDULES[fam]}
        best = min(sch, key=lambda s: sch[s][0])
        meas = [(what, sms * mhz * 1e6 / rate * f, rate, mhz) for f_, what, rate, sms, mhz, f, _ in pb.SHA_MEASURED
                if f_ == fam]
        sms, mhz = pb.SMS_MHZ[fam]
        out[name] = dict(model=THETA_SHA[name], d5_best=(best, sch[best][0], sch[best][1]),
                         schedules={s: (v[0], v[1], sms * mhz * 1e6 / v[0]) for s, v in sch.items()},
                         measured=meas, pinned=meas[0])
    return out


def _uniform(lo: int, hi: int) -> np.ndarray:
    p = ((pb.INT8 >= lo) & (pb.INT8 <= hi)).astype(float)
    return p / p.sum()


def _plus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """pmf of the sum, wrapped to int8 (the rows here stay inside)."""
    out = np.zeros(256)
    for i, pa in enumerate(a):
        if pa:
            np.add.at(out, pb.wrap8(pb.INT8[i] + pb.INT8) + 128, pa * b)
    return out


A21 = {   # ALGO-COST-2 A21 (CMP-118 d0ac4cc8): its figure for each row, outside D5's int8 menu where it says so
    ("H200", "b c^T"): "0.835-0.938 (A16, b1 planes)",
    ("RTX PRO 6000", "b c^T"): "0.5 (one FP4 product: b - 3 is exact E2M1; not in D5's menu)",
    ("H200", "b U^T"): "1.0 (INT8; b1 planes would take 3.0)", ("RTX PRO 6000", "b U^T"): "1.0 (INT8; U isn't E2M1)",
    ("H200", "c_t, t 2"): "2.503 (nofold C_T 2.502)", ("H200", "c_t, t 3"): "2.553 (nofold C_T 2.548)",
    ("RTX PRO 6000", "c_t, t 3"): "not priced", ("RTX PRO 6000", "c_t, t 4"): "2.543 (nofold C_T 2.539)",
}


def phyb3_rows(quick: bool = False) -> dict:
    """A21's three rows for P-HYB3 (3 fresh bits an entry): b c^T at 3 x 3 bits, b U^T (and V c^T) at 3 bits x ±80,
    and the tile route's c_t at |b + V| <= 87 (Z U^T's price plus one int8 product), each standalone at θ (lanes
    idle) with the fast methods; the tile route's c_t at v2's 1-bit b beside it, against nofold's C_T."""
    b3, u80 = _uniform(0, 7), _uniform(-80, 80)
    zero = pb.data_pmf("zero")
    menu = {} if quick else pb.methods(pb.orbit(), pb.catalogue())
    out = {}
    for name, c in CARDS.items():
        for row, eX, eW in (("b c^T, 3 x 3 bits", b3, b3), ("b U^T, 3 bits x ±80", b3, u80)):
            L, n = 2 * 512, 16512
            chosen = {"naive": (None, 0)}
            for mn, (schemes, d) in menu.items():
                if mn != "naive":
                    chosen[mn] = (pb.best_scheme(c, d, eX, eW, L, n, pb.TOKENS, schemes, beside=True)[1], d)
            h = pb.price_hull(c, chosen, eX, eW, L, n, pb.TOKENS, beside=True)
            tens = pb.price_hull(c, chosen, eX, eW, L, n, pb.TOKENS, beside=False)
            out[name, row] = dict(c=h.c, c_tensor=tens.c, mix=h.mix, binding=h.binding, routes=h.routes,
                                  a21=A21[name, row[:5]])
        for t in ((2, 3) if name == "H200" else (3, 4)):
            for label, bpmf in (("3-bit b", b3), ("1-bit b (v2)", _uniform(0, 1))):
                eX = _plus(bpmf, u80)
                eU = pb.entry_pmf(zero, pb.V2_U)
                zleaf = pb.Leaf(1.0, pb._stats("A", eX, (1,) * t), pb._stats("B", eU, (1,)), 1)
                z = pb.price(c, [zleaf], 256, 16512, pb.TOKENS, beside=True)
                out[name, f"c_t, t {t}, {label}"] = dict(c=1 + z.c, z=z.c, binding=z.binding,
                                                         max_abs=int(pb._stats("A", eX, (1,)).max_abs),
                                                         a21=A21[name, f"c_t, t {t}"])
    return out


def advice_rows(nf) -> dict:
    """The best program with advice on W' (P14: weight-side coins reused across a round's decode calls; L28: advice
    fixed before the period, of bounded size): W'-side forms, excess packing, conversions and lookup tables over W'
    paid before the coins. Priced at v3's frozen qkv shape for prefill (m = 4096) and decode (m = 32): standalone (the
    fast methods, lanes idle) and in the job (job-only and served prices), with advice against without."""
    out = {}
    for name, c in CARDS.items():
        eX, eW, _ = pb._tile_operands(T_V3[name], pb.V2_V)
        K, N = 16384, 16512
        Lp = V3_STATIC[name][K, N]
        L = T_V3[name] * Lp
        chosen = {"naive": (None, 0), **dict(fast_schemes(name))}
        meths = [pb.scheme_method(lb, s, d, eX, eW) for lb, (s, d) in chosen.items()]
        for m in (pb.TOKENS, 32):
            row = {}
            for adv in (False, True):
                p = pb.price_methods(c, meths, L, N, m, beside=True, lookups=True, advice=adv)
                row[adv] = dict(c=p.c, mix=p.mix, binding=p.binding)
            out[name, "standalone", m] = row
        r, j = job(nf, name, K, N, T_V3[name], Lp)
        for label, mu in (("job-only", j["route"].mu),
                          ("served", period(c, j, N, first_tensor_bound(c, j, N) + 1e-4 * j["route"].c).mu)):
            row = {}
            for adv in (False, True):
                p = pb.price_methods(c, j["methods"], L, N, pb.TOKENS, beside=True, items=(j["sha"],), lookups=True,
                                     mu=mu, advice=adv)
                row[adv] = dict(c=p.c, mix=p.mix)
            out[name, label, pb.TOKENS] = row
    return out


ADVICE_SIZE = (   # advice a W' entry (int8, one byte), as a multiple of the weights' bytes
    ("Strassen forms of W', depth 1 / 2", "7/4 / 49/16 (int8 forms; the ±160 sums take a second byte: 7/2 / 49/8)"),
    ("2:4 excess packing of W' forms", "≈ 2 q (q the excess density, ≤ 0.3 at ±80)"),
    ("lut1's tables T[r] = r U (O-IO)", "512 (256 int16 entries an entry): 4 TB for Llama-3.1-8B's 8 GB, past any HBM"),
    ("Four-Russians over W' rows (g-bit chunks of X)", "2^(8g)/g int32 an entry for int8 X: past any HBM from g = 2"),
)


def named(res: dict, design: str = "CMP-118") -> list:
    """WORKSOUND-INST-2's PubBest (L3): per card and world model, the hull's cheapest program at the period's dual
    prices over the static rounds at the design's caps, MAC-weighted, as model cost a credited MAC (MAC-times at the
    card's θ s8 rate): pub_mm its matmul, pub_sha its hashing at the model's schedule (c_sha = 1 in-model), and the
    methods' MAC shares; served-period and job-only. pub_sha.<card> is the schedule's model cost a compression."""
    rows = []
    src_job = "benchmarks/pouw/theory/public_best.py: tile_methods, tile_job"
    src_d5 = "benchmarks/pouw/theory/d5_v2.py: analyse, at_prices"
    for card in CARDS:
        tag = card.replace(" ", "")
        for model in ("Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro"):
            st = [r for r in res["static"][design][card][model] if r["w"] > 0]
            for col in ("served", "job"):
                w = weighted(st, col)
                if not w:
                    continue
                W = sum(r["w"] for r in st)

                def avg(k):
                    return sum(r["w"] * r[col][k] for r in st) / W
                mix = {}
                for r in st:
                    for k, v in r[col]["mix"].items():
                        mix[k] = mix.get(k, 0.0) + r["w"] * v / W
                rows.append(dict(name=f"pub_mm.{tag}.v2", design=design, model=model, prices=col,
                                 pub_mm=avg("pub_mm"), H_mm=avg("H_mm"), c_mm=w["c_mm"], pub_sha=avg("pub_sha"),
                                 c=w["c"], mix={k: v for k, v in mix.items() if v > 1e-3},
                                 source=f"{src_job}; {src_d5}"))
        s = res["sha"][card]
        if design != "CMP-118":
            continue
        rows.append(dict(name=f"pub_sha.{tag}", model="all", prices="θ", pub_sha_clocks=s["model"],
                         d5_schedule=s["d5_best"][:2], measured_clocks=s["pinned"][1],
                         source="θ sha_ceiling (CALIB #4513 at 188907341d), U13; public_best.SHA_SCHEDULES"))
    return rows


# --------------------------------------------------------------------------------------------------------------------

def run(nf, quick: bool = False, jobs: int = 1) -> dict:
    res = {}
    res["static"], res["pv"] = rounds(nf, quick, jobs)
    res["sweep"] = sweep(nf, quick, jobs)
    res["sha"] = sha_rows()
    res["matching"] = {}
    for card in CARDS:
        best = res["sha"][card]["d5_best"][1]
        for model, shapes in SHAPES.items():
            K, N, _ = shapes[0]
            t, Lp = T_V3[card], V3_STATIC[card][K, N]
            kw = {f"{m}, {s}": dict(match=m, model_sha=s == "model sha") for m in ("comps", "lane")
                  for s in ("model sha", "measured sha")}
            kw["comps at θ's ceiling"] = dict(sha_scale=THETA_SHA[card] / best)
            kw["comps at U13's one-lane IMAD.WIDE"] = dict(sha_scale=U13_ONE_LANE[card] / best)
            res["matching"][card, model] = {}
            for k, a in kw.items():
                o = analyse(nf, card, K, N, t, Lp, sens=(), fixed=(), **a)
                res["matching"][card, model][k] = dict(served=o["served"], job=o["job"])
    res["world_x"] = world_x(nf)
    res["phyb3"] = phyb3_rows(quick)
    res["advice"] = advice_rows(nf)
    res["named"] = named(res) + named(res, "06:37Z")
    return res


def f4(x) -> str:
    return "unpriced" if x is None else f"{x:.4f}"


def report(res: dict) -> None:
    pct = "{:.2%}".format
    for kind in ("static", "pv"):
        for design, cards in res[kind].items():
            print(f"\n== {kind} rounds at {design}'s caps: per model (MAC-weighted), served-period | job-only; c = pub/H, "
                  "c_mm the matmul's, leak = the routes (1 - pub/OPT_honest), eps_alg = the deployed gap; worst cell")
            for card, models in cards.items():
                for model, rows in models.items():
                    s, j = weighted(rows, "served"), weighted(rows, "job")
                    if not s:
                        continue
                    print(f"  {card:13} {model:16} served c {s['c']:.4f} (c_inc {s['c_inc']:.4f}) c_mm "
                          f"{f4(s['c_mm'])} leak {pct(s['leak'])} "
                          f"eps_alg {pct(s['eps_alg'])} worst {s['worst_c']:.4f} | job c {j['c']:.4f} c_mm "
                          f"{f4(j['c_mm'])} leak {pct(j['leak'])} eps_alg {pct(j['eps_alg'])} | any-prices "
                          f"{pct(s['any'])} | γ {s['gamma']:.2f}")
                    for r in rows if kind == "static" else []:
                        sv, jb = r["served"], r["job"]
                        mix = ",".join(f"{k} {v:.2f}" for k, v in sv["mix"].items() if v > 1e-3)
                        print(f"      {r['K']}x{r['N']} t{r['t']} L'{r['Lp']} d{r['digits']}: served {sv['c']:.4f} "
                              f"(c_mm {f4(sv['c_mm'])}, X0 {sv['X0']:.2f}, {'+'.join(sv['binding'])}; {mix}) "
                              f"worst {r['worst']['c']:.4f} at X {r['worst']['X']:.2f} | job {jb['c']:.4f} "
                              f"({'+'.join(r['job_binding'])}) | lane {r['lane']:.2f}")
                        print("        X past X0 (round-times): c (c_inc): " + "; ".join(
                            f"{x['X'] - sv['X0']:g}: {x['c']:.3f} ({x['c_inc']:.3f}) {'+'.join(x['binding'])}"
                            for x in r["sens"]))
                    if kind == "pv":
                        lo = min(rows, key=lambda r: r["served"]["c"])
                        print(f"      every n to {max(r['K'] for r in rows)}: least served c {lo['served']['c']:.4f} "
                              f"at n {lo['K']} (L' {lo['Lp']}), least job c {min(r['job']['c'] for r in rows):.4f}")
                        cells = [f"{r['K']}:{r['t']}x{r['Lp']} {r['served']['c']:.3f} ({r['served']['c_inc']:.3f})/"
                                 f"{r['job']['c']:.3f}" for r in rows]
                        for i in range(0, len(cells), 8):
                            print("        " + "; ".join(cells[i:i + 8]))
    print("\n== pub(L') for PROTO-DESIGN-4: per shape and L' (t), γ | job c | served c (c_inc) | at X = 1: c (c_inc); "
          "pub and H at the served prices are in the JSON")
    def cell(card, r):
        t = "" if r["t"] == T_V3[card] else f" t{r['t']}"
        s, x = r["served"], r["fixed"][FIXED_X[0]]
        return (f"{r['Lp']}{t}: {r['gamma']:.2f} | {r['job']['c']:.3f} | {s['c']:.3f} ({s['c_inc']:.3f}) | "
                f"{x['c']:.3f} ({x['c_inc']:.3f})")

    for card, shapes in res["sweep"].items():
        for shape, rows in shapes.items():
            cells = [cell(card, r) for r in rows]
            print(f"  {card:13} {shape:16} " + "; ".join(cells[:4]))
            for i in range(4, len(cells), 4):
                print(" " * 32 + "; ".join(cells[i:i + 4]))
    print("\n== X0 (MAC-times of other tensor work a credited MAC before the tensor pipe has a price), against a world "
          "model's own QK^T (+ indexer) a credited MAC")
    for design, cards in res["static"].items():
        for card, models in cards.items():
            for model, rows in models.items():
                wx = res["world_x"].get(model)
                own = f"; own QK^T {wx['qk']:.4f} + indexer {wx['indexer']:.4f}" if wx else ""
                x0 = [r["served"]["X0_macs"] for r in rows if r["w"] > 0]
                print(f"  {design:8} {card:13} {model:16} X0 {min(x0):.3f}-{max(x0):.3f}{own}")
    print("\n== hashing priced six ways (frozen-first shape of each model, CMP-118's caps), served c / job c; D5 v2 "
          "prices 'comps, model sha'")
    for (card, model), d in res["matching"].items():
        print(f"  {card:13} {model:16} " + "; ".join(f"{k}: {v['served']['c']:.4f}/{v['job']['c']:.4f}"
                                                   for k, v in d.items()))
    print("\n== pub_sha (SM-clocks a compression)")
    for card, s in res["sha"].items():
        print(f"  {card}: model {s['model']} (θ), D5's best schedule {s['d5_best'][0]} {s['d5_best'][1]:.2f} "
              f"({s['d5_best'][2]}-bound); measured: " + "; ".join(f"{w} {c_:.1f}" for w, c_, _, _ in s["measured"])
              + "; paper: " + "; ".join(f"{k} {v[0]:.2f} ({v[2] / 1e9:.2f} G/s)" for k, v in s["schedules"].items()))
    print("\n== P-HYB3 (A21's rows), standalone at θ; A21's figure beside")
    for (card, row), d in res["phyb3"].items():
        mix = ",".join(f"{k} {v:.2f}" for k, v in d.get("mix", {}).items() if v > 1e-3)
        print(f"  {card:13} {row:30} c {d['c']:.3f}" + (f" (tensor only {d['c_tensor']:.3f}; {mix})"
                                                         if "c_tensor" in d else f" (Z U^T {d['z']:.3f})")
              + f" {'+'.join(d['binding'])}; A21: {d['a21']}")
    print("\n== advice on W' (P14, L28): standalone c, or the job's pub (MAC-times a credited MAC); without | with")
    for (card, label, m), row in res["advice"].items():
        mix = ", ".join(f"{k} {v:.2f}" for k, v in row[True]["mix"].items() if v > 1e-3)
        print(f"  {card:13} {label:10} m {m:5}: {row[False]['c']:.4f} | {row[True]['c']:.4f} "
              f"(gain {1 - row[True]['c'] / row[False]['c']:.2%}; with: {mix})")
    print("\n== named programs")
    for r in res["named"]:
        print("  " + json.dumps(_f(r)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quick", action="store_true", help="one shape a model, a few PV lengths, P-HYB3 without schemes")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    nf = nofold()
    res = run(nf, a.quick, a.jobs)
    res["inputs"] = dict(nofold_sha256=hashlib.sha256(Path(nf.__file__).read_bytes()).hexdigest(),
                         public_best_sha256=hashlib.sha256(Path(pb.__file__).read_bytes()).hexdigest())
    report(res)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        flat = _f({"|".join(map(str, k)) if isinstance(k, tuple) else k: v for k, v in res.items()})
        (a.out / "d5_v2.json").write_text(json.dumps(flat, indent=1, default=str))


if __name__ == "__main__":
    main()
