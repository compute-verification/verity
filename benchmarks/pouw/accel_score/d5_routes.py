"""PD3-4: D5's `tile_job` hull (BET-ALGO-LAB, public_best.py at 126ba960ca) plus two routes it lacks at A = B = 0:
PD2-1's O-route (the dense words, O = bc^T at `bit` from the 1-bit dither, no peels) and the zero-high-digit peel
(Y_W's high digit is 0, so the peels cost 3 digits of inner L' instead of 4).

    PUBLIC_BEST_DIR=<checkout of cursor/compute-research-accel-bet-algo-lab-ea24>/benchmarks/pouw/theory \
        python3 benchmarks/pouw/accel_score/d5_routes.py
"""
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, os.environ.get("PUBLIC_BEST_DIR", str(Path(__file__).resolve().parents[1] / "theory")))
import public_best as pb  # noqa: E402


def rows(card) -> list:
    """`price_methods`' inequality rows in order: each finite-rate resource, then θ's joint facets."""
    return [r for r in pb.RES if not math.isinf(pb.cap(card)[r])] + [f"joint{i}" for i in range(len(card.facets))]


@contextmanager
def _lp(hook):
    """Route `price_methods`' one LP through hook(linprog, c, **kw)."""
    orig = pb.linprog
    pb.linprog = lambda c, **kw: hook(orig, c, **kw)
    try:
        yield
    finally:
        pb.linprog = orig


def methods(card, t, Lp, K, n, bit, depth, v, hash_scale=1.0, o_route=True, zhd=True, digits=(2, 2), comps=None,
            model_sha=False):
    """The job's pieces: L, the hash item, the dense method (the honest words with nofold's peel digits), every method
    of the hull, and the deployed kernel's price. See `job`. model_sha: the LP's hash item takes every SHA-512 schedule
    θ admits, measured or not (U13; ALGO-COST-3's A27b: R-hi on sm_120); the deployed kernel's price keeps the
    measured ones."""
    L = t * Lp
    if comps is not None:
        hash_scale = comps / pb.comps_per_mac(L)
    L, zleaf, sha, peels, dense, honest = pb._tile_honest(card, t, Lp, K, n, v, hash_scale)
    if model_sha:
        sha = (sha[0], sha[1], {s: pb.sha_loads(card, s) for s in pb.SHA_SCHEDULES[card.family]})
    dz, dy = digits
    peels = (peels[0], (dz + dy) * Lp / K, peels[2])
    honest = pb.price(card, dense, L, n, pb.TOKENS, beside=False, items=(peels, sha))
    dense_m = pb.Method("dense", dense, items=(peels,))
    mma = dict(smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf, issue=card.mma_issue)

    def bc_fixed(x):
        f = {r: y * x for r, y in mma.items()}
        f["tensor"] = x / card.s8
        if depth:
            ops = (pb.assembly(pb.STRASSEN, depth) + 1) / L
            for r, y in (("alu", ops), ("issue", ops), ("rf", 3 * ops), ("vec", ops)):
                f[r] = f.get(r, 0.0) + y
        return f

    ms = [pb.Method("tiles", [replace(zleaf, weight=1 / t)], fixed=bc_fixed(bit + 1 / t)), dense_m]
    if o_route:
        ms.append(pb.Method("o_route", dense, fixed=bc_fixed(bit)))
    if zhd:
        ms.append(pb.Method("zhd", dense, items=((peels[0], (dz + 1) * Lp / K, peels[2]),)))
    return dict(L=L, n=n, sha=sha, dense=[dense_m], hull=ms, honest=honest)


def priced(card, m, which, hook=None):
    """`price_methods` over m[which] ('dense': the honest words' best program, OPT*; 'hull': every route), with O-IO's
    lookup routes where m['lookups'] says so."""
    kw = dict(beside=True, items=(m["sha"],), lookups=m.get("lookups", False))
    if hook is None:
        return pb.price_methods(card, m[which], m["L"], m["n"], pb.TOKENS, **kw)
    with _lp(hook):
        return pb.price_methods(card, m[which], m["L"], m["n"], pb.TOKENS, **kw)


def row_min(card, m, which, row, c_max):
    """The least load on one row (a resource or a facet, in honest-kernel units) over m[which] at makespan ≤ c_max:
    the job's price at that row's single-row dual, with the job held to its makespan."""
    j = rows(card).index(row)

    def hook(lp, c, A_ub, bounds, **kw):
        return lp([0.0] + list(A_ub[j][1:]), A_ub=A_ub, bounds=[(0, c_max / card.s8 * (1 + 1e-9))] + bounds[1:], **kw)

    return priced(card, m, which, hook).loads[row]


def freed(card, m, best=None):
    """Per row, what the routes free at makespan ≤ OPT*'s: (least load over OPT*'s own programs − least over the
    hull) / OPT*, so a row OPT* itself can lower at its makespan frees nothing. Also RT-PROTO-2's reading against
    OPT*'s one LP solution, (best.loads − least over the hull) / OPT*."""
    best = best or priced(card, m, "dense")
    out = {}
    for r in rows(card):
        h = row_min(card, m, "hull", r, best.c)
        out[r] = ((row_min(card, m, "dense", r, best.c) - h) / best.c, (best.loads[r] - h) / best.c)
    return out


def mu_price(card, m, which, mu):
    """The job's least cost at fixed dual prices μ over m[which] (no makespan bound): min Σ_r μ_r load_r, with loads in
    time (load / rate) and μ as `period` returns it (Σ μ_r cap_r = 1). In honest-kernel units."""
    names = rows(card)
    box = {}

    def hook(lp, c, A_ub, b_ub, **kw):
        tc = [-a[0] for a in A_ub]
        obj = [0.0] + list(sum(mu.get(r, 0.0) / tc[j] * np.array(A_ub[j][1:]) for j, r in enumerate(names)))
        box["fixed"] = sum(mu.get(r, 0.0) / tc[j] * -b_ub[j] for j, r in enumerate(names))
        res = lp(obj, A_ub=A_ub, b_ub=b_ub, **kw)
        box["fun"] = res.fun
        return res

    priced(card, m, which, hook)
    return (box["fun"] + box["fixed"]) * card.s8


def captured(card, m, which):
    """One job's LP as `price_methods` builds it (c, A_ub, b_ub, A_eq, b_eq), and its price."""
    box = {}

    def hook(lp, c, **kw):
        box.update(kw, c=c)
        return lp(c, **kw)

    return box, priced(card, m, which, hook)


def period(card, jobs, fixed=None):
    """One period-level LP: the least makespan T (SM-clocks a credited MAC) over every job's programs at once, each job
    weighted by its share of the credited MACs, so the prover may balance jobs that bind different rows against each
    other. jobs: [(weight, captured LP)]. fixed: loads a credited MAC with no choice of route (the period's uncredited
    served work), {resource: load}. Returns T·s8 (honest-kernel units), the rows' loads and their duals μ
    (Σ μ_r cap_r = 1)."""
    names, caps = rows(card), pb.cap(card)
    A_ub, b_ub, A_eq, b_eq, tcol = _period_lp(card, jobs, fixed)
    nv = A_ub.shape[1]
    res = pb.linprog(np.r_[1.0, np.zeros(nv - 1)], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    T = res.x[0]
    load = (A_ub[:, 1:] @ res.x[1:] - b_ub) / -tcol * card.s8
    mu = -res.ineqlin.marginals * -tcol
    return dict(c=T * card.s8, loads=dict(zip(names, load)), mu=dict(zip(names, mu)),
                binding=[r for r, x in zip(names, load) if x >= T * card.s8 * (1 - 1e-6)], caps=caps)


def _period_lp(card, jobs, fixed):
    """`period`'s LP: (A_ub, b_ub, A_eq, b_eq, the makespan column), the fixed loads moved into b_ub."""
    names = rows(card)
    tcol = np.array([jobs[0][1]["A_ub"][j][0] for j in range(len(names))])
    ub = [sparse.csr_matrix(np.array(lp["A_ub"])[:, 1:] * w) for w, lp in jobs]
    b_ub = sum(w * np.array(lp["b_ub"]) for w, lp in jobs)
    f = dict(fixed or {})
    for j, r in enumerate(names):
        if r.startswith("joint"):
            a, b = card.facets[int(r[5:])]
            b_ub[j] -= a * f.get("tensor", 0.0) + b * f.get(card.facet_other, 0.0) / 32
        else:
            b_ub[j] -= f.get(r, 0.0)
    A_ub = sparse.hstack([sparse.csr_matrix(tcol[:, None])] + ub).tocsr()
    eq = sparse.block_diag([sparse.csr_matrix(np.array(lp["A_eq"])[:, 1:]) for _, lp in jobs])
    A_eq = sparse.hstack([sparse.csr_matrix((eq.shape[0], 1)), eq]).tocsr()
    return A_ub, b_ub, A_eq, np.concatenate([lp["b_eq"] for _, lp in jobs]), tcol


def period_row_min(card, jobs, fixed, row, T_max):
    """The least load the jobs put on one row (a resource, honest-kernel units a credited MAC, the fixed loads' share
    of it left out) over every job's programs at once, with the period's makespan held to T_max (honest-kernel
    units): `period`'s LP with T bounded and that row's load as the objective."""
    j = rows(card).index(row)
    A_ub, b_ub, A_eq, b_eq, tcol = _period_lp(card, jobs, fixed)
    obj = A_ub[j].toarray().ravel()
    obj[0] = 0.0
    res = pb.linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, T_max / card.s8 * (1 + 1e-9))] + [(0, None)] * (A_ub.shape[1] - 1), method="highs")
    assert res.status == 0, res.message
    b_free = sum(w * lp["b_ub"][j] for w, lp in jobs)
    return float(((A_ub[j, 1:] @ res.x[1:])[0] - b_free) / -tcol[j] * card.s8)


def job(card, t, Lp, K, n, bit, depth, v, hash_scale, o_route=True, zhd=True, digits=(2, 2), comps=None,
        model_sha=False):
    """D5's tile job with every route at A = B = 0. digits: the peels' base-128 digits of Z_X and Y_W (nofold's dz, dy;
    D5 fixes 2 + 2); the zero-high-digit route keeps one digit of Y_W. comps: SHA-512 compressions a MAC (nofold's word
    model), which sets the hash scale over D5's 32-bit words instead of `hash_scale`. model_sha: see `methods`."""
    m = methods(card, t, Lp, K, n, bit, depth, v, hash_scale, o_route, zhd, digits, comps, model_sha)
    best, hull = priced(card, m, "dense"), priced(card, m, "hull")
    return dict(c_opt=min(1.0, hull.c / best.c), eps_alg=1 - best.c / m["honest"].c, mix=hull.mix, best=best,
                honest=m["honest"], hull=hull, methods=m)


def job_at(card, t, Lp, K, n, bit, depth, v, alu, **kw):
    lo, hi = 0.0, 4.0
    load = lambda s: pb._alu_load(pb._tile_honest(card, t, Lp, K, n, v, s)[-1])  # noqa: E731
    if load(hi) < alu:
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if load(mid) < alu else (lo, mid)
    return job(card, t, Lp, K, n, bit, depth, v, hi, **kw)


def report(ts=None, **kw):
    for card in (pb.H200, pb.SM120):
        t = (ts or pb.V2_T)[card.family]
        _, bit, depth = pb.BIT_COST[card.family][0]
        for model in ("frozen", "Llama-3.1-8B", "Llama-3.1-70B", "DeepSeek-V4-Pro"):
            num = den = 0.0
            cells = []
            for mdl, K, N, reps, Lp, alu in pb.V2_PICKS[card.family]:
                if mdl != model:
                    continue
                r = job_at(card, t, Lp, K, N, bit, depth, pb.V2_V, alu, **kw)
                w = K * N * reps
                num, den = num + w * r["best"].c * (1 - r["c_opt"]), den + w * r["best"].c
                mix = ",".join(f"{k.split(':')[-1]} {x:.2f}" for k, x in r["mix"].items() if x > 1e-3)
                cells.append(f"{K}x{N} L'{Lp}: {r['c_opt']:.3f} [{mix}]")
            print(f"{card.name:12} t {t} {model:16} leak {num / den:6.2%} | " + "; ".join(cells))


if __name__ == "__main__":
    for o, z in ((False, False), (True, False), (False, True), (True, True)):
        print(f"\n-- o_route {o} zhd {z}")
        report(o_route=o, zhd=z)
