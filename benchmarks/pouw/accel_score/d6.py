"""D6, the number table for protocol v3 (D1, CMP-118 e0cd4061) and v3.1, printed from code (PROTO-DESIGN-4).

Credit is pub, the public best at the period's dual prices (06:10Z, 06:37Z, 07:04Z): one LP over every credited round
of the period with every public method and route of D5's hull (`build_job`: its fast schemes, O-IO's lookups, the peels
on DP4A beside the MMA and, from D5 v2.1, route (iii), b cᵀ with b1 emulated on the RTX PRO 6000's lanes;
`d5_routes.period`), priced at the honest served period's duals μ°, so a winning route lowers pub and leaves honest
time uncredited, never ε. The world rows are μ°-priced device capacity scaled to the world's C (12:02Z), with the
tensor-only price μ_t as a labelled sensitivity and, beside them, credit at a fixed served price μ¹ (the honest period's
duals at beside load x = 1, published as a protocol parameter: P22's alternative). v3.1 is v3 with L′ chosen by the
coordinator's 07:04Z rule, taken at the period (`rule`). ε, in model units, is the sampling term (P15/P20's Poisson
draw, normalised by the period's registered total, published before its first check (P21); η from the measured
free-guess rate; the union over P16's replica groups, V4's the least replica that holds its weights (P23), and over the
window's periods (P26)) plus ε_net (P18), times 1 + H_o/H for the timed answers (P25). γ is from measured hashing, with
the paper ceiling beside it. Every LP (v3.1's picks, the period, the routes) hashes at every SHA-512 schedule θ admits
(U13; ALGO-COST-3's A27b: R-hi on sm_120, unmeasured), so it is one for both hashing rates; v3's own picks stay as
adopted (measured schedules).

    PUBLIC_BEST_DIR=<checkout of cursor/compute-research-accel-bet-algo-lab-ea24 at 0588f18015 (D5 v2.1) or later>/
    benchmarks/pouw/theory python3 benchmarks/pouw/accel_score/d6.py [--json OUT] [--pickle OUT]

--from PKL prints the report from a build's pickle (--pickle) without rebuilding. --census-vector JSON
[--census-vector-ref REF] adds the RTX PRO 6000's beside-load rows with its joint issue rows from a machine-model vector
(ALGO-COST-3's A36; {"vector": name, "rows": {"design | card | model | x": {leak_served, c_inc}}}).
"""
import contextlib
import io
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d5_routes as dr  # noqa: E402
import nofold as nf  # noqa: E402
import v3_caps as vc  # noqa: E402

pb = dr.pb
import d5_v2 as d5v2  # PUBLIC_BEST_DIR's, beside public_best

# ---------------------------------------------------------------------------------------------------------------------
# η (RT-LEAN-2 L38, coordinator 06:34Z): η ≤ 1 − p, p the best free guess's success rate on one opened unit.

def segment_word_pmf_max(Lp: int, t: int, samples: int, seed: int = 0, sigma: float | None = None,
                         bin_width: int = 64) -> dict:
    """The first segment word T_1 = Σ_{k<L} X_k W′_k mod 2^16 of one cell (the fewest terms, so the most predictable
    segment word), L = t·L′, X = wrap8(A + b + V_rep), W′ = wrap8(B + c + U[π_s]), U and V uniform on ±80, b, c
    uniform bits, U's L′ values permuted afresh in each tile. A = B = 0, or Gaussian entries of sd sigma (served-like).
    Its pmf's maximum is estimated from `samples` cells, binned at bin_width values (the pmf is smooth at that scale:
    sums of L products of values spread over ±80), so the estimate's noise is that of a bin count, not of 2^16 cells."""
    rng = np.random.default_rng(seed)
    v = nf.V_WIDTH["H200"]
    counts = np.zeros(2 ** 16 // bin_width, dtype=np.int64)
    chunk = max(1, 2 ** 22 // (t * Lp))
    done = 0
    while done < samples:
        n = min(chunk, samples - done)
        V = rng.integers(-v, v + 1, size=(n, 1, Lp))
        U = rng.integers(-v, v + 1, size=(n, 1, Lp))
        Up = rng.permuted(np.broadcast_to(U, (n, t, Lp)), axis=2)
        b = rng.integers(0, 2, size=(n, t, Lp))
        c = rng.integers(0, 2, size=(n, t, Lp))
        if sigma is None:
            X, W = b + V, c + Up
        else:
            A = np.clip(np.rint(rng.normal(0, sigma, (n, t, Lp))), -127, 127).astype(np.int64)
            B = np.clip(np.rint(rng.normal(0, sigma, (n, t, Lp))), -127, 127).astype(np.int64)
            X, W = (A + b + V + 128) % 256 - 128, (B + c + Up + 128) % 256 - 128
        T1 = (X.astype(np.int64) * W).sum(axis=(1, 2)) % 2 ** 16
        counts += np.bincount(T1 // bin_width, minlength=len(counts))
        done += n
    top = counts.max()
    return dict(L=t * Lp, samples=samples, bin_width=bin_width, p_max=top / samples / bin_width,
                p_max_hi=(top + 3 * math.sqrt(top)) / samples / bin_width, uniform=2.0 ** -16)


def o_pair_pmf_max(K: int) -> float:
    """O = (A + b)(B + c)ᵀ at A = B = 0 is bcᵀ, Binomial(K, 1/4): the 32-bit pair's pmf maximum (its high 16 bits are
    0 for K < 2^18, so the high word alone has p = 1)."""
    return float(binom.pmf(np.arange(K + 1), K, 0.25).max())


def eta(static: dict, pv: dict, samples: int = 400_000) -> dict:
    """η for v3's opened unit. A draw opens one segment word of a cell; the last, T_S, opens together with O's two
    16-bit halves (one 64-bit unit), so O's predictable high word never stands alone. p is the most predictable unit
    over every credited round: T_1 at the least L (S ≥ 2), or (T_S, O) with p ≤ p(T_S) (S = 1). Also reported: O's
    32-bit pair alone at the least credited K, and O's high word alone (p = 1), which is why they open with T_S."""
    least = {}
    for (card, K, N), p in {**static, **pv}.items():
        if p is None:
            continue
        L = p["t"] * p["Lp"]
        if card not in least or L < least[card][0]:
            least[card] = (L, p["t"], p["Lp"], K, N)
    out = {}
    for card, (L, t, Lp, K, N) in least.items():
        seg = segment_word_pmf_max(Lp, t, samples)
        served = segment_word_pmf_max(Lp, t, samples // 4, seed=1, sigma=30.7)
        k_min = min(K for (c, K, N), p in {**static, **pv}.items() if c == card and p is not None)
        out[card] = dict(least_L=dict(L=L, t=t, Lp=Lp, K=K, N=N), segment=seg, served=served,
                         o_pair_alone=dict(K=k_min, p=o_pair_pmf_max(k_min)), o_high_alone=1.0,
                         p=seg["p_max_hi"], eta=1 - seg["p_max_hi"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Picks: v3's (D1), and the dual-safe alternative (RT-PROTO-2's P12 re-raise and P19).

DUAL_TOL = {"H200": 0.01, "RTX PRO 6000": 0.01}    # most a route may free at any single-row dual, a share of OPT*
PV_GAMMA_MAX = 10.0                               # P19: PV is served plain, uncredited, where its round's γ exceeds it
HEADLINE = "v3.1"
DESIGNS = ("v3", "v3.1", "v3.1, PV γ ≤ 10", "v3.1, PV L′ ≤ n/32 from n 2048", "v3.1, PV uncredited",
           "v3.1, RTX PRO 6000 static L′ 64", "v3.1, RTX PRO 6000 PV L′ 64", "dual-safe")
RTX_ONLY = ("v3.1, RTX PRO 6000 static L′ 64", "v3.1, RTX PRO 6000 PV L′ 64")   # the H200's rows are v3.1's


def quiet(f, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **kw)


def build_job(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True, peel_lanes: bool = True):
    """D5 v2's job at a pick (`public_best.tile_job`'s hull, PUBLIC-BEST-2's): PD3-4's routes, D5 v2's fast schemes (the
    2×2 orbit at depth 1 and 2 and the integral catalogue, `d5_v2.fast_schemes`) and O-IO's lookups, hashing at
    nofold's word model with every SHA-512 schedule θ admits in the LP (A27b: R-hi on sm_120, unmeasured). peel_lanes:
    the peels and the zero-high-digit peel may also run on DP4A at D5's own load (`job_items`' route beside the MMA),
    which `tile_job` leaves out (08:24Z: price them); the deployed kernel keeps them on the MMA. The LP is the model's,
    so it is one for both hashing rates; the measured and paper rates differ only in the honest kernel's time (γ). In
    `d5_routes`' form: m['dense'] is OPT*'s method, m['hull'] every method; m['alone'] each one's makespan with the
    job alone; m['honest_tensor'] the deployed kernel's tensor MAC-times a MAC."""
    r = nf.v2_round(card, K, N, t * Lp, 16, tiles=t, weight_coins=weight_coins)
    c = vc.CARDS[card][0]
    _, bit, depth = pb.BIT_COST[c.family][0]
    v, digits = nf.V_WIDTH[card], r["digits"]
    L, zleaf, sha, peels, dense, honest = pb._tile_honest(c, t, Lp, K, N, v, r["comps"] / pb.comps_per_mac(t * Lp),
                                                          digits, True)
    if peel_lanes:
        peels = (peels[0], peels[1], {**peels[2], "dp4a": pb.job_items(c, L, K, False, True, [])[1][2]["dp4a"]})
    ms = pb.tile_methods(c, t, Lp, K, bit, depth, L, zleaf, peels, dense, digits,
                         schemes=dict(d5v2.fast_schemes(card)), v=v)
    best, hull = (pb.price_methods(c, x, L, N, pb.TOKENS, beside=True, items=(sha,), lookups=True) for x in (ms[:1], ms))
    return c, dict(L=L, n=N, sha=sha, dense=ms[:1], hull=ms, lookups=True, alone=(best.c, hull.c),
                   honest_tensor=honest.loads["tensor"]), r


_JM: dict = {}


def job_methods(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True, peel_lanes: bool = True):
    """`build_job`, cached (the picks')."""
    key = (card, K, N, t, Lp, weight_coins, peel_lanes)
    if key not in _JM:
        _JM[key] = build_job(*key)
    return _JM[key]


_FREED: dict = {}


def freed(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True) -> dict:
    key = (card, K, N, t, Lp, weight_coins)
    if key not in _FREED:
        _, lps = job_lps(card, K, N, t, Lp, weight_coins)
        _FREED[key] = {r: x[0] for r, x in dr.freed_lp(vc.CARDS[card][0], lps).items()}
    return _FREED[key]


def dual_safe(card: str, K: int, N: int, weight_coins: bool = True) -> dict | None:
    """The largest L′ (v3's grid, t·L′ ≤ K, under the card's limit) at which no route frees more than DUAL_TOL of
    OPT* at any single-row dual; None if no L′ ≥ 64 does."""
    _, t, lmax, _ = vc.CARDS[card]
    t = t if t * vc.LPS[0] <= K else 1
    for Lp in sorted((x for x in vc.LPS if t * x <= K and (lmax is None or x <= lmax)), reverse=True):
        f = freed(card, K, N, t, Lp, weight_coins)
        if max(f.values()) <= DUAL_TOL[card]:
            return dict(t=t, Lp=Lp, freed=f)
    return None


def round_gamma(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True, built: bool = True) -> float:
    r = nf.v2_round(card, K, N, t * Lp, 16, tiles=t, weight_coins=weight_coins, built=built)
    return max(r["T"], r["A_nw"] + r["wrap"])


def pv_gamma(card: str, n: int, dv: int, p: dict, built: bool = True) -> float:
    return round_gamma(card, n, dv, p["t"], p["Lp"], weight_coins=False, built=built)


TAUS = (1e-4, 0.01, 0.05, 0.2, 1.0)    # job-only leak tolerances the 07:04Z rule sweeps
BESIDE = (0.0, 0.25, 1.0, 4.0)          # other plain INT8 tensor work beside the period, MACs a credited MAC (A29)
_GRID: dict = {}


def grid(card: str, K: int, N: int, weight_coins: bool = True) -> list[dict]:
    """Every L′ on v3's grid (t·L′ ≤ K, under the card's limit) for one round class, with its job-only leak (the job
    alone at its makespan, `build_job`'s hull against OPT*) and its γ at measured hashing (overlapped). Uncached
    jobs: the grid keeps two numbers an L′, and `job_methods` caches only the picks."""
    key = (card, K, N, weight_coins)
    if key not in _GRID:
        _, t, lmax, _ = vc.CARDS[card]
        t = t if t * vc.LPS[0] <= K else 1
        _GRID[key] = []
        for Lp in [x for x in vc.LPS if t * x <= K and (lmax is None or x <= lmax)] or [K]:
            best, hull = build_job(card, K, N, t, Lp, weight_coins)[1]["alone"]
            lk = 1 - min(1.0, hull / best)
            _GRID[key].append(dict(t=t, Lp=Lp, leak=lk, gamma=round_gamma(card, K, N, t, Lp, weight_coins)))
    return _GRID[key]


def least_gamma(card: str, K: int, N: int, weight_coins: bool = True) -> dict:
    """The least-γ L′ within v3's job-only leak tolerance (H200 1e-4, RTX PRO 6000 1%), else the least-leak L′."""
    g, tol = grid(card, K, N, weight_coins), vc.CARDS[card][3]
    ok = [x for x in g if x["leak"] < tol]
    return dict(min(ok, key=lambda x: x["gamma"]) if ok else min(g, key=lambda x: x["leak"]))


def rule_pick(card: str, K: int, N: int, tau: float, weight_coins: bool = True) -> dict:
    """The 07:04Z rule's candidate at job-only leak tolerance τ: the least-γ L′ with γ ≤ PV_GAMMA_MAX and leak ≤ τ,
    else the least-leak L′ with γ ≤ PV_GAMMA_MAX. Where no L′ has γ ≤ PV_GAMMA_MAX, the least-γ L′, marked
    infeasible: `rule` credits it only where that adds credited value per honest second."""
    g = grid(card, K, N, weight_coins)
    ok = [x for x in g if x["gamma"] <= PV_GAMMA_MAX]
    if not ok:
        return dict(min(g, key=lambda x: x["gamma"]), feasible=False)
    within = [x for x in ok if x["leak"] <= tau]
    return dict(min(within, key=lambda x: x["gamma"]) if within else min(ok, key=lambda x: x["leak"]), feasible=True)


def value_test(card: str, model: dict, k: tuple, p: dict, ev: dict) -> float:
    """Credited value per honest second a round class adds over serving it plain, less the period's: CLOCK·η·pub_k
    / (γ_k − 1) − the period's credited share, pub_k at the period's served duals (plain PV costs 1 a MAC)."""
    c = vc.CARDS[card][0]
    _, lps = job_lps(card, k[1], k[2], p["t"], p["Lp"], False)
    pub = dr.mu_price_lp(c, lps["hull"][0], ev["period"]["dense"]["served_mu"])
    return CLOCK[card] * nf.ETA * pub / max(p["gamma"] - 1, 1e-9) - ev["share"]["share"]


def rule(static_keys, pv_keys) -> tuple[dict, dict, dict]:
    """The coordinator's 07:04Z rule (maximize credited value per honest second, pub/T_honest, at served-period
    prices, γ ≤ 10 per round class), taken at the period: the period's credited share at served duals is exactly
    its credited value per honest second, and the picks move those duals, so a per-round ratio at fixed prices is
    myopic (at the H200's all-ALU served dual it rewards hashing: the smallest L′ with γ ≤ 10). Per card, the τ in
    TAUS whose picks (`rule_pick`) give the largest mean credited share over the three models, ties within 0.1 point
    to the lower mean γ. A round class with no L′ at γ ≤ 10 (short PV) is credited at its least-γ L′ where
    `value_test` is positive at every model whose period holds it, else served plain. Returns (static, pv, sweep)."""
    sweep, chosen, tau_picks = {}, {}, {}
    for tau in TAUS:
        st = {k: rule_pick(*k, tau) for k in static_keys}
        pv = {k: rule_pick(*k, tau, weight_coins=False) for k in pv_keys}
        ev = evaluate(st, pv, rates=(True,))
        users = {}
        for (card, _, mname), e in ev.items():
            model = model_of(mname)
            for n in {nf.pv_bucket(n) for n in nf.world_macs(model)[0]}:
                k = (card, n, model["dv"])
                if not pv[k]["feasible"]:
                    users.setdefault(k, []).append(value_test(card, model, k, pv[k], e))
        pv = {k: (None if k in users and min(users[k]) <= 0 else p) for k, p in pv.items()}
        tau_picks[tau] = (st, pv)
        ev = evaluate(st, pv, rates=(True,))
        for card in vc.CARDS:
            es = [e for (c, _, _), e in ev.items() if c == card]
            sweep[card, tau] = dict(share=[e["share"]["share"] for e in es], gamma=[e["world"]["gamma"] for e in es],
                                    cov=[1 - e["world"]["uncredited"] for e in es],
                                    infeasible=sum(1 for k in users if k[0] == card),
                                    plain=sum(1 for k in users if k[0] == card and pv[k] is None), picks=(st, pv))
    for card in vc.CARDS:
        best = max(TAUS, key=lambda t: np.mean(sweep[card, t]["share"]))
        near = [t for t in TAUS if np.mean(sweep[card, t]["share"]) >= np.mean(sweep[card, best]["share"]) - 1e-3]
        chosen[card] = min(near, key=lambda t: np.mean(sweep[card, t]["gamma"]))
    st = {k: sweep[k[0], chosen[k[0]]]["picks"][0][k] for k in static_keys}
    pv = {k: sweep[k[0], chosen[k[0]]]["picks"][1][k] for k in pv_keys}
    return st, pv, dict(sweep={k: {x: y for x, y in v.items() if x != "picks"} for k, v in sweep.items()},
                        tau=chosen, tau_picks=tau_picks)


def p19_pv(pv_keys) -> dict:
    """RT-PROTO-2's P19 policy: PV credited from n 2048 at the largest L′ ≤ n/32 on the card's grid, else plain."""
    out = {}
    for card, n, dv in pv_keys:
        _, t, lmax, _ = vc.CARDS[card]
        lps = [x for x in vc.LPS if x <= n / 32 and t * x <= n and (lmax is None or x <= lmax)]
        out[card, n, dv] = (dict(t=t, Lp=max(lps), gamma=round_gamma(card, n, dv, t, max(lps), False))
                            if n >= 2048 and lps else None)
    return out


def picks() -> tuple[dict, dict]:
    """{design: (static, pv)}, and the rule's sweep. v3: v3's caps (`v3_caps`, the largest job-only leak-free L′),
    PV credited at every n. v3.1: the 07:04Z rule (`rule`). Beside it, P19's two ways and RT-PROTO-2's: v3.1 with PV
    credited only where its round's γ ≤ 10, v3.1 with PV uncredited, and PV at L′ ≤ n/32 from n 2048 (`p19_pv`).
    dual-safe: every pick rescored at every single-row dual (P12 re-raised), PV left uncredited where no L′ is
    dual-safe or its γ exceeds PV_GAMMA_MAX."""
    static, pv = quiet(vc.static_caps), quiet(vc.pv_caps)
    s31, p31, sweep = rule(list(static), list(pv))
    ds_static = {k: dual_safe(*k) or static[k] for k in static}
    ds_pv = {}
    for (card, n, dv) in pv:
        p = dual_safe(card, n, dv, weight_coins=False)
        ds_pv[card, n, dv] = p if p and pv_gamma(card, n, dv, p) <= PV_GAMMA_MAX else None
    out = {"v3": (static, pv), "v3.1": (s31, p31),
           "v3.1, PV γ ≤ 10": (s31, {k: p if p and p["gamma"] <= PV_GAMMA_MAX else None for k, p in p31.items()}),
           "v3.1, PV uncredited": (s31, dict.fromkeys(p31)), "v3.1, PV L′ ≤ n/32 from n 2048": (s31, p19_pv(pv)),
           "v3.1, RTX PRO 6000 static L′ 64": ({k: (next(x for x in grid(*k) if x["Lp"] == 64)
                                                   if k[0] == "RTX PRO 6000" else p) for k, p in s31.items()}, p31),
           "v3.1, RTX PRO 6000 PV L′ 64": (s31, {k: (next((dict(x) for x in grid(*k, weight_coins=False)
                                                           if x["Lp"] == 64 and x["gamma"] <= PV_GAMMA_MAX), p)
                                                     if k[0] == "RTX PRO 6000" else p) for k, p in p31.items()}),
           "dual-safe": (ds_static, ds_pv)}
    for static_, pv_ in out.values():
        for k, p in {**static_, **pv_}.items():
            if p is not None and "freed" not in p:
                p["freed"] = freed(k[0], k[1], k[2], p["t"], p["Lp"], weight_coins=k in static_)
    return out, sweep


def as_picks(static: dict, pv: dict) -> dict:
    return {k: (p["t"], p["Lp"]) if p else None for k, p in {**static, **pv}.items()}


# ---------------------------------------------------------------------------------------------------------------------
# The period: one LP over the credited rounds (credit at pub), and the served period around them.

_JOB: dict = {}


def job_lps(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool,
            peel_lanes: bool = True) -> tuple[dict, dict]:
    """A job's methods and its two LPs (dense, hull) as `price_methods` builds them, and m['at_tensor'], each one's
    price at μ_t, cached across designs."""
    key = (card, K, N, t, Lp, weight_coins, peel_lanes)
    if key not in _JOB:
        c, m, _ = job_methods(card, K, N, t, Lp, weight_coins=weight_coins, peel_lanes=peel_lanes)
        lps = {wh: dr.captured(c, m, wh) for wh in ("dense", "hull")}
        for box, _ in lps.values():
            box.update({k: np.asarray(box[k], dtype=float) for k in ("A_ub", "b_ub", "A_eq", "b_eq")})
        m["at_tensor"] = {wh: dr.mu_price_lp(c, lps[wh][0], MU_T) for wh in lps}
        _JOB[key] = (m, lps)
    return _JOB[key]


MU_T = {"tensor": 1.0}   # 08:24Z: the tensor-only price, in tensor MAC-times (the tensor row's cap is 1)
BOUND_SLACK = (0.0, 0.25, 1.0)   # the bounded tensor minimum's deadlines: OPT*'s served makespan × (1 + s)


def serve_loads(card) -> dict:
    """Loads a MAC of plain INT8 served work (QKᵀ's two softmax passes, V4's indexer, plain PV): the s8 MMA."""
    return dict(tensor=1 / card.s8, smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf,
                issue=card.mma_issue)


def fixed_alone(c, lp: dict, F: dict) -> float:
    """The makespan of the period's fixed load alone, no credited round (honest-kernel units), on `lp`'s rows."""
    f = []
    for r in dr.rows(c):
        if r.startswith("joint"):
            a, b = c.facets[int(r[5:])]
            f.append(a * F.get("tensor", 0.0) + b * F.get(c.facet_other, 0.0) / 32)
        else:
            f.append(F.get(r, 0.0))
    return float(max(np.array(f) / -np.array([row[0] for row in lp["A_ub"]]))) * c.s8


def mu_load(c, served: dict, F: dict) -> float:
    """μ·F, load F's price at `served`'s duals (`d5_routes.period`'s mu and caps), honest-kernel units."""
    x = 0.0
    for r, mu in served["mu"].items():
        if r.startswith("joint"):
            a, b = c.facets[int(r[5:])]
            x += mu * (a * F.get("tensor", 0.0) + b * F.get(c.facet_other, 0.0) / 32) / 4
        else:
            x += mu * F.get(r, 0.0) / served["caps"][r]
    return x * c.s8


def period(card: str, model: dict, static: dict, pv: dict, beside: float = 0.0, peel_lanes: bool = True,
           bounded: bool = False) -> dict:
    """Per credited MAC, in honest-kernel units at θ (c = SM-clocks · s8): OPT*'s period (`dense`: the honest words'
    best programs) and pub (`hull`: every route), each over the credited rounds alone and in the served period (plus
    QKᵀ, the indexer and plain PV as fixed load, and `beside` plain INT8 MACs a credited MAC of other tensor work:
    uncredited decode, other jobs; A29). Credit is each one's price at the honest served period's duals μ°
    (`at_served`, P20's C(μ) at the dual where the honest period sits). `at_tensor`: each one's price at the
    tensor-only price μ_t (08:24Z: tensor MAC-times a credited MAC, lanes free), beside the deployed kernel's tensor
    (`honest_tensor`); bounded: also pub's least tensor with the period held to OPT*'s served makespan × (1 + s)
    (`tensor_bounded`, for s in BOUND_SLACK: a sensitivity on a deadline the protocol doesn't enforce). `net_most`:
    max_k N_k × round k's price at each μ, for ε_net (`eps_terms`). `static` and `pv`: the same
    prices over each kind of credited round alone (v3.1's PV picks priced as a whole). `freed`: per row, the share of
    OPT* the routes free at that row's single-row dual (`d5_routes.freed`, makespan ≤ OPT*'s), cost-weighted over the
    credited rounds (P12 re-raised). Also Σ_w q_w (P20's draw multiplier): each round's own makespan share of the
    honest period, and PB4's c_inc, the credited rounds' share of the period's makespan increment over its fixed load
    alone (pub's period over OPT*'s, no duals). `unit`: one plain INT8 MAC of beside work, its standalone time and its
    price at μ°, so beside work hidden from credit at μ° is x (alone − at_served) a credited MAC (P22)."""
    c = vc.CARDS[card][0]
    hist, _, qk, _, ix = nf.world_macs(model)
    jobs, rounds = {"dense": [], "hull": []}, {"dense": 0.0, "hull": 0.0}
    items = [(K, N, K * N * reps * nf.WORLD_TOKENS, static[card, K, N], True)
             for K, N, reps in nf.SHAPES[model["name"]]]
    bins = {}
    for n, cnt in hist.items():
        bins[nf.pv_bucket(n)] = bins.get(nf.pv_bucket(n), 0) + cnt * model["hq"] * model["dv"] * n
    plain_pv = 0.0
    for nb, x in sorted(bins.items()):
        p = pv[card, nb, model["dv"]]
        if p is None:
            plain_pv += x
        else:
            items.append((nb, model["dv"], x, p, False))
    cred = sum(it[2] for it in items)
    ms, freed_ = [], {}
    for K, N, w, p, wc in items:
        m, lps = job_lps(card, K, N, p["t"], p["Lp"], wc, peel_lanes)
        ms.append((w / cred, m, wc, N, lps))
        for wh in jobs:
            box, pr = lps[wh]
            jobs[wh].append((w / cred, box))
            rounds[wh] += w / cred * pr.c
            if wh == "dense":
                for r, x in p.get("freed", {}).items():
                    freed_[r] = freed_.get(r, 0.0) + w / cred * pr.c * x
    F_macs = (qk * (1 + nf.SOFTMAX_PASS) + ix + plain_pv) / cred + beside
    F = {r: v * F_macs for r, v in serve_loads(c).items()}
    out = dict(cred_macs=cred, plain_pv=plain_pv, rounds=rounds,
               freed={r: x / rounds["dense"] for r, x in freed_.items()})
    for wh in jobs:
        alone, served = dr.period(c, jobs[wh]), dr.period(c, jobs[wh], F)
        out[wh] = dict(alone=alone["c"], alone_binding=alone["binding"], alone_mu=alone["mu"], served=served["c"],
                       served_binding=served["binding"], served_mu=served["mu"],
                       own_dual=served["c"] - mu_load(c, served, F))
        if wh == "dense":
            # One plain INT8 MAC of beside work: its standalone time and its price at μ° (P22's hidden work).
            out["unit"] = dict(alone=fixed_alone(c, jobs[wh][0][1], serve_loads(c)),
                               at_served=mu_load(c, served, serve_loads(c)))
    # Both priced at the honest served period's duals μ°, where the honest prover's period sits.
    mu0 = out["dense"]["served_mu"]
    kinds = {"static": True, "pv": False}
    for wh in jobs:
        each = [(w, wc, dr.mu_price_lp(c, lps[wh][0], mu0), m["at_tensor"][wh], N) for w, m, wc, N, lps in ms]
        out[wh]["at_served"] = sum(w * a for w, _, a, _, _ in each)
        out[wh]["at_tensor"] = sum(w * b for w, _, _, b, _ in each)
        out[wh]["net_most"] = dict(served=max(N * a for _, _, a, _, N in each),
                                   tensor=max(N * b for _, _, _, b, N in each))
        for kind, wc_ in kinds.items():
            sel = [(w, a, b) for w, wc, a, b, _ in each if wc == wc_]
            out[wh][kind] = dict(w=sum(w for w, _, _ in sel), at_served=sum(w * a for w, a, _ in sel),
                                 at_tensor=sum(w * b for w, _, b in sel))
    assert abs(out["dense"]["at_served"] / out["dense"]["own_dual"] - 1) < 1e-4, (out["dense"], card, model["name"])
    out["honest_tensor"] = sum(w * m["honest_tensor"] for w, m, _, _, _ in ms)
    if bounded:
        out["hull"]["tensor_bounded"] = {s: dr.period_row_min(c, jobs["hull"], F, "tensor",
                                                              out["dense"]["served"] * (1 + s)) for s in BOUND_SLACK}
    out["leak"] = 1 - out["hull"]["alone"] / out["dense"]["alone"]
    out["leak_served"] = 1 - out["hull"]["at_served"] / out["dense"]["at_served"]
    out["leak_kind"] = {k: (1 - out["hull"][k]["at_served"] / out["dense"][k]["at_served"]
                            if out["dense"][k]["w"] > 0 else None) for k in kinds}
    out["leak_job"] = 1 - rounds["hull"] / rounds["dense"]
    out["q_sum"] = rounds["dense"] / out["dense"]["served"]
    out["fixed_alone"] = T_F = fixed_alone(c, jobs["dense"][0][1], F)
    out["c_inc"] = (out["hull"]["served"] - T_F) / (out["dense"]["served"] - T_F)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ε, the credited share, the world (rule L2).

WORLD_FLOP = {"week": 1.8e29, "month": 7.8e29, "year": 9.36e30}   # FP8 FLOP of the world fleet (brief rule L2)
YEAR_S = 365.25 * 86400
DELTAS = (0.01, 1e-6)
EPS_S_AT_1E6 = 0.001                                      # the sampling term's target at δ 1e-6, which sets k
K_DRAWS = math.ceil(math.log(1e6) / EPS_S_AT_1E6)         # 13,816
GROUP_GPUS = (1, 8, 64)                                   # P16's table: group sizes beside each model's own
NODE_GPUS = 8
GPU_MEM = {"H200": 141e9, "RTX PRO 6000": 96e9}           # HBM bytes a GPU
PARAMS = {"Llama-3.1-8B": 8.0e9, "Llama-3.1-70B": 70.6e9, "DeepSeek-V4-Pro": 1.6e12}   # weights, INT8 (P23)
EP_BITS = 1e7                                             # P23: V4's expert-parallel bits a token out of one node
WINDOW_S = {"week": 7 * 86400, "month": YEAR_S / 12, "year": YEAR_S}
# P25: the timed answers. A group's checks go through the firewall one at a time, so a period holds them all only if
# its draws N satisfy N(Δ + τ_ans) ≤ T.
PERIOD_S = 7 * 86400                                      # T, one period: a week
DELTA_S = 30.0                                            # Δ: a round's coins to its commitment at the certifier
TAU_ANS_S = 10.0                                          # τ_ans: a draw to its answer
ANS_GPU_S, ANS_FLOOR_S, ANS_RUN = 1.0, 8.0, "r20261024-092255-2edd"   # C-Flock CUDA --zk: GPU-s a proof, latency floor
WIRE_BITS = 32                                            # bits a token across the group's boundary (RT-PROTO-2 P18)
HTTP_FACTOR = 6.5                                         # RT-PROTO-2: wire bytes a token if HTTP/SSE crosses it
CLOCK = {"H200": 1797 / 1980, "RTX PRO 6000": 1.0}        # M14: the throttled honest kernel over θ's clock
R_SHA = {"H200": 1.54, "RTX PRO 6000": 1.55}              # U13: θ's SHA-512 over the card's (PD3's share formula)
HOST = {"H200": (0.0315, 0.0490), "RTX PRO 6000": (0.0951, 0.0507)}   # U9 (CALIB): AMX / INT8, CPU / GPU hashing
SHA_MEASURED = {"H200": (4.79e9, 1980, "r20261018-062010-1c6b"), "RTX PRO 6000": (7.13e9, 2092, "r20261016-110413-c5d1")}
# H1's perplexity (Δ over bf16) for v3's arithmetic: both operands 1-bit dithered (H1's design A; the noise U, V peels
# off exactly), and PV with the fresh bit on P in one block (COVER-ATTN's credited PV).
PPL = {
    "Llama-3.1-8B": ("+1.38% (WT-2 2K, 3 seeds, r20261017-065449-b05b); +3.39% at 8K (seed 0, r20261017-070733-6851)",
                     "+0.83% (r20261017-065449-b05b's adopted A)", "+1.38% at 2K, +3.39% at 8K"),
    "Llama-3.1-70B": ("+3.01% (C4 2K, seed 0, r20261017-063215-c9b2); 8K not run", "+1.68% (adopted A)",
                      "+3.01% at 2K (8K not run)"),
    "DeepSeek-V4-Pro": ("not run", "not run", "not run"),
}
PPL_ART = "art:02211b98fe2c3320134413f91f7e5a68b5d2fdbefc69675b354728f3648add35"


def n_gpus(card: str) -> float:
    return WORLD_FLOP["year"] / (2 * nf.V2_CARDS[card][0] * YEAR_S)


def model_of(name: str) -> dict:
    return next(m for m in nf.WORLD if m["name"] == name)


def group_gpus(card: str, model: dict) -> int:
    """P23: a group is a serving replica, its coins released in lockstep. Where replicas fit in a node (the Llamas),
    a node's 8 GPUs, its replicas released together; else the least replica that holds the INT8 weights (V4: 12 H200s,
    17 RTX PRO 6000s, before KV cache), the most groups such replicas allow, with the network certifier's boundary
    around the whole group."""
    return max(NODE_GPUS, math.ceil(PARAMS[model["name"]] / GPU_MEM[card]))


def answer_time(card: str, model: dict, per: dict) -> float:
    """P25: the honest answers' GPU time over the group's, H_o/H in time: k Σ_w q_w draws a period, each one C-Flock
    --zk proof of the opened unit (ANS_GPU_S, run beside later rounds), over g·T with the group busy all period."""
    return K_DRAWS * per["q_sum"] * ANS_GPU_S / (group_gpus(card, model) * PERIOD_S)


def eps_terms(card: str, model: dict, eta_: float, per: dict, share_: dict) -> dict:
    """ε in model units, a share of the claim at every dual (P20), the claim's draw normalised by the period's
    registered total C_reg, published before the period's first check (P21): ε_S = ln(1/δ)/(ηk) with P15's Poisson
    draw (expected draws k Σ_w q_w); P16 per replica group, the union bound over G groups (ε_S at k a group, and the k
    a group that holds ε_S at 0.1%), at the model's group (`group_gpus`, P23); ε_net = n_max E / credited MACs (P18,
    n_max the largest N of a credited round, E the wire bits across the group's boundary). `at`: ε at the model's
    group as a share of the claim at μ° and at μ_t, with ε_net priced at that μ: a bit exported in round k stands for
    N_k of its MACs at its price, so ε_net = E max_k(N_k price_k) / (credited MACs × the mean price), n_max E /
    credited MACs where prices are equal; None where the claim at that μ is 0. `head`: at μ°, times 1 + H_o/H in
    model units (P25: the answers' time over the claim, s·g·T); `no_c_reg`: P21's variant, the draw normalised by the
    wall-clock period instead, so ε_S is a share of s·T, ε_S/s + ε_net; `union`: ln(P)/(ηk) over a window's P periods
    (P26), to add to ε_S."""
    cred_macs, q_sum = per["cred_macs"], per["q_sum"]
    g_m = group_gpus(card, model)
    out = dict(eta=eta_, k=K_DRAWS, draws=K_DRAWS * q_sum, q_sum=q_sum, g=g_m)
    out["eps_S"] = {d: math.log(1 / d) / (eta_ * K_DRAWS) for d in DELTAS}
    n_max = max([N for _, N, _ in nf.SHAPES[model["name"]]] + [model["dv"]])
    E = WIRE_BITS * nf.WORLD_TOKENS
    out["n_max"], out["eps_net"] = n_max, n_max * E / cred_macs
    out["eps_net_http"] = out["eps_net"] * HTTP_FACTOR
    out["eps_net_node"] = n_max * EP_BITS * nf.WORLD_TOKENS / cred_macs if g_m > NODE_GPUS else None
    out["eps"] = {d: out["eps_S"][d] + out["eps_net"] for d in DELTAS}
    groups = {}
    for g in sorted({*GROUP_GPUS, g_m, NODE_GPUS * math.ceil(g_m / NODE_GPUS)}):
        G = n_gpus(card) / g
        groups[g] = dict(G=G, eps_S={d: math.log(G / d) / (eta_ * K_DRAWS) for d in DELTAS},
                         k_group=math.ceil(math.log(G / 1e-6) / (eta_ * EPS_S_AT_1E6)))
        groups[g]["draws_world"] = G * groups[g]["k_group"] * q_sum
        groups[g]["eps"] = {d: groups[g]["eps_S"][d] + out["eps_net"] for d in DELTAS}
    out["groups"] = groups
    h = per["hull"]
    out["at"] = {}
    for mu, mean in (("served", h["at_served"]), ("tensor", h["at_tensor"])):
        if mean <= 1e-12:
            out["at"][mu] = None
            continue
        net = E * h["net_most"][mu] / (cred_macs * mean)
        out["at"][mu] = dict(net_scale=net / out["eps_net"], eps_net=net,
                             eps={d: groups[g_m]["eps_S"][d] + net for d in DELTAS})
    s = share_["share"]
    out["ho_claim"] = ho = share_["answers"] / s
    out["head"] = {d: out["at"]["served"]["eps"][d] * (1 + ho) for d in DELTAS}
    out["no_c_reg"] = {d: (groups[g_m]["eps_S"][d] / s + out["at"]["served"]["eps_net"]) * (1 + ho) for d in DELTAS}
    out["union"] = {w: math.log(WINDOW_S[w] / PERIOD_S) / (eta_ * K_DRAWS) for w in WINDOW_S}
    return out


def answers(eps: dict) -> dict:
    """P25's timing at the model's group: its draws N (Poisson, mean k Σ_w q_w) at their 1 − 1e-6 quantile must fit
    the period one at a time, N(Δ + τ_ans) ≤ T; with the group stalled for each answer instead of pipelined, the
    answers would take N τ_ans / T of it; and rounds in the period's last Δ + τ_ans can't be checked, so they go
    uncredited (no coins near the period's end, P5)."""
    n_hi = float(poisson.ppf(1 - 1e-6, eps["draws"]))
    return dict(draws=eps["draws"], draws_hi=n_hi, busy=n_hi * (DELTA_S + TAU_ANS_S) / PERIOD_S,
                t_min_h=n_hi * (DELTA_S + TAU_ANS_S) / 3600, serial=eps["draws"] * TAU_ANS_S / PERIOD_S,
                tail=(DELTA_S + TAU_ANS_S) / PERIOD_S)


def share(card: str, row: dict, per: dict, model: dict) -> dict:
    """The credited share of honest device time (06:45Z: r and the clock kept in it): credit at pub (the served
    period's duals, D5's θ) over the measured honest time, decomposed into factors that multiply to it: the clock
    (M14), the wraps (outside credit), the uncredited served ops (QKᵀ, the indexer, plain PV, where they don't hide in
    a slack pipe), θ's generosity (D5's price of the credited rounds' best programs over the built kernel's time:
    hashing r and the matmul's η together), the routes (pub over OPT* at the served duals) and the timed answers'
    GPU time (P25, `answer_time`)."""
    cov = 1 - row["uncredited"]
    honest, credit_b = row["honest"], row["credit"]
    ho = answer_time(card, model, per)
    # nofold's units are a MAC at η of θ's INT8 rate; D5's are a MAC at θ's rate.
    f = {"clock": CLOCK[card],
         "wraps": max(row["T_all"], row["A_nw"]) / honest,
         "uncredited ops": credit_b / max(row["T_all"], row["A_nw"]),
         "θ's generosity": per["dense"]["at_served"] * cov * nf.ETA / credit_b,
         "routes": per["hull"]["at_served"] / per["dense"]["at_served"],
         "answers": 1 / (1 + ho)}
    s = math.prod(f.values())
    pd3 = max(row["T_c"], row["A_nw"] / R_SHA[card]) / max(row["T_c"], row["A_nw"])
    # 08:24Z: the same honest time credited in tensor MAC-times at μ_t (FP8 FLOP / 2), pub's against OPT*'s, the
    # deployed kernel's and, where `period` bounds it, pub's least tensor at OPT*'s served makespan.
    to_t = s / per["hull"]["at_served"]
    tensor = dict(share=to_t * per["hull"]["at_tensor"], opt=to_t * per["dense"]["at_tensor"],
                  honest=to_t * per["honest_tensor"])
    if "tensor_bounded" in per["hull"]:
        tensor["bounded"] = {s: to_t * x for s, x in per["hull"]["tensor_bounded"].items()}
    return dict(share=s, factors=f, pd3_r_reading=pd3, routes_job=1 - per["leak_job"], cov=cov,
                with_hosts=s / (1 + HOST[card][1]), tensor=tensor, answers=ho)


def worst_gamma(card: str, model: dict, static: dict, pv: dict, built: bool) -> dict:
    """The worst credited round's γ (overlapped: max(T, A + wraps)) over the model's static shapes and PV lengths."""
    cells = []
    for K, N, _ in nf.SHAPES[model["name"]]:
        p = static[card, K, N]
        r = nf.v2_round(card, K, N, p["t"] * p["Lp"], 16, tiles=p["t"], built=built)
        cells.append((max(r["T"], r["A_nw"] + r["wrap"]), f"{K}x{N} L' {p['Lp']}"))
    for n in sorted({nf.pv_bucket(n) for n in nf.world_macs(model)[0]}):
        p = pv[card, n, model["dv"]]
        if p is not None:
            cells.append((pv_gamma(card, n, model["dv"], p, built), f"PV n {n} L' {p['Lp']}"))
    g, where = max(cells)
    return dict(gamma=g, where=where)


def streaming(card: str, model: dict, row: dict, pv: dict) -> dict:
    """P2: what the network certifier streams (b, P's b included) at the measured honest rate, a GPU and an 8-GPU
    node, and three options with what each does to n_max (MACs an exported unpredictable bit) and so to ε_net."""
    hist, st, qk, pvm, ix = nf.world_macs(model)
    toks = nf.WORLD_TOKENS
    t_req = row["gamma"] * (st + qk + pvm + ix) / (nf.ETA * nf.V2_CARDS[card][0]) / CLOCK[card]
    shapes = nf.SHAPES[model["name"]]
    static_b = toks * sum(K * reps for K, _, reps in shapes)
    credited_n = {n: c for n, c in hist.items() if pv.get((card, nf.pv_bucket(n), model["dv"])) is not None}
    p_seed = sum(c * model["hq"] * min(n, 512) for n, c in credited_n.items())
    p_b = sum(c * model["hq"] * n for n, c in credited_n.items())
    v_side = toks * sum(nl for nl, _ in model["att"](toks)) * model["hkv"] * model["dv"] * 9 if credited_n else 0
    n_static = max(N for _, N, _ in shapes)
    pv_macs = sum(c * model["hq"] * model["dv"] * n for n, c in credited_n.items())
    cred = st + pv_macs
    n_max_seed_pv = max([max(n / 512, 1) * model["dv"] for n in credited_n] + [0])

    def seeded(K: int, N: int) -> tuple[int, float]:
        """A static row's streamed bits with 512-bit seeds covering at most 512·N_max MACs each (raw b if fewer bits),
        and the MACs one of those bits stands for."""
        bits = min(K, 512 * math.ceil(K * N / (512 * n_static)))
        return bits, K * N / bits

    opts = {
        "raw (v3: b and P's b)": (static_b + p_b, max(n_static, model["dv"])),
        "P's b seeded per (head, row)": (static_b + p_seed, max(n_static, n_max_seed_pv)),
        "every credited coin seeded, ≤ 512·N_max MACs a seed (RT-PROTO-2's; adopted 07:04Z)": (
            toks * sum(seeded(K, N)[0] * reps for K, N, reps in shapes) + p_seed,
            max([seeded(K, N)[1] for K, N, _ in shapes] + [n_max_seed_pv])),
        "floor (credited MACs / N_max)": (cred / n_static, n_static),
    }
    E = WIRE_BITS * toks
    out = {k: dict(gpu=bits / 8 / t_req, node=8 * bits / 8 / t_req, n_max=nm, eps_net=nm * E / cred)
           for k, (bits, nm) in opts.items()}
    out["Ṽ's dither and noise, if streamed"] = dict(gpu=v_side / 8 / t_req, node=8 * v_side / 8 / t_req,
                                                    n_max=None, eps_net=None)
    return out


def decode(static: dict) -> dict:
    """Static rounds at decode batches m (v3's prefill caps; D5's LP prices them at m = 4096), measured hashing:
    γ with fresh and reused weight-side coins, and the roofline (INT8 weights from HBM, +1/8 B a weight for reused c)."""
    out = {}
    picks_ = {k: (p["t"], p["Lp"]) for k, p in static.items()}
    for card in vc.CARDS:
        mem_mac = nf.ETA * nf.V2_CARDS[card][0] / nf.HBM[card]
        for m in nf.WORLD:
            for b in (1, 32, 256):
                f = nf.static_pipes(card, m["name"], 0.0, m=b, picks=picks_, built=True)
                r = nf.static_pipes(card, m["name"], 0.0, m=b, picks=picks_, built=True, weight_coins=False)
                gf, gr = max(f["T"], f["A_nw"] + f["wrap"]), max(r["T"], r["A_nw"] + r["wrap"])
                plain = max(1.0, mem_mac / b)
                out[card, m["name"], b] = dict(fresh=gf, reused=gr, roof_fresh=max(gf, mem_mac / b) / plain,
                                               roof_reused=max(gr, 1.125 * mem_mac / b) / plain)
    return out


def evaluate(static: dict, pv: dict, rates=(True, False), bounded: bool = False) -> dict:
    """One design at the world model: {(card, rate, model): its nofold row, period, credited share, worst round}."""
    out, pers = {}, {}
    pk = as_picks(static, pv)
    for built in rates:
        for row in nf.v2_world(0.0, built=built, picks=pk):
            card = row["card"].split(" built")[0]
            model = model_of(row["model"])
            if (card, model["name"]) not in pers:
                pers[card, model["name"]] = period(card, model, static, pv, bounded=bounded)
            per = pers[card, model["name"]]
            out[card, "measured" if built else "paper", model["name"]] = dict(
                card=card, rate="measured" if built else "paper", model=model["name"], world=row, period=per,
                share=share(card, row, per, model), worst=worst_gamma(card, model, static, pv, built))
    return out


def beside_sweep(P: dict, tau_picks: dict) -> dict:
    """A29: the served duals move with what else the period holds. For v3, the headline, dual-safe and each τ of the
    rule's sweep, at BESIDE plain INT8 MACs of other tensor work a credited MAC (uncredited decode, other jobs) on top
    of the period's own uncredited work: what the routes leave uncredited at those duals (`leak`) and the honest
    period's binding rows. The credited share at x is the share at x = 0 with its routes factor moved to x's: beside
    work overlaps the credited rounds, and at the period's duals their OPT* price is at least the period time they
    add, so dividing it by their standalone honest time would count the overlap as a loss. The rule's value test
    stays at x = 0. Also PB4's c_inc at each x (`period`), and PB5's case, the RTX PRO 6000's PV at L′ 64. For P22,
    each x's pub and OPT* at its own duals μ_x, the dearest credited round's N × price (ε_net), the unit beside MAC's
    time and price at μ_x, and the honest period's makespan and its fixed load's alone."""
    sets = {d: P[d] for d in ("v3", HEADLINE, *RTX_ONLY, "dual-safe")}
    sets.update({f"τ {tau:g}": p for tau, p in tau_picks.items()})
    out = {}
    for name, (static, pv) in sets.items():
        for (card, _, mname), e in evaluate(static, pv, rates=(True,)).items():
            if name in RTX_ONLY and card != "RTX PRO 6000":
                continue
            model = model_of(mname)
            for x in BESIDE:
                per = e["period"] if x == 0 else period(card, model, static, pv, beside=x)
                out[name, card, mname, x] = dict(leak=per["leak_served"], binding=per["dense"]["served_binding"],
                                                 c_inc=per["c_inc"], leak_kind=per["leak_kind"],
                                                 gamma=e["world"]["gamma"],
                                                 share0=e["share"]["share"], leak0=e["period"]["leak_served"],
                                                 pub=per["hull"]["at_served"], opt=per["dense"]["at_served"],
                                                 net_most=per["hull"]["net_most"]["served"], unit=per["unit"],
                                                 served=per["dense"]["served"], fixed_alone=per["fixed_alone"],
                                                 cred_macs=per["cred_macs"], q_sum=per["q_sum"])
                out[name, card, mname, x]["share"] = (e["share"]["share"] * (1 - per["leak_served"])
                                                      / (1 - e["period"]["leak_served"]))
    return out


def best_tau(B: dict, card: str, x: float) -> str:
    """The rule's τ at beside load x (`rule`'s test there): the τ design with the largest mean credited share over the
    models at x, ties within 0.1 point to the lower mean γ."""
    models = [m["name"] for m in nf.WORLD]
    taus = list(dict.fromkeys(k[0] for k in B if k[0].startswith("τ ")))
    mean = {n: np.mean([B[n, card, m, x]["share"] for m in models]) for n in taus}
    return min((n for n in taus if mean[n] >= max(mean.values()) - 1e-3),
               key=lambda n: np.mean([B[n, card, m, x]["gamma"] for m in models]))


def build() -> dict:
    P, sweep = picks()
    tau_picks = sweep.pop("tau_picks")
    eta_ = eta(*P[HEADLINE])
    eta_min = min(e["eta"] for e in eta_.values())
    out = dict(eta=eta_, rows={}, k=K_DRAWS, rule=sweep)
    for design, (static, pv) in P.items():
        for (card, rate, mname), e in evaluate(static, pv, bounded=design == HEADLINE).items():
            model = model_of(mname)
            e["design"] = design
            e["eps"] = eps_terms(card, model, eta_min, e["period"], e["share"])
            e["answers"] = answers(e["eps"])
            if design == HEADLINE and rate == "measured":
                e["streaming"] = streaming(card, model, e["world"], pv)
                # 08:24Z's gap: the peels and the zero-high-digit peel on the MMA only, as `tile_job` has them.
                h = period(card, model, static, pv, peel_lanes=False)["hull"]
                e["mma_peels"] = dict(at_served=h["at_served"], at_tensor=h["at_tensor"],
                                      share=e["share"]["share"] * h["at_served"] / e["period"]["hull"]["at_served"])
            out["rows"][design, card, rate, mname] = e
    out["beside"] = beside_sweep(P, tau_picks)
    out["decode"] = decode(P[HEADLINE][0])
    out["picks"] = {design: {k: p and dict(t=p["t"], Lp=p["Lp"], freed=p["freed"], leak=p.get("leak"),
                                           gamma=round_gamma(*k, p["t"], p["Lp"], weight_coins=k in P[design][0]))
                             for k, p in {**P[design][0], **P[design][1]}.items()} for design in P}
    return out


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def report(D: dict) -> str:
    rows, L = D["rows"], []
    models = [m["name"] for m in nf.WORLD]
    short = {"Llama-3.1-8B": "8B", "Llama-3.1-70B": "70B", "DeepSeek-V4-Pro": "V4"}

    def mma_t(r):
        return r["share"]["share"] * r["mma_peels"]["at_tensor"] / r["period"]["hull"]["at_served"]

    L.append("## D6 v1.1: protocol v3 (D1) and v3.1 (v3's L′ by the 07:04Z rule), printed by "
             "`benchmarks/pouw/accel_score/d6.py`\n")
    L.append(f"The headline is {HEADLINE}. Columns are 8B / 70B / V4 unless stated. ε is in model units, a share of "
             "the credited claim at every dual price (P20). "
             "Credit is pub at the period's dual prices: one LP over the period's credited rounds with every public "
             "method and route of D5 v2.1's hull (its fast schemes and lookups, the peels on DP4A beside the MMA and, "
             "on the RTX PRO 6000, route (iii), b cᵀ with b1 emulated on the lanes; `d5_routes.period`). A route that "
             "wins there leaves honest time uncredited and never adds to ε.\n")
    L.append("What changed from v1 (RT-PROTO-2's P21–P26, CMP-118 71d22b4c; the coordinator's 12:36Z decisions): "
             "D5 v2.1 (0588f18015) carries route (iii) (x = 0 doesn't move), but θ's sm_120 joint issue rows are "
             "closed rows (ALGO-COST-3's A36), so the RTX PRO 6000's x > 0 cells stay upper bounds until CALIB moves "
             "θ (`theta.toml`) to a current vector, GPU work after M3; the period's registered total is published before its first check (P21, in the "
             "spec below, with the no-publication variant beside ε); a group is a serving replica, V4's 12 H200s or "
             "17 RTX PRO 6000s with the network certifier's boundary around it (P23); V4's static MACs are its "
             "checkpoint's matmuls at their own shapes, 48.85e9 a token, with no credit for the input embedding and "
             "no shape scale (P24, A39); the timed answers are priced (P25); the week, month and year rows carry the "
             "union over periods (P26); and the world rows say what they bound, with credit at a fixed served price "
             "μ¹ as one alternative row (P22).\n")

    def gamma_ans(r):
        return r["world"]["gamma"] * (1 + r["share"]["answers"])

    L.append(f"### The goals at the headline ({HEADLINE}, measured hashing, a period, replica groups)\n")
    L.append("ε is a period's, at the model's replica group, times 1 + H_o/H for the timed answers (P25); γ includes the "
             "answers' GPU time.\n")
    L.append("| card, model | GPUs a group | ε at δ 1e-6 (goal ≤ 1%) | γ overlapped, measured / paper (goal ≤ 10) | "
             "coverage (goal ≥ 95%) | perplexity (goal < 5%) | credited share at μ° |")
    L.append("|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            r, p = rows[HEADLINE, card, "measured", m], rows[HEADLINE, card, "paper", m]
            cov = 1 - r["world"]["uncredited"]
            L.append(f"| {card}, {short[m]} | {r['eps']['g']} | {pct(r['eps']['head'][1e-6], 3)} | "
                     f"{gamma_ans(r):.2f} / {gamma_ans(p):.2f} | {pct(cov, 1)}{'' if cov >= 0.95 else ' (miss)'} | "
                     f"{PPL[m][2]} | {pct(r['share']['share'], 1)} |")
    v4 = model_of("DeepSeek-V4-Pro")
    _, st, qk, pvm, ix = nf.world_macs(v4)
    f32 = nf.WORLD_TOKENS * sum(K * N * reps for K, N, reps in nf.SHAPES[v4["name"]] if (K, N) in nf.V4_F32)
    cov4 = {card: 1 - rows[HEADLINE, card, "measured", v4["name"]]["world"]["uncredited"] for card in vc.CARDS}
    no_f32 = {card: x - f32 / (st + qk + pvm + ix) for card, x in cov4.items()}

    def v4_goal(card):
        least = min(cov4[card], no_f32[card])
        if least >= 0.95:
            return f"V4 on the {card} meets the 95% goal in both readings, by {(least - 0.95) * 100:.2f} points at the least."
        return f"That is a goal miss for the {card}'s V4 row" + (" with the F32 GEMMs uncredited." if cov4[card] >= 0.95
                                                                   else ".")

    L.append(f"\n{short[v4['name']]}'s static MACs are its pinned checkpoint's matmuls a token at their own shapes, "
             f"{v4['static'] / 1e9:.3f}e9 (ALGO-COST-3's A39, CMP-118 ecee1a19), every one listed and credited at "
             "its own shape: D6 v1's list left out the compressors, the indexer's compressor and query projection, "
             "the router and mHC's pre GEMMs (1.42e9), and had wkv at 576 columns, V4's being 512. The input "
             "embedding is a lookup and is credited no MACs; the stated 49e9 active parameters already exclude it "
             "and sit 0.31% above the real sum, so P24's gap was real matmuls, not the embedding. The world model "
             "credits V4's BF16 and F32 weights at INT8, as it does V4's BF16 head, an assumption. With mHC's F32 "
             "GEMMs uncredited, V4's coverage is "
             + " and ".join(f"{pct(x, 1)} ({card})" for card, x in no_f32.items()) + ". "
             + " ".join(v4_goal(card) for card in vc.CARDS))

    def g_max(designs):
        return max((r["world"]["gamma"], k) for k, r in rows.items() if k[0] in designs and k[2] == "measured")

    g, g_all = g_max((HEADLINE, "v3")), g_max(DESIGNS)
    L.append(f"\nP-HYB3's condition is a world-mix γ above 10 in the adopted protocol (v3 and {HEADLINE}). The largest "
             f"there, at measured hashing, is {g[0]:.2f} ({g[1][0]}, {g[1][1]}, {short[g[1][3]]}), so P-HYB3 "
             + ("is not needed." if g[0] <= 10 else "is needed.")
             + (f" Over every design it is {g_all[0]:.2f} ({g_all[1][0]}, {g_all[1][1]}, {short[g_all[1][3]]}), a "
                "sensitivity that isn't adopted." if g_all[1][0] not in (HEADLINE, "v3") else "") + "\n")
    a_h = rows[HEADLINE, "H200", "measured", models[0]]["answers"]
    v4h = {card: rows[HEADLINE, card, "measured", v4["name"]]["eps"]["g"] for card in vc.CARDS}
    L.append(f"### Protocol {HEADLINE} as D6 prices it (v3's spec, CMP-118 e0cd4061, with the 07:04Z rule's L′ and "
             "P21, P23, P25)\n")
    L.append("- **Words.** v2's: 16-bit prefix sums of the noised product every L = t·L′ inner indices, committed as "
             "SHA-512 hiding leaves (8 blocks, a 16-ary frame-v3-sha512 tree), O the last 32-bit word, bound to them. "
             "HYB's noise U, V uniform on ±80, U's columns freshly permuted every noise tile (t tiles a segment), and a "
             "1-bit dither on both sides. PV is credited with O bound and a fresh bit on P; QKᵀ and V4's indexer are "
             "served plain, uncredited. L′ per round class by the 07:04Z rule (the picks below).\n"
             "- **Credit.** pub(μ°) of every credited round: its least price over D5 v2.1's menu at the honest served "
             "period's duals μ°, which follow from the period's public shapes.\n"
             "- **Coins.** Live, from the challenger. A round is the calls whose coins go out together (P13). Every "
             "credited coin is drawn from seeds the network certifier streams (at most 512·N_max MACs a seed; "
             "07:04Z); weight-side coins may be reused across decode calls (P14, P17).\n"
             "- **The registered total (P21).** Before the period's first check, the prover registers the period's "
             "rounds (each round's registered instance, L27) and the verifier publishes their total C_reg = Σ_w "
             "H(reg w), at most the group's capacity at the card's rate. Each round w is then checked independently "
             "with probability π_w = k·H_w / C_reg (P15, P20's Poisson draw, H_w its public cost). The bound is ε·C_reg, "
             "so a prover that registers more than it serves only raises its own ε. Without the publication, the draw "
             "has to be normalised by the wall-clock period, and ε_S becomes a share of s·T (the ε table's variant "
             "column).\n"
             f"- **Groups (P23).** A group is a serving replica with its coins in lockstep, so its rounds are totally "
             f"ordered, and the theorem holds per group with a union bound over the world's G groups. A Llama replica "
             f"fits in a node, so the group is a node's {NODE_GPUS} GPUs, its replicas released together. V4-Pro's "
             f"{PARAMS[v4['name']] / 1e12:.1f}e12 INT8 weights need at least {v4h['H200']} H200s or "
             f"{v4h['RTX PRO 6000']} RTX PRO 6000s (before KV cache), so its group is that replica (the least, so the "
             "most groups: the larger ε), with whole nodes beside it in the P16 table. The network certifier's "
             "boundary goes around the whole group: every port out of it is behind the certifier, so the "
             "expert-parallel traffic inside it doesn't count against ε_net.\n"
             f"- **Timing (P25).** A period is T = {PERIOD_S / 86400:g} days. A round's commitment reaches the "
             f"certifier within Δ = {DELTA_S:g} s of its coins, and a draw's answer within τ_ans = {TAU_ANS_S:g} s: "
             f"one C-Flock --zk proof of the opened unit (a session's latency floor is about {ANS_FLOOR_S:g} s, its "
             f"GPU busy about {ANS_GPU_S:g} s: {ANS_RUN}), run beside the group's later rounds. A group's checks go "
             f"through the firewall one at a time, so k(Δ + τ_ans) ≤ T: {K_DRAWS:,} × {DELTA_S + TAU_ANS_S:g} s = "
             f"{K_DRAWS * (DELTA_S + TAU_ANS_S) / 86400:.2f} days, and {a_h['draws_hi']:,.0f} draws (the H200 8B "
             f"group's at its 1 − 1e-6 quantile) take {a_h['t_min_h'] / 24:.2f} days of the {PERIOD_S / 86400:g}. "
             f"The margin is thin: at that quantile Δ + τ_ans may be at most {PERIOD_S / a_h['draws_hi']:.1f} s, so "
             f"with Δ = {DELTA_S:g} s an answer has {PERIOD_S / a_h['draws_hi'] - DELTA_S:.1f} s against the proof's "
             f"{ANS_FLOOR_S:g} s floor; a slower answer needs a longer period (the union over periods then shrinks) "
             "or a shorter Δ. No coins go out in a period's last Δ + τ_ans.\n"
             "- **ε.** ε_S + ε_net at the model's group, times 1 + H_o/H for the honest answers (P25), with "
             "ln(P)/(ηk) added over a window's P periods (P26).\n")
    L.append(f"### The world (rule L2): pooled, every window, both δ, {HEADLINE}, V4 as the world's request (8B, 70B "
             "below)\n")
    L.append("**How to read the world rows (P22).** They are μ°-priced device capacity: credited is the honest device "
             "time the claim covers, valued at pub(μ°) and scaled to the world's C as FP8 FLOP-equivalents of that "
             "time. They are not FP8 FLOP of other work. εC bounds other work valued at the same μ° (at a fixed μ°, "
             "other work's μ°-price plus the claim is at most the period's makespan, times 1 + ε, whatever routes or "
             "mix the prover runs). At x = 0, μ° prices the H200's tensor pipe at 0 (its honest period binds the ALU "
             "and the FMA pipe) and prices the RTX PRO 6000's low (ALU and register file), so plain INT8 tensor work "
             "beside the credited rounds is valued at little or nothing: neither εC nor the uncredited column bounds "
             "it. The P22 table below gives that hidden work, and the one alternative, credit at a fixed served "
             "price μ¹ published as a protocol parameter.\n")
    L.append("Two tables in different units. C is the world's FP8 FLOP in the window (brief rule L2). What is scaled: "
             "the whole fleet is taken to be one card serving one model, so every figure is that card's share times "
             "C, and P16's groups are C's GPUs at the card's FP8 rate over the model's group (not a census). V4's "
             "static MACs are its checkpoint's matmuls, 48.85e9 a token at their own shapes (A39). εC is ε times the "
             "claim beside it: ε is a share of the claim at every dual (P20), at the model's group, with ε_net's MACs each at the dearest credited round's price at that "
             "table's μ over the period's mean, times 1 + H_o/H, plus the union over the window's periods (P26: "
             + ", ".join(f"{win} {WINDOW_S[win] / PERIOD_S:.2f}" for win in WINDOW_S)
             + " one-week periods). Credit at a period's duals adds up only if everything credited in the period is "
             "priced at those duals (ALGO-COST-3's A30b); D6 credits one workload a card in one LP, with the beside "
             "work uncredited, so it holds here.\n")

    def eps_win(r, d, win):
        e = r["eps"]
        return e["head"][d] + e["union"][win] * (1 + e["ho_claim"])

    def var_win(r, d, win):
        e = r["eps"]
        return e["no_c_reg"][d] + e["union"][win] * (1 + e["ho_claim"]) / r["share"]["share"]

    L.append("**Model units at μ°, the cards' headline.** Credited is honest device time at pub(μ°), the honest served "
             "period's duals, as FP8 FLOP-equivalents of that time (share × C).\n")
    L.append("| card, model | window | C (FP8 FLOP) | credited (FLOP-eq.) | εC δ 1% | εC δ 1e-6 | "
             "εC δ 1e-6 without C_reg (P21's variant) | honest time uncredited (FLOP-eq.) | "
             "QKᵀ + indexer + plain PV, uncredited MACs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in reversed(models):
            r = rows[HEADLINE, card, "measured", m]
            for win, F in WORLD_FLOP.items():
                s = r["share"]["share"]
                L.append(f"| {card}, {short[m]} | {win} | {F:.2e} | {s * F:.2e} | {eps_win(r, 0.01, win) * s * F:.1e} | "
                         f"{eps_win(r, 1e-6, win) * s * F:.1e} | {var_win(r, 1e-6, win) * s * F:.1e} | "
                         f"{(1 - s) * F:.1e} | {pct(r['world']['uncredited'], 1)} |")
    L.append("")
    B = D["beside"]

    def fixed_price(card, m, xp, design):
        """Credit at the fixed price μ_xp (the honest period's duals at beside load xp, with `design`'s picks),
        published: the credited share with the honest period at x = 0 and at x = 1, ε at δ 1e-6, and the plain INT8
        beside work it leaves unpriced, x′ (alone − μ_xp-price) a credited MAC over pub(μ_xp), at x′ in BESIDE. ε_S and
        the union take the headline's η, as every design's rows do (`build`)."""
        b0, bx, b1 = (B[design, card, m, x] for x in (0.0, xp, 1.0))
        pub, s0 = bx["pub"], b0["share0"]
        sh0, sh1 = s0 * pub / b0["pub"], s0 * pub / b1["opt"] * b0["opt"] / b0["pub"]
        e = rows[HEADLINE, card, "measured", m]["eps"]
        net = WIRE_BITS * nf.WORLD_TOKENS * bx["net_most"] / (b0["cred_macs"] * pub)
        ho = answer_time(card, model_of(m), b0) / sh0
        eps = (e["groups"][e["g"]]["eps_S"][1e-6] + net) * (1 + ho)
        hid = {x: x * (bx["unit"]["alone"] - bx["unit"]["at_served"]) / pub for x in BESIDE if x > 0}
        rtp = {x: (B[design, card, m, x]["fixed_alone"] - b0["fixed_alone"]
                   - (B[design, card, m, x]["served"] - b0["served"])) / b0["pub"] for x in BESIDE if x > 0}
        return dict(sh0=sh0, sh1=sh1, eps=eps, eps_year=eps + e["union"]["year"] * (1 + ho), hid=hid, rtp=rtp,
                    binding=bx["binding"])

    h, t = vc.CARDS
    tau1 = {card: best_tau(B, card, 1.0) for card in vc.CARDS}
    fp = {(card, m, xp): fixed_price(card, m, xp, HEADLINE if xp == 0 else tau1[card])
          for card in vc.CARDS for m in models for xp in (0.0, 1.0)}

    def tau_label(card):
        n = tau1[card]
        return f"{n} ({HEADLINE}'s picks)" if n == f"τ {D['rule']['tau'][card]:g}" else f"{n}'s picks"

    L.append("**P22: credit at a fixed served price.** The headline credits at μ°, the honest period's duals with only "
             "its own uncredited work beside (x = 0). The alternative publishes μ¹, the honest period's duals at x = 1 "
             "(one plain INT8 MAC of other tensor work a credited MAC beside the period), as a protocol parameter "
             "and credits pub(μ¹) whatever the period holds. Its picks are the rule's at x = 1, the τ whose picks give "
             "the largest mean credited share with the honest period at x = 1 (on D5 v2.1 τ 0.2 wins only at x = 0, "
             f"ALGO-COST-3's A38): {tau_label(h)} on the {h} and {tau_label(t)} on the {t}, with μ¹ that τ's own "
             "period's duals. ε holds at either fixed price (P15/P20). Hidden beside "
             "work is plain INT8 tensor work beside the credited rounds that the price doesn't value: x (its "
             "standalone time − its price) a credited MAC, over the claim. RT-PROTO-2's measure (its standalone time "
             "less the honest period's growth, over the claim at μ°) is beside it for μ°. A year's figures are "
             "FLOP-equivalents at that row's credited share with the honest period at x = 0.\n")
    L.append("| card, model | credit at | binding rows at that price | credited share, honest period at x = 0 / "
             "x = 1 | ε δ 1e-6, a period | hidden beside work, x = "
             + " / ".join(f"{x:g}" for x in BESIDE if x > 0) + " | RT-PROTO-2's measure, x = "
             + " / ".join(f"{x:g}" for x in BESIDE if x > 0)
             + " | a year: credited / εC δ 1e-6 / hidden at x = 0.25 (FLOP-eq.) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            for xp in (0.0, 1.0):
                f = fp[card, m, xp]
                lab = "μ° (the headline)" if xp == 0 else f"μ¹ (x = 1, published), {tau_label(card)}"
                cy = f["sh0"] * WORLD_FLOP["year"]
                L.append(f"| {card}, {short[m]} | {lab} | {', '.join(f['binding'])} | {pct(f['sh0'], 1)} / "
                         f"{pct(f['sh1'], 1)} | {pct(f['eps'], 3)} | "
                         + " / ".join(pct(v, 1) for v in f["hid"].values()) + " | "
                         + (" / ".join(pct(v, 1) for v in f["rtp"].values()) if xp == 0 else "—")
                         + f" | {cy:.2e} / {f['eps_year'] * cy:.1e} / {f['hid'].get(0.25, 0.0) * cy:.1e} |")
    L.append("")

    def span(vals, d=0):
        lo, hi = min(vals), max(vals)
        return pct(lo, d) if f"{lo:.{d + 2}f}" == f"{hi:.{d + 2}f}" else f"{pct(lo, d)}–{pct(hi, d)}"

    def unit_ratio(card, x):
        d = HEADLINE if x == 0 else tau1[card]
        return [B[d, card, m, x]["unit"]["at_served"] / B[d, card, m, x]["unit"]["alone"] for m in models]

    def removed(card):
        return [1 - fp[card, m, 1.0]["hid"][1.0] / fp[card, m, 0.0]["hid"][1.0] for m in models]

    def cost(card):
        return [fp[card, m, 1.0]["sh0"] / fp[card, m, 0.0]["sh0"] for m in models]

    def pub_ratio(card):
        return [B[tau1[card], card, m, 1.0]["pub"] / B[tau1[card], card, m, 0.0]["pub"] for m in models]

    def picks_cost(card):
        if tau1[card] == f"τ {D['rule']['tau'][card]:g}":
            return ""
        s = [B[tau1[card], card, m, 0.0]["share0"] / B[HEADLINE, card, m, 0.0]["share0"] for m in models]
        return f"; on the {card} the rest is {tau1[card]}'s picks, whose share at x = 0 is {span(s, 1)} of {HEADLINE}'s"

    L.append(f"A plain INT8 MAC beside the claim is valued at {span(unit_ratio(h, 0.0))} of its standalone time by "
             f"μ° on the {h} and {span(unit_ratio(t, 0.0))} on the {t}, and at {span(unit_ratio(h, 1.0))} and "
             f"{span(unit_ratio(t, 1.0))} by μ¹, where the joint facets bind. So μ¹ removes {span(removed(t))} of the "
             f"{t}'s hidden work and {span(removed(h))} of the {h}'s, whose ALU still binds at μ¹. The cost is the "
             f"credited share with nothing beside (x = 0): the μ¹ row's is {span(cost(h), 1)} of the headline's on "
             f"the {h} and {span(cost(t), 1)} on the {t}. At the μ¹ row's picks pub(μ¹) is {span(pub_ratio(h), 1)} of "
             f"pub(μ°) on the {h} and {span(pub_ratio(t), 1)} on the {t}, because μ¹ values the credited rounds' lane "
             f"work below μ°{picks_cost(h)}{picks_cost(t)}. On the {t}, μ¹ is an x = 1 dual of θ's joint issue rows, "
             "so its row carries the x > 0 cells' caveat (the beside table below). With the "
             "honest period at x = 1, credit at μ° reads above its x = 0 share: the credited rounds' own price falls "
             "at μ¹'s duals while their credit stays at pub(μ°), so the rise is the beside work's time counted in the "
             "claim (the hidden work again), not a gain. Whether the claim moves to μ¹ is Daniel's call (19:00Z); the "
             "coordinator recommends it.\n")
    L.append("**Tensor units at μ_t, a labelled sensitivity, not a claim (12:02Z reversed 08:24Z's tensor-unit "
             "claim).** Credited FP8 FLOP is twice the tensor MACs of "
             "the cheapest public route at the tensor-only price μ_t: every lane is free, so it counts only work the "
             "tensor pipe must do however long the lanes take. Every product in D5's menu may run on the lanes "
             "(DP4A, the integer ALU), so where that column is 0, no tensor work is forced at all. A nonzero tensor "
             "claim needs a deadline: the bounded columns are pub's least tensor with the period held to OPT*'s served "
             "makespan × (1 + s). That deadline is not in the protocol or in P20's theorem, so they are sensitivities "
             "with no ε. The MMA-only column is `tile_job`'s menu, whose peels and zero-high-digit peel have no lane "
             "route (08:24Z: price them or name the gap); D6 prices them at D5's own DP4A load.\n")
    sl = BOUND_SLACK
    L.append("| card, model | window | credited FP8 FLOP at μ_t | εC δ 1% / 1e-6 | peels on the MMA only | "
             + " | ".join(f"deadline OPT* × {1 + s:g}" for s in sl) + " | the deployed kernel's tensor FP8 FLOP |")
    L.append("|---|---|---|---|---|" + "---|" * len(sl) + "---|")
    for card in vc.CARDS:
        for m in reversed(models):
            r = rows[HEADLINE, card, "measured", m]
            t, a = r["share"]["tensor"], r["eps"]["at"]["tensor"]
            for win, F in WORLD_FLOP.items():
                ec = (f"{a['eps'][0.01] * t['share'] * F:.1e} / {a['eps'][1e-6] * t['share'] * F:.1e}" if a
                      else "0 / 0")
                L.append(f"| {card}, {short[m]} | {win} | {t['share'] * F:.2e} | {ec} | "
                         f"{mma_t(r) * F:.2e} | "
                         + " | ".join(f"{t['bounded'][s] * F:.2e}" for s in sl) + f" | {t['honest'] * F:.2e} |")
    L.append("")
    L.append("The same as shares of C, beside the model-unit share, with ε at each μ (the model's group, δ 1e-6, before "
             "the answers' 1 + H_o/H; in brackets ε_net's scale over the equal-price n_max E / credited MACs). OPT* is "
             "the honest words' best program.\n")
    L.append("| card, model | model units at μ° | ε at μ° | μ_t: pub | μ_t: OPT* | ε at μ_t | peels on the MMA only | "
             + " | ".join(f"deadline × {1 + s:g}" for s in sl) + " | deployed kernel's tensor |")
    L.append("|---|---|---|---|---|---|---|" + "---|" * len(sl) + "---|")

    def eps_at(a):
        return f"{pct(a['eps'][1e-6], 3)} (×{a['net_scale']:.2f})" if a else "— (claim 0)"

    for card in vc.CARDS:
        for m in models:
            r = rows[HEADLINE, card, "measured", m]
            t, a = r["share"]["tensor"], r["eps"]["at"]
            L.append(f"| {card}, {short[m]} | {pct(r['share']['share'], 1)} | {eps_at(a['served'])} | "
                     f"{pct(t['share'], 1)} | {pct(t['opt'], 1)} | {eps_at(a['tensor'])} | "
                     f"{pct(mma_t(r), 1)} | " + " | ".join(pct(t["bounded"][s], 1) for s in sl)
                     + f" | {pct(t['honest'], 1)} |")
    L.append("")
    L.append("### ε (goal ≤ 1% at δ = 1e-6)\n")
    L.append("ε = (ε_S + ε_net)(1 + H_o/H), a period's, as a share of the claim at μ°. P16 (07:04Z): the theorem holds "
             "per serving-replica group (coins in lockstep inside a group), with a union bound over the world fleet's "
             f"G groups, so ε_S = ln(G/δ)/(ηk) at k = {K_DRAWS:,} a group; G is the world model's GPUs at the card's "
             "FP8 rate over the model's group (P23; scaled from one card, not a census); other group sizes are in the "
             "P16 table. ε_net: RT-PROTO-2's assumptions (n_max the largest credited N, 4 bytes a token crossing the "
             "group's boundary, 4,500-token requests; ×6.5 if HTTP/SSE crosses it), each MAC at its round's price at "
             "μ° (in brackets, the scale over the equal-price n_max E / credited MACs). H_o/H is the timed answers' "
             "time over the claim (P25). P21's variant is ε with the draw normalised by the wall-clock period "
             "instead of the published C_reg: ε_S/s + ε_net, s the credited share (RT-PROTO-2: 0.42 / 0.38 / 0.37% "
             "H200, 0.32 / 0.27 / 0.28% RTX PRO 6000 at v1's figures). P26 adds ln(P)/(ηk) over a window's P one-week "
             "periods: " + ", ".join(f"{win} +{pct(math.log(WINDOW_S[win] / PERIOD_S) / (rows[HEADLINE, 'H200', 'measured', models[0]]['eps']['eta'] * K_DRAWS), 3)}"
                                     for win in WINDOW_S) + " (times 1 + H_o/H). V4's node-boundary column is ε_net with "
             f"the certifier's boundary at one node instead of the group, where expert-parallel traffic (about "
             f"{EP_BITS:.0e} bits a token, RT-PROTO-2's estimate) crosses it: vacuous, so the boundary goes around the "
             "group.\n")
    L.append("| card, model | η | GPUs a group (G) | k a group (expected draws, P20) | ε_S δ 1% / 1e-6 | ε_net at μ° | "
             "H_o/H | ε δ 1% | **ε δ 1e-6** | P21's variant (no C_reg), δ 1e-6 | one sequence (no P16), δ 1e-6 | "
             "a year (P26), δ 1e-6 | ε_net, boundary at a node |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            r = rows[HEADLINE, card, "measured", m]
            e = r["eps"]
            gg = e["groups"][e["g"]]
            a = e["at"]["served"]
            L.append(f"| {card}, {short[m]} | 1 − {1 - e['eta']:.1e} | {e['g']} ({gg['G']:.2e}) | {e['k']:,} "
                     f"({e['draws']:,.0f}) | {pct(gg['eps_S'][0.01], 3)} / {pct(gg['eps_S'][1e-6], 3)} | "
                     f"{pct(a['eps_net'], 4)} (×{a['net_scale']:.2f}) | {pct(e['ho_claim'], 2)} | "
                     f"{pct(e['head'][0.01], 3)} | **{pct(e['head'][1e-6], 3)}** | {pct(e['no_c_reg'][1e-6], 3)} | "
                     f"{pct(e['eps'][1e-6], 3)} | {pct(eps_win(r, 1e-6, 'year'), 3)} | "
                     + (pct(e["eps_net_node"], 0) if e["eps_net_node"] else "— (the group is a node)") + " |")
    L.append("")
    L.append("### The timed answers (P25), at each model's group\n")
    L.append(f"T = {PERIOD_S / 86400:g} days, Δ = {DELTA_S:g} s, τ_ans = {TAU_ANS_S:g} s, {ANS_GPU_S:g} GPU-s an "
             f"answer ({ANS_RUN}). The draws must fit the period one at a time, N(Δ + τ_ans) ≤ T; RT-PROTO-2's bound is "
             f"T ≥ {K_DRAWS / 3600:.2f} h per second of Δ + τ_ans. The answers' cost is in γ and the credited share (the "
             "answers' GPU time over the group's) and in ε (over the claim). If each answer stalled the whole group "
             "instead of running beside its later rounds, it would cost the share in the stall column. Rounds in "
             "the period's last Δ + τ_ans go uncredited.\n")
    L.append("| card, model | GPUs a group | draws a period, mean / 1 − 1e-6 | N(Δ + τ_ans), a share of T | "
             "H_o/H in time / over the claim | γ without / with the answers | if answers stalled the group | "
             "uncredited tail |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            r = rows[HEADLINE, card, "measured", m]
            a, e = r["answers"], r["eps"]
            L.append(f"| {card}, {short[m]} | {e['g']} | {a['draws']:,.0f} / {a['draws_hi']:,.0f} | "
                     f"{pct(a['busy'], 1)}{'' if a['busy'] <= 1 else ' (does not fit)'} | "
                     f"{pct(r['share']['answers'], 3)} / {pct(e['ho_claim'], 3)} | "
                     f"{r['world']['gamma']:.3f} / {gamma_ans(r):.3f} | {pct(a['serial'], 1)} | {pct(a['tail'], 4)} |")
    L.append("")
    L.append("### γ (goal ≤ 10), credited share and coverage (goal ≥ 95%)\n")
    L.append("γ here is the serving overhead before the timed answers (the goals table adds their H_o/H, at most "
             f"{pct(max(r['share']['answers'] for r in rows.values()), 2)}); the credited share includes them.\n")
    L.append("| design, card, SHA-512 rate | γ overlapped | γ additive | worst round's γ | tensor util. | "
             "credited share (served duals) | routes uncredited, period / job | coverage of matmul MACs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for design in sorted({k[0] for k in rows}, key=DESIGNS.index):
        for card in vc.CARDS:
            if design in RTX_ONLY and card != "RTX PRO 6000":
                continue
            for rate in ("measured", "paper"):
                rr = [rows[design, card, rate, m] for m in models]
                sha = SHA_MEASURED[card][0] if rate == "measured" else nf.V2_CARDS[card][1]
                L.append(f"| {design}, {card}, {sha / 1e9:.2f} G/s ({rate}) | "
                         + " / ".join(f"{r['world']['gamma']:.2f}" for r in rr) + " | "
                         + " / ".join(f"{r['world']['gamma_add']:.2f}" for r in rr) + " | "
                         + " / ".join(f"{r['worst']['gamma']:.1f} ({r['worst']['where']})" for r in rr) + " | "
                         + " / ".join(f"{r['world']['util']:.0%}" for r in rr) + " | "
                         + (" / ".join(pct(r["share"]["share"], 1) for r in rr) if rate == "measured" else "—")
                         + " | " + " / ".join(pct(r["period"]["leak_served"]) for r in rr) + "; job "
                         + " / ".join(pct(r["period"]["leak_job"]) for r in rr) + " | "
                         + " / ".join(pct(1 - r["world"]["uncredited"], 1) for r in rr) + " |")
    L.append("")
    L.append("The paper-rate rows are the same period LP as the measured rows: the LP hashes at every schedule θ admits "
             "(A27b), so one LP serves both rates and the routes column is that LP's. This is how D6 prices the RTX PRO "
             "6000's paper-rate 70B row as one period LP (PROTO-DESIGN-3's open item). nofold's world-level 1.13% there "
             "was a per-round ε_alg; under credit at pub a route's saving is uncredited honest time, never ε. The "
             "paper-rate share is not printed, because credit is per second of the measured honest kernel and the "
             "paper schedules are unbuilt.\n")
    L.append("### The credited share, decomposed (measured hashing; factors multiply to the share)\n")
    fac = list(next(iter(rows.values()))["share"]["factors"])
    L.append("| card, model | " + " | ".join(fac) + " | share | with hosts in C (U9) | PD3's r-reading of θ's hashing |")
    L.append("|---|" + "---|" * (len(fac) + 3))
    for card in vc.CARDS:
        for m in models:
            s = rows[HEADLINE, card, "measured", m]["share"]
            L.append(f"| {card}, {short[m]} | " + " | ".join(f"{s['factors'][k]:.3f}" for k in fac)
                     + f" | {pct(s['share'], 1)} | {pct(s['with_hosts'], 1)} | {s['pd3_r_reading']:.3f} |")
    L.append("")
    L.append("### The 07:04Z rule: credited share at served duals (credited value per honest second) by τ\n")
    L.append("Per round class, the least-γ L′ with γ ≤ 10 and job-only leak ≤ τ (`rule_pick`); round classes with no "
             "L′ at γ ≤ 10 are credited at their least-γ L′ only where that adds credited value per honest second "
             "(`value_test`). v3.1 takes, per card, the τ of the largest mean share (ties within 0.1 point: lower γ).\n")
    L.append("| card | τ | credited share | γ overlapped | coverage | PV classes with no L′ at γ ≤ 10 / served plain |")
    L.append("|---|---|---|---|---|---|")
    for (card, tau), x in D["rule"]["sweep"].items():
        mark = " (v3.1)" if D["rule"]["tau"][card] == tau else ""
        L.append(f"| {card} | {tau:g}{mark} | " + " / ".join(pct(v, 1) for v in x["share"]) + " | "
                 + " / ".join(f"{v:.2f}" for v in x["gamma"]) + " | " + " / ".join(pct(v, 1) for v in x["cov"])
                 + f" | {x['infeasible']} / {x['plain']} |")
    L.append("")
    L.append("### A29: the served duals at a stated beside load\n")
    L.append("The rows above price the period with only its own uncredited work beside (QKᵀ's two softmax passes, "
             "V4's indexer, plain PV). There the H200's honest period is bound by ALU and the RTX PRO 6000's by ALU "
             "and register file, so the routes' tensor saving is worth nothing. x more plain INT8 MACs of tensor work a "
             "credited MAC (uncredited decode, other jobs) move the duals onto the joint facet and tensor. Cells: "
             "credited share (routes uncredited at those duals), 8B / 70B / V4. The share is x = 0's with its routes "
             "factor moved to x's (beside work overlaps the credited rounds, whose OPT* price at the period's duals "
             "is at least the period time they add); the value test stays at x = 0. `v3.1, RTX PRO 6000 PV L′ 64` is "
             "PB5's case: v3.1's static picks and the RTX PRO 6000's PV at L′ 64 wherever its round's γ ≤ 10 (else "
             "v3.1's pick). At small x the routes' figure at μ° is mostly μ°'s duality gap in the hull's LP, not "
             "makespan the routes save (ALGO-COST-3's A30a: OPT(μ) − pub(μ) = S + gap_hull(μ) − gap_OPT(μ), "
             "gap_OPT(μ°) = 0); PB4's table below gives S.\n")
    B = D["beside"]
    names = list(dict.fromkeys(k[0] for k in B))
    L.append("| picks, card | " + " | ".join(f"x = {x:g}" for x in BESIDE) + " |")
    L.append("|---|" + "---|" * len(BESIDE))
    for card in vc.CARDS:
        for name in names:
            if (name, card, models[0], 0.0) not in B:
                continue
            L.append(f"| {name}, {card} | " + " | ".join(
                " / ".join(pct(B[name, card, m, x]["share"], 1) for m in models) + " ("
                + " / ".join(pct(B[name, card, m, x]["leak"], 1) for m in models) + ")" for x in BESIDE) + " |")
    L.append("\nOn the RTX PRO 6000 these cells carry route (iii), b cᵀ with b1 emulated on the lanes, which D5 v2.1 "
             "prices from the sm_120 census (ALGO-COST-3's A35, A38). Its x > 0 cells are still upper bounds: beside "
             "work moves the period onto the joint issue rows (MMA beside lane work), and θ's sm_120 joint rows are "
             "closed rows (A36), so the cells stand as bounds until CALIB moves θ (`theta.toml`) to a current vector, "
             "GPU work after M3. On the H200 b1 is native.")

    def census_rows(rows, label):
        cz = {k if k.count(" | ") == 3 else f"{HEADLINE} | {k}": v for k, v in rows.items()}
        L.append("| picks, card | " + " | ".join(f"x = {x:g}" for x in BESIDE) + " |")
        L.append("|---|" + "---|" * len(BESIDE))
        card = "RTX PRO 6000"
        for name in dict.fromkeys(k.split(" | ")[0] for k in cz):
            def cen(m, x, name=name):
                return cz[f"{name} | {card} | {m} | {x}"]

            L.append(f"| {name}, {card}, {label} | " + " | ".join(
                " / ".join(pct(B[name, card, m, x]["share0"] * (1 - cen(m, x)["leak_served"])
                               / (1 - B[name, card, m, x]["leak0"]), 1) for m in models) + " ("
                + " / ".join(pct(cen(m, x)["leak_served"], 1) for m in models) + ")" for x in BESIDE) + " |")
            L.append(f"| {name}, {card}, {label}: 1 − c_inc | " + " | ".join(
                " / ".join(pct(1 - cen(m, x)["c_inc"], 1) for m in models) for x in BESIDE) + " |")

    if "census_vector" in D:
        cv = D["census_vector"]
        L.append(f"\nThe same with the RTX PRO 6000's joint issue rows (MMA beside lane work) from {cv['vector']}, the "
                 "machine model's vector, at IMMA16832's coefficient in D5's form, instead of θ's JI3e_0–2, which "
                 "D5's card prices and which the machine model closed at VF3n-16 (ALGO-COST-3's A36's swap, on D5 "
                 f"v2.1's menu with route (iii) in it; {cv['ref']}). A sensitivity: D5 v2.1's card is the hull of "
                 "record (08:24Z).\n")
        census_rows(cv["rows"], cv["vector"])
        card = "RTX PRO 6000"
        side = {"lower": [], "higher": []}
        for x in BESIDE:
            if x > 0:
                d = np.mean([B[HEADLINE, card, m, x]["share0"] * (1 - cv["rows"][f"{HEADLINE} | {card} | {m} | {x}"]["leak_served"])
                             / (1 - B[HEADLINE, card, m, x]["leak0"]) - B[HEADLINE, card, m, x]["share"]
                             for m in models])
                side["lower" if d < 0 else "higher"].append(f"{x:g}")
        L.append(f"\nOn {cv['vector']}, {HEADLINE}'s x > 0 cells read "
                 + " and ".join(f"{k} than θ's at x = {', '.join(v)}" for k, v in side.items() if v)
                 + f". {cv['vector']} isn't current either, so it doesn't settle the bound's direction; CALIB's vector "
                 "does.")
    L.append("\nPB4: the same periods by makespan, which needs no duals. Cells: 1 − c_inc, the share of the credited "
             "rounds' honest makespan increment over the period's fixed load alone that pub's period saves, 8B / 70B / "
             "V4, beside the dual-price figure above. Credit stays at the duals (P12); c_inc shows how much of that "
             "figure is a marginal price on slack the period still has.\n")
    L.append("| picks, card | " + " | ".join(f"x = {x:g}" for x in BESIDE) + " |")
    L.append("|---|" + "---|" * len(BESIDE))
    for card in vc.CARDS:
        for name in ("v3", HEADLINE, *RTX_ONLY, "dual-safe"):
            if (name, card, models[0], 0.0) in B:
                L.append(f"| {name}, {card} | " + " | ".join(
                    " / ".join(pct(1 - B[name, card, m, x]["c_inc"], 1) for m in models) for x in BESIDE) + " |")
    L.append("\nv3.1's PV picks priced as a whole (08:24Z), beside its static rounds, in the same period LP and at the "
             "same duals: each kind's share of the credited MACs, the routes' leak over that kind's rounds at x = 0 "
             "and x = 1 (pub over OPT* at the period's μ°), and its pub at μ_t (tensor MAC-times a credited MAC). "
             "8B / 70B / V4.\n")
    L.append("| picks, card, kind | credited MACs | leak x = 0 | leak x = 1 | pub at μ_t |")
    L.append("|---|---|---|---|---|")

    def kind_leak(name, card, kind, x):
        return " / ".join("—" if B[name, card, m, x]["leak_kind"][kind] is None
                          else pct(B[name, card, m, x]["leak_kind"][kind]) for m in models)

    for card in vc.CARDS:
        for name in (HEADLINE, "v3.1, RTX PRO 6000 PV L′ 64"):
            if (name, card, models[0], 0.0) not in B:
                continue
            for kind in ("static", "pv"):
                pers = [rows[name, card, "measured", m]["period"] for m in models]
                L.append(f"| {name}, {card}, {kind} | " + " / ".join(pct(p["hull"][kind]["w"], 1) for p in pers)
                         + f" | {kind_leak(name, card, kind, 0.0)} | {kind_leak(name, card, kind, 1.0)} | "
                         + " / ".join(
                             f"{p['hull'][kind]['at_tensor'] / p['hull'][kind]['w']:.3f}" if p["hull"][kind]["w"]
                             else "—" for p in pers) + " |")
    for card in vc.CARDS:
        best = {x: best_tau(B, card, x) for x in BESIDE}
        bind = {x: B[HEADLINE, card, models[0], x]["binding"] for x in BESIDE}
        L.append(f"\n{card}: the rule's τ by load: " + "; ".join(f"x {x:g} → {best[x][2:]}" for x in BESIDE)
                 + f". {HEADLINE}'s 8B period binds " + "; ".join(f"x {x:g}: {', '.join(bind[x])}" for x in BESIDE)
                 + ".")
    L.append("")
    L.append("### v3.1's picks against v3's (P19): L′ and the round's γ at measured hashing\n")
    L.append("Static, per model's shapes (K×N: v3 L′ → v3.1 L′, γ v3 → v3.1):\n")
    pk = D["picks"]
    for card in vc.CARDS:
        for m in models:
            cells = [f"{K}×{N}: {pk['v3'][card, K, N]['Lp']} → {pk[HEADLINE][card, K, N]['Lp']}, "
                     f"{pk['v3'][card, K, N]['gamma']:.2f} → {pk[HEADLINE][card, K, N]['gamma']:.2f}"
                     for K, N, _ in nf.SHAPES[m]]
            L.append(f"- {card}, {short[m]}: " + "; ".join(cells))
    L.append("\nPV, per attended length n (rounded to 64), runs of n with one L′ (v3's L′ ⇒ v3.1's; γ at the run's "
             "ends):\n")
    for card in vc.CARDS:
        for dv in sorted({m["dv"] for m in nf.WORLD}):
            ns = sorted(k[1] for k in pk[HEADLINE] if k[0] == card and k[2] == dv and k in pk["v3"]
                        and k[1] % 64 == 0 and pk[HEADLINE][k] is not None)
            runs, cur = [], None
            for n in ns:
                a, b = pk["v3"][card, n, dv], pk[HEADLINE][card, n, dv]
                tag = (a["Lp"], b["Lp"])
                if cur and cur[0] == tag:
                    cur[2], cur[4] = n, (a["gamma"], b["gamma"])
                else:
                    cur = [tag, n, n, (a["gamma"], b["gamma"]), (a["gamma"], b["gamma"])]
                    runs.append(cur)
            if not runs:
                continue
            p10 = [n for n in ns if pk["v3.1, PV γ ≤ 10"][card, n, dv] is not None]
            plain = sorted(k[1] for k in pk[HEADLINE] if k[0] == card and k[2] == dv and pk[HEADLINE][k] is None)
            L.append(f"- {card}, d_v {dv}" + (f" (served plain at n {plain[0]}–{plain[-1]}, {len(plain)} lengths)"
                                              if plain else "") + ": " + "; ".join(
                f"n {lo}–{hi}: {ta} ⇒ {tb} (γ {g0[0]:.1f}→{g0[1]:.1f} … {g1[0]:.1f}→{g1[1]:.1f})"
                for (ta, tb), lo, hi, g0, g1 in runs)
                + f". Round γ ≤ 10 from n {min(p10) if p10 else '—'}.")
    L.append("")
    L.append("### P12 re-raised: what the routes free at each single-row dual, over the period (makespan ≤ OPT*'s)\n")
    L.append("Cost-weighted over the period's credited rounds. Under credit at pub this is honest time left uncredited "
             "at that dual (P20's C(μ) is lower there), not ε. Largest row, then tensor and issue: measured against "
             "OPT*'s own least load at its makespan, the issue residual RT-PROTO-2 saw (1.56%, against OPT*'s one LP "
             "solution) is LP degeneracy (issue doesn't bind; ALGO-COST-3's A27).\n")
    L.append("| design, card | 8B | 70B | V4 |")
    L.append("|---|---|---|---|")
    for design in ("v3", HEADLINE, "v3.1, PV L′ ≤ n/32 from n 2048", "dual-safe"):
        for card in vc.CARDS:
            cells = []
            for m in models:
                f = rows[design, card, "measured", m]["period"]["freed"]
                top = max(f, key=f.get)
                cells.append(f"{top} {pct(f[top])}; tensor {pct(f.get('tensor', 0.0))}; issue "
                             f"{pct(f.get('issue', 0.0), 3)}")
            L.append(f"| {design}, {card} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### P16: per replica group (union bound), G from the world model at the card's FP8 rate (scaled)\n")
    L.append("V4's groups, with each model's own marked (P23: the Llamas' a node, V4's its least replica; whole nodes "
             "beside it).\n")
    L.append(f"| card | GPUs a group | G | ε_S δ 1% / 1e-6 at k = {K_DRAWS:,} a group | k a group for ε_S 0.1% at 1e-6 | "
             "world draws a period (V4) |")
    L.append("|---|---|---|---|---|---|")
    for card in vc.CARDS:
        own = {}
        for m in models:
            own.setdefault(rows[HEADLINE, card, "measured", m]["eps"]["g"], []).append(short[m])
        for g, x in rows[HEADLINE, card, "measured", models[-1]]["eps"]["groups"].items():
            mark = f" ({', '.join(own[g])}'s)" if g in own else ""
            L.append(f"| {card} | {g}{mark} | {x['G']:.2e} | {pct(x['eps_S'][0.01], 3)} / "
                     f"{pct(x['eps_S'][1e-6], 3)} | {x['k_group']:,} | {x['draws_world']:.2e} |")
    L.append("")
    L.append("### Streaming b (P2): what the network certifier sends, measured honest rate, 100% of time credited\n")
    L.append("| card, model | option | GB/s a GPU | GB/s an 8-GPU node | n_max | ε_net |")
    L.append("|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            for opt, x in rows[HEADLINE, card, "measured", m]["streaming"].items():
                L.append(f"| {card}, {short[m]} | {opt} | {x['gpu'] / 1e9:.2f} | {x['node'] / 1e9:.1f} | "
                         + (f"{x['n_max']:,.0f} | {pct(x['eps_net'], 4)} |" if x["n_max"] else "— | — |"))
    L.append("")
    L.append(f"### Decode under {HEADLINE} (static rounds at its prefill picks, measured hashing): γ fresh / reused "
             "weight-side coins; roofline\n")
    L.append("The roofline is time over the plain decode's, the plain decode reading INT8 weights from HBM. Reused "
             "coins are stored, 1/8 B a weight read with the weights, so a fresh coin hidden under the weight read "
             "can beat a reused one (RTX PRO 6000, m ≤ 32).\n")
    L.append("| card, model | m = 1 | m = 32 | m = 256 |")
    L.append("|---|---|---|---|")
    for card in vc.CARDS:
        for m in models:
            cells = [D["decode"][card, m, b] for b in (1, 32, 256)]
            L.append(f"| {card}, {short[m]} | " + " | ".join(
                f"{c['fresh']:.2f} / {c['reused']:.2f}; roof {c['roof_fresh']:.2f} / {c['roof_reused']:.2f}"
                for c in cells) + " |")
    L.append("")
    L.append("### Perplexity (goal < 5%; H1, " + PPL_ART + ")\n")
    L.append("| model | v3 (both sides dithered, PV's fresh bit on P, one block) | static only (design A) |")
    L.append("|---|---|---|")
    for m in models:
        L.append(f"| {m} | {PPL[m][0]} | {PPL[m][1]} |")
    L.append("")
    L.append("### η: the measured free-guess rate (A = B = 0, U and V uniform on ±80)\n")
    for card, e in D["eta"].items():
        s, sv = e["segment"], e["served"]
        L.append(f"- {card}: the most predictable unit is T_1 at L = {e['least_L']['L']} (t {e['least_L']['t']}, "
                 f"L′ {e['least_L']['Lp']}, K {e['least_L']['K']}): p ≤ {s['p_max_hi']:.2e} ({s['samples']:,} cells, "
                 f"bins of {s['bin_width']}; uniform {s['uniform']:.2e}); served-like data (sd 30.7) {sv['p_max']:.2e}. "
                 f"O's 32-bit pair alone would be {e['o_pair_alone']['p']:.4f} (K {e['o_pair_alone']['K']}), its high "
                 f"word alone 1, so O opens with T_S.")
    return "\n".join(L)


def jsonable(x):
    if isinstance(x, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


if __name__ == "__main__":
    arg = {k: sys.argv[sys.argv.index(k) + 1] for k in ("--json", "--pickle", "--from", "--census-vector",
                                                         "--census-vector-ref") if k in sys.argv}
    D = pickle.loads(Path(arg["--from"]).read_bytes()) if "--from" in arg else build()
    if "--census-vector" in arg:
        cv = json.loads(Path(arg["--census-vector"]).read_text())
        D["census_vector"] = dict(rows=cv["rows"], vector=cv["vector"],
                                  ref=arg.get("--census-vector-ref", "unpublished"))
    if "--pickle" in arg:
        Path(arg["--pickle"]).write_bytes(pickle.dumps(D))
    if "--json" in arg:
        Path(arg["--json"]).write_text(json.dumps(jsonable(D), indent=1, default=str))
    print(report(D))
