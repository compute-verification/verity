"""D6, the number table for protocol v3 (D1, CMP-118 e0cd4061) and v3.1, printed from code (PROTO-DESIGN-4).

Credit is pub, the public best at the period's dual prices (06:10Z, 06:37Z, 07:04Z): one LP over every credited round
of the period with every public route (`d5_routes.period`), priced at the honest served period's duals, so a winning
route lowers pub and leaves honest time uncredited, never ε. v3.1 is v3 with L′ chosen by the coordinator's 07:04Z
rule, taken at the period (`rule`). ε, in model units, is the sampling term (P15/P20's Poisson draw, η from the
measured free-guess rate, P16's replica groups) plus ε_net (P18). γ is from measured hashing, with the paper ceiling
beside it.

    PUBLIC_BEST_DIR=<checkout of cursor/compute-research-accel-bet-algo-lab-ea24>/benchmarks/pouw/theory \\
        python3 benchmarks/pouw/accel_score/d6.py [--json OUT]
"""
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d5_routes as dr  # noqa: E402
import nofold as nf  # noqa: E402
import v3_caps as vc  # noqa: E402

pb = dr.pb

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
           "v3.1, RTX PRO 6000 static L′ 64", "dual-safe")


def quiet(f, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **kw)


def job_methods(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True, paper: bool = False):
    """D5's job at a pick with every route (d5_routes.methods), hashing at nofold's word model. paper: θ's SHA-512
    menu with the unmeasured IMAD.HI schedule in it (the paper ceiling); else the measured schedules only."""
    r = nf.v2_round(card, K, N, t * Lp, 16, tiles=t, weight_coins=weight_coins)
    c = vc.CARDS[card][0]
    _, bit, depth = pb.BIT_COST[c.family][0]
    m = dr.methods(c, t, Lp, K, N, bit, depth, nf.V_WIDTH[card], 1.0, digits=r["digits"], comps=r["comps"])
    if paper:
        name, w, _ = m["sha"]
        m["sha"] = (name, w, {s: pb.sha_loads(c, s) for s in pb.SHA_SCHEDULES[c.family]})
    return c, m, r


_FREED: dict = {}


def freed(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool = True) -> dict:
    key = (card, K, N, t, Lp, weight_coins)
    if key not in _FREED:
        c, m, _ = job_methods(card, K, N, t, Lp, weight_coins)
        _FREED[key] = {r: x[0] for r, x in dr.freed(c, m).items()}
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
_GRID: dict = {}


def grid(card: str, K: int, N: int, weight_coins: bool = True) -> list[dict]:
    """Every L′ on v3's grid (t·L′ ≤ K, under the card's limit) for one round class, with its job-only leak
    (`v3_caps.leak`: D5's LP at the round's makespan) and its γ at measured hashing (overlapped)."""
    key = (card, K, N, weight_coins)
    if key not in _GRID:
        _, t, lmax, _ = vc.CARDS[card]
        t = t if t * vc.LPS[0] <= K else 1
        _GRID[key] = []
        for Lp in [x for x in vc.LPS if t * x <= K and (lmax is None or x <= lmax)] or [K]:
            lk, j, _ = vc.leak(card, K, N, t, Lp, weight_coins=weight_coins)
            _GRID[key].append(dict(t=t, Lp=Lp, leak=lk, gamma_lp=j["honest"].c,
                                   gamma=round_gamma(card, K, N, t, Lp, weight_coins)))
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
    _, m, _ = job_methods(card, k[1], k[2], p["t"], p["Lp"], weight_coins=False)
    pub = dr.mu_price(c, m, "hull", ev["period"]["dense"]["served_mu"])
    return CLOCK[card] * nf.ETA * pub / max(p["gamma"] - 1, 1e-9) - ev["share"]["share"]


def rule(static_keys, pv_keys) -> tuple[dict, dict, dict]:
    """The coordinator's 07:04Z rule (maximize credited value per honest second, pub/T_honest, at served-period
    prices, γ ≤ 10 per round class), taken at the period: the period's credited share at served duals is exactly
    its credited value per honest second, and the picks move those duals, so a per-round ratio at fixed prices is
    myopic (at the H200's all-ALU served dual it rewards hashing: the smallest L′ with γ ≤ 10). Per card, the τ in
    TAUS whose picks (`rule_pick`) give the largest mean credited share over the three models, ties within 0.1 point
    to the lower mean γ. A round class with no L′ at γ ≤ 10 (short PV) is credited at its least-γ L′ where
    `value_test` is positive at every model whose period holds it, else served plain. Returns (static, pv, sweep)."""
    sweep, chosen = {}, {}
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
                        tau=chosen)


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


def job_lps(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool, paper: bool) -> tuple[dict, dict]:
    """A job's methods and its two LPs (dense, hull) as `price_methods` builds them, cached across designs."""
    key = (card, K, N, t, Lp, weight_coins, paper)
    if key not in _JOB:
        c, m, _ = job_methods(card, K, N, t, Lp, weight_coins=weight_coins, paper=paper)
        _JOB[key] = (m, {wh: dr.captured(c, m, wh) for wh in ("dense", "hull")})
    return _JOB[key]


def serve_loads(card) -> dict:
    """Loads a MAC of plain INT8 served work (QKᵀ's two softmax passes, V4's indexer, plain PV): the s8 MMA."""
    return dict(tensor=1 / card.s8, smem=1 / card.honest_tile[0] + 1 / card.honest_tile[1], rf=card.mma_rf,
                issue=card.mma_issue)


def period(card: str, model: dict, static: dict, pv: dict, paper: bool = False) -> dict:
    """Per credited MAC, in honest-kernel units at θ (c = SM-clocks · s8): OPT*'s period (`dense`: the honest words'
    best programs) and pub (`hull`: every route), each over the credited rounds alone and in the served period (plus
    QKᵀ, the indexer and plain PV as fixed load). Credit is each one's price at the honest served period's duals μ°
    (`at_served`, P20's C(μ) at the dual where the honest period sits). `freed`: per row, the share of OPT* the routes
    free at that row's single-row dual (`d5_routes.freed`, makespan ≤ OPT*'s), cost-weighted over the credited rounds
    (P12 re-raised). Also Σ_w q_w (P20's draw multiplier): each round's own makespan share of the honest period."""
    c = vc.CARDS[card][0]
    hist, st, qk, pvm, ix = nf.world_macs(model)
    jobs, rounds = {"dense": [], "hull": []}, {"dense": 0.0, "hull": 0.0}
    items = [(K, N, K * N * reps * nf.WORLD_TOKENS * st / (model["static"] * nf.WORLD_TOKENS), static[card, K, N],
              True) for K, N, reps in nf.SHAPES[model["name"]]]
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
        m, lps = job_lps(card, K, N, p["t"], p["Lp"], wc, paper)
        ms.append((w / cred, m))
        for wh in jobs:
            box, pr = lps[wh]
            jobs[wh].append((w / cred, box))
            rounds[wh] += w / cred * pr.c
            if wh == "dense":
                for r, x in p.get("freed", {}).items():
                    freed_[r] = freed_.get(r, 0.0) + w / cred * pr.c * x
    F_macs = (qk * (1 + nf.SOFTMAX_PASS) + ix + plain_pv) / cred
    F = {r: v * F_macs for r, v in serve_loads(c).items()}
    out = dict(cred_macs=cred, plain_pv=plain_pv, rounds=rounds,
               freed={r: x / rounds["dense"] for r, x in freed_.items()})
    for wh in jobs:
        alone, served = dr.period(c, jobs[wh]), dr.period(c, jobs[wh], F)
        muF = 0.0
        for r, mu in served["mu"].items():
            if r.startswith("joint"):
                a, b = c.facets[int(r[5:])]
                muF += mu * (a * F.get("tensor", 0.0) + b * F.get(c.facet_other, 0.0) / 32) / 4
            else:
                muF += mu * F.get(r, 0.0) / served["caps"][r]
        out[wh] = dict(alone=alone["c"], alone_binding=alone["binding"], alone_mu=alone["mu"], served=served["c"],
                       served_binding=served["binding"], served_mu=served["mu"], own_dual=served["c"] - muF * c.s8)
    # Both priced at the honest served period's duals μ°, where the honest prover's period sits.
    mu0 = out["dense"]["served_mu"]
    for wh in jobs:
        out[wh]["at_served"] = sum(w * dr.mu_price(c, m, wh, mu0) for w, m in ms)
    assert abs(out["dense"]["at_served"] / out["dense"]["own_dual"] - 1) < 1e-4, (out["dense"], card, model["name"])
    out["leak"] = 1 - out["hull"]["alone"] / out["dense"]["alone"]
    out["leak_served"] = 1 - out["hull"]["at_served"] / out["dense"]["at_served"]
    out["leak_job"] = 1 - rounds["hull"] / rounds["dense"]
    out["q_sum"] = rounds["dense"] / out["dense"]["served"]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ε, the credited share, the world (rule L2).

WORLD_FLOP = {"week": 1.8e29, "month": 7.8e29, "year": 9.36e30}   # FP8 FLOP of the world fleet (brief rule L2)
YEAR_S = 365.25 * 86400
DELTAS = (0.01, 1e-6)
EPS_S_AT_1E6 = 0.001                                      # the sampling term's target at δ 1e-6, which sets k
K_DRAWS = math.ceil(math.log(1e6) / EPS_S_AT_1E6)         # 13,816
GROUP_GPUS = (1, 8, 64)                                   # P16: a serving replica group's GPUs (scaled, not a census)
WIRE_BITS = 32                                            # bits a token across the node's boundary (RT-PROTO-2 P18)
HTTP_FACTOR = 6.5                                         # RT-PROTO-2: wire bytes a token if HTTP/SSE crosses it
CLOCK = {"H200": 1797 / 1980, "RTX PRO 6000": 1.0}        # M14: the throttled honest kernel over θ's clock
R_SHA = {"H200": 1.54, "RTX PRO 6000": 1.55}              # U13: θ's SHA-512 over the card's (PD3's share formula)
HOST = {"H200": (0.0315, 0.0490), "RTX PRO 6000": (0.0951, 0.0507)}   # U9 (CALIB): AMX / INT8, CPU / GPU hashing
SHA_MEASURED = {"H200": (4.79e9, 1980, "r20261018-062010-1c6b"), "RTX PRO 6000": (7.13e9, 2092, "r20261016-110413-c5d1")}
# H1's perplexity (Δ over bf16) for v3's arithmetic: both operands 1-bit dithered (H1's design A; the noise U, V peels
# off exactly), and PV with the fresh bit on P in one block (COVER-ATTN's credited PV).
PPL = {
    "Llama-3.1-8B": ("+1.38% (WT-2 2K, 3 seeds, r20261017-065449-b05b); +3.39% at 8K (seed 0, r20261017-070733-6851)",
                     "+0.83% (r20261017-065449-b05b's adopted A)"),
    "Llama-3.1-70B": ("+3.01% (C4 2K, seed 0, r20261017-063215-c9b2); 8K not run", "+1.68% (adopted A)"),
    "DeepSeek-V4-Pro": ("not run", "not run"),
}
PPL_ART = "art:02211b98fe2c3320134413f91f7e5a68b5d2fdbefc69675b354728f3648add35"


def n_gpus(card: str) -> float:
    return WORLD_FLOP["year"] / (2 * nf.V2_CARDS[card][0] * YEAR_S)


def model_of(name: str) -> dict:
    return next(m for m in nf.WORLD if m["name"] == name)


def eps_terms(card: str, model: dict, eta_: float, cred_macs: float, q_sum: float) -> dict:
    """ε in model units, a share of C at every dual (P20): ε_S = ln(1/δ)/(ηk) with P15's Poisson draw (expected draws
    k Σ_w q_w); P16 per replica group, the union bound over G groups (ε_S at k a group, and the k a group that holds
    ε_S at 0.1%); ε_net = n_max E / credited MACs (P18, n_max the largest N of a credited round, E the wire bits)."""
    out = dict(eta=eta_, k=K_DRAWS, draws=K_DRAWS * q_sum, q_sum=q_sum)
    out["eps_S"] = {d: math.log(1 / d) / (eta_ * K_DRAWS) for d in DELTAS}
    n_max = max([N for _, N, _ in nf.SHAPES[model["name"]]] + [model["dv"]])
    E = WIRE_BITS * nf.WORLD_TOKENS
    out["n_max"], out["eps_net"] = n_max, n_max * E / cred_macs
    out["eps_net_http"] = out["eps_net"] * HTTP_FACTOR
    out["eps"] = {d: out["eps_S"][d] + out["eps_net"] for d in DELTAS}
    groups = {}
    for g in GROUP_GPUS:
        G = n_gpus(card) / g
        groups[g] = dict(G=G, eps_S={d: math.log(G / d) / (eta_ * K_DRAWS) for d in DELTAS},
                         k_group=math.ceil(math.log(G / 1e-6) / (eta_ * EPS_S_AT_1E6)))
        groups[g]["draws_world"] = G * groups[g]["k_group"] * q_sum
        groups[g]["eps"] = {d: groups[g]["eps_S"][d] + out["eps_net"] for d in DELTAS}
    out["groups"] = groups
    return out


def share(card: str, row: dict, per: dict) -> dict:
    """The credited share of honest device time (06:45Z: r and the clock kept in it): credit at pub (the served
    period's duals, D5's θ) over the measured honest time, decomposed into factors that multiply to it: the clock
    (M14), the wraps (outside credit), the uncredited served ops (QKᵀ, the indexer, plain PV, where they don't hide in
    a slack pipe), θ's generosity (D5's price of the credited rounds' best programs over the built kernel's time:
    hashing r and the matmul's η together) and the routes (pub over OPT* at the served duals)."""
    cov = 1 - row["uncredited"]
    honest, credit_b = row["honest"], row["credit"]
    # nofold's units are a MAC at η of θ's INT8 rate; D5's are a MAC at θ's rate.
    f = {"clock": CLOCK[card],
         "wraps": max(row["T_all"], row["A_nw"]) / honest,
         "uncredited ops": credit_b / max(row["T_all"], row["A_nw"]),
         "θ's generosity": per["dense"]["at_served"] * cov * nf.ETA / credit_b,
         "routes": per["hull"]["at_served"] / per["dense"]["at_served"]}
    s = math.prod(f.values())
    pd3 = max(row["T_c"], row["A_nw"] / R_SHA[card]) / max(row["T_c"], row["A_nw"])
    return dict(share=s, factors=f, pd3_r_reading=pd3, routes_job=1 - per["leak_job"], cov=cov,
                with_hosts=s / (1 + HOST[card][1]))


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


def evaluate(static: dict, pv: dict, rates=(True, False)) -> dict:
    """One design at the world model: {(card, rate, model): its nofold row, period, credited share, worst round}."""
    out = {}
    pk = as_picks(static, pv)
    for built in rates:
        for row in nf.v2_world(0.0, built=built, picks=pk):
            card = row["card"].split(" built")[0]
            model = model_of(row["model"])
            per = period(card, model, static, pv, paper=not built)
            out[card, "measured" if built else "paper", model["name"]] = dict(
                card=card, rate="measured" if built else "paper", model=model["name"], world=row, period=per,
                share=share(card, row, per), worst=worst_gamma(card, model, static, pv, built))
    return out


def build() -> dict:
    P, sweep = picks()
    eta_ = eta(*P[HEADLINE])
    eta_min = min(e["eta"] for e in eta_.values())
    out = dict(eta=eta_, rows={}, k=K_DRAWS, rule=sweep)
    for design, (static, pv) in P.items():
        for (card, rate, mname), e in evaluate(static, pv).items():
            model = model_of(mname)
            e["design"] = design
            e["eps"] = eps_terms(card, model, eta_min, e["period"]["cred_macs"], e["period"]["q_sum"])
            if design == HEADLINE and rate == "measured":
                e["streaming"] = streaming(card, model, e["world"], pv)
            out["rows"][design, card, rate, mname] = e
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
    w = lambda f, design, card, rate: " / ".join(f(rows[design, card, rate, m]) for m in models)  # noqa: E731

    L.append("## D6: protocol v3 (D1) and v3.1 (v3's L′ by the 07:04Z rule), printed by "
             "`benchmarks/pouw/accel_score/d6.py`\n")
    L.append(f"The headline is {HEADLINE}. Columns are 8B / 70B / V4 unless stated. ε is in model units, a share of C "
             "at every dual price (P20). "
             "Credit is pub at the period's dual prices: one LP over the period's credited rounds with every public "
             "route (`d5_routes.period`). A route that wins there leaves honest time uncredited and never adds to ε.\n")
    L.append("### ε (goal ≤ 1% at δ = 1e-6)\n")
    L.append("ε = ε_S + ε_net. P16 (07:04Z): the theorem holds per serving-replica group (coins in lockstep inside a "
             "group), with a union bound over the world fleet's G groups, so ε_S = ln(G/δ)/(ηk) at k = 13,816 a group; "
             "G is the world model's GPUs at the card's FP8 rate over the group size (scaled from one card, not a "
             "census). 8-GPU groups lead; 1 and 64 GPUs are in the P16 table. ε_net: RT-PROTO-2's assumptions (n_max "
             "the largest credited N, 4 bytes a token crossing the node's boundary, 4,500-token requests; ×6.5 if "
             "HTTP/SSE crosses it).\n")
    L.append("| card | η | k a group (expected draws, P20) | ε_S δ 1% / 1e-6, 8-GPU groups | ε_net | "
             "ε δ 1%, 8-GPU groups | **ε δ 1e-6, 8-GPU groups** | ε δ 1e-6, one sequence (no P16) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        r0 = rows[HEADLINE, card, "measured", models[0]]["eps"]
        g8 = r0["groups"][8]
        L.append(f"| {card} | 1 − {1 - r0['eta']:.1e} | {r0['k']:,} "
                 f"({w(lambda r: f'{r['eps']['draws']:,.0f}', HEADLINE, card, 'measured')}) | "
                 f"{pct(g8['eps_S'][0.01], 3)} / {pct(g8['eps_S'][1e-6], 3)} | "
                 f"{w(lambda r: pct(r['eps']['eps_net'], 4), HEADLINE, card, 'measured')} | "
                 f"{w(lambda r: pct(r['eps']['groups'][8]['eps'][0.01], 3), HEADLINE, card, 'measured')} | "
                 f"{w(lambda r: pct(r['eps']['groups'][8]['eps'][1e-6], 3), HEADLINE, card, 'measured')} | "
                 f"{w(lambda r: pct(r['eps']['eps'][1e-6], 3), HEADLINE, card, 'measured')} |")
    L.append("")
    L.append("### γ (goal ≤ 10), credited share and coverage (goal ≥ 95%)\n")
    L.append("| design, card, SHA-512 rate | γ overlapped | γ additive | worst round's γ | tensor util. | "
             "credited share (served duals) | routes uncredited, period / job | coverage of matmul MACs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for design in sorted({k[0] for k in rows}, key=DESIGNS.index):
        for card in vc.CARDS:
            if "RTX PRO 6000 static" in design and card != "RTX PRO 6000":
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
    L.append("### The world (rule L2): pooled, every window, both δ, V4 as the world's request (8B, 70B below)\n")
    L.append("ε at P16's 8-GPU groups. C is the world's FP8 FLOP (brief rule L2); the credited share and ε are one "
             "card's, scaled to the world.\n")
    L.append("| card, model | window | C (FP8 FLOP) | credited (FLOP-eq.) | εC δ 1% | εC δ 1e-6 | "
             "honest time uncredited (FLOP-eq.) | QKᵀ + indexer + plain PV, uncredited MACs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for m in reversed(models):
            r = rows[HEADLINE, card, "measured", m]
            for win, F in WORLD_FLOP.items():
                s = r["share"]["share"]
                e = r["eps"]
                g8 = e["groups"][8]["eps"]
                L.append(f"| {card}, {short[m]} | {win} | {F:.2e} | {s * F:.2e} | {g8[0.01] * F:.1e} | "
                         f"{g8[1e-6] * F:.1e} | {(1 - s) * F:.1e} | {pct(r['world']['uncredited'], 1)} |")
    L.append("")
    L.append("### P16: per replica group (union bound), G from the world model at the card's FP8 rate (scaled)\n")
    L.append("| card | GPUs a group | G | ε_S δ 1% / 1e-6 at k = 13,816 a group | k a group for ε_S 0.1% at 1e-6 | "
             "world draws a period (V4) |")
    L.append("|---|---|---|---|---|---|")
    for card in vc.CARDS:
        for g, x in rows[HEADLINE, card, "measured", models[-1]]["eps"]["groups"].items():
            L.append(f"| {card} | {g} | {x['G']:.2e} | {pct(x['eps_S'][0.01], 3)} / {pct(x['eps_S'][1e-6], 3)} | "
                     f"{x['k_group']:,} | {x['draws_world']:.2e} |")
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
    D = build()
    print(report(D))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(jsonable(D), indent=1, default=str))
