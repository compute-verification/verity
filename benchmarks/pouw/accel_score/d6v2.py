"""D6 v2: ACCEL's goals in circuit size (CMP-158; Daniel, 26 Oct 11:46 AM and 12:11 PM PDT: Boolean circuits
throughout, size in gates in place of work, FLOPs and pricing; top's 19:33Z notation), printed by code (PROTO-DESIGN-4).

Credit is size. A sampling block's public cost S_b is the gate count of the smallest public program for its products
over D5 v2.1's menu (0588f18015): `d6.job_parts`' methods, priced by `public_best.price_methods` at each resource's
gate price (`gates.d5_sigma`, the cheapest instruction on it: the least size errs low), with b cᵀ's tensor term at its
own gates (`gates.bit_size`), no hash item (the commitment is the service's; `size_job`'s hashed sensitivity adds the
round's SHA-512 at `gates.sha512_gates`) and no table routes (𝓜 has no indexed
load: a table read is its multiplexer tree, `gates.lookup_gates`, dearer than the product it stands for). The
registration declares S_reg = Σ_b S_b at the monitoring period's start; ε_skip is a share of S_reg. Φ is gates a
second (`gates.phi`, one LP over 𝓜's units), s_× the fewest gates of a counted operation (`gates.s_x`). μ°, the period
LP, its duals, μ¹ and the beside load x are retired as the credit's unit.

The picks are v3.1's rule taken in size: per card, the τ of `d6.TAUS` whose `d6.rule_pick` picks give the largest mean
credited share in size over the three models (ties within 0.1 point to the lower mean γ), a PV class with no L′ at
γ ≤ 10 credited at its least-γ L′ only where that adds credited size per honest second (`value_test`).

    PUBLIC_BEST_DIR=<checkout at 0588f18015 (D5 v2.1)>/benchmarks/pouw/theory [ACCEL_GATE_MEASURE=gates|and] \\
        python3 benchmarks/pouw/accel_score/d6v2.py [--json OUT] [--pickle OUT] [--quick]

ACCEL_GATE_MEASURE (`gates.MEASURE`) picks the size: all gates (the default) or ANDs alone; the build of record is
both, one run each.

--from PKL prints the report from a build's pickle without rebuilding. --quick: the 8B model, 512-token requests and
two τ only (a smoke test, not a result).
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6  # noqa: E402
import gates as G  # noqa: E402
import nofold as nf  # noqa: E402
import v3_caps as vc  # noqa: E402

pb = d6.pb
HEAD_BOUND = "lo"                          # the gate prices of the headline: each resource at its cheapest instruction
Q = d6.K_DRAWS                             # q checks a replica's registration (13,816: ε_S 0.1% at δ_coin 1e-6, η 1)
DELTAS = d6.DELTAS                         # δ_coin
MONITOR_S = d6.WINDOW_S                    # the monitoring periods: a week, a month, a year
F_W = d6.WORLD_FLOP                        # census world-inference-2027 (#4731): FP8 FLOP a monitoring period
CENSUS = "world-inference-2027 (#4731, benchmarks/census/adversary.json): 9.36e30 e4m3 FLOP a year, 2 FLOP a MAC"
DELTA_WIPE_S = 86400.0                     # Δ_wipe: a wipe every 24 h (Daniel's operating point)
K_WIPE_BITS = 4e10                         # k_wipe: 5 GB a server left by the wipe (Daniel, 12:16 PM PDT)
R_SIDE = 1e3                               # r_side: the assumed side-channel cap, bits a second
SERVER_GPUS = 8
EPS_DARK = (1e-3, 1e-2)                    # ε_dark, the dark leaves' share of Φ_total (ruled)
GOALS = {"eps": 0.01, "gamma": 10.0, "cov": 0.95, "ppl": 0.05}
PPL_WORST = 0.0339                         # d6.PPL's largest measured Δ (8B at 8K, r20261017-070733-6851)
V11 = dict(art="art:6926be1c98b7855aca699278551040538e83843d28b72fdf85044cfd291bf3ef", run="r20261026-151601-d0c1",
           tau={"H200": 1e-4, "RTX PRO 6000": 0.2},
           share={("H200", "Llama-3.1-8B"): 0.599, ("H200", "Llama-3.1-70B"): 0.590, ("H200", "DeepSeek-V4-Pro"): 0.607,
                  ("RTX PRO 6000", "Llama-3.1-8B"): 0.840, ("RTX PRO 6000", "Llama-3.1-70B"): 0.855,
                  ("RTX PRO 6000", "DeepSeek-V4-Pro"): 0.843},
           eps={("H200", "Llama-3.1-8B"): 0.00275, ("H200", "Llama-3.1-70B"): 0.00228,
                ("H200", "DeepSeek-V4-Pro"): 0.00225, ("RTX PRO 6000", "Llama-3.1-8B"): 0.00282,
                ("RTX PRO 6000", "Llama-3.1-70B"): 0.00233, ("RTX PRO 6000", "DeepSeek-V4-Pro"): 0.00228},
           hidden={("H200", "Llama-3.1-8B"): (0.061, 0.243, 0.972), ("H200", "Llama-3.1-70B"): (0.073, 0.291, 1.164),
                   ("H200", "DeepSeek-V4-Pro"): (0.051, 0.203, 0.811),
                   ("RTX PRO 6000", "Llama-3.1-8B"): (0.111, 0.445, 1.779),
                   ("RTX PRO 6000", "Llama-3.1-70B"): (0.130, 0.521, 2.083),
                   ("RTX PRO 6000", "DeepSeek-V4-Pro"): (0.108, 0.432, 1.730)})
UNIT = {"gates": "all gates (And2 + Xor2 + Not)",
        "and": "ANDs (And2 alone, the multiplicative size C-Flock's proof pays for; Xor2 and Not free)"}[G.MEASURE]
SHORT = {"Llama-3.1-8B": "8B", "Llama-3.1-70B": "70B", "DeepSeek-V4-Pro": "V4"}
MODELS = [m["name"] for m in nf.WORLD]
TAUS = d6.TAUS


# ---------------------------------------------------------------------------------------------------------------------
# S_b: a sampling block's public size, gates a naive MAC of its products.

_SJ: dict = {}


def size_job(card: str, K: int, N: int, t: int, Lp: int, weight_coins: bool, bound: str = HEAD_BOUND,
             hashed: bool = False, wired: bool = False) -> dict:
    """S_b a naive MAC of one round class at a pick: the least size over the menu (hull) and over the honest words'
    programs alone (dense, OPT* in size), each at the gate prices `bound`, with the winning methods and routes.
    hashed: the round's SHA-512 compressions added at their traced gates (`gates.sha512_gates`; nofold's `comps`, the
    leaves, A and the coin expansion a naive MAC), a sensitivity on what v1.1's credit carried. Size adds across
    subcircuits, so the hash adds to the least size and doesn't move the winning method. wired: D5's table routes
    kept, a read at its resources' gate prices (smem and issue carry none), so a table read is wiring: the actual
    circuit is one run's (`Accounting.Actual`), its addresses fixed (RT-SIZE's RS1)."""
    key = (card, K, N, t, Lp, weight_coins, bound, hashed, wired)
    if key not in _SJ:
        c, ms, L, _, _, r = d6.job_parts(card, K, N, t, Lp, weight_coins, bit=G.bit_size(card))
        kw = dict(beside=True, items=(), lookups=wired, mu=G.d5_sigma(card, c, bound))
        dense, hull = (pb.price_methods(c, x, L, N, pb.TOKENS, **kw) for x in (ms[:1], ms))
        h = r["comps"] * G.sha512_gates()["total"] if hashed else 0.0
        _SJ[key] = dict(dense=dense.c / c.s8 + h, hull=hull.c / c.s8 + h, comps=r["comps"],
                        mix={k: float(v) for k, v in hull.mix.items() if v > 1e-3},
                        routes={k: float(v) for k, v in hull.routes.items() if v > 1e-3})
    return _SJ[key]


def credited_items(card: str, model: dict, static: dict, pv: dict) -> tuple[list, float]:
    """`d6.period`'s credited round classes of one request: (K, N, MACs, pick, weight coins), and plain PV's MACs."""
    hist = nf.world_macs(model)[0]
    items = [(K, N, K * N * reps * nf.WORLD_TOKENS, static[card, K, N], True) for K, N, reps in nf.SHAPES[model["name"]]]
    bins: dict = {}
    for n, cnt in hist.items():
        bins[nf.pv_bucket(n)] = bins.get(nf.pv_bucket(n), 0) + cnt * model["hq"] * model["dv"] * n
    plain = 0.0
    for nb, x in sorted(bins.items()):
        p = pv[card, nb, model["dv"]]
        if p is None:
            plain += x
        else:
            items.append((nb, model["dv"], x, p, False))
    return items, plain


def size_period(card: str, model: dict, static: dict, pv: dict, sens: bool = False) -> dict:
    """S_reg a credited MAC (MAC-weighted S_b over the credited classes) at the headline prices; sens: also at the
    upper gate prices, OPT* in size, and with the commitment's hash, and the static and PV classes apart."""
    items, plain = credited_items(card, model, static, pv)
    cred = sum(w for _, _, w, _, _ in items)
    out = dict(cred_macs=cred, plain_pv=plain, classes=len(items))
    variants = [("s", HEAD_BOUND, False, False, "hull")]
    if sens:
        variants += [("s_hi", "hi", False, False, "hull"), ("s_dense", HEAD_BOUND, False, False, "dense"),
                     ("s_hashed", HEAD_BOUND, True, False, "hull"), ("s_dense_hi", "hi", False, False, "dense"),
                     ("s_wired", HEAD_BOUND, False, True, "hull")]
    for name, bound, hashed, wired, wh in variants:
        out[name] = sum(w * size_job(card, K, N, p["t"], p["Lp"], wc, bound, hashed, wired)[wh]
                        for K, N, w, p, wc in items) / cred
    if sens:
        for kind, wc_ in (("static", True), ("pv", False)):
            sel = [(K, N, w, p) for K, N, w, p, wc in items if wc == wc_]
            wsum = sum(w for _, _, w, _ in sel)
            out[kind] = dict(w=wsum / cred, s=(sum(w * size_job(card, K, N, p["t"], p["Lp"], wc_)["hull"]
                                                    for K, N, w, p in sel) / wsum) if wsum else None)
        K, N, _, p, wc = max(items, key=lambda it: it[2])
        out["largest"] = dict(K=K, N=N, t=p["t"], Lp=p["Lp"], **size_job(card, K, N, p["t"], p["Lp"], wc))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The credited share in size, the rule, the goals.

def phi_s8(card: str) -> float:
    """Φ in gates a tensor s8 MAC-time (clock-free: Φ a SM-clock over 𝓜's s8 MACs a SM-clock), so Φ at the clock the
    honest kernel's time is measured at."""
    return G.phi(card)["per_s8_mac"]


def answers_share(card: str, model: dict) -> float:
    """H_o/H: the q answers' GPU time (C-Flock --zk, `d6.ANS_GPU_S` each) over the replica's, a week."""
    return Q * d6.ANS_GPU_S / (d6.group_gpus(card, model) * d6.PERIOD_S)


def share_size(card: str, row: dict, sp: dict, model: dict) -> dict:
    """S_reg over Φ·T of the honest period, as factors that multiply to it: the clock (M14), the served efficiency η
    (nofold's unit, a MAC at η of θ's INT8 rate), coverage (credited MACs a MAC), S_b a credited MAC over Φ a tensor
    s8 MAC-time, 1/γ, and the timed answers."""
    f = {"clock": d6.CLOCK[card], "served η": nf.ETA, "coverage": 1 - row["uncredited"],
         "S_b / Φ an s8 MAC-time": sp["s"] / phi_s8(card), "1/γ": 1 / row["gamma"],
         "answers": 1 / (1 + answers_share(card, model))}
    return dict(share=math.prod(f.values()), factors=f)


def value_test(card: str, k: tuple, p: dict, share: float) -> float:
    """Credited size per honest second a PV class adds over serving it plain, less the period's share."""
    s = size_job(card, k[1], k[2], p["t"], p["Lp"], False)["hull"]
    return d6.CLOCK[card] * nf.ETA * s / (phi_s8(card) * max(p["gamma"] - 1, 1e-9)) - share


def evaluate(static: dict, pv: dict, rates=(True, False), sens: bool = False) -> dict:
    """{(card, rate, model): nofold's row, S_reg a credited MAC, the share in size, the worst round}."""
    out, sps = {}, {}
    for built in rates:
        for row in nf.v2_world(0.0, built=built, picks=d6.as_picks(static, pv)):
            card = row["card"].split(" built")[0]
            if row["model"] not in MODELS:
                continue
            model = d6.model_of(row["model"])
            if (card, model["name"]) not in sps:
                sps[card, model["name"]] = size_period(card, model, static, pv, sens)
            sp = sps[card, model["name"]]
            out[card, "measured" if built else "paper", model["name"]] = dict(
                world=row, size=sp, share=share_size(card, row, sp, model),
                worst=d6.worst_gamma(card, model, static, pv, built))
    return out


def rule(static_keys: list, pv_keys: list) -> tuple[dict, dict, dict]:
    """v3.1's rule in size (module docstring). Returns (static, pv, sweep)."""
    sweep, picks_ = {}, {}
    for tau in TAUS:
        st = {k: d6.rule_pick(*k, tau) for k in static_keys}
        pv = {k: d6.rule_pick(*k, tau, weight_coins=False) for k in pv_keys}
        ev = evaluate(st, pv, rates=(True,))
        users: dict = {}
        for (card, _, mname), e in ev.items():
            model = d6.model_of(mname)
            for n in {nf.pv_bucket(n) for n in nf.world_macs(model)[0]}:
                k = (card, n, model["dv"])
                if not pv[k]["feasible"]:
                    users.setdefault(k, []).append(value_test(card, k, pv[k], e["share"]["share"]))
        pv = {k: (None if k in users and min(users[k]) <= 0 else p) for k, p in pv.items()}
        ev = evaluate(st, pv, rates=(True,))
        picks_[tau] = (st, pv)
        for card in vc.CARDS:
            es = [e for (c, _, _), e in ev.items() if c == card]
            sweep[card, tau] = dict(share=[e["share"]["share"] for e in es], gamma=[e["world"]["gamma"] for e in es],
                                    s=[e["size"]["s"] for e in es], infeasible=sum(1 for k in users if k[0] == card),
                                    plain=sum(1 for k in users if k[0] == card and pv[k] is None))
    chosen = {}
    for card in vc.CARDS:
        best = max(TAUS, key=lambda t: np.mean(sweep[card, t]["share"]))
        near = [t for t in TAUS if np.mean(sweep[card, t]["share"]) >= np.mean(sweep[card, best]["share"]) - 1e-3]
        chosen[card] = min(near, key=lambda t: np.mean(sweep[card, t]["gamma"]))
    st = {k: picks_[chosen[k[0]]][0][k] for k in static_keys}
    pv = {k: picks_[chosen[k[0]]][1][k] for k in pv_keys}
    return st, pv, dict(sweep=sweep, tau=chosen)


def eps_skip(card: str, model: dict, eta_: float) -> dict:
    """ε_skip, a share of S_reg: q checks a replica's registration, each a gate of S_reg drawn by the challenger's live
    coins (a proof block in proportion to its public size), η the opened unit's free-guess bound; the union over the
    census's G replicas (POOL-THM, CMP-152, may drop it), ln(G/δ_coin)/(ηq). One registration a monitoring period,
    at its start, so a week, a month and a year carry the same ε_skip; `weekly`: the variant with a registration
    each week, the union over the monitoring period's weeks added (v1.1's P26 reading)."""
    g = d6.group_gpus(card, model)
    Gr = d6.n_gpus(card) / g
    base = {d: math.log(Gr / d) / (eta_ * Q) for d in DELTAS}
    weekly = {w: {d: base[d] + math.log(s / d6.PERIOD_S) / (eta_ * Q) for d in DELTAS} for w, s in MONITOR_S.items()}
    return dict(g=g, G=Gr, eta=eta_, q=Q, eps=base, weekly=weekly,
                one_replica={d: math.log(1 / d) / (eta_ * Q) for d in DELTAS})


def answers(card: str, model: dict) -> dict:
    """The q answers through the firewall one at a time (P25): draws N ~ Poisson(q) at its 1 − 1e-6 quantile, so
    N(Δ + τ_ans) ≤ T a monitoring period; the share of a week they keep the firewall busy."""
    n_hi = float(poisson.ppf(1 - 1e-6, Q))
    return dict(draws=Q, draws_hi=n_hi, busy={w: n_hi * (d6.DELTA_S + d6.TAU_ANS_S) / s for w, s in MONITOR_S.items()},
                ho=answers_share(card, model))


def coverage_gates(card: str, model: dict, row: dict) -> dict:
    """Coverage of the model's circuit's gates: credited MACs at the s8 dot's gates over every MAC at those gates plus
    the elementwise steps (`gates.elementwise_gates`; an estimate of the serving circuit, stated): a static matmul's
    input quantised (q) and its output dequantised with one F32 multiply and add (dq + mul + add) an element; a score
    dequantised and scaled, its max subtracted, exponentiated, summed and quantised (dq + mul + 2 add + ex2 + q); a
    query head's row normalised (rcp, and dq + mul a PV output element); an indexer score dq + mul + add."""
    e = G.elementwise_gates()
    hist, st, qk, pvm, ix = nf.world_macs(model)
    tot = st + qk + pvm + ix
    g8 = G.dot_gates(card, "s8")["per_mac"]
    static = nf.WORLD_TOKENS * sum(reps * (K * e["q"] + N * (e["dq"] + e["mul"] + e["add"]))
                                   for K, N, reps in nf.SHAPES[model["name"]])
    scores = qk / model["dqk"]
    rows = sum(c for c in hist.values()) * model["hq"]
    attn = scores * (e["dq"] + e["mul"] + 2 * e["add"] + e["ex2"] + e["q"]) + rows * (
        e["rcp"] + model["dv"] * (e["dq"] + e["mul"]))
    idx = ix / 128 * (e["dq"] + e["mul"] + e["add"])
    ew = static + attn + idx
    cred = (1 - row["uncredited"]) * tot
    return dict(cov=cred * g8 / (tot * g8 + ew), ew_share=ew / (tot * g8 + ew), static=static, attn=attn, idx=idx,
                ew_per_mac=ew / tot, g8=g8)


def exclusion(card: str, share: float, eps: float) -> dict:
    """Per monitoring period over the census's fleet: Φ_fleet T = F/2 · Φ an FP8 MAC-time (𝓜's E4M3 rate is its s8
    rate, weight 1, on both cards: the census's FLOP as the fleet's FP8 tensor capacity, a stated conversion), the
    registered S_reg = share · Φ_fleet T, U = Φ_fleet T − (1 − ε_skip) S_reg, and U/s_× counted operations against the
    census's own F/2 MACs; ε_dark's dark leaves add ε_dark Φ_fleet T / s_×."""
    sx = G.s_x()["s_x"]
    out = {}
    for w, F in F_W.items():
        cap = F / 2 * phi_s8(card)
        S = share * cap
        U = cap - (1 - eps) * S
        out[w] = dict(cap=cap, S_reg=S, U=U, ops=U / sx, census_ops=F / 2, ratio=U / sx / (F / 2),
                      census_gates=F / 2 * G.dot_gates(card, "e4m3")["per_mac"],
                      dark={d: d * cap / sx for d in EPS_DARK})
    return out


def serving_block(card: str, model: dict, row: dict, share: float, eps: float) -> dict:
    """The padded serving block of an 8-GPU server: c_S = ⌊Φ_s Δ_wipe⌋ gates (Φ_s at 𝓜's tick), k_S = k_wipe +
    (r_in + r_side) Δ_wipe bits, r_in the server's token ingress at the honest rate; U_S = c_S − certified gates and its
    U_S/s_× counted operations; blocks a monitoring period (the fleet's Φ_fleet T over c_S, each padded to c_S)."""
    phi_s = SERVER_GPUS * G.phi(card)["per_s"]
    c_S = math.floor(phi_s * DELTA_WIPE_S)
    _, st, qk, pvm, ix = nf.world_macs(model)
    t_req = row["gamma"] * (st + qk + pvm + ix) / (nf.ETA * nf.V2_CARDS[card][0]) / d6.CLOCK[card]
    r_in = d6.WIRE_BITS * nf.WORLD_TOKENS / t_req * SERVER_GPUS
    U_S = c_S * (1 - (1 - eps) * share)
    ex = exclusion(card, share, eps)
    return dict(phi_s=phi_s, c_S=c_S, k_S=K_WIPE_BITS + (r_in + R_SIDE) * DELTA_WIPE_S, r_in=r_in, U_S=U_S,
                ops_S=U_S / G.s_x()["s_x"], blocks={w: x["cap"] / c_S for w, x in ex.items()}, t_req=t_req)


def channel(card: str, model: dict, sb: dict) -> dict:
    """k_infil / k_exfil, bits across a replica's network-certifier boundary: the tokens in and out (WIRE_BITS each,
    RT-PROTO-2's P18 wire) and the assumed side channel r_side, a week and a serving block's window. No longer priced
    in MACs or gates (top's 19:33Z notation): the channel term stands beside ε_skip."""
    g = d6.group_gpus(card, model)
    per_s = d6.WIRE_BITS * nf.WORLD_TOKENS / sb["t_req"] * g
    return dict(g=g, week=per_s * d6.PERIOD_S, side_week=R_SIDE * d6.PERIOD_S, request=d6.WIRE_BITS * nf.WORLD_TOKENS,
                block=sb["k_S"] - K_WIPE_BITS)


def beside(card: str, e: dict, eps: float) -> list:
    """v1.1's hidden beside work in size. S_b is public and fixed: no dual prices the period, so plain INT8 work beside
    the credited rounds (x MACs a credited MAC, x·g_s8 gates) changes no credit, and it is uncertified capacity, inside
    U, counted at its gates. Hidden is 0 at every x; `fits`: whether x·g_s8 fits in U at the honest period's T (else
    the period lengthens and U grows with it)."""
    g8 = G.dot_gates(card, "s8")["per_mac"]
    cov, s = 1 - e["world"]["uncredited"], e["size"]["s"]
    cap = e["size"]["s"] / e["share"]["share"]              # Φ·T a credited MAC
    U = cap - (1 - eps) * s
    return [dict(x=x, gates=x * g8, U=U, fits=x * g8 <= U, hidden=0.0, cov=cov) for x in d6.BESIDE]


# ---------------------------------------------------------------------------------------------------------------------

def keys(models: list) -> tuple[list, list]:
    st, pv = [], []
    for card in vc.CARDS:
        for m in models:
            model = d6.model_of(m)
            st += [(card, K, N) for K, N, _ in nf.SHAPES[m] if (card, K, N) not in st]
            pv += [(card, nb, model["dv"]) for nb in sorted({nf.pv_bucket(n) for n in nf.world_macs(model)[0]})
                   if (card, nb, model["dv"]) not in pv]
    return st, pv


def build(quick: bool = False) -> dict:
    global MODELS, TAUS
    if quick:
        MODELS, TAUS = ["Llama-3.1-8B"], (0.2, 1.0)
        nf.WORLD = [m for m in nf.WORLD if m["name"] in MODELS]
        nf.WORLD_TOKENS = 512
    st_keys, pv_keys = keys(MODELS)
    static, pv, R = rule(st_keys, pv_keys)
    eta_ = d6.eta(static, pv, samples=40_000 if quick else 400_000)
    eta_min = min(x["eta"] for x in eta_.values())
    out = dict(models=MODELS, taus=TAUS, rule=R, eta=eta_, q=Q, rows={}, census=CENSUS)
    for (card, rate, m), e in evaluate(static, pv, sens=True).items():
        model = d6.model_of(m)
        e["eps"] = eps_skip(card, model, eta_min)
        e["answers"] = answers(card, model)
        e["gate_cov"] = coverage_gates(card, model, e["world"])
        if rate == "measured":
            eps = e["eps"]["eps"][1e-6]
            e["exclusion"] = exclusion(card, e["share"]["share"], eps)
            e["serving"] = serving_block(card, model, e["world"], e["share"]["share"], eps)
            e["channel"] = channel(card, model, e["serving"])
            e["beside"] = beside(card, e, eps)
        out["rows"][card, rate, m] = e
    out["picks"] = {k: p and dict(t=p["t"], Lp=p["Lp"], gamma=p["gamma"], leak=p.get("leak"))
                    for k, p in {**static, **pv}.items()}
    out["measure"] = G.MEASURE
    out["phi"] = {c: dict(full=G.phi(c), pinned=G.phi(c, pinned=True), serving=G.phi(c, serving=True))
                  for c in G.ACCEL}
    out["s_x"] = G.s_x()
    out["sha512"] = G.sha512_gates()
    out["lanes"] = G.lane_gates()
    out["elementwise"] = G.elementwise_gates()
    out["sigma"] = {c: {b: G.d5_sigma(c, vc.CARDS[c][0], b) for b in ("lo", "hi")} for c in vc.CARDS}
    out["bit"] = {c: G.bit_size(c) for c in vc.CARDS}
    out["tensor_checks"] = {c: G.tensor_checks(c, vc.CARDS[c][0]) for c in vc.CARDS}
    out["dots"] = {c: {f: G.dot_gates(c, f)["per_mac"] for f in G.ACCEL[c]["weight"]} for c in G.ACCEL}
    out["lookups"] = dict(lut1=G.lookup_gates(256, 16, 1), four_russians={g: G.lookup_gates(2 ** g, 32, 2 * g)
                                                                         for g in range(1, 14)})
    return out


# ---------------------------------------------------------------------------------------------------------------------

def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def sci(x: float, d: int = 2) -> str:
    return f"{x:.{d}e}"


def verdict(D: dict) -> str:
    rows = {k: r for k, r in D["rows"].items() if k[1] == "measured"}
    eps = max(r["eps"]["eps"][1e-6] for r in rows.values())
    gam = max(r["world"]["gamma"] * (1 + r["answers"]["ho"]) for r in rows.values())
    cov = min(1 - r["world"]["uncredited"] for r in rows.values())
    cg = min(r["gate_cov"]["cov"] for r in rows.values())
    sh = [r["share"]["share"] for r in rows.values()]
    ratio = max(r["exclusion"]["year"]["ratio"] for r in rows.values())

    def ok(b):
        return "meets" if b else "MISSES"
    return (f"VERDICT D6 v2 (size in {'all gates' if D['measure'] == 'gates' else 'ANDs'}): ε_skip at δ_coin 1e-6 ≤ "
            f"{pct(eps, 3)} ({ok(eps <= GOALS['eps'])} ≤ 1%, ratio "
            f"{eps / GOALS['eps']:.3f}); γ ≤ {gam:.2f} ({ok(gam <= GOALS['gamma'])} ≤ 10, ratio {gam / 10:.3f}); "
            f"coverage of MACs ≥ {pct(cov, 1)} ({ok(cov >= GOALS['cov'])} ≥ 95%, ratio {cov / 0.95:.3f}); coverage "
            f"of gates ≥ {pct(cg, 1)} ({ok(cg >= GOALS['cov'])} ≥ 95%, ratio {cg / 0.95:.3f}); perplexity carried, "
            f"worst +{100 * PPL_WORST:.2f}% (8B at 8K; {ok(PPL_WORST < GOALS['ppl'])} < 5%, ratio "
            f"{PPL_WORST / GOALS['ppl']:.3f}; V4 not run); credited share in size {pct(min(sh), 2)}–{pct(max(sh), 2)} "
            f"(no target); U/s_× up to {ratio:.2f}× the census's counted operations a monitoring period")


def report(D: dict) -> str:
    rows, L = D["rows"], []
    cards = list(vc.CARDS)
    models = D["models"]
    sx = D["s_x"]
    L.append("## D6 v2: ACCEL's goals in circuit size, printed by `benchmarks/pouw/accel_score/d6v2.py` (CMP-158)\n")
    L.append(f"Every size is in {UNIT} of the canonical Boolean circuit (`verity.ml.circuits.boolean.trace`; this "
             f"build is ACCEL_GATE_MEASURE={D['measure']}, and the other measure is its companion build), called "
             "gates below. Credit is size: a sampling block's public cost S_b is the least gate count over D5 v2.1's "
             "menu of a program for its products, the registration declares S_reg = Σ_b S_b at the monitoring "
             "period's start, and ε_skip is a share of S_reg. Φ is gates a second from 𝓜 (#4662's `Accel`), s_× = "
             f"{sx['s_x']:.1f} gates (an s8 MAC). Results are per monitoring period (a week, a month, a year) over "
             f"the adversary resource model, census {CENSUS}. Columns are 8B / 70B / V4 unless stated; the headline "
             "is measured hashing.\n")
    L.append(verdict(D) + "\n")

    def gam(r):
        return r["world"]["gamma"] * (1 + r["answers"]["ho"])

    L.append("### The goals, each against its target and its ratio\n")
    L.append("ε_skip is one registration's, at the model's replica, the union over the census's replicas in it. γ "
             "includes the answers' GPU time. Ratio is value over target (≤ 1 meets an upper target; ≥ 1 meets "
             "coverage's lower one).\n")
    L.append("| card, model | GPUs a replica | ε_skip δ_coin 1e-6 / 1% (≤ 1%) | ratio | γ measured / paper (≤ 10) | "
             "ratio | coverage of MACs (≥ 95%) | ratio | coverage of gates (≥ 95%) | ratio | perplexity (< 5%) | "
             "credited share in size |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            r, p = rows[card, "measured", m], rows[card, "paper", m]
            e = r["eps"]["eps"]
            cov, cg = 1 - r["world"]["uncredited"], r["gate_cov"]["cov"]
            L.append(f"| {card}, {SHORT[m]} | {r['eps']['g']} | {pct(e[1e-6], 3)} / {pct(e[0.01], 3)} | "
                     f"{e[1e-6] / GOALS['eps']:.3f} | {gam(r):.2f} / {gam(p):.2f} | {gam(r) / 10:.3f} | {pct(cov, 1)} | "
                     f"{cov / 0.95:.3f} | {pct(cg, 1)} | {cg / 0.95:.3f} | {d6.PPL[m][2]} | "
                     f"{pct(r['share']['share'], 2)} |")
    L.append("")
    miss = [(c, SHORT[m]) for c in cards for m in models if rows[c, "measured", m]["gate_cov"]["cov"] < 0.95]
    L.append("Coverage of gates " + (f"misses 95% on {', '.join(f'{c} {m}' for c, m in miss)}: the uncredited "
                                     "elementwise steps (softmax above all) and QKᵀ. " if miss else "meets 95% everywhere. ")
             + "The elementwise estimate is " + ", ".join(
                 f"{SHORT[m]} {pct(rows[cards[0], 'measured', m]['gate_cov']['ew_share'], 2)}" for m in models)
             + " of the circuit's gates (`gates.elementwise_gates`, each step the repository's Boolean form; the "
             "count of steps a token is `coverage_gates`', a stated model of the serving circuit, not a traced one).\n")

    # Credited share.
    L.append("### The credited share in size, S_reg over Φ·T of the honest period\n")
    L.append("It is low, and that is the result: Φ counts every gate the card can complete, and its most gate-dense "
             "instructions (FFMA and DFMA lanes, 2:4 FP16 MMA) complete far more gates a second than the INT8 "
             "products the protocol credits. The share is the product of its factors:\n")
    L.append("| card, model | clock | served η | coverage | S_b a credited MAC / Φ an s8 MAC-time | 1/γ | answers | "
             "share | at Φ without FP64 and stand-ins | v1.1's at μ° |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for card in cards:
        sv = phi_s8(card) / D["phi"][card]["serving"]["per_s8_mac"]
        for m in models:
            r = rows[card, "measured", m]
            f = r["share"]["factors"]
            L.append(f"| {card}, {SHORT[m]} | {f['clock']:.3f} | {f['served η']:.2f} | {pct(f['coverage'], 1)} | "
                     f"{r['size']['s']:.1f} / {phi_s8(card):.0f} = {f['S_b / Φ an s8 MAC-time']:.3f} | "
                     f"{f['1/γ']:.3f} | {f['answers']:.4f} | {pct(r['share']['share'], 2)} | "
                     f"{pct(r['share']['share'] * sv, 2)} | {pct(V11['share'][card, m], 1)} |")
    L.append("")
    L.append("Φ without FP64 and stand-ins leaves out DMMA, DFMA and the dots 𝓜 leaves open, which serving doesn't run "
             "(RT-SIZE's RS2): a sensitivity, since Φ is what the card can complete, whatever serving runs.\n")

    # Φ.
    L.append("### Φ per card, and how it is derived\n")
    L.append("Each instruction class of 𝓜 is the canonical Boolean circuit of its semantics, traced: the tensor dots "
             "are the repository's silicon steps (one accumulator coordinate of one MMA step, gates over its k), the "
             "lane functions `Isa.lean`'s `Lane.eval` as `forms` builders, DMMA and DFMA as 𝓜 over-approximates "
             "them (exact integer MACs within ±2^53). Φ = max Σ_i x_i g_i over instruction rates x_i ≥ 0 under 𝓜's "
             "units a tick (`opUse` against ρ: tensor, issue, alu, imad, shfl, f64, smem, l2, dram), one LP, then "
             "times SMs and 𝓜's tick. Copies and shuffles compute nothing. Formats 𝓜 leaves `Open.dot` (f16, tf32, "
             "sm_120's e2m1, every 2:4 form) take a stand-in, so the pinned-dots line leaves them out.\n")
    L.append("| card | Φ (gates/s) | a SM-clock | an s8 MAC-time | the LP's mix | binding units | pinned dots only: "
             "Φ, an s8 MAC-time, mix | without FP64 too: Φ, an s8 MAC-time, mix |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in G.ACCEL:
        p, q, v = D["phi"][c]["full"], D["phi"][c]["pinned"], D["phi"][c]["serving"]

        def mix(x):
            return ", ".join(f"{k} {pct(v, 1)}" for k, v in sorted(x.items(), key=lambda kv: -kv[1]))
        L.append(f"| {c} | {sci(p['per_s'], 3)} | {p['per_sm_clock']:.4g} | {p['per_s8_mac']:.0f} | {mix(p['mix'])} | "
                 f"{', '.join(p['binding'])} | {sci(q['per_s'], 3)}, {q['per_s8_mac']:.0f}, {mix(q['mix'])} | "
                 f"{sci(v['per_s'], 3)}, {v['per_s8_mac']:.0f}, {mix(v['mix'])} |")
    L.append("")
    L.append(f"𝓜's ticks: H200 {G.ACCEL['H200']['mhz']} MHz × {G.ACCEL['H200']['sms']} SMs, RTX PRO 6000 "
             f"{G.ACCEL['RTX PRO 6000']['mhz']} MHz × {G.ACCEL['RTX PRO 6000']['sms']} SMs ({G.ACCEL_REF}). The share "
             "uses Φ an s8 MAC-time, clock-free, so Φ at the clock the honest kernel's time is measured at (θ's "
             "2617 MHz on the RTX PRO 6000, above 𝓜's tick).\n")
    L.append("Gates a MAC of each dot step (`gates.dot_gates`): " + "; ".join(
        f"{c}: " + ", ".join(f"{f} {v:.1f}" for f, v in D["dots"][c].items()) for c in D["dots"]) + ". s_× = "
             f"{sx['s_x']:.1f} ({sx['at'][0]} {sx['at'][1]}); every counted format: " + ", ".join(
                 f"{k} {v:.1f}" for k, v in sx["each"].items()) + f"; FP4, not counted, {sx['fp4']:.1f}. s_× is "
             "read as the least size of any circuit for a counted operation: an assumption, since a builder bounds "
             "that least size from above only.\n")
    L.append("Lane functions, gates an instruction a lane: " + ", ".join(f"{k} {v}" for k, v in D["lanes"].items())
             + ". Elementwise steps: " + ", ".join(f"{k} {v}" for k, v in D["elementwise"].items()) + ".\n")

    # S_b.
    L.append("### S_b, a sampling block's public size\n")
    sg = D["sigma"]
    L.append("D5's resources priced in gates (`gates.d5_sigma`), a unit each, headline (lo) / upper (hi): "
             + "; ".join(f"{c}: " + ", ".join(f"{r} {sg[c]['lo'][r]:.6g} / {sg[c]['hi'][r]:.6g}" for r in sg[c]['lo'])
                         for c in cards)
             + ". tensor is a SM-clock at θ's s8 rate; issue, rf, vec and smem carry no gates. b cᵀ's term is at "
             "its own gates: " + ", ".join(f"{c} {D['bit'][c]:.4f} s8 MAC-times" for c in cards) + " (the H200's "
             "b1 MAC; sm_120's FP4 Strassen depth 2, (7/8)² FP4 MACs). Formats D5 prices on the tensor row at s8 "
             "MAC-time, against their own gates (< 1 errs low): " + "; ".join(
                 f"{c}: " + ", ".join(f"{k} {v:.3f}" for k, v in D["tensor_checks"][c].items()) for c in cards)
             + f". Table routes are left out: a 256-entry int16 read is {D['lookups']['lut1']:.0f} gates a MAC, "
             "a Four Russians read at least " + f"{min(D['lookups']['four_russians'].values()):.0f}"
             + " a bit MAC, each above the product it stands for.\n")
    sha = D["sha512"]
    L.append(f"The hashed column adds the round's SHA-512 compressions (the leaves, A and the coin expansion, "
             f"nofold's `comps` a naive MAC) at one compression's traced gates, {sha['total']:,} ({sha['and']:,} And2, "
             f"{sha['xor']:,} Xor2, {sha['not']:,} Not; `gates.sha512_gates`): what v1.1's credit carried, and the "
             "share it would give.\n")
    L.append("The wired column keeps D5's table routes with a read at its resources' gate prices (smem and issue carry "
             "none): the actual circuit is one run's (`Accounting.Actual`), so a read's address is wiring, not a "
             "multiplexer (RT-SIZE's RS1). The headline leaves the table routes out; the wired column is what the "
             "least size over the menu is if that reading holds.\n")
    L.append("| card, model | S_b a credited MAC: lo / hi | OPT* (dense) lo / hi | with the round's SHA-512 (lo), share "
             "| table reads as wiring (lo), share | static / PV | credited classes | the largest class's winning method |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            r = rows[card, "measured", m]
            s = r["size"]
            lg = s["largest"]
            meth = ", ".join(f"{k} {v:.2f}" for k, v in lg["mix"].items())
            L.append(f"| {card}, {SHORT[m]} | {s['s']:.1f} / {s['s_hi']:.1f} | {s['s_dense']:.1f} / "
                     f"{s['s_dense_hi']:.1f} | {s['s_hashed']:.1f}, "
                     f"{pct(r['share']['share'] * s['s_hashed'] / s['s'], 2)} | {s['s_wired']:.1f}, "
                     f"{pct(r['share']['share'] * s['s_wired'] / s['s'], 2)} | "
                     f"{s['static']['s']:.1f} / " + (f"{s['pv']['s']:.1f}" if s['pv']['s'] else "— (none credited)")
                     + f" | {s['classes']} | {lg['K']}×{lg['N']} L′ "
                     f"{lg['Lp']}: {meth} |")
    L.append("")

    # Exclusion.
    L.append("### The exclusion bound, per monitoring period over the census\n")
    L.append("U = Φ·T − certified size, at most U/s_× counted operations: Φ_fleet T = F/2 × Φ an FP8 MAC-time (the "
             "census's FLOP as the fleet's FP8 tensor capacity, 𝓜's E4M3 rate being its s8 rate on both cards), "
             "certified = (1 − ε_skip) S_reg, S_reg = share × Φ_fleet T. The serving-block bound summed over the "
             "monitoring period's serving blocks is the same sum (the next table). ε_dark's dark leaves add "
             "ε_dark Φ_fleet T / s_×.\n")
    L.append("| card, model | period | census F (FLOP) | Φ_fleet T (gates) | S_reg | U | U/s_× (counted ops) | "
             "census F/2 (MACs) | U/s_× over F/2 | + dark at ε_dark 1e-3 / 1e-2 (ops) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            ex = rows[card, "measured", m]["exclusion"]
            for w, x in ex.items():
                L.append(f"| {card}, {SHORT[m]} | {w} | {sci(F_W[w])} | {sci(x['cap'])} | {sci(x['S_reg'])} | "
                         f"{sci(x['U'])} | {sci(x['ops'])} | {sci(x['census_ops'])} | {x['ratio']:.2f} | "
                         f"{sci(x['dark'][1e-3], 1)} / {sci(x['dark'][1e-2], 1)} |")
    L.append("")
    L.append("The bound is weak: the uncertified gates of a monitoring period could complete "
             + ", ".join(f"{c} {min(rows[c, 'measured', m]['exclusion']['year']['ratio'] for m in models):.1f}–"
                         f"{max(rows[c, 'measured', m]['exclusion']['year']['ratio'] for m in models):.1f}×"
                         for c in cards)
             + " the census's counted operations, because Φ an FP8 MAC-time is "
             + ", ".join(f"{phi_s8(c) / sx['s_x']:.2f}× s_× on the {c}" for c in cards)
             + " and the credited share is a few percent. The census's own work as gates (its F/2 E4M3 MACs at the "
             "card's E4M3 dot) is " + ", ".join(
                 f"{pct(rows[c, 'measured', models[0]]['exclusion']['year']['census_gates'] / rows[c, 'measured', models[0]]['exclusion']['year']['cap'], 1)} of Φ_fleet T on the {c}"
                 for c in cards) + ".\n")
    L.append("U/s_× bounds counted operations only if each needs s_× gates of its own. Under gate sharing it doesn't: "
             "a constant-operand or Four Russians product takes fewer gates an operation (RT-SIZE's RS3), and the "
             "S_b table's wired column is that sharing in S_b. U/s_× over F/2 is RT-SIZE's vacuity V at the s8 "
             "rate: above 1, the bound allows more hidden counted work than the fleet's FP8 peak.\n")

    # Serving blocks.
    L.append("### Serving blocks (8-GPU servers, a wipe every 24 h)\n")
    L.append("| card, model | Φ_s (gates/s) | c_S (gates, padded to exactly) | k_S (bits) | U_S | U_S/s_× | serving "
             "blocks a week / month / year |")
    L.append("|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            sb = rows[card, "measured", m]["serving"]
            L.append(f"| {card}, {SHORT[m]} | {sci(sb['phi_s'], 3)} | {sci(sb['c_S'], 3)} | {sci(sb['k_S'], 3)} | "
                     f"{sci(sb['U_S'])} | {sci(sb['ops_S'])} | " + " / ".join(sci(sb['blocks'][w], 2)
                                                                             for w in MONITOR_S) + " |")
    L.append("")

    # ε_skip detail, channel, answers.
    L.append("### ε_skip, the channel term and the answers\n")
    L.append(f"q = {Q} checks a replica's registration; η = {pct(min(x['eta'] for x in D['eta'].values()), 4)} (the "
             "opened unit's free-guess bound at the picks' least L). ε_skip = ln(G/δ_coin)/(ηq), G the census's "
             "replicas; one replica alone is " + ", ".join(
                 f"{pct(rows[cards[0], 'measured', models[0]]['eps']['one_replica'][d], 3)} at δ_coin {d:g}"
                 for d in DELTAS) + ". Network outsourcing is the channel term k_infil/k_exfil in bits, not in "
             "ε_skip.\n")
    L.append("| card, model | G replicas | ε_skip 1e-6, weekly registrations: week / month / year | k_infil + k_exfil a "
             "replica a week (bits) | side channel a week | a serving block's ingress (bits) | answers: N at 1 − 1e-6,"
             " busy share of a week | H_o/H |")
    L.append("|---|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            r = rows[card, "measured", m]
            ch, a = r["channel"], r["answers"]
            L.append(f"| {card}, {SHORT[m]} | {sci(r['eps']['G'])} | " + " / ".join(
                pct(r["eps"]["weekly"][w][1e-6], 3) for w in MONITOR_S)
                + f" | {sci(ch['week'])} | {sci(ch['side_week'])} | {sci(ch['block'])} | {a['draws_hi']:.0f}, "
                  f"{pct(a['busy']['week'], 1)} | {pct(a['ho'], 3)} |")
    L.append("")

    # Beside.
    L.append("### v1.1's hidden beside work, in size\n")
    L.append("Size removes v1.1's pricing gap, and the beside work moves into U (RT-SIZE's RS4). S_b is public and "
             "fixed when the registration is declared, and no dual prices the period, so x plain INT8 MACs of other "
             "work a credited MAC beside the credited rounds change no credit: hidden in the credit is 0 at every x. "
             "They are uncertified capacity, inside U at their own gates (x · g_s8 a credited MAC), so U/s_× counts "
             "them only as far as s_× holds (the exclusion section). The tensor pipe idles 1 − 1/γ of a credited "
             "block, room for γ − 1 hidden s8 MACs a credited MAC at the s8 rate. v1.1's column (μ°, its art) is "
             "beside for comparison.\n")
    L.append("| card, model | hidden in the credit at x = 0.25 / 1 / 4 | beside gates a credited MAC | U a credited MAC "
             "(fits?) | tensor pipe idle, 1 − 1/γ | hidden s8 MACs a credited MAC it holds | v1.1's hidden at μ°, "
             "x = 0.25 / 1 / 4 |")
    L.append("|---|---|---|---|---|---|---|")
    for card in cards:
        for m in models:
            r = rows[card, "measured", m]
            b = [x for x in r["beside"] if x["x"] > 0]
            g = r["world"]["gamma"]
            L.append(f"| {card}, {SHORT[m]} | 0 / 0 / 0 | " + " / ".join(f"{x['gates']:.0f}" for x in b)
                     + f" | {b[0]['U']:.0f} (" + " / ".join("yes" if x["fits"] else "no" for x in b) + ") | "
                     + f"{pct(1 - 1 / g, 1)} | {g - 1:.2f} | "
                     + " / ".join(pct(v, 1) for v in V11["hidden"][card, m]) + " |")
    L.append("")

    # The rule.
    R = D["rule"]
    L.append("### The picks: v3.1's rule in size\n")
    L.append("| card | τ | mean share in size | mean γ | mean S_b | PV classes past γ 10 / served plain |")
    L.append("|---|---|---|---|---|---|")
    for card in cards:
        for t in D["taus"]:
            s = R["sweep"][card, t]
            mark = " (chosen)" if R["tau"][card] == t else ""
            mark += " (v1.1's)" if V11["tau"][card] == t else ""
            L.append(f"| {card} | {t:g}{mark} | {pct(np.mean(s['share']), 3)} | {np.mean(s['gamma']):.3f} | "
                     f"{np.mean(s['s']):.1f} | {s['infeasible']} / {s['plain']} |")
    L.append("")

    # What moved.
    L.append("### What moved since v1.1, and why\n")
    L.append("- **Credit's unit.** v1.1 credited pub(μ°), the public best at the honest period's duals in s8 "
             "MAC-times; v2 credits S_b in gates. The credited share falls from "
             + ", ".join(f"{pct(min(V11['share'][c, m] for m in models), 1)}–{pct(max(V11['share'][c, m] for m in models), 1)}"
                         f" to {pct(min(rows[c, 'measured', m]['share']['share'] for m in models), 2)}–"
                         f"{pct(max(rows[c, 'measured', m]['share']['share'] for m in models), 2)} ({c})" for c in cards)
             + ". Three things carry it. Φ counts the card's most gate-dense mix, so an s8 MAC-time is worth "
             + ", ".join(f"{phi_s8(c):.0f} gates on the {c}" for c in cards) + " against an s8 MAC's "
             + f"{sx['s_x']:.0f} (the share's S_b/Φ factor). γ now divides the share in full: the honest time past "
             "the products (hashing on the lanes above all) completes few gates of S_b, where μ° priced that time at "
             "the lanes' duals (the round's SHA-512 at its traced gates adds "
             + ", ".join(f"{min(rows[c, 'measured', m]['size']['s_hashed'] - rows[c, 'measured', m]['size']['s'] for m in models):.1f}–"
                         f"{max(rows[c, 'measured', m]['size']['s_hashed'] - rows[c, 'measured', m]['size']['s'] for m in models):.1f}"
                         f" gates a MAC on the {c}" for c in cards)
             + ", the S_b table's hashed column). And the routes win in gates, where a bit product is a few gates, "
             "so S_b sits below the dense program's size (OPT*).")
    L.append("- **ε.** v1.1's ε was the sampling term plus ε_net (exported bits priced in MACs), times 1 + H_o/H. "
             "v2's ε_skip is the sampling term alone, a share of S_reg; outsourcing is the channel term k_infil/"
             "k_exfil in bits; the answers' GPU time is uncertified capacity, inside U. One registration a monitoring "
             "period, so no union over weeks (the weekly variant keeps it).")
    L.append("- **The world rows are gone.** Each result is per monitoring period over the census: U, U/s_× and "
             "the census's own counted operations, the serving-block bound summed over its blocks.")
    L.append("- **Hidden beside work** in the credit is 0, since no dual prices the period; the work moves into U "
             "(above).")
    L.append("- **The picks.** The rule scores each τ's picks by the share in size; S_b barely moves with L′ (the "
             "menu's best route doesn't use the peels), so it favours the least γ.")
    L.append("- **Coverage of gates** is new: the elementwise steps, softmax above all, are uncredited gates.\n")

    # RT-SIZE.
    def rng(f):
        return ", ".join(f"{min(f(rows[c, 'measured', m]) for m in models):.1f}–"
                         f"{max(f(rows[c, 'measured', m]) for m in models):.1f} on the {c}" for c in cards)
    L.append("### RT-SIZE's checkpoint 1 (CMP-158, 19:56Z) against this build\n")
    L.append("- **RS1, values don't certify gates.** The wired column measures it here: with D5's table routes and a "
             "read as wiring, S_b (gates a credited MAC) is " + rng(lambda r: r["size"]["s_wired"])
             + ", against the headline's " + rng(lambda r: r["size"]["s"]) + ". The headline leaves table routes out because 𝓜 has no indexed "
             "load; in the actual circuit of one run that reason fails, so the headline S_b overstates the least "
             "size unless the credit rests on a premise in 𝓜's resources (RS1's fix: a claim, Daniel's).")
    L.append("- **RS2, a scalar Φ.** The share at Φ without FP64 and stand-ins shows how much of the low share is Φ "
             "counting what serving doesn't run; the companion build gives Φ in RT-SIZE's measure (ANDs).")
    L.append("- **RS3, s_× under sharing.** Stated with the exclusion bound: U/s_× holds only without gate sharing.")
    L.append("- **RS4, beside work moves into U.** The beside table gives the tensor pipe's idle share and what it "
             "holds.")
    L.append("- **RS5, late answers.** The answers' busy share of a week (the ε_skip table) is the serialized "
             "queue's load; near 95% its lag runs to hours. Queued check coins are ACCEL-SPEC's fix, and D6 v2 "
             "carries no timing change.\n")

    # Lean.
    L.append("### What the Lean needs (for the Lean worker after CMP-154)\n")
    for x in LEAN:
        L.append(f"- {x}")
    L.append("")
    L.append(verdict(D))
    return "\n".join(L)


LEAN = [
    "`PubBest` → public size: the menu's hull at a resource's gate price (`gates.d5_sigma`'s cheapest instruction a "
    "unit), no makespan variable and no duals; `SAKMatmul` becomes: no program for the block's products is smaller "
    "than S_b (an assumption on the menu, as today's hull statement is).",
    "`Hmac` (the credited MACs' price at μ°) → `Hsize`: S_b = s_b · (the block's MACs), s_b public, read from the "
    "registration; delete every μ°, μ¹ and dual-price lemma that only `Hmac` used.",
    "`W_b` (a block's weight in the draw) → S_b / S_reg: the challenger draws a gate of S_reg, so a proof block in "
    "proportion to its public size; `ε` → `ε_skip`, a share of S_reg, with δ_coin for the draw and δ_crypto beside.",
    "`W_reg` → S_reg = Σ_b S_b, declared at the monitoring period's start (P21 as structure).",
    "Φ: a per-card constant in gates a second, `Accel`'s LP over its units with each `Op`'s gate content (its "
    "`Lane.eval`/`dmmaEval` circuit's size) as the objective, and s_× (the fewest gates of a counted operation) as a "
    "named assumption (a lower bound the builders don't prove).",
    "U(X, J) = Φ_X|J| − |A[X,J] ∩ W_cert| and its U/s_× counted operations, per serving block of exactly c_S gates "
    "(padded), summed over a monitoring period's blocks; ε_dark's leaves as q_U(L_dark).",
    "ε_net → the channel term k_infil/k_exfil in bits at the network certifier's boundary, no MAC price.",
    "Everything on LEAN-63's single circuit type and partition.",
]


def jsonable(x):
    if isinstance(x, dict):
        return {" | ".join(map(str, k)) if isinstance(k, tuple) else str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--pickle")
    ap.add_argument("--from", dest="src")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.src:
        D = pickle.loads(Path(a.src).read_bytes())
        MODELS, TAUS = D["models"], D["taus"]
        assert D["measure"] == G.MEASURE, f"this pickle is in {D['measure']}: set ACCEL_GATE_MEASURE={D['measure']}"
    else:
        D = d6.quiet(build, a.quick)
    if a.pickle:
        Path(a.pickle).write_bytes(pickle.dumps(D))
    if a.json:
        Path(a.json).write_text(json.dumps(jsonable(D), indent=1, default=str))
    print(report(D))
