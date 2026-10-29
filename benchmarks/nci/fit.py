"""The fit bound: when no server window can hold a model that reaches the target loss, a token's work spans windows, and
every window it enters reads its state.

The memory cap's closed form in ``tiers.py``, mu = 4 R_wipe lambda_in / (kappa d b_w b_a F), needs (d, e)-threading at the
scale of a wipe window. The proved layered threading (``ThreadedOfLayered``) gives it only while a region holds less than
one embedding column's worth of state: a window holding the embedding and the first layers computes a token's work from
its 17-bit id. The fit bound needs threading only at server scale, where the loss premise makes it hold. Its premises, per
server window of whole wipe periods (``tiers.windows_bound``'s windows):

* F1, capacity (C1 per weight, not per region): a window that multiplies p distinct updated weights and works on e tokens
  it didn't start reads at least b_w p + w_e e bits: the weight half of ``WeightsIncompressible`` at window scale, with
  F4's w_e as its thread half. Master weights, optimizer state and stored activations only add to it; ``B_TRAIN`` is the
  training-storage premise that charges them.
* F2, state: a window of L seconds reads at most S = R_wipe + lambda_in L bits, lambda_in = sigma + tau + alpha F
  (``tiers.ingress``): the residual at the wipe that opens it and its intake. What crosses a later wipe inside it isn't
  novel; what crosses the one that opens it is, so a token carried across a wipe in the residual pays like one carried in.
* F3, loss (empirical): every token's computation in a dense model that reaches the target loss L* depends on at least
  N* updated weights beyond the registered models (``ReachSlots`` at token threads, which the policy adopts for dense
  recipes; false for a token-routed MoE, whose tokens use fewer); ``n_star`` reads N* off a loss law, at no data limit
  unless given one. Each server's first window, which builds its weights from a seed, is charged (``left_out``).
* F4, boundary (empirical): one window starts each token, from its id; any other window that does work on it reads at
  least w_e bits of its values: its hidden state crossing in and its gradient crossing back, or its context's keys and
  values, which cost more. An ensemble, whose members each start the token, is outside it.

A token's start window gives it at most S / b_w of its N* weights, so the windows it enters give it at least the rest,
and its work is at most k' / (k' - 1) times its work there, k' = N* b_w / S. A window that enters e tokens with p weights
does at most 2 p e of their work, and b_w p + w_e e <= S caps p e at S^2 / (4 b_w w_e). A window therefore does at most

    2 N* S^2 / (4 w_e (N* b_w - S))

counted multiplications: ``tiers.region_cap``'s 2 K^2 / (4 d b_w b_a) with the thread's entry fee d b_a replaced by
w_e (k' - 1) / k', paid at window scale. Over its length, at the least over window lengths, it bounds counted
multiplications per server-second, as the closed form does at d b_a (which reaches its least at L = R_wipe / lambda_in).
When a one-period window holds N* weights (S >= N* b_w) the bound is void, and what remains is the replica attack
(``replica``): a server-sized model synced through intake, which reaches L* only if it holds N* weights at its training
precision.

    python benchmarks/nci/fit.py            # the table at the ladder's r-hat (PoUW's floor credit) and mfu 0.4
    python benchmarks/nci/fit.py --json     # the same rows, for other renderers
"""

from __future__ import annotations

import argparse
import json
import math

import tiers as T

#: loss laws L(N, D) = E + A / N^alpha + B / D^beta in nats per token: Besiroglu et al.'s refit of Chinchilla's Approach 3
#: (besiroglu-2024-chinchilla-replication/2) and Hoffmann et al.'s original Approach 3 fit, which the refit refutes and
#: which stays as the low case
FITS = {
    "besiroglu": {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658},
    "hoffmann": {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28},
}

#: frontier targets as a dense model and its tokens with the same loss: Llama 3.1 405B base, which DeepSeek-V3 base
#: matches on Pile-test (0.542 vs 0.548 bits per byte, deepseekai-2024-deepseek-v3/6); and Kimi K2 base (1.04T total, 32B
#: active, 15.5T tokens), at least as good as DeepSeek-V3 and at most as good as a dense 1T model on its tokens
TARGETS = {"405b": (405e9, 15.6e12), "k2": (1.0e12, 15.5e12)}

#: F4's w_e, the bits a token's crossing reads, forward plus backward, at d_model = 7168
#: (activation.boundary_bits/d7168): a d/2 bottleneck at 4 bits forward with gradients at 1.5x the forward bits (central);
#: d at 8 bits both ways, today's practice (high); Subspace Networks' contested 100x forward code with local learning,
#: which sends no gradient back (low)
BOUNDARY = {"low": 1147.0, "central": 14336.0 * 2.5, "high": 57344.0 * 2.0}

#: bits a server holds per weight it trains: the C1 floor b_w; BF16 weights under a factored or stateless optimizer; and
#: mixed-precision Adam (FP32 master and two moments, BF16 copy)
B_TRAIN = {"floor": 2.5, "central": 16.0, "adam": 128.0}

#: sequence-routed compositions of small models, as dense parameters per path parameter: SmallTalk's 32 x 335M experts
#: match a 1.3B dense model at about equal FLOPs (filippova-2024-smalltalk/1); DiPaCo's 256 paths of 150M match it at
#: about 21x its FLOPs (douillard-2024-dipaco/1); neither is tested above 1.3B, and BTX finds BTM below dense at 7B
#: (sukhbaatar-2024-btx/1)
PATH_GAIN = {"none": 1.0, "smalltalk": 1.3e9 / 335e6, "dipaco": 1.3e9 / 150e6}
PATH_MULTIPLIER = {"none": 1.0, "smalltalk": 0.9, "dipaco": 0.05}

#: DiLoCo's multiplier at M = 4 replicas, 2.4B-10B (charles-2025-diloco-scaling/2, /4); nothing tests M above 8
DILOCO = 0.94

WIPES_GB = (1.0, 4.0, 6.0, 13.0, 50.0, 166.0)
#: wipe residuals past the table's, where a replica first holds N* weights at 16 bits
REPLICA_GB = (203.0, 230.0, 300.0, 350.0)
DATA_LIMITS = (("inf", math.inf), ("1e15", 1e15), ("1e14", 1e14))


def loss(fit: str, N: float, D: float = math.inf) -> float:
    f = FITS[fit]
    return f["E"] + f["A"] / N ** f["alpha"] + (f["B"] / D ** f["beta"] if math.isfinite(D) else 0.0)


def target_loss(fit: str, target: str) -> float:
    return loss(fit, *TARGETS[target])


def n_star(fit: str, L: float, D_max: float = math.inf) -> float:
    """F3's N*: the fewest parameters with which the loss law reaches ``L`` on at most ``D_max`` tokens."""
    f = FITS[fit]
    room = L - f["E"] - (f["B"] / D_max ** f["beta"] if math.isfinite(D_max) else 0.0)
    return (f["A"] / room) ** (1 / f["alpha"]) if room > 0 else math.inf


def tokens_to_reach(fit: str, N: float, L: float) -> float:
    f = FITS[fit]
    room = L - f["E"] - f["A"] / N ** f["alpha"]
    return (f["B"] / room) ** (1 / f["beta"]) if room > 0 else math.inf


def optimal_compute(fit: str, L: float) -> float:
    """6 N D of the cheapest run reaching ``L``, where the parameter and data terms stand in ratio beta : alpha."""
    f = FITS[fit]
    room = L - f["E"]
    a = room * f["beta"] / (f["alpha"] + f["beta"])
    return 6 * (f["A"] / a) ** (1 / f["alpha"]) * (f["B"] / (room - a)) ** (1 / f["beta"])


def undersizing(fit: str, N: float, L: float) -> float:
    """m_N: the cheapest run's FLOP to reach ``L`` over those of a run fixed at ``N`` parameters; 0 when ``N`` never gets
    there."""
    D = tokens_to_reach(fit, N, L)
    return optimal_compute(fit, L) / (6 * N * D) if math.isfinite(D) else 0.0


def params(wipe_gb: float, **over: float) -> dict:
    """``tiers``' defaults with a wipe residual of ``wipe_gb`` (memory the fill doesn't reach, no proof-of-space slack):
    the monitored residual is the ladder's, PoUW's floor credit at the locked clock."""
    return T.DEFAULTS | {"unfilled_gb": wipe_gb, "pous_rounds": 1e9, "pous_floor": 0.0} | over


def intake(p: dict) -> float:
    return T.ingress(p, T.TIERS["T3"])


def state(p: dict, wipe_bits: float, periods: int = 1) -> float:
    """F2's S: what a server window of ``periods`` whole wipe periods reads."""
    return wipe_bits + intake(p) * periods * p["wipe_s"]


def visits(S: float, N: float, b: float) -> float:
    """k': the windows' worth of weights a token's ``N`` needs when each window holds at most ``S / b``."""
    return N * b / S


def window_mults(S: float, N: float, w_e: float, b: float) -> float:
    """The most counted multiplications a window reading ``S`` bits does under F1-F4, or ``inf`` once it holds ``N``
    weights."""
    return 2 * N * S ** 2 / (4 * w_e * (N * b - S)) if S < N * b else math.inf


def fit_rate(p: dict, wipe_bits: float, N: float, w_e: float, b: float | None = None) -> tuple[float, int]:
    """Counted multiplications per server-second under F1-F4, the least over windows of whole wipe periods, and the
    periods of the window that gives it; ``(inf, 0)`` when a one-period window holds ``N`` weights."""
    b = p["b_w"] if b is None else b
    best, periods, m = math.inf, 0, 1
    while (S := state(p, wipe_bits, m)) < N * b:
        rate = window_mults(S, N, w_e, b) / (m * p["wipe_s"])
        if rate < best:
            best, periods = rate, m
        m += 1
    return best, periods


def share(p: dict, rate: float) -> float:
    """Counted multiplications per server-second as a share of server FLOP, the unit of ``tiers``' efficiency terms."""
    return rate * 2 / p["kappa"] / p["node_flops"]


def left_out(p: dict, periods: int) -> float:
    """What the fit bound leaves out, in ``share``'s unit: each server's first window of ``periods`` wipe periods, at mfu,
    over the horizon. That window builds its weights from a seed, so F1's weight half fails there, and
    ``Nci.SecurityProofs.Fit.fitBoundSlots_init`` adds the counted multiplications inside it to the bound. The threads
    it starts need no F3: their work in later windows is paid for by their entry fees there. F3 is asked, outside the
    first windows, of the threads started later, which do no work in a first window when every server starts with the
    run. A server that joins later, or a recipe that seeds weights after a server's first window (progressive growth),
    is outside it."""
    return p["mfu"] * periods * p["wipe_s"] / (p["years"] * T.YEAR)


def fit_mu(p: dict, wipe_bits: float, N: float, w_e: float, b: float | None = None) -> float:
    """The fit bound's cap on monitored efficiency: ``fit_rate``'s share plus ``left_out``; ``inf`` where it's void."""
    rate, periods = fit_rate(p, wipe_bits, N, w_e, b)
    return share(p, rate) + left_out(p, periods)


def closed_form(p: dict, wipe_bits: float) -> float:
    return 4 * wipe_bits * intake(p) / (p["kappa"] * p["d"] * p["b_w"] * p["b_a"] * p["node_flops"])


def residual_with_lag(p: dict) -> float:
    return min(1.0, T.residual_share(p) + p["identities"] * p["lag_s"] / (p["years"] * T.YEAR))


def dampener(p: dict, mu: float, declared: float, cheat: float = 0.0) -> float:
    """D under a memory cap ``mu``: 1/D = (min(r + declared, m mu) + cheat) / mfu, as in ``tiers.evaluate``."""
    return max(1.0, p["mfu"] / (min(residual_with_lag(p) + declared, p["m"] * mu) + cheat))


def attack_dampener(p: dict, efficiency: float) -> float:
    """D of a developer whose only route to the target is an attack of this efficiency; ``inf`` when it never gets there."""
    return max(1.0, p["mfu"] / efficiency) if efficiency > 0 else math.inf


def replica(p: dict, S: float, L: float, b_train: float, fit: str = "besiroglu", gain: float = 1.0,
            multiplier: float = DILOCO, b_sync: float = 4.0) -> dict:
    """The replica attack: each server trains the largest model it holds at ``b_train`` bits per weight (worth ``gain``
    dense weights each, for a path composition) and syncs a pseudo-gradient of ``b_sync`` bits per weight through
    intake. Between syncs a replica trains r F b_sync / (6 lambda_in) tokens, whatever its size."""
    N = S / b_train
    lam = intake(p)
    r = residual_with_lag(p)
    m = multiplier * undersizing(fit, gain * N, L)
    sync_s = N * b_sync / lam
    return {
        "params": N, "dense_params": gain * N, "tokens": tokens_to_reach(fit, gain * N, L), "m": m,
        "sync_s": sync_s, "tokens_per_sync": r * p["node_flops"] * b_sync / (6 * lam),
        "outer_steps": p["years"] * T.YEAR / sync_s if sync_s > 0 else math.inf, "efficiency": r * m,
    }


def bite_gb(p: dict, N: float, b: float) -> float:
    """The largest wipe residual at which the fit bound holds: S < N b."""
    return (N * b - intake(p) * p["wipe_s"]) / T.GB


def row(wipe_gb: float, N: float, L: float, fit: str = "besiroglu") -> dict:
    p = params(wipe_gb)
    closed = T.evaluate(p, T.TIERS["T4"])
    R = closed["wipe_state_bits"]
    S = state(p, R)
    declared = closed["declared"]
    fits = {k: fit_rate(p, R, N, w) for k, w in BOUNDARY.items()}
    mu = {k: share(p, rate) + left_out(p, periods) for k, (rate, periods) in fits.items()}
    rate_train, periods_train = fit_rate(p, R, N, BOUNDARY["central"], B_TRAIN["central"])
    mu_train = share(p, rate_train) + left_out(p, periods_train)
    rep = replica(p, S, L, B_TRAIN["central"], fit)
    path = replica(p, S, L, B_TRAIN["central"], fit, PATH_GAIN["smalltalk"], PATH_MULTIPLIER["smalltalk"], 0.0)
    return {
        "wipe_gb": wipe_gb, "state_gb": S / T.GB, "visits": visits(S, N, p["b_w"]), "periods": fits["central"][1],
        "mu_closed": closed_form(p, R), "D_closed": closed["dampener"],
        "mu_fit": mu["central"], "D_fit": dampener(p, mu["central"], declared),
        "D_fit_low": dampener(p, mu["low"], declared), "D_fit_high": dampener(p, mu["high"], declared),
        "fit_over_closed": mu["central"] / closed_form(p, R),
        "visits_train": visits(S, N, B_TRAIN["central"]), "periods_train": periods_train, "mu_fit_train": mu_train,
        "D_fit_train": dampener(p, mu_train, declared),
        "replica": rep, "D_replica": attack_dampener(p, rep["efficiency"]),
        "path": path, "D_path": attack_dampener(p, path["efficiency"]),
        "D_none": dampener(p, math.inf, declared),
    }


def summary(fit: str = "besiroglu") -> dict:
    p = params(WIPES_GB[0])
    L = {t: target_loss(fit, t) for t in TARGETS}
    N = {t: {k: n_star(fit, L[t], D) for k, D in DATA_LIMITS} for t in TARGETS}
    n405 = N["405b"]["inf"]
    return {
        "fit": fit, "target_loss": L, "n_star": N,
        "n_star_hoffmann": {t: n_star("hoffmann", target_loss("hoffmann", t)) for t in TARGETS},
        "intake": intake(p), "rows": [row(R, n405, L["405b"], fit) for R in WIPES_GB],
        "replica_rows": [row(R, n405, L["405b"], fit) for R in REPLICA_GB],
        "bite_gb": {
            f"{t}/{b}/{g}": bite_gb(p, N[t]["inf"] / PATH_GAIN[g], B_TRAIN[b])
            for t in TARGETS for b in ("floor", "central") for g in ("none", "smalltalk")
        },
        "replica_onset_gb": {k: bite_gb(p, N["405b"][k], B_TRAIN["central"]) for k, _ in DATA_LIMITS},
    }


def _d(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:,.0f}" if x < 1e5 else f"{x:.2g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="targets, N*, the table and the regime as JSON")
    args = ap.parse_args(argv)
    s = summary()
    if args.json:
        print(json.dumps(s, indent=1, default=str))
        return 0
    L, N = s["target_loss"], s["n_star"]
    print(f"405B-class target L* = {L['405b']:.4f}: N* = {N['405b']['inf'] / 1e9:.1f}B at no data limit, "
          f"{N['405b']['1e15'] / 1e9:.0f}B at 1e15 tokens, {N['405b']['1e14'] / 1e9:.0f}B at 1e14; "
          f"Hoffmann's fit {s['n_star_hoffmann']['405b'] / 1e9:.1f}B")
    print(f"K2-class target L* = {L['k2']:.4f}: N* at most {N['k2']['inf'] / 1e9:.0f}B")
    print(f"r-hat = {T.audited_residual(T.DEFAULTS):.3g}, mfu = {T.DEFAULTS['mfu']}, lambda_in = {s['intake']:.4g} bit/s, T_wipe = "
          f"{T.DEFAULTS['wipe_s']:.0f} s, b_w = {T.DEFAULTS['b_w']}, w_e = "
          + " / ".join(f"{w:.0f}" for w in BOUNDARY.values()) + " bits (low / central / high)\n")
    print("S and k' = N* b_w / S at one wipe period; L, the window in wipe periods at which the fit bound is least\n")
    print(f"{'R_wipe':>7} {'S':>8} {'k':>5} {'mu_closed':>10} {'D_closed':>8} {'L':>3} {'mu_fit':>9} {'D_fit':>6} "
          f"{'D_fit, w_e low-high':>19} {'fit/closed':>10}")
    for r in s["rows"]:
        void = r["periods"] == 0
        fit_cols = (f"{'-':>3} {'void':>9} {_d(r['D_none']):>6}" if void else
                    f"{r['periods']:3d} {r['mu_fit']:9.3g} {_d(r['D_fit']):>6} "
                    f"{_d(r['D_fit_low']) + ' - ' + _d(r['D_fit_high']):>19} {r['fit_over_closed']:10.2f}")
        print(f"{r['wipe_gb']:5.0f}GB {r['state_gb']:6.1f}GB {r['visits']:5.2f} {r['mu_closed']:10.3g} "
              f"{_d(r['D_closed']):>8} {fit_cols}")
    print(f"\nAt {B_TRAIN['central']:.0f} bits per trained weight (the training-storage premise): the fit bound, the "
          f"replica attack, and a x{PATH_GAIN['smalltalk']:.1f} path composition at m {PATH_MULTIPLIER['smalltalk']}")
    print(f"{'R_wipe':>7} {'k':>5} {'L':>3} {'D_fit':>6} {'replica':>8} {'tokens to L*':>12} {'m':>7} {'D':>6} "
          f"{'path, dense':>11} {'m':>7} {'D':>6}")
    for r in s["rows"] + s["replica_rows"]:
        rep, path = r["replica"], r["path"]
        void = r["periods_train"] == 0
        print(f"{r['wipe_gb']:5.0f}GB {r['visits_train']:5.2f} {'-' if void else r['periods_train']:>3} "
              f"{'void' if void else _d(r['D_fit_train']):>6} "
              f"{rep['params'] / 1e9:7.0f}B {rep['tokens']:12.3g} {rep['m']:7.2g} {_d(r['D_replica']):>6} "
              f"{path['dense_params'] / 1e9:10.0f}B {path['m']:7.2g} {_d(r['D_path']):>6}")
    rep = s["replica_rows"][0]["replica"]
    print(f"\nA replica trains {rep['tokens_per_sync']:.3g} tokens between FP4 syncs whatever its size; the "
          f"{rep['params'] / 1e9:.0f}B one at {s['replica_rows'][0]['wipe_gb']:.0f} GB syncs every "
          f"{rep['sync_s'] / 3600:.0f} h, {rep['outer_steps']:.0f} outer steps a year.")
    print("A 16-bit replica first reaches the 405B-class target at "
          + ", ".join(f"{gb:.0f} GB ({k} tokens)" for k, gb in s["replica_onset_gb"].items()) + ".")
    print("The fit bound holds while S < N* b, at wipe residuals up to:")
    for key, gb in s["bite_gb"].items():
        t, b, g = key.split("/")
        print(f"  {t:4} at {B_TRAIN[b]:4.1f} bits per weight, path gain x{PATH_GAIN[g]:.1f}: {gb:6.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
