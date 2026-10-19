"""The fit bound: when no server can hold a model that reaches the target loss, training must cross servers, and every
crossing is a server's intake.

The memory cap's closed form in ``tiers.py``, mu = 4 R_wipe lambda_in / (kappa d b_w b_a F), needs (d, e)-threading at the
scale of a wipe window. The proved layered threading (``ThreadedOfLayered``) gives it only while a region holds less than
one embedding column's worth of state: a window holding the embedding and the first layers computes a token's work from
its 17-bit id. The fit bound needs threading only at server scale, where the loss premise makes it hold. Its premises, per
server window from one wipe to the next:

* F1, capacity (C1 per weight, not per region): a window that multiplies p distinct updated weights reads at least b_w
  bits for each of them, the weight half of ``WeightsIncompressible`` at window scale. Master weights, optimizer state and
  stored activations only add to it; ``B_TRAIN`` is the training-storage premise that charges them.
* F2, state: a window reads at most S = R_wipe + lambda_in T_wipe bits.
* F3, loss (empirical): every sequence's computation in a model that reaches the target loss L* depends on at least N*
  updated weights beyond the registered models; ``n_star`` reads N* off a loss law, at no data limit unless given one.
* F4, boundary (empirical): a window that does work on a token it didn't start reads at least w_e bits of that token's
  values: its hidden state crossing in and its gradient crossing back, or its context's keys and values, which cost more.
* F5, intake: a server reads at most lambda_in = sigma + tau + alpha F bits per second (``tiers.ingress``).

A token then visits k >= k0 = ceil(N* b_w / S) windows, gets at most 2 S / b_w counted multiplications in each, and pays
w_e in all but the one that starts it, so when k0 >= 2 the fleet's counted multiplications per server-second are at most

    2 lambda_in S k0 / ((k0 - 1) b_w w_e),

a share 4 lambda_in S k0 / (kappa (k0 - 1) b_w w_e F) of server FLOP. Against the closed form the ratio is
(S / R_wipe) (d b_a / w_e) k0 / (k0 - 1): the window bound with the thread's entry fee d b_a measured at server scale.
When S >= N* b_w the bound is void, and what remains is the replica attack (``replica``): a server-sized model synced
through intake, which reaches L* only if it holds N* weights at its training precision.

    python benchmarks/nci/fit.py            # the table at r = 0.255 and mfu 0.4
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

R_SHARE = 0.255
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
    """``tiers``' defaults with a wipe residual of ``wipe_gb`` (memory the fill doesn't reach, no proof-of-space slack) and
    the monitored residual at ``R_SHARE``."""
    p = T.DEFAULTS | {"unfilled_gb": wipe_gb, "pous_rounds": 1e9, "pous_floor": 0.0}
    return p | {"residual": R_SHARE - p["host_share"]} | over


def intake(p: dict) -> float:
    return T.ingress(p, T.TIERS["T3"])


def state(p: dict, wipe_bits: float) -> float:
    """F2's S: what a server window from one wipe to the next reads."""
    return wipe_bits + intake(p) * p["wipe_s"]


def servers(S: float, N: float, b: float) -> int:
    """k0: the fewest windows a token visits when each holds at most ``S / b`` of its ``N`` weights."""
    return math.ceil(N * b / S)


def fit_rate(p: dict, S: float, N: float, w_e: float, b: float | None = None) -> float:
    """Counted multiplications per server-second under F1-F5, or ``inf`` when one window holds ``N`` weights."""
    b = p["b_w"] if b is None else b
    k = servers(S, N, b)
    return 2 * intake(p) * S * k / ((k - 1) * b * w_e) if k >= 2 else math.inf


def share(p: dict, rate: float) -> float:
    """Counted multiplications per server-second as a share of server FLOP, the unit of ``tiers``' efficiency terms."""
    return rate * 2 / p["kappa"] / p["node_flops"]


def closed_form(p: dict, wipe_bits: float) -> float:
    return 4 * wipe_bits * intake(p) / (p["kappa"] * p["d"] * p["b_w"] * p["b_a"] * p["node_flops"])


def residual_with_lag(p: dict) -> float:
    return min(1.0, T.residual_share(p) + p["identities"] * p["lag_s"] / (p["years"] * T.YEAR))


def dampener(p: dict, mu: float, declared: float, cheat: float = 0.0) -> float:
    """D under a memory cap ``mu``: 1/D = (min(r + declared, mu) + cheat) / mfu, as in ``tiers.evaluate``."""
    return max(1.0, p["mfu"] / (min(residual_with_lag(p) + declared, mu) + cheat))


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
    mu = {k: share(p, fit_rate(p, S, N, w)) for k, w in BOUNDARY.items()}
    mu_train = share(p, fit_rate(p, S, N, BOUNDARY["central"], B_TRAIN["central"]))
    rep = replica(p, S, L, B_TRAIN["central"], fit)
    path = replica(p, S, L, B_TRAIN["central"], fit, PATH_GAIN["smalltalk"], PATH_MULTIPLIER["smalltalk"], 0.0)
    return {
        "wipe_gb": wipe_gb, "state_gb": S / T.GB, "k0": servers(S, N, p["b_w"]),
        "mu_closed": closed_form(p, R), "D_closed": closed["dampener"],
        "mu_fit": mu["central"], "D_fit": dampener(p, mu["central"], declared),
        "D_fit_low": dampener(p, mu["low"], declared), "D_fit_high": dampener(p, mu["high"], declared),
        "fit_over_closed": mu["central"] / closed_form(p, R),
        "k0_train": servers(S, N, B_TRAIN["central"]), "mu_fit_train": mu_train,
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
    print(f"r = {R_SHARE}, mfu = {T.DEFAULTS['mfu']}, lambda_in = {s['intake']:.4g} bit/s, T_wipe = "
          f"{T.DEFAULTS['wipe_s']:.0f} s, b_w = {T.DEFAULTS['b_w']}, w_e = "
          + " / ".join(f"{w:.0f}" for w in BOUNDARY.values()) + " bits (low / central / high)\n")
    print(f"{'R_wipe':>7} {'S':>8} {'k0':>3} {'mu_closed':>10} {'D_closed':>8} {'mu_fit':>9} {'D_fit':>6} "
          f"{'D_fit, w_e low-high':>19} {'fit/closed':>10}")
    for r in s["rows"]:
        void = r["k0"] < 2
        fit_cols = (f"{'void':>9} {_d(r['D_none']):>6}" if void else
                    f"{r['mu_fit']:9.3g} {_d(r['D_fit']):>6} {_d(r['D_fit_low']) + ' - ' + _d(r['D_fit_high']):>19} "
                    f"{r['fit_over_closed']:10.2f}")
        print(f"{r['wipe_gb']:5.0f}GB {r['state_gb']:6.1f}GB {r['k0'] if not void else '-':>3} {r['mu_closed']:10.3g} "
              f"{_d(r['D_closed']):>8} {fit_cols}")
    print(f"\nAt {B_TRAIN['central']:.0f} bits per trained weight (the training-storage premise): the fit bound, the "
          f"replica attack, and a x{PATH_GAIN['smalltalk']:.1f} path composition at m {PATH_MULTIPLIER['smalltalk']}")
    print(f"{'R_wipe':>7} {'k0':>3} {'D_fit':>6} {'replica':>8} {'tokens to L*':>12} {'m':>7} {'D':>6} "
          f"{'path, dense':>11} {'m':>7} {'D':>6}")
    for r in s["rows"] + s["replica_rows"]:
        rep, path = r["replica"], r["path"]
        print(f"{r['wipe_gb']:5.0f}GB {r['k0_train']:>3} {_d(r['D_fit_train']) if r['k0_train'] >= 2 else 'void':>6} "
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
