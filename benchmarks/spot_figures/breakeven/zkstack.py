"""The two-stage audit's weight-theft budget in its per-window form: the evasion threshold paid in each of the period's T
audit windows, so the replay blocks each window may hide, K_δ − 1, add up over the windows, where
`optimize.two_stage_rankin` (V5's form) pays δ once over the period:

    log2|S| = log2(T + 1) + log2 Σ_{j <= K} C(T|O|, j) |V|^j,   K = E_ρ M,   M = T (K_δ − 1) + 1,

K_δ the least K with x^K <= δ/(c² T), x = 1 − r₁ r₂, r₁ the first stage's rate and r₂ the least min(k_t, n_t)/n_t over
strata.  A replay block (a request) reaches E_ρ served slots of |V| values, its served bits; T|O| is every served slot
of the period.  |Adv| and |Tick| are 1 (no advice, the certified tick), and the lag term (L + 1)|O|, which V5's form has
too, is left out.  The design is the planner's family (`optimize.two_stage_design`'s thresholds and κ, uniform and
stratified, and every stratum proved whole), each at the most q the budget pays for, the one of least budget taken.
"""
from math import ceil, e, inf, log, log1p, log2
from dataclasses import replace

from common import c, f1, opt

#: the period's audit windows: its seconds over the window's
WINDOW_S = c["window"].value
PERIOD_S = dict(f1.PERIODS)
#: a served slot's bits, the token select's output word
SLOT_BITS = 8 * c["token_bytes"].value
#: the extractor's constant c in δ/(c² T)
C_EXTRACT = 1.0


def windows(scope: f1.Scope, per_machine: bool = False) -> float:
    """T: the period's audit windows, fleet-wide, or each machine's counted apart (`per_machine`)."""
    t = PERIOD_S[scope.period] / WINDOW_S
    return t * scope.gpu_seconds / PERIOD_S[scope.period] / c["machine_gpus"].value if per_machine else t


def k_delta(r: float, delta: float) -> float:
    """The least K with (1 − r)^K <= delta."""
    if r >= 1:
        return 1
    if r <= 0:
        return inf
    return max(1, ceil(log(delta) / log1p(-r) - 1e-9))


def bits(r: float, t: float, delta: float, served: float, replay_blocks: float) -> float:
    """log2|S| at detection rate r = r₁ r₂ over T = `t` windows, each replay block serving `served` bits."""
    kd = k_delta(r, delta / (C_EXTRACT ** 2 * t))
    if kd == inf:
        return inf
    e_rho = served / SLOT_BITS
    k = e_rho * (t * (kd - 1) + 1)
    slots = replay_blocks * e_rho
    every = slots * log2(1 + 2.0 ** SLOT_BITS)
    return log2(t + 1) + (every if k >= slots else min(every, k * (SLOT_BITS + log2(e * slots / k))))


def design(g, m, budget: float, span: float, t: float) -> dict:
    """The planner's designs at `budget`, each priced as `optimize.two_stage_design` prices it, under this form."""
    cells = list(g.cells)
    classes = [opt._size_class(x.bits) for x in cells]
    rows = []
    for merged in sorted(set(classes)):
        split = [replace(x, stratum=0 if w <= merged else w) for x, w in zip(cells, classes)]
        strata: dict[int, float] = {}
        for x in split:
            strata[x.stratum] = strata.get(x.stratum, 0.0) + x.n
        charge = sum(x.n * x.bits for x in split)
        widest = {s: max(min(x.bits, m.served) for x in split if x.stratum == s) for s in strata}
        ks = [{s: min(ceil(n), max(1, ceil(kappa * n * (widest[s] + 1 / log(2) + log2(g.replay_blocks * n)) / charge)))
               for s, n in strata.items()} for kappa in opt.KAPPAS]
        for k in [*ks, {s: ceil(n) for s, n in strata.items()}]:
            left = budget / g.replay_blocks
            per = g.replay + opt.two_stage_price(split, k)[2]
            q = min(1.0, left / per) if per > 0 else 1.0
            if span and q < 1:
                q = opt._registration_q(split, k, span, g.replay, left, q)
            r2 = min(min(k[s], n) / n for s, n in strata.items())
            rows.append({"merged": merged, "k": k, "q": q, "r2": r2,
                         "kdelta": k_delta(q * r2, m.delta / (C_EXTRACT ** 2 * t)),
                         "bits": bits(q * r2, t, m.delta, m.served, g.replay_blocks),
                         "replay_sample": q * g.replay_blocks, "registration_sample": q * span})
    return min(rows, key=lambda r: r["bits"])


def bound(m, g, budget: float, span: float, full: bool = False, per_machine: bool = False):
    """The line's exfiltration bound at `budget` GPU-seconds under this form, capped at what the scope serves."""
    t = windows(m.scope, per_machine)
    row = design(g, m, budget, span, t)
    y = min(row["bits"], f1.served_total(m))
    pt = {"y": y, "count": row["bits"], "windows": t, "plan": {"replay_sample": row["replay_sample"]},
          "registration_sample": row["registration_sample"], "q": row["q"], "r2": row["r2"], "kdelta": row["kdelta"]}
    return (y, pt) if full else y
