"""The price-vector LP: a convex combination B_θ = Σ θ_r·row_r of the machine's rows.

Charnes–Cooper over λ ≥ 0 with p_MMA = 4096 (a tensor MAC costs 1); every constraint is homogeneous:

- (i) ρ_k := (p_k/32)/(p_MMA/128) ≥ ρ for every CUDA lane kind;
- every credit (kind, c): p_kind ≥ c MACs at the MAC price;
- every attack: its program costs X ≥ x_min at the MAC price.

Stage 1 minimizes the honest GEMM's non-MMA price (U/F_low - 1); stage 2 minimizes Σλ (maximizes τ) at that optimum, or at
U/F_low - 1 ≤ infl_max; stage 3 is stage 2 without the cap (τ alone). Kinds with no coefficient on the vector's rows are
refused, not gated: they leave (i) and every attack that uses them.

The arithmetic, constraint order and solver calls are fixed, so every published vector re-solves to the vertex it was
published at (tests/test_vectors.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import linprog

from verity_machine_model.model import Attack, Census, Model

MACS = 4096  # one MMA does 4096 MACs


@dataclass(frozen=True)
class Solution:
    lam: dict[str, float]          # row: λ_r (> 1e-12), in the vector's row order
    theta: dict[str, float]        # λ_r / σ
    sigma: float                   # Σλ
    infl: float                    # U/F_low - 1
    tau: float                     # 1 / (σ · honest_real)
    tau_max: float                 # stage 3's τ
    infl_at_tau_max: float
    attacks: tuple[Attack, ...]    # the attacks the LP constrained
    refused: tuple[str, ...]       # kinds with no coefficient on the rows


def coef(census: Census, k: str, r: str) -> float:
    return census[k].get(r, 0.0)


def price(census: Census, lam: dict[str, float], k: str) -> float:
    return sum(x * coef(census, k, r) for r, x in lam.items())


def cost(census: Census, lam: dict[str, float], prog: dict[str, float]) -> float:
    return sum(v * price(census, lam, k) for k, v in prog.items())


def solve(model: Model, census: Census, status: dict[str, str], attacks: list[Attack], rows: list[str],
          credits: list[tuple[str, float]], *, rho: float, x_min: float, tie: float, designs: set[str] | None = None,
          tau_min: float | None = None, infl_max: float | None = None) -> Solution | None:
    """The vector over `rows` with `credits`, or None when the LP is infeasible. `status` gives each row's status: a row that
    isn't `measured` costs `tie` more in stage 2, so a measured row wins a tie."""
    mac = 1 / MACS
    R = list(rows)
    priced = {k for k in census if any(coef(census, k, r) for r in R)}

    def col(k):
        return [coef(census, k, r) for r in R]

    mma = col("MMA")
    A_ub, b_ub = [], []
    for k in model.cuda:
        if k in priced:
            A_ub.append([-(a - rho * 32 / 128 * m) for a, m in zip(col(k), mma)])
            b_ub.append(0.0)
    for k, cr in credits:
        if k in priced:
            A_ub.append([-(a - cr * mac * m) for a, m in zip(col(k), mma)])
            b_ub.append(0.0)
    used = tuple(a for a in attacks if a.on and (designs is None or a.design in designs) and set(a.program) <= priced)
    for a in used:
        row = [0.0] * len(R)
        for k, v in a.program.items():
            row = [x + v * c for x, c in zip(row, col(k))]
        A_ub.append([-(x - x_min * mac * m) for x, m in zip(row, mma)])
        b_ub.append(0.0)
    if tau_min is not None:
        A_ub.append([1.0] * len(R))
        b_ub.append(1 / (tau_min * model.honest_real))
    hon = [0.0] * len(R)
    for k, v in model.honest.items():
        if k != "MMA":
            hon = [x + v * c for x, c in zip(hon, col(k))]
    kw = dict(A_eq=[mma], b_eq=[float(MACS)], bounds=(0, None), method="highs")
    s1 = linprog(hon, A_ub=A_ub, b_ub=b_ub, **kw)
    if s1.status:
        return None
    c2 = [1.0 + tie * (status[r] != "measured") for r in R]
    cap = s1.fun * (1 + 1e-9) + 1e-12 if infl_max is None else infl_max * MACS
    s2 = linprog(c2, A_ub=A_ub + [hon], b_ub=b_ub + [cap], **kw)
    if s2.status:
        return None
    s3 = linprog(c2, A_ub=A_ub, b_ub=b_ub, **kw)
    lam = {r: x for r, x in zip(R, s2.x) if x > 1e-12}
    sig = sum(lam.values())
    lam3 = {r: x for r, x in zip(R, s3.x) if x > 1e-12}
    return Solution(
        lam=lam, theta={r: x / sig for r, x in lam.items()}, sigma=sig,
        infl=sum(h * x for h, x in zip(hon, s2.x)) / MACS, tau=1 / (sig * model.honest_real),
        tau_max=1 / (sum(lam3.values()) * model.honest_real),
        infl_at_tau_max=sum(v * price(census, lam3, k) for k, v in model.honest.items() if k != "MMA") / MACS,
        attacks=used, refused=tuple(sorted(set(census) - priced)))
