"""ACCEL ALGO-COST-2 (CMP-114): the RTX PRO 6000 honest kernel's joint shared-memory port and lane allocation under P1.

P1 (PROTO-DESIGN-3, `nofold.py` at f31a299d96 on cursor/compute-research-accel-proto-design-ea24) runs every static
shape at one word a cell, so its periods on the RTX PRO 6000 are tensor-bound or balanced and the lanes and the
shared-memory port have slack. Any method that trades tensor time for lane or port work then lowers the period. A prover
can choose a method per tile, so the reachable (tensor, lanes, port) loads are the convex hull of the pure methods',
and the cheapest period is a linear program:

    minimize τ  subject to  tensor ≤ τ,  lanes ≤ τ,  port ≤ τ,

over mixtures of the standard product and every 2:4/lane split of depth-1 Strassen ⟨2,2,2;7⟩, HK ⟨2,3,3;15⟩ and the
catalogue ⟨3,3,3;23⟩ (`algo_cost.scheme_overlapped`'s route per summed form: one 2:4 pass with lane leftovers, or sparse
lane adds), and, pooled over a request, moving PV MACs to Four-Russians lookups (16 port MAC-times and ℓ lane
MAC-times a moved MAC, A15). Units: plain INT8 tensor time a credited MAC (γ). Constants are A8's: 6 lane and 8 port
MAC-times a correction add, staging 0.22 of tensor time on the port; overflow rates at data 0 (U = V = ±80).

The honest kernel takes the program's optimum on its own instance (wraps on the lanes); credit is that allocation's
period without wraps (`nofold`'s convention); the prover at A = B = 0 solves the same program without wraps. ε_alg is
1 − prover/credit. A kernel restricted to one pure method (A14's Strassen kernel, or the cheaper of Strassen and HK)
shows what the mixture is worth.

    uv run python benchmarks/pouw/theory/joint_alloc.py
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
from algo_cost import dist, overflow
from scipy.optimize import linprog

C_OP, LANE_PER_ADD, PORT_PER_ADD, STAGING, PORT_LOOKUP = 8, 6, 8, 0.22, 16
SCHEMES = {  # name: (rank, mkn, every summed form's term count, both sides)
    "Strassen <2,2,2;7>": (7, 8, [2] * 10),
    "HK <2,3,3;15>": (15, 18, [2] * 10 + [3] * 8 + [4] * 2),
    "<3,3,3;23>": (23, 27, [2] * 16 + [3] * 8 + [7] * 4),
}

# P1 on the RTX PRO 6000 at the paper SHA-512 rate (9.82 G/s), from `nofold.py` at f31a299d96: per static shape
# (K, N, layers, tensor 1 + peels, lanes without wraps), the static wraps 0.054, and pooled over COVER-ATTN's request:
# MAC shares, static means, PV and QKᵀ lane loads, ATT's wraps and the lookup price ℓ.
P1_SHAPES = {
    "Llama-3.1-8B": [(4096, 6144, 32, 1.0625, 0.5768), (4096, 4096, 32, 1.0625, 0.5845),
                     (4096, 28672, 32, 1.0625, 0.5646), (14336, 4096, 32, 1.0268, 0.1891),
                     (4096, 128256, 1, 1.0625, 0.5620)],
    "Llama-3.1-70B": [(8192, 10240, 80, 1.0469, 0.2946), (8192, 8192, 80, 1.0469, 0.2968),
                      (8192, 57344, 80, 1.0469, 0.2874), (28672, 8192, 80, 1.0134, 0.0995),
                      (8192, 128256, 1, 1.0469, 0.2865)],
    "DeepSeek-V4-Pro": [(7168, 43008, 61, 1.0446, 0.3273), (3072, 50176, 61, 1.0833, 0.7469),
                        (7168, 1536, 61, 1.0446, 0.3842), (1536, 65536, 61, 1.1667, 1.4813),
                        (7168, 576, 61, 1.0446, 0.4825), (4096, 16384, 61, 1.0625, 0.5671),
                        (16384, 7168, 61, 1.0234, 0.1602), (7168, 129280, 1, 1.0446, 0.3259)],
}
STATIC_WRAP = 0.054
P1_POOLED = {
    "Llama-3.1-8B": {"st": 0.927118, "pv": 0.036441, "qk": 0.036441, "ix": 0.0, "s_T": 1.05356, "s_A": 0.47314,
                         "pv_a": 5.72952, "qk_a": 1.13976, "att_wrap": 0.0504, "ell": 9.3652},
    "Llama-3.1-70B": {"st": 0.959286, "pv": 0.020357, "qk": 0.020357, "ix": 0.0, "s_T": 1.03782, "s_A": 0.23800,
                          "pv_a": 5.69196, "qk_a": 1.10220, "att_wrap": 0.0504, "ell": 9.3652},
    "DeepSeek-V4-Pro": {"st": 0.934941, "pv": 0.031211, "qk": 0.031211, "ix": 0.002637, "s_T": 1.06644, "s_A": 0.55681,
                            "pv_a": 3.33187, "qk_a": 3.13140, "att_wrap": 0.0504, "ell": 12.4300},
}


@functools.cache
def p0(t: int) -> float:
    """Overflow of an all-plus t-term sum at data 0 (`algo_cost.rates`' first entry)."""
    return overflow([dist(0)] * t, (1,) * t)


def leftover(p: float) -> float:
    return (4 * p**3 * (1 - p) + 2 * p**4) / 4


def methods(lane_per_add: float = LANE_PER_ADD, port_per_add: float = PORT_PER_ADD,
            schemes=tuple(SCHEMES)) -> list[tuple[str, float, float, float]]:
    """(name, tensor, extra lanes, port) a standard MAC of the product: the standard product and every split."""
    out = [("standard", 1.0, 0.0, STAGING)]
    for name in schemes:
        rank, mkn, terms = SCHEMES[name]
        counts = {t: terms.count(t) for t in sorted(set(terms)) if t > 1}
        ps = {t: p0(min(t, 8)) for t in counts}
        for js in itertools.product(*[range(c + 1) for c in counts.values()]):
            split = dict(zip(counts, js, strict=True))
            tensor = (rank + 0.5 * sum(js)) / mkn
            adds = sum((counts[t] - j) * ps[t] for t, j in split.items()) / mkn
            lanes = lane_per_add * adds + C_OP * sum(j * leftover(ps[t]) for t, j in split.items()) / mkn
            out.append((f"{name} {split}", tensor, lanes, port_per_add * adds + STAGING * tensor))
    return out


def solve(base: dict, menu: list, wraps: float) -> tuple[float, np.ndarray, float]:
    """The period's least τ over mixtures of `menu` on the static share (and lookups when base has pv > 0).

    base: st (static share), s_T (static tensor a MAC, 1 + peels), s_A (static lanes without wraps), and optionally the
    attention block (pv, qk, ix shares, pv_a, qk_a lanes, ell). Returns (τ, weights, lookups moved)."""
    st, rm = base["st"], base["s_T"] - 1
    pv, qk, ix = base.get("pv", 0.0), base.get("qk", 0.0), base.get("ix", 0.0)
    t_rest = st * rm + 2 * pv + 3 * qk + ix
    a_rest = st * base["s_A"] + pv * base.get("pv_a", 0.0) + qk * base.get("qk_a", 0.0) + wraps
    M = len(menu)
    t = np.array([m[1] for m in menu]) * st
    a = np.array([m[2] for m in menu]) * st
    p = np.array([m[3] for m in menu]) * st
    ell = base.get("ell", 0.0)
    # x = (w_1..w_M, y, τ)
    A_ub = np.vstack([np.r_[t, -1.0, -1.0], np.r_[a, ell, -1.0], np.r_[p, PORT_LOOKUP, -1.0]])
    b_ub = np.array([-t_rest, -a_rest, -STAGING * t_rest])
    A_eq = np.r_[np.ones(M), 0.0, 0.0][None, :]
    bounds = [(0, None)] * M + [(0, pv), (None, None)]
    r = linprog(np.r_[np.zeros(M), 0.0, 1.0], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                method="highs")
    assert r.status == 0, r.message
    return r.x[-1], r.x[:M], r.x[M]


def period(base: dict, menu: list, w: np.ndarray, y: float, wraps: float) -> float:
    """max(tensor, lanes, port) of a given allocation."""
    st, rm = base["st"], base["s_T"] - 1
    pv, qk, ix = base.get("pv", 0.0), base.get("qk", 0.0), base.get("ix", 0.0)
    t_rest = st * rm + 2 * pv + 3 * qk + ix
    T = st * sum(wi * m[1] for wi, m in zip(w, menu, strict=True)) + t_rest - y
    A = (st * (base["s_A"] + sum(wi * m[2] for wi, m in zip(w, menu, strict=True))) + pv * base.get("pv_a", 0.0)
         + qk * base.get("qk_a", 0.0) + wraps + base.get("ell", 0.0) * y)
    P = st * sum(wi * m[3] for wi, m in zip(w, menu, strict=True)) + STAGING * t_rest + PORT_LOOKUP * y
    return max(T, A, P)


def pure(base: dict, menu: list, wraps: float) -> tuple[float, np.ndarray, float, list]:
    """The cheapest single method of `menu` on the whole static share, lookups optimized (a 1-D convex problem)."""
    st, rm = base["st"], base["s_T"] - 1
    pv, qk, ix, ell = base.get("pv", 0.0), base.get("qk", 0.0), base.get("ix", 0.0), base.get("ell", 0.0)
    t_rest = st * rm + 2 * pv + 3 * qk + ix
    m = np.array([x[1:] for x in menu])
    T0 = st * m[:, 0] + t_rest
    A0 = st * (base["s_A"] + m[:, 1]) + pv * base.get("pv_a", 0.0) + qk * base.get("qk_a", 0.0) + wraps
    P0 = st * m[:, 2] + STAGING * t_rest
    y = np.clip(np.minimum((T0 - A0) / (1 + ell), (T0 - P0) / (1 + PORT_LOOKUP)), 0, pv) if pv else np.zeros(len(m))
    tau = np.maximum.reduce([T0 - y, A0 + ell * y, P0 + PORT_LOOKUP * y])
    i = int(np.argmin(tau))
    return float(tau[i]), np.array([1.0]), float(y[i]), [menu[i]]


def kernels(base: dict, wraps: float, lane_per_add: float = LANE_PER_ADD, port_per_add: float = PORT_PER_ADD) -> dict:
    """Honest kernels (standard only; one pure Strassen split; the cheaper pure split of Strassen and HK; the mixture)
    against the prover's mixture at A = B = 0. Each value: (honest τ, gap 1 − prover/honest, nofold-convention ε_alg
    with credit at the honest allocation's period without wraps). The gap is ε_alg when credit is priced at the
    deployed kernel's cost, and the honest busy time left uncredited when credit is priced at the public best."""
    full = methods(lane_per_add, port_per_add)
    tau_p, w_p, _ = solve(base, full, 0.0)

    def sub(keep) -> list:
        return [x for x in full if x[0] == "standard" or x[0].split(" {")[0] in keep]

    out = {}
    for label, (tau, w, y, menu) in {
        "standard": pure(base, sub(()), wraps),
        "Strassen (A14)": pure(base, sub(("Strassen <2,2,2;7>",)), wraps),
        "Strassen or HK": pure(base, sub(("Strassen <2,2,2;7>", "HK <2,3,3;15>")), wraps),
        "mixture": (*solve(base, full, wraps), full),
    }.items():
        credit = period(base, menu, w, y, 0.0)
        out[label] = (tau, max(0.0, 1 - tau_p / tau), max(0.0, 1 - tau_p / credit))
    out["prover"] = (tau_p, 0.0, 0.0)
    out["prover mix"] = {full[i][0]: round(float(x), 3) for i, x in enumerate(w_p) if x > 1e-6}
    return out


KERNELS = ("standard", "Strassen (A14)", "Strassen or HK", "mixture")


def per_shape(lane_per_add: float = LANE_PER_ADD, port_per_add: float = PORT_PER_ADD, rows: bool = True) -> None:
    print(f"\nP1, RTX PRO 6000, per static shape (kernel-level overlap), {lane_per_add} lane and {port_per_add} port "
          "MAC-times an add; gap = 1 − prover/honest")
    if rows:
        print(f"  {'model':16} {'K':>6} {'N':>6} {'load':>5}  {'standard':>8} {'Strassen':>8} {'S or HK':>8} "
              f"{'mixture':>8}  prover   gap: standard / Strassen / S or HK / mixture")
    for model, shapes in P1_SHAPES.items():
        tot = {k: [0.0, 0.0] for k in KERNELS}
        for K, N, layers, T, A in shapes:
            r = kernels({"st": 1.0, "s_T": T, "s_A": A}, STATIC_WRAP, lane_per_add, port_per_add)
            if rows:
                print(f"  {model:16} {K:6} {N:6} {(A + STATIC_WRAP) / T:5.2f}  "
                      + " ".join(f"{r[k][0]:8.4f}" for k in KERNELS) + f"  {r['prover'][0]:.4f}   "
                      + " / ".join(f"{r[k][1]:.2%}" for k in KERNELS))
            for k, acc in tot.items():
                acc[0] += K * N * layers * r[k][0]
                acc[1] += K * N * layers * r[k][0] * r[k][1]
        print(f"  {model:16} static gap, honest-time-weighted: "
              + ", ".join(f"{k} {v[1] / v[0]:.2%}" for k, v in tot.items()))


def pooled(lane_per_add: float = LANE_PER_ADD, port_per_add: float = PORT_PER_ADD) -> None:
    print(f"\nP1, RTX PRO 6000, pooled over the request (the model's overlapped accounting), {lane_per_add} lane and "
          f"{port_per_add} port MAC-times an add; PV lookups at {PORT_LOOKUP} port MAC-times a moved MAC")
    for model, base in P1_POOLED.items():
        wraps = base["st"] * STATIC_WRAP + (base["pv"] + base["qk"]) * base["att_wrap"]
        r = kernels(base, wraps, lane_per_add, port_per_add)
        print(f"  {model:16} honest γ (gap; nofold ε_alg): "
              + ", ".join(f"{k} {r[k][0]:.4f} ({r[k][1]:.2%}; {r[k][2]:.2%})" for k in KERNELS)
              + f"; prover {r['prover'][0]:.4f}")
        print(f"  {'':16} prover's mixture: {r['prover mix']}")


def main() -> None:
    per_shape()
    pooled()
    for lpa, ppa in ((12, 8), (6, 12), (6, 16), (12, 16)):
        per_shape(lpa, ppa, rows=False)
        pooled(lpa, ppa)


if __name__ == "__main__":
    main()
