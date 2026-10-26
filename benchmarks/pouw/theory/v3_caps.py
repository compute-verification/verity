"""ACCEL ALGO-COST-2 (CMP-114): v3's caps priced at the period's dual prices (P12), static rounds and PV at every n.

v3 is v2 with (t, L′) capped so that no cheaper route wins: the RTX PRO 6000 at t ≤ 3 and L′ ≤ 256 (PB2, A23), the
H200 at t = 2 with L′ capped per shape (A20, PB1). The routes at A = B = 0 are D5's tile route plus PD3-4's two
(PD2-1's O-route and the zero-high-digit peel), priced together in D5's job-level LP over θ's facets
(`d5_routes.job_at`, hashing matched to nofold's honest lane load): the leak is 1 − c_opt against the honest words'
best program. Two readings:

- **job LP**: c_opt from `d5_routes.job_at` at each pick.
- **any prices**: a route that saves any resource wins once that resource has a positive dual price (other work beside
  it). nofold's routes save tensor time only, so this is the largest route saving over the honest tensor time.

For each shape and n the pick is the least-γ (t, L′) whose job-LP leak is 0, else the least leak.

    git show origin/cursor/compute-research-accel-proto-design-ea24:benchmarks/pouw/accel_score/nofold.py > /tmp/nofold.py
    git show origin/cursor/compute-research-accel-proto-design-ea24:benchmarks/pouw/accel_score/d5_routes.py \
        > /tmp/d5_routes.py
    git show origin/cursor/compute-research-accel-bet-algo-lab-ea24:benchmarks/pouw/theory/public_best.py \
        > /tmp/public_best.py
    uv run python benchmarks/pouw/theory/v3_caps.py /tmp
"""

from __future__ import annotations

import sys

DESIGNS = {"H200": ((2, (64, 128, 256, 512, 1024)),), "RTX PRO 6000": ((3, (64, 128, 192, 256)), (2, (64, 128, 192, 256)))}
PV_N = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
PV_MODELS = (("Llama d_v 128 GQA 4", 128, 4), ("V4 d_v 512 MQA 128", 512, 128))


def price(n, pb, d, card: str, K: int, N: int, m: int, designs=None) -> dict:
    sm = pb.SM120 if card == "RTX PRO 6000" else pb.H200
    _, bit, depth = pb.BIT_COST[sm.family][0]
    picks = []
    for t, lps in designs or DESIGNS[card]:
        for lp in lps:
            if t * lp > K:
                continue
            r = n.v2_round(card, K, N, t * lp, 16, m=m, tiles=t)
            j = d.job_at(sm, t, lp, K, N, bit, depth, pb.V2_V, r["A_nw"] / r["T"])
            c = 1.0 if j is None else float(j["c_opt"])
            picks.append({"t": t, "Lp": lp, "gamma": r["gamma_max"], "leak": 1 - c, "any": r["saved"] / r["T"],
                          "load": r["A_nw"] / r["T"], "mix": {} if j is None else j["mix"]})
    clean = [p for p in picks if p["leak"] < 5e-4]
    return min(clean, key=lambda p: p["gamma"]) if clean else min(picks, key=lambda p: p["leak"])


def mix(p: dict) -> str:
    return ",".join(f"{k.split(':')[-1]} {x:.2f}" for k, x in p["mix"].items() if x > 1e-3) or "-"


def static(n, pb, d) -> None:
    print("\nstatic rounds at v3's caps: pick (t, L′), γ, lane load, job-LP leak [hull mix], any-prices tensor saving")
    for card in n.V2_CARDS:
        for model, shapes in n.SHAPES.items():
            W = G = leak = anyp = 0.0
            cells = []
            for K, N, reps in shapes:
                p = price(n, pb, d, card, K, N, n.M_PREFILL)
                w = K * N * reps
                W, G, leak, anyp = W + w, G + w * p["gamma"], leak + w * p["leak"], anyp + w * p["any"]
                cells.append(f"{K}x{N} t{p['t']} L′{p['Lp']} γ{p['gamma']:.2f} ld{p['load']:.2f} "
                             f"{p['leak']:.1%}[{mix(p)}] any {p['any']:.1%}")
            print(f"  {card:13} {model:16} γ {G / W:.2f} leak {leak / W:.2%} any {anyp / W:.2%}")
            for c in cells:
                print(f"      {c}")


def pv(n, pb, d) -> None:
    print("\nPV at every n (K = n attended keys): pick, γ, job-LP leak, any-prices tensor saving")
    for card in n.V2_CARDS:
        for name, d_v, group in PV_MODELS:
            cells, worst = [], (0.0, None)
            for K in PV_N:
                p = price(n, pb, d, card, K, d_v, n.M_PREFILL * group)
                worst = max(worst, (p["leak"], K), key=lambda x: x[0])
                cells.append(f"{K} t{p['t']}/{p['Lp']} γ{p['gamma']:.2f} {p['leak']:.1%}/{p['any']:.1%}")
            print(f"  {card:13} {name:20} worst job-LP leak {worst[0]:.2%} at n {worst[1]}")
            for i in range(0, len(cells), 5):
                print("      " + "; ".join(cells[i:i + 5]))


def main(argv: list[str]) -> None:
    sys.path.insert(0, argv[1] if len(argv) > 1 else "/tmp")
    import d5_routes as d
    import nofold as n
    import public_best as pb

    static(n, pb, d)
    pv(n, pb, d)


if __name__ == "__main__":
    main(sys.argv)
