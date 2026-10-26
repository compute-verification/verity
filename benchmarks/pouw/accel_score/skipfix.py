"""ACCEL SKIP-FIX (CMP-137): the serving skip's routes in protocol v2, the designs that move its cap, and the latency.

Reuses `nofold.v2_round` (protocol v2: O bound to the words, ρ = 4 peel products of inner size L, NF3 leaf and tree,
honest SHA-512 at the public best). At A = B = 0 the served output is the bit product O = bcᵀ (K/s_b a cell), and a
prover saves, per cell, with z the zero-digit peel products (`nofold.ZERO_DIGITS`):

- R1, derivation: one segment's word from a prefix's bit product and peel, O by the bit product: L − K/s_b. Any coin
  schedule allows it, SEG-COINS's late final dither included (`segcoins.py` checks that exactly);
- R2, bit-route O (PD2-1): every word honest, O by the bit product, no peel: ρL − K/s_b;
- R3, zero digits (PD2-1): zL; R1 + R3: (1 + z)L − K/s_b.

v2's ε_alg is max(zL, ρL − K/s_b)/(γK), which covers all four for ρ ≥ 1 + z. `routes()` prints them at v2's pick.

Designs beside v2 (`table()`), each picking L as `nofold.v2_pick` does:

- P-HYB3 (O-IO): 3-bit dither, so bcᵀ's cheapest public price is INT8 Strassen to depth 4 (sums of 16 three-bit values
  stay inside int8), c_O = (7/8)^4 on both cards (b1 bit planes cost 9/8; 0..7 doesn't fit E2M1). Its dither bits
  triple, and Y_W's high digit is zero in a tile while its sums of S dithers (mean 3.5S) stay below 128
  (`zero_share`). Its 21× dither variance and its SAK status are not priced here;
- P1 (RT-PROTO): O out of the round. No route frees credited work (ε 0); O isn't hashed in the round; the peel becomes
  serving work, replaced by one plain product once ρL > K (then no wraps).

`latency()`: one challenger round trip per credited unit and step when each unit's coins must follow its input's
commitment (live coins), on the critical path; whether micro-batches hide it.

    uv run python benchmarks/pouw/accel_score/skipfix.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nofold

C_O_HYB3 = (7 / 8) ** 4
BUDGETS = (0.005, 0.01, 0.02)
HBM = {"H200": 4.8e12, "RTX PRO 6000": 1.792e12}   # bytes/s, data sheets
RTTS = (20e-6, 100e-6, 500e-6)                       # certifier round trips: same rack, same hall, across sites


def zero_share(S: int, top: int = 7, tile: int = 128 * 64) -> float:
    """The share of a tile's high-digit products a prover skips: Y_W's entries, sums of S uniform 0..top dithers, all
    below 128 in the tile (Gaussian tail; exact 1 when top·S < 128)."""
    if top * S < 128:
        return 1.0
    mean, var = top / 2 * S, S * ((top + 1) ** 2 - 1) / 12
    p = 0.5 * math.erfc((127.5 - mean) / math.sqrt(2 * var))
    return (1 - p) ** tile


def per_comp(card: str) -> float:
    mac, sha, _, _ = nofold.V2_CARDS[card]
    return nofold.ETA * mac / sha


def leak_routes(card: str, K: int, L: int, g: float) -> dict:
    s_b, z, rho = nofold.V2_CARDS[card][3], nofold.ZERO_DIGITS, nofold.RHO_V2
    raw = {"R1": L - K / s_b, "R2": rho * L - K / s_b, "R3": z * L, "R1+R3": (1 + z) * L - K / s_b}
    return {k: max(0.0, v) / (g * K) for k, v in raw.items()}


def round_(card: str, K: int, N: int, L: int, word_bits: int, design: str, m: int = nofold.M_PREFILL) -> dict:
    r = nofold.v2_round(card, K, N, L, word_bits, m=m)
    if design == "v2":
        return r
    pc = per_comp(card)
    if design == "P-HYB3":
        coin = r["coin"] + pc * 2 * (m + N) * K / (512 * m * N * K)
        g = r["gamma"] - r["coin"] + coin
        saved = max(zero_share(r["S"]) * L, r["rm"] * K - C_O_HYB3 * K)
        return dict(r, gamma=g, coin=coin, eps_alg=saved / (g * K))
    if design == "P1":
        g_hash = r["hash"] - pc * nofold.LEAF_BLOCKS * nofold.TREE / nofold.LEAF_DATA_BITS * 32 / K
        rm, wrap = (r["rm"], r["wrap"]) if r["rm"] <= 1 else (1.0, 0.0)
        g = 1 + rm + g_hash + r["A"] + r["coin"] + wrap
        return dict(r, gamma=g, gamma_max=max(1 + rm, g - 1 - rm), hash=g_hash, rm=rm, wrap=wrap,
                    h=(g_hash + r["A"] + r["coin"]) / g, eps_alg=0.0)
    raise ValueError(design)


def pick(card: str, K: int, N: int, word_bits: int, budget: float, design: str) -> dict:
    rounds = [round_(card, K, N, L, word_bits, design) for L in range(64, K + 1, 64)]
    ok = [r for r in rounds if r["eps_alg"] <= budget]
    return min(ok, key=lambda r: r["gamma"]) if ok else min(rounds, key=lambda r: r["eps_alg"])


def static(card: str, model: str, word_bits: int, budget: float, design: str) -> dict:
    """As `nofold.v2_static`: MAC-weighted γ and h, ε_alg weighted by honest cost, the worst shape."""
    macs = cost = hashed = saved = 0.0
    worst, shapes = (0.0, None), []
    for K, N, reps in nofold.SHAPES[model]:
        r = pick(card, K, N, word_bits, budget, design)
        w = K * N * reps
        macs += w
        cost += w * r["gamma"]
        hashed += w * r["gamma"] * r["h"]
        saved += w * r["gamma"] * r["eps_alg"]
        shapes.append(dict(K=K, N=N, L=r["L"], gamma=r["gamma"], eps_alg=r["eps_alg"]))
        if r["gamma"] > worst[0]:
            worst = (r["gamma"], (K, N, r["L"]))
    return {"gamma": cost / macs, "h": hashed / cost, "eps_alg": saved / cost, "worst": worst, "shapes": shapes}


def routes(word_bits: int = 16, budget: float = 0.01) -> list[dict]:
    print(f"\nroutes at A = B = 0 at v2's pick ({word_bits}-bit words, budget {budget:.1%}), as ε = saved/(γK); "
          f"rho {nofold.RHO_V2}, z {nofold.ZERO_DIGITS}")
    print(f"  {'card':13} {'model':16} {'K':>6} {'N':>7} {'L':>5} {'gamma':>6} {'R1':>6} {'R2':>6} {'R3':>6} "
          f"{'R1+R3':>6} {'v2':>6}  binds")
    rows = []
    for card in nofold.V2_CARDS:
        for model, shapes in nofold.SHAPES.items():
            for K, N, _ in shapes:
                r = nofold.v2_pick(card, K, N, word_bits, budget)
                e = leak_routes(card, K, r["L"], r["gamma"])
                binds = max(e, key=e.get)
                assert max(e.values()) <= r["eps_alg"] + 1e-12, "a route beats v2's ε_alg"
                print(f"  {card:13} {model:16} {K:6} {N:7} {r['L']:5} {r['gamma']:6.2f} {e['R1']:6.2%} "
                      f"{e['R2']:6.2%} {e['R3']:6.2%} {e['R1+R3']:6.2%} {r['eps_alg']:6.2%}  {binds}")
                rows.append(dict(card=card, model=model, K=K, N=N, L=r["L"], gamma=r["gamma"], routes=e,
                                 eps_alg=r["eps_alg"]))
    return rows


def table() -> list[dict]:
    print("\n(L, gamma, eps) by design: MAC-weighted gamma, h, eps_alg; the worst shape (K, N, L): gamma")
    print(f"  {'card':13} {'model':16} {'design':7} {'words':>5} {'budget':>6} {'gamma':>6} {'h':>5} {'eps_alg':>7}"
          f"  worst shape")
    rows = []
    for card in nofold.V2_CARDS:
        for model in nofold.SHAPES:
            for design in ("v2", "P-HYB3", "P1"):
                for wb in (32, 16):
                    for budget in (BUDGETS if design != "P1" else (0.0,)):
                        s = static(card, model, wb, budget, design)
                        g, (K, N, L) = s["worst"]
                        print(f"  {card:13} {model:16} {design:7} {wb:5} {budget:6.1%} {s['gamma']:6.2f} "
                              f"{s['h']:5.2f} {s['eps_alg']:7.2%}  K {K} N {N} L {L}: {g:.2f}")
                        rows.append(dict(card=card, model=model, design=design, word_bits=wb, budget=budget,
                                         **{k: s[k] for k in ("gamma", "h", "eps_alg", "shapes")},
                                         worst=dict(gamma=g, K=K, N=N, L=L)))
    return rows


def latency() -> list[dict]:
    """Per step: credited units, the step's matmul time, and units × RTT on the critical path."""
    print("\nlatency: one certifier round trip per credited unit a step (its coins follow its input's commitment)")
    print(f"  {'card':13} {'model':16} {'m':>5} {'units':>5} {'step ms':>8} " +
          " ".join(f"{'+' + str(int(t * 1e6)) + 'us':>9}" for t in RTTS) + "  hidden by micro-batches")
    rows = []
    for card, bw in HBM.items():
        mac = nofold.V2_CARDS[card][0]
        crossover = nofold.ETA * mac / bw
        for model, shapes in nofold.SHAPES.items():
            for m in (64, nofold.M_PREFILL):
                units = sum(reps for _, _, reps in shapes)
                step = sum(reps * max(K * N / bw, m * K * N / (nofold.ETA * mac)) for K, N, reps in shapes)
                added = [units * t for t in RTTS]
                if m >= 2 * crossover:
                    t_unit = step / units
                    hide = f"yes: {1 + math.ceil(RTTS[0] / t_unit)} at 20us (each still compute-bound)"
                else:
                    hide = f"no: m {m} < {crossover:.0f} is memory-bound, a micro-batch rereads the weights"
                print(f"  {card:13} {model:16} {m:5} {units:5} {step * 1e3:8.2f} " +
                      " ".join(f"{a * 1e3:6.1f}ms" for a in added) + f"  {hide}")
                rows.append(dict(card=card, model=model, m=m, units=units, step_s=step,
                                 added_s=dict(zip((f"{int(t * 1e6)}us" for t in RTTS), added)), hidden=hide))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = {"routes": routes(), "table": table(), "latency": latency(),
           "params": dict(rho=nofold.RHO_V2, z=nofold.ZERO_DIGITS, cards={k: list(v) for k, v in nofold.V2_CARDS.items()},
                          c_o_hyb3=C_O_HYB3, m_prefill=nofold.M_PREFILL)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "skipfix.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
