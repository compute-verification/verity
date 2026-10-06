"""Lower bounds on a dependent squaring mod p = 2^16448 - 21065 for representations that defer work across squarings.

A construction that does J squarings per round of exchanges costs, per squaring,
    (hops * hop + W / T + steps * t_mul) / J        cycles,
where hop is one cross-SM one-way latency, W the multiply-adds a round needs in total (spread perfectly over the GPU at
throughput T, in 26 x 26-bit products per cycle), and steps the dependent modular multiplications a residue does
locally. Every term is the optimistic end of what was measured, so each row is a floor, not a forecast:
  hop    447 cycles: half the L2 ping-pong round trip between two SMs (893); 324 within a 16-SM cluster (DSMEM 647 / 2)
  T      188 SMs x (30 IMAD.WIDE + 1,004 int8 mma MACs / 16 per 26-bit product), both units at once, at full use
  t_mul  37 cycles: four dependent IMADs (9.25 each) per modular multiplication, Barrett reduction included
Constructions:
  positional   633 radix-2^26 digits, lazy carries (sqchain4): every output digit depends on every input, 1 hop per
               squaring, W = 633 * 634 / 2.
  positional-J x^(2^J) per hop: after one all-gather each SM squares the full value J - 1 times redundantly (its own
               throughput, Karatsuba to depth 3) before its share of the last one.
  rns-crt      residues mod K primes below 2^26 with range for y^(2^J); explicit CRT reduction, 1 hop, W = K^2.
  rns-mont     the same with Bajard-Montgomery reduction: 2 hops (B to B', back to B), W = 2 K_B K_B'.
    python floor_model.py [--hop 447] [--clock-mhz 2430] [--json]
"""
from __future__ import annotations

import argparse
import json
import math

W_BITS, RB, SMS = 16448, 26, 188
E_SQ = W_BITS - 2
T_SM = 30 + 1004 / 16
WINDOW_US, CPU_US = 854.0, 1708.0


def k_for(j: int) -> int:
    k = 700
    for _ in range(8):
        bits = (1 << j) * (W_BITS + RB + math.log2(k) + 1) + 20
        k = math.ceil(bits / (RB - 0.001))
    return k


def rows(hop: float, t_mul: float, clock_mhz: float) -> list[dict]:
    t = SMS * T_SM
    wpos = 633 * 634 / 2
    out = []

    def add(name: str, j: int, hops: int, w: float, steps: int, note: str = "", local: float = 0.0) -> None:
        rest = w / t + local + steps * t_mul
        cyc = (hops * hop + rest) / j
        ns = cyc / clock_mhz * 1e3
        mark = {f"hop_below_for_{m}ns": round((m * clock_mhz / 1e3 * j - rest) / hops, 1) for m in (103.9, 51.9)}
        out.append({"construction": name, "J": j, "hops_per_sq": hops / j, "cyc_per_sq": round(cyc, 1), **mark,
                    "ns_per_sq": round(ns, 1), "root_us": round(ns * E_SQ / 1e3, 1),
                    "x_window": round(ns * E_SQ / 1e3 / WINDOW_US, 2), "x_cpu": round(ns * E_SQ / 1e3 / CPU_US, 2),
                    "note": note})

    add("positional", 1, 1, wpos, 2)
    kara = wpos * 4 * (3 / 4) ** 3
    for j in (2, 3):
        add("positional-J", j, 1, wpos, 0, local=(j - 1) * kara / T_SM)
    for j in range(1, 6):
        k = k_for(j)
        add("rns-crt", j, 1, k * k, j + 2, f"K={k}")
    k0 = k_for(0) + 2
    for j in range(1, 7):
        kb = k_for(j) - k0
        add("rns-mont", j, 2, 2 * kb * k0, j + 4, f"K_B={kb}, K_B'={k0}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=float, default=447.0)
    ap.add_argument("--t-mul", type=float, default=37.0)
    ap.add_argument("--clock-mhz", type=float, default=2430.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rs = rows(a.hop, a.t_mul, a.clock_mhz)
    if a.json:
        print(json.dumps({"hop": a.hop, "t_mul": a.t_mul, "clock_mhz": a.clock_mhz, "rows": rs}))
        return
    print(f"hop {a.hop} cycles, t_mul {a.t_mul}, {a.clock_mhz} MHz; marks 252.5 cycles (103.9 ns), 126 (51.9 ns)")
    for r in rs:
        print(f"{r['construction']:13s} J={r['J']}  hops/sq {r['hops_per_sq']:.2f}  {r['cyc_per_sq']:7.1f} cyc  "
              f"{r['ns_per_sq']:6.1f} ns  root {r['root_us']:8.1f} us  {r['x_window']:5.2f}x window  {r['x_cpu']:5.2f}x cpu  "
              f"hop < {r['hop_below_for_103.9ns']:7.1f} / {r['hop_below_for_51.9ns']:7.1f} for the marks  {r['note']}")


if __name__ == "__main__":
    main()
