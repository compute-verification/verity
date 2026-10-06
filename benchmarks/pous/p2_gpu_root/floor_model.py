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
               throughput, Karatsuba to depth 3) before its share of the last one. Each redundant squaring takes
               the larger of its work at the SM's throughput and the dependent chain from the last digit of one
               squaring to the next (one mma, 27 cycles, one modular multiplication for the fold, one t_smem pass),
               as if the two overlapped perfectly.
  single-sm    the limit of positional-J: one SM does the whole chain, no hop.
  rns-crt      residues mod K primes below 2^26 with range for y^(2^J); explicit CRT reduction, 1 hop, W = K^2.
  rns-mont     the same with Bajard-Montgomery reduction: 2 hops (B to B', back to B), W = 2 K_B K_B'.
  rns-tree     the same residues, reduced through position: each SM rebuilds its s = K/C residues as one integer with a
               product tree (Borodin-Moenck, Bernstein's scaled remainder trees) and multiplies it by its 633-word
               CRT constant mod p; a reduce-scatter sums the C products (hop 1), each SM folds its slice mod p, an
               all-gather spreads y (hop 2), and each SM reduces y to its residues with a remainder tree. Either
               direction may instead be dense (633 products per residue, depth one modular multiplication). A tree
               level of size n is charged n/2 log2 n + n products (one transform and a pointwise pass, a third of
               the usual three transforms) and log2 n + 1 dependent butterflies of t_bfly cycles plus one t_smem
               pass inside the SM; transforms count one product per butterfly at T. All of that favors the attacker.
    python floor_model.py [--hop 447] [--clock-mhz 2430] [--t-bfly 4] [--t-smem 20] [--sms 188] [--t-sm 92.75] [--json]
--sms and --t-sm (26-bit products per cycle per SM) describe another part, e.g. a B200 estimate:
--sms 148 --t-sm 511.25 --clock-mhz 1965 (30 IMAD + about 7,700 dense int8 MACs / 16).
"""
from __future__ import annotations

import argparse
import json
import math

W_BITS, RB, SMS = 16448, 26, 188
E_SQ = W_BITS - 2
T_SM = 30 + 1004 / 16
T_MMA = 27
WINDOW_US, CPU_US = 854.0, 1708.0


def k_for(j: int) -> int:
    k = 700
    for _ in range(8):
        bits = (1 << j) * (W_BITS + RB + math.log2(k) + 1) + 20
        k = math.ceil(bits / (RB - 0.001))
    return k


def tree(s: int, b: int, t_mul: float, t_bfly: float, t_smem: float) -> tuple[float, float]:
    """Products and dependent cycles of one SM's product or remainder tree over s one-word residues: dense blocks of
    b residues (s * b products, one modular multiplication deep), then binary levels up to s; a level whose nodes
    span more than a warp's 32 words also pays one t_smem pass."""
    w, dep = float(s * b), t_mul + (t_smem if b > 32 else 0.0)
    for lv in range(int(math.log2(b)) + 1, math.ceil(math.log2(s)) + 1):
        w += s / 2 * lv + s
        dep += (lv + 1) * t_bfly + (t_smem if (1 << lv) > 32 else 0.0)
    return w, dep


def edge(s: int, t_mul: float, t_bfly: float, t_smem: float) -> list[tuple[float, float]]:
    """The s-word tree root times its 633-word constant: in the transform domain, or dense."""
    n = 1 << math.ceil(math.log2(s + 633))
    return [(n / 2 * math.log2(n) + n, (math.log2(n) + 1) * t_bfly + t_smem), (633.0 * s, t_mul)]


def reduce_ways(s: int, t_mul: float, t_bfly: float, t_smem: float) -> list[tuple[float, float, str]]:
    """Every (products per SM, dependent cycles, label) for one direction of the rns-tree reduction."""
    out = [(633.0 * s, t_mul + t_smem, "dense")]
    for i in range(math.ceil(math.log2(s)) + 1 if s > 1 else 1):
        b = 1 << i
        tw, td = tree(s, b, t_mul, t_bfly, t_smem)
        for k, (ew, ed) in enumerate(edge(s, t_mul, t_bfly, t_smem)):
            out.append((tw + ew, td + ed, f"tree b={b} edge={'ntt' if k == 0 else 'dense'}"))
    return out


def rows(hop: float, t_mul: float, clock_mhz: float, t_bfly: float = 4.0, t_smem: float = 20.0,
         sms: int = SMS, t_sm: float = T_SM) -> list[dict]:
    t = sms * t_sm
    wpos = 633 * 634 / 2
    out = []

    def add(name: str, j: int, hops: int, w: float, steps: int, note: str = "", local: float = 0.0) -> None:
        rest = w / t + local + steps * t_mul
        cyc = (hops * hop + rest) / j
        ns = cyc / clock_mhz * 1e3
        mark = {f"hop_below_for_{m}ns": round((m * clock_mhz / 1e3 * j - rest) / hops, 1) if hops else None
                for m in (103.9, 51.9)}
        out.append({"construction": name, "J": j, "hops_per_sq": hops / j, "cyc_per_sq": round(cyc, 1), **mark,
                    "ns_per_sq": round(ns, 1), "root_us": round(ns * E_SQ / 1e3, 1),
                    "x_window": round(ns * E_SQ / 1e3 / WINDOW_US, 2), "x_cpu": round(ns * E_SQ / 1e3 / CPU_US, 2),
                    "note": note})

    add("positional", 1, 1, wpos, 2)
    kara = wpos * (3 / 4) ** 3
    one = max(kara / t_sm, T_MMA + t_mul + t_smem)
    for j in range(2, 13):
        add("positional-J", j, 1, wpos, 2, local=(j - 1) * one)
    add("single-sm", 1, 0, 0.0, 0, local=one)
    for j in range(1, 6):
        k = k_for(j)
        add("rns-crt", j, 1, k * k, j + 2, f"K={k}")
    k0 = k_for(0) + 2
    for j in range(1, 7):
        kb = k_for(j) - k0
        add("rns-mont", j, 2, 2 * kb * k0, j + 4, f"K_B={kb}, K_B'={k0}")
    for j in range(1, 10):
        k = k_for(j)
        s = math.ceil(k / sms)
        ways = reduce_ways(s, t_mul, t_bfly, t_smem)
        w1, d1, how = min(ways, key=lambda x: 2 * sms * x[0] / t + 2 * x[1])
        w, local = 2 * sms * w1, 2 * d1
        add("rns-tree", j, 2, w, j + 2, f"K={k}, s={s}, both ways {how}, W/K={w / k:.0f}", local=local)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=float, default=447.0)
    ap.add_argument("--t-mul", type=float, default=37.0)
    ap.add_argument("--clock-mhz", type=float, default=2430.0)
    ap.add_argument("--t-bfly", type=float, default=4.0)
    ap.add_argument("--t-smem", type=float, default=20.0)
    ap.add_argument("--sms", type=int, default=SMS)
    ap.add_argument("--t-sm", type=float, default=T_SM)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rs = rows(a.hop, a.t_mul, a.clock_mhz, a.t_bfly, a.t_smem, a.sms, a.t_sm)
    if a.json:
        print(json.dumps({"hop": a.hop, "t_mul": a.t_mul, "clock_mhz": a.clock_mhz, "t_bfly": a.t_bfly,
                          "t_smem": a.t_smem, "sms": a.sms, "t_sm": a.t_sm, "rows": rs}))
        return
    marks = " ".join(f"{m * a.clock_mhz / 1e3:.1f} cycles ({m} ns)" for m in (103.9, 51.9))
    print(f"hop {a.hop} cycles, t_mul {a.t_mul}, t_bfly {a.t_bfly}, t_smem {a.t_smem}, {a.sms} SMs x {a.t_sm} "
          f"products/cycle, {a.clock_mhz} MHz; marks {marks}")
    for r in rs:
        hb = "  ".join("-" if r[k] is None else f"{r[k]:7.1f}" for k in ("hop_below_for_103.9ns", "hop_below_for_51.9ns"))
        print(f"{r['construction']:13s} J={r['J']}  hops/sq {r['hops_per_sq']:.2f}  {r['cyc_per_sq']:7.1f} cyc  "
              f"{r['ns_per_sq']:6.1f} ns  root {r['root_us']:8.1f} us  {r['x_window']:5.2f}x window  {r['x_cpu']:5.2f}x cpu  "
              f"hop < {hb} for the marks  {r['note']}")


if __name__ == "__main__":
    main()
