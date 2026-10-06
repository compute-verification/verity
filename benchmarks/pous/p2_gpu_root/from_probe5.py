"""floor_model.py on probe5's measured latencies, for this GPU and as a B200 extrapolation from one measured SM.

Inputs taken from probe5.jsonl, each at the attacker's end of what was measured:
  t_bfly     the faster of the two dependent butterfly levels (Shoup u32, FP32 complex), arithmetic only
  t_smem     a dependent shared-memory load (ldchase), as if a level's exchange cost no store and no barrier; the
             models are also run at t_smem_pass, the fastest full pass (read a neighbour, write, barrier)
  t_mul      a Barrett modular multiplication's chain: two IMAD.HI (product and quotient) and two IMAD (q*m and the
             correction)
  t_sm       30 IMAD.WIDE + the densest measured round's int8 MACs per cycle / 16 (this GPU only)
  sm_sq_cyc  the fastest block size's dependent round of 330 m16n8k32 int8 mma (1.35M MACs, one depth-3 Karatsuba
             squaring of 633 words; rounded up to a multiple of the warps), with its smem write-back and barrier; it
             omits carries and the fold
A B200 SM's squaring is extrapolated three ways. `scaled` keeps the round's fixed part c(0) in cycles and scales its
mma part c(330) - c(0) by the measured dense rate here over the B200's peak dense int8 rate per SM, as if a B200 SM
ran a dependent round at the same fraction of its peak. `mma_tail` replaces c(0) by the dependent chain from one
squaring's products to the next one's (t_mma + t_mul + t_smem, at this GPU's latencies), and `mma_only` drops it
too, as if all of it overlapped.
    python from_probe5.py probe5.jsonl [--t-mma 27] [--b200-macs-per-clk 7737] [--json]
"""
from __future__ import annotations

import argparse
import json

from floor_model import T_MMA, rows

K_MMA = 330


def inputs(lines: list[dict]) -> dict:
    chain = {r["op"]: r["cyc_per_link"] for r in lines if r["probe"] == "chain"}
    smem = [r for r in lines if r["probe"] == "smem"]
    gemm = [r for r in lines if r["probe"] == "gemm"]
    passes = {r["threads"]: r["cyc"] for r in smem if r["mode"] == "pass"}
    by_w: dict[int, dict[int, float]] = {}
    for r in gemm:
        by_w.setdefault(r["warps"], {})[r["mma_per_round"]] = r["cyc_per_round"]
    sq = {w: (m[min(k for k in m if k >= K_MMA)], m[0]) for w, m in by_w.items()}
    w_best = min(sq, key=lambda w: sq[w][0])
    return {
        "t_bfly": min(chain["bfly_shoup_u32"], chain["bfly_fp32_complex"]),
        "t_smem": next(r["cyc"] for r in smem if r["mode"] == "ldchase"),
        "t_smem_pass": min(passes.values()), "t_smem_pass_1024": passes.get(1024),
        "t_mul": 2 * chain["imad_hi"] + 2 * chain["imad"],
        "sm_sq_cyc": sq[w_best][0], "sm_sq_fixed_cyc": sq[w_best][1], "sm_sq_warps": w_best,
        "dense_macs_per_clk": max(r["macs_per_clk"] for r in gemm),
        "chain": chain, "smem": smem,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe5")
    ap.add_argument("--t-mma", type=float, default=T_MMA)
    ap.add_argument("--b200-macs-per-clk", type=float, default=7737.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with open(a.probe5) as f:
        m = inputs([json.loads(x) for x in f if x.strip()])
    c, c0 = m["sm_sq_cyc"], m["sm_sq_fixed_cyc"]
    mma = (c - c0) * m["dense_macs_per_clk"] / a.b200_macs_per_clk
    b200 = {"scaled": c0 + mma, "mma_tail": mma + a.t_mma + m["t_mul"] + m["t_smem"], "mma_only": mma}
    out = {"inputs": m, "b200_sm_sq_cyc": {k: round(v, 1) for k, v in b200.items()},
           "b200_macs_per_clk": a.b200_macs_per_clk, "t_mma": a.t_mma, "models": {}}
    rtx_t_sm = 30 + m["dense_macs_per_clk"] / 16
    for hop in (324, 447):
        for sname, ts in (("ld", m["t_smem"]), ("pass", m["t_smem_pass"])):
            out["models"][f"rtx_hop{hop}_smem_{sname}"] = rows(hop, m["t_mul"], 2430, m["t_bfly"], ts, t_sm=rtx_t_sm,
                                                               t_mma=a.t_mma, sm_sq_cyc=c)
        for bname, sq in b200.items():
            out["models"][f"b200_hop{hop}_{bname}"] = rows(hop, m["t_mul"], 1965, m["t_bfly"], m["t_smem"], sms=148,
                                                           t_sm=511.25, t_mma=a.t_mma, sm_sq_cyc=sq)
    if a.json:
        print(json.dumps(out))
        return
    print(json.dumps({k: v for k, v in m.items() if k not in ("chain", "smem")}), json.dumps(out["b200_sm_sq_cyc"]))
    for name, rs in out["models"].items():
        best: dict[str, dict] = {}
        for r in rs:
            if r["construction"] not in best or r["ns_per_sq"] < best[r["construction"]]["ns_per_sq"]:
                best[r["construction"]] = r
        for r in best.values():
            print(f"{name:22s} {r['construction']:13s} J={r['J']:<2d} {r['cyc_per_sq']:7.1f} cyc {r['ns_per_sq']:6.1f} ns "
                  f"root {r['root_us']:7.1f} us {r['x_window']:5.2f}x window {r['x_cpu']:5.2f}x cpu  {r['note']}")


if __name__ == "__main__":
    main()
