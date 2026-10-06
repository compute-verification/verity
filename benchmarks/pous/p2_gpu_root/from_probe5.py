"""floor_model.py on probe5's measured latencies, for this GPU and as a B200 extrapolation from one measured SM.

Inputs taken from probe5.jsonl, each at the attacker's end of what was measured:
  t_bfly     the faster of the two dependent butterfly levels (Shoup u32, FP32 complex), arithmetic only
  t_smem     the fastest double-buffered shared-memory pass (read a neighbour, write, barrier) over the block sizes
  t_mul      four links of the fastest dependent IMAD form
  t_sm       30 IMAD.WIDE + the densest measured round's int8 MACs per cycle / 16 (this GPU only)
  sm_sq_cyc  the fastest block size's dependent round of 330 m16n8k32 int8 mma (1.35M MACs, one depth-3 Karatsuba
             squaring of 633 words; rounded up to a multiple of the warps), with its smem write-back and barrier; it
             omits carries and the fold
The B200 single SM keeps the round's fixed part c(0) in cycles and scales its mma part c(330) - c(0) by the ratio
of the B200's peak dense int8 rate per SM to the dense rate measured here, as if a B200 SM ran a dependent round at
the same fraction of its peak.
    python from_probe5.py probe5.jsonl [--b200-macs-per-clk 7737] [--json]
"""
from __future__ import annotations

import argparse
import json

from floor_model import rows

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
        "t_bfly_with_shfl": min(chain["bfly_shoup_u32"], chain["bfly_fp32_complex"]) + chain["shfl_xor"],
        "t_smem": min(passes.values()), "t_smem_1024": passes.get(1024),
        "t_mul": 4 * min(chain["imad"], chain["imad_wide"], chain["imad_hi"]),
        "sm_sq_cyc": sq[w_best][0], "sm_sq_fixed_cyc": sq[w_best][1], "sm_sq_warps": w_best,
        "dense_macs_per_clk": max(r["macs_per_clk"] for r in gemm),
        "chain": chain, "smem": smem,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe5")
    ap.add_argument("--b200-macs-per-clk", type=float, default=7737.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with open(a.probe5) as f:
        m = inputs([json.loads(x) for x in f if x.strip()])
    c, c0 = m["sm_sq_cyc"], m["sm_sq_fixed_cyc"]
    b200_sq = c0 + (c - c0) * m["dense_macs_per_clk"] / a.b200_macs_per_clk
    lat = dict(t_mul=m["t_mul"], t_bfly=m["t_bfly"], t_smem=m["t_smem"])
    out = {"inputs": m, "b200_sm_sq_cyc": round(b200_sq, 1), "b200_macs_per_clk": a.b200_macs_per_clk, "models": {}}
    for hop in (324, 447):
        out["models"][f"rtx_hop{hop}"] = rows(hop, clock_mhz=2430, t_sm=30 + m["dense_macs_per_clk"] / 16, sm_sq_cyc=c,
                                              **lat)
        out["models"][f"b200_hop{hop}"] = rows(hop, clock_mhz=1965, sms=148, t_sm=511.25, sm_sq_cyc=b200_sq, **lat)
    if a.json:
        print(json.dumps(out))
        return
    print(json.dumps({k: v for k, v in m.items() if k not in ("chain", "smem")}),
          f"b200_sm_sq_cyc {b200_sq:.1f}")
    for name, rs in out["models"].items():
        best: dict[str, dict] = {}
        for r in rs:
            if r["construction"] not in best or r["ns_per_sq"] < best[r["construction"]]["ns_per_sq"]:
                best[r["construction"]] = r
        for r in best.values():
            print(f"{name:13s} {r['construction']:13s} J={r['J']:<2d} {r['cyc_per_sq']:7.1f} cyc {r['ns_per_sq']:6.1f} ns "
                  f"root {r['root_us']:7.1f} us {r['x_window']:5.2f}x window {r['x_cpu']:5.2f}x cpu  {r['note']}")


if __name__ == "__main__":
    main()
