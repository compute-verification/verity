"""Markdown tables from one overlap run's outputs (bench.jsonl, nsys_cuda_gpu_trace.csv).

    python3 summary.py OUT
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys


def rows(path: str) -> list[dict]:
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main(o: str) -> int:
    b = rows(os.path.join(o, "bench.jsonl"))
    dev = next((r for r in b if "device" in r), {})
    print(f"# overlap: {dev.get('device', '?')} ({dev.get('sms', '?')} SMs)\n")
    print("| kernel | regs | smem | local |\n|---|---|---|---|")
    for r in b:
        if "kernel" in r:
            print(f"| {r['kernel']} | {r['regs']} | {r['smem']} | {r['local']} |")
    print()
    clocks = [c for r in b if "sm_mhz" in r for c in r["sm_mhz"]]
    if clocks:
        print(f"SM clock over every timed item: min {min(clocks)}, median {statistics.median(clocks)}, max {max(clocks)} MHz\n")
    gates = [r for r in b if "gate" in r]
    print(f"device gates: {sum(r['ok'] for r in gates)} of {len(gates)} pass\n")
    print("| shape | GEMM | hash | leaves | cuBLASLt ms | GEMM ms | hash ms | seq ms | par ms | hidden (own GEMM) | hidden (vs cuBLASLt) | "
          "par / (cuBLASLt + hash) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in b:
        if "summary" in r:
            print(f"| {r['summary']} | {r['gemm']} | {r['variant']} | {r['leaves']} | {r['lt_ms']:.4f} | {r['gemm_ms']:.4f} | {r['hash_ms']:.4f} | "
                  f"{r['seq_ms']:.4f} | {r['par_ms']:.4f} | {100 * r['hidden_vs_own_gemm']:.1f}% | {100 * r['hidden_vs_lt']:.1f}% | "
                  f"{r['net_vs_lt_plus_hash']:.3f} |")
    print("\n| chain (per layer) | hash | cuBLASLt ms | GEMM ms | hash ms | seq ms | par ms | hidden (own GEMM) | hidden (vs cuBLASLt) | "
          "par / (cuBLASLt + hash) | exposed GB/s |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in b:
        if "chain_summary" in r:
            exp = r["exposed_hash_ms"]
            gbps = r["bytes_per_layer"] / (exp * 1e-3) / 1e9 if exp > 0 else float("inf")
            print(f"| {r['chain_summary']} | {r['variant']} | {r['lt_ms']:.4f} | {r['gemm_ms']:.4f} | {r['hash_ms']:.4f} | {r['seq_ms']:.4f} | "
                  f"{r['par_ms']:.4f} | {100 * r['hidden_vs_own_gemm']:.1f}% | {100 * r['hidden_vs_lt']:.1f}% | {r['net_vs_lt_plus_hash']:.3f} | "
                  f"{gbps:,.0f} |")
    for path in glob.glob(os.path.join(o, "nsys*_cuda_gpu_trace.csv")):
        seen: dict[str, tuple] = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                name = r.get("Name", "")
                if not r.get("GrdX"):
                    continue
                key = name[:110]
                if key not in seen:
                    seen[key] = (r["GrdX"], r["GrdY"], r["GrdZ"], r["BlkX"], r.get("Reg/Trd", ""), r.get("StcSMem (MB)", ""),
                                 r.get("DymSMem (MB)", ""), r.get("Duration (ns)", ""))
        print("\n| kernel (nsys) | grid | block | regs | static smem MB | dynamic smem MB | ns |\n|---|---|---|---|---|---|---|")
        for k, v in seen.items():
            print(f"| `{k}` | {v[0]}x{v[1]}x{v[2]} | {v[3]} | {v[4]} | {v[5]} | {v[6]} | {v[7]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
