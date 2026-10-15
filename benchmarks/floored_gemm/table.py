"""The four-kernel table from fgemm's bench.jsonl (and ncu's tensor-pipe counts when the run dir has them), as Markdown.

python3 table.py BENCH.jsonl [RUN_OUT_DIR]

Per shape: eta (useful MACs m n k over the time, against 1,024 int8 MACs/SM/clk x SMs x the SM clock NVML read at each
rep), and each arm's time over (a)'s minus 1. (a) cuBLASLt int8; (b) noised, no receipts (form + GEMM + peel); (c) duo at
stride 32; (d) duo at 64; c2 / d2 duo2. `plain` is this kernel's mainloop on the un-noised operands, `form` the forming
passes alone, `g0` (b)'s GEMM and peel alone.
"""

import csv
import json
import sys
from pathlib import Path

ARMS = [("lt", "(a) cuBLASLt"), ("plain", "mine, plain"), ("b", "(b) noised"), ("c", "(c) duo /32"), ("d", "(d) duo /64"),
        ("c2", "duo2 /32"), ("d2", "duo2 /64")]


def ncu_tensor(path: Path):
    """Duration-weighted tensor-pipe active % (of elapsed) over the kernels ncu profiled, and the kernels' total ns."""
    if not path.exists():
        return None
    rows = [r for r in csv.reader(l for l in path.open() if l.startswith('"'))]
    if not rows:
        return None
    head, body = rows[0], rows[1:]
    try:
        ki, mi, vi = head.index("Kernel Name"), head.index("Metric Name"), head.index("Metric Value")
        ii = head.index("ID")
    except ValueError:
        return None
    per: dict[str, dict[str, float]] = {}
    for r in body:
        try:
            per.setdefault(r[ii] + ":" + r[ki], {})[r[mi]] = float(r[vi].replace(",", ""))
        except ValueError:
            pass
    tot = sum(v.get("gpu__time_duration.sum", 0) for v in per.values())
    if not tot:
        return None
    pipe = sum(v.get("gpu__time_duration.sum", 0) * v.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
               for v in per.values()) / tot
    gemm = max(per.values(), key=lambda v: v.get("gpu__time_duration.sum", 0))
    return {"pipe_pct": pipe, "ns": tot, "kernels": len(per),
            "gemm_pipe_pct": gemm.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0),
            "gemm_alu_pct": gemm.get("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", 0),
            "gemm_issue_pct": gemm.get("smsp__issue_active.avg.pct_of_peak_sustained_active", 0),
            "gemm_dram_pct": gemm.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)}


def main() -> int:
    bench = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print("| shape | m n k | " + " | ".join(f"{t} η" for _, t in ARMS) + " | (b) vs (a) | (c) vs (a) | (d) vs (a) | duo2/32 vs (a) "
          "| duo2/64 vs (a) | (c) vs (b) | form share of (b) | MHz |")
    print("|" + "---|" * (2 + len(ARMS) + 8))
    for s in bench:
        a = s["arms"]
        t = {k: a[k]["ms"] for k in a}
        ov = lambda x, y="lt": f"{100 * (t[x] / t[y] - 1):+.1f}%"
        print(f"| {s['shape']} | {s['m']} {s['n']} {s['k']} | " + " | ".join(f"{a[k]['eta']:.3f}" for k, _ in ARMS) +
              f" | {ov('b')} | {ov('c')} | {ov('d')} | {ov('c2')} | {ov('d2')} | {ov('c', 'b')} | "
              f"{100 * t['form'] / t['b']:.1f}% | {a['c']['mhz']:.0f} |")
    if out:
        print()
        print("ncu (--clock-control none), one call: tensor pipe active % of elapsed, duration-weighted over the call's kernels; "
              "the GEMM kernel's tensor / ALU (of active) / issue (of active) / DRAM %")
        print()
        print("| shape | arm | call tensor % | GEMM tensor % | GEMM ALU % | GEMM issue % | GEMM DRAM % | kernels | ns |")
        print("|---|---|---|---|---|---|---|---|---|")
        for f in sorted(out.glob("ncu-*.csv")):
            nm, arm = f.stem[4:].rsplit("-", 1)
            r = ncu_tensor(f)
            if r:
                print(f"| {nm} | {arm} | {r['pipe_pct']:.1f} | {r['gemm_pipe_pct']:.1f} | {r['gemm_alu_pct']:.1f} | "
                      f"{r['gemm_issue_pct']:.1f} | {r['gemm_dram_pct']:.1f} | {r['kernels']} | {r['ns']:.0f} |")
            else:
                print(f"| {nm} | {arm} | (no data) | | | | | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
