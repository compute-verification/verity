"""The tensor-core receipts' cost table from tb_run.sh's bench.jsonl (and ncu's counts when the run dir has them), as Markdown.

python3 tb_table.py BENCH.jsonl [RUN_OUT_DIR]

η is m n k over the time, against 1,024 int8 MACs/SM/clk x SMs x the SM clock NVML read at each rep. Arms: lt cuBLASLt
int8; b the noised call without receipts (form + GEMM + peel); c duo /32 and c2 duo2 /32 (W13's); t4, t2, t1 the
tensor-core receipts TB4, TB2, TB1 /32 (form + GEMM), t2w and t1w their k32 packings, a trailing c the chained kinds;
g0 and every arm ending in g the GEMM alone.
"""

import json
import sys
from pathlib import Path

from table import LINEAR, ncu_tensor

ORDER = ["lt", "b", "c", "c2", "t4", "t2", "t1", "t2w", "t1w", "t4c", "t1c", "t2wc", "t1wc"]


def label(s):
    return "4096³" if s["m"] == s["n"] == s["k"] else f"{LINEAR.get((s['n'], s['k']), s['shape'])} m={s['m']}"


def best(bench):
    by = {}
    for s in bench:
        row = by.setdefault((s["m"], s["n"], s["k"]), {"label": label(s), "arms": {}})
        for a, v in s["arms"].items():
            if a not in row["arms"] or v["ms"] < row["arms"][a]["ms"]:
                row["arms"][a] = dict(v, cfg=s["tile"])
    return list(by.values())


def main() -> int:
    bench = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    call = [a for a in ORDER if all(a in s["arms"] for s in bench)]
    gemm = [a + "g" for a in call if a.startswith("t") and all(a + "g" in s["arms"] for s in bench)]
    print("Each arm at its fastest tile config: η, and time over cuBLASLt's (lt)")
    print()
    print("| shape | " + " | ".join(f"{a} η" for a in call) + " | " + " | ".join(f"{a} vs lt" for a in call[1:]) + " |")
    print("|" + "---|" * (2 * len(call)))
    for r in best(bench):
        a = r["arms"]
        print(f"| {r['label']} | " + " | ".join(f"{a[x]['eta']:.3f}" for x in call) + " | " +
              " | ".join(f"{100 * (a[x]['ms'] / a['lt']['ms'] - 1):+.0f}%" for x in call[1:]) + " |")
    print()
    print("Every shape and tile config: η per arm, each receipted GEMM over the noised GEMM g0, each receipted call over (b)")
    print()
    cols = ["g0"] + call + gemm
    rcv = [x for x in call if x not in ("lt", "b")]
    print("| shape | tile | " + " | ".join(cols) + " | " + " | ".join(f"{g}/g0" for g in gemm) + " | " +
          " | ".join(f"{x}/b" for x in rcv) + " | MHz |")
    print("|" + "---|" * (3 + len(cols) + len(gemm) + len(rcv)))
    for s in bench:
        a = s["arms"]
        t = {k: a[k]["ms"] for k in a}
        print(f"| {s['shape']} | {s['tile']} | " + " | ".join(f"{a[k]['eta']:.3f}" for k in cols) + " | " +
              " | ".join(f"{t[g] / t['g0']:.3f}" for g in gemm) + " | " + " | ".join(f"{t[x] / t['b']:.3f}" for x in rcv) +
              f" | {a['lt']['mhz']:.0f} |")
    if out:
        print()
        print("ncu (--clock-control none), one call of the GEMM alone: tensor pipe active % of elapsed, ALU / issue % of active")
        print()
        print("| shape | arm | tensor % | ALU % | issue % | DRAM % | ns |")
        print("|---|---|---|---|---|---|---|")
        for f in sorted(out.glob("ncu-*.csv")):
            nm, arm = f.stem[4:].rsplit("-", 1)
            r = ncu_tensor(f)
            if r:
                print(f"| {nm} | {arm} | {r['gemm_pipe_pct']:.1f} | {r['gemm_alu_pct']:.1f} | {r['gemm_issue_pct']:.1f} | "
                      f"{r['gemm_dram_pct']:.1f} | {r['ns']:.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
