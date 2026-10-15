"""The tensor-core receipts' cost table from tb_run.sh's bench.jsonl (and ncu's counts when the run dir has them), as Markdown.

python3 tb_table.py BENCH.jsonl [RUN_OUT_DIR]

η is m n k over the time, against 1,024 int8 MACs/SM/clk x SMs x the SM clock NVML read at each rep. Arms: lt cuBLASLt
int8; b the noised call without receipts (form + GEMM + peel); c duo /32 and c2 duo2 /32 (W13's); t4, t2, t1 the
tensor-core receipts TB4, TB2, TB1 /32 (form + GEMM); g0, t4g, t2g, t1g the GEMMs alone.
"""

import sys
from pathlib import Path

from table import LINEAR, ncu_tensor

CALL = ["lt", "b", "c", "c2", "t4", "t2", "t1"]
GEMM = ["g0", "t4g", "t2g", "t1g"]


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
    import json

    bench = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print("Each arm at its fastest tile config: η, and time over cuBLASLt's (lt)")
    print()
    print("| shape | " + " | ".join(f"{a} η" for a in CALL) + " | " + " | ".join(f"{a} vs lt" for a in CALL[1:]) + " | tiles (t4, t2, t1) |")
    print("|" + "---|" * (2 * len(CALL) + 1))
    for r in best(bench):
        a = r["arms"]
        print(f"| {r['label']} | " + " | ".join(f"{a[x]['eta']:.3f}" for x in CALL) + " | " +
              " | ".join(f"{100 * (a[x]['ms'] / a['lt']['ms'] - 1):+.0f}%" for x in CALL[1:]) +
              f" | {a['t4']['cfg']}, {a['t2']['cfg']}, {a['t1']['cfg']} |")
    print()
    print("Every shape and tile config: η per arm; the receipted GEMM over the plain noised GEMM (g0) and the call over (b)")
    print()
    cols = ["lt", "plain", "g0", "b", "c", "c2", "t4", "t2", "t1", "t4g", "t2g", "t1g"]
    print("| shape | tile | " + " | ".join(cols) + " | t4g/g0 | t2g/g0 | t1g/g0 | c/b | c2/b | t4/b | t2/b | t1/b | MHz |")
    print("|" + "---|" * (2 + len(cols) + 9))
    for s in bench:
        a = s["arms"]
        t = {k: a[k]["ms"] for k in a}
        rat = lambda x, y: f"{t[x] / t[y]:.3f}"
        print(f"| {s['shape']} | {s['tile']} | " + " | ".join(f"{a[k]['eta']:.3f}" for k in cols) + " | " +
              " | ".join([rat("t4g", "g0"), rat("t2g", "g0"), rat("t1g", "g0"), rat("c", "b"), rat("c2", "b"),
                          rat("t4", "b"), rat("t2", "b"), rat("t1", "b")]) + f" | {a['t4']['mhz']:.0f} |")
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
