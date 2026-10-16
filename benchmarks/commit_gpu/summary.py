"""One line per measurement of a gpu.sh run's outputs: `python3 summary.py OUT`."""
from __future__ import annotations

import json
import os
import sys


def lines(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.startswith("{")]


def main(out: str) -> None:
    for f in sorted(x for x in os.listdir(out) if x.startswith("check-") and x.endswith(".jsonl")):
        rows = lines(os.path.join(out, f))
        done = [r for r in rows if "summary" not in r]
        bad = [r for r in done if not r["ok"]]
        print(f"{f}: {len(done)} cases, {len(bad)} failed", *(f"\n  FAILED {r}" for r in bad[:5]))
    for r in lines(os.path.join(out, "sha.jsonl")):
        s = r["sha"]
        print(f"sha add={s['sha_add']}: {s['gcomp_per_s']:.2f} G compressions/s ({s['gb_per_s']:.0f} GB/s of blocks) at {s['mhz']:.0f} MHz")
    for name in ("bench-e2e.jsonl", "bench-sweep.jsonl"):
        for r in lines(os.path.join(out, name)):
            if "bench" in r:
                print(f"{r['bench']} n={r['n']} bits={r['lbits']} add={r['sha_add']}: commit {r['commit_ms']:.3f} ms "
                      f"(leaves {r['leaf_ms']:.3f}, nodes {r['nodes_ms']:.3f}), {r['operand_gb_per_s']:.1f} GB/s of operand, "
                      f"{r['gcomp_per_s']:.2f} G compressions/s, {r['mhz']:.0f} MHz")
            else:
                g = r["beside_gemm"]
                print(f"  beside GEMM {g['m']}x{g['n']}x{g['k']} ({g['gemm_ms']:.3f} ms, x{g['gemms']}): commit/GEMM "
                      f"{g['commit_over_gemm']:.2f}; both {g['both_wall_ms']:.3f} ms vs serial {g['serial_wall_ms']:.3f}; "
                      f"hidden {g['hidden_fraction']:.2f}; commit beside {g['both_commit_ms']:.3f} ms "
                      f"({g['commit_gb_per_s_beside']:.1f} GB/s)")


if __name__ == "__main__":
    main(sys.argv[1])
