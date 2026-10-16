"""A gpu.sh run's outputs, one line a measurement: `python3 summary.py OUT`; `python3 summary.py best OUT` prints the tag of
the checked build whose e2e commitment was fastest."""
from __future__ import annotations

import json
import os
import sys


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.startswith("{")] if os.path.exists(path) else []


def tagged(out: str, prefix: str) -> list[tuple[str, list[dict]]]:
    names = sorted(x for x in os.listdir(out) if x.startswith(prefix) and x.endswith(".jsonl"))
    return [(x[len(prefix):-len(".jsonl")], lines(os.path.join(out, x))) for x in names]


def best(out: str) -> str:
    runs = [(r[0]["commit_ms"], tag) for tag, r in tagged(out, "bench-e2e-") if r and "nomask" not in tag]
    return min(runs)[1] if runs else ""


def bench_line(tag: str, r: dict) -> str:
    if "bench" in r:
        return (f"{tag} {r['bench']} n={r['n']} bits={r['lbits']}: commit {r['commit_ms']:.3f} ms (leaves {r['leaf_ms']:.3f}, "
                f"nodes {r['nodes_ms']:.3f}), {r['operand_gb_per_s']:.1f} GB/s of operand, {r['gcomp_per_s']:.2f} G compressions/s, "
                f"{r['mhz']:.0f} MHz")
    g = r["beside_gemm"]
    return (f"  beside GEMM {g['m']}x{g['n']}x{g['k']} ({g['gemm_ms']:.3f} ms, x{g['gemms']}): commit/GEMM {g['commit_over_gemm']:.2f}; "
            f"both {g['both_wall_ms']:.3f} ms vs serial {g['serial_wall_ms']:.3f}; hidden {g['hidden_fraction']:.2f}; "
            f"commit beside {g['both_commit_ms']:.3f} ms ({g['commit_gb_per_s_beside']:.1f} GB/s)")


def main(out: str) -> None:
    for tag, rows in tagged(out, "check-"):
        done = [r for r in rows if "summary" not in r]
        bad = [r for r in done if not r["ok"]]
        print(f"check {tag}: {len(done)} cases, {len(bad)} failed", *(f"\n  FAILED {r}" for r in bad[:5]))
    for tag, rows in tagged(out, "sha-"):
        for r in rows:
            s = r["sha"]
            print(f"sha {tag}: {s['gcomp_per_s']:.2f} G compressions/s ({s['gb_per_s']:.0f} GB/s of blocks) at {s['mhz']:.0f} MHz")
    for prefix in ("bench-e2e-", "bench-sweep-", "bench-seg-"):
        for tag, rows in tagged(out, prefix):
            for r in rows:
                print(bench_line(tag, r))


if __name__ == "__main__":
    print(best(sys.argv[2])) if sys.argv[1] == "best" else main(sys.argv[1])
