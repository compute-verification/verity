"""Where a step's commit work sits, from a served_commit nsys report: for one arm's last profiled run and one phase (decode: steps
of at most 64 tokens, prefill: the rest), every kernel launched while a step ran, grouped by name, with its launches a step, mean
duration, grid and block, stream, and the median start of its first launch relative to the step's first and last model kernel
(ms; negative: before). Also, per step, the GPU's idle time between the step's model kernels (the gaps a hash could fill without
sharing SMs). Measurement only.

    python3 benchmarks/hash/loop/kernels.py OUT/served.nsys-rep [--arm served-commit-rows] [--phase decode] [--top 20]
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "served_commit"))
import nsys_summary as NS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--arm", default="served-commit-rows")
    ap.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    db_path = a.report
    if a.report.suffix != ".sqlite":
        db_path = Path(tempfile.mkdtemp()) / "served.sqlite"
        subprocess.run(["nsys", "export", "--type", "sqlite", "-f", "true", "-o", str(db_path), str(a.report)], check=True,
                       stdout=subprocess.DEVNULL)
    db = sqlite3.connect(db_path)
    q = lambda s, *p: db.execute(s, p).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t is not None else strings.get(tid), g) for s, e, t, tid, g in
            q("SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    runs = sorted((r for r in nvtx if r[2] == f"run:{a.arm}"), key=lambda r: r[0])
    if not runs:
        print(f"no run range run:{a.arm}", file=sys.stderr)
        return 1
    r0, r1, _, gtid = runs[-1]
    pid = gtid >> 24
    launch = {c: s for s, c in q("SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ?", pid)}
    kern = [(s, e, strings.get(n, str(n)), launch.get(c, -1), st, (gx, gy, gz, bx)) for s, e, n, c, st, gx, gy, gz, bx in
            q("SELECT start, end, shortName, correlationId, streamId, gridX, gridY, gridZ, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL "
              "WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?", pid, r0, r1 + 5_000_000_000)]
    steps = sorted((s, int(t.split(":")[1])) for s, e, t, g in nvtx if g >> 24 == pid and r0 <= s <= r1 and t and t.startswith("exec:"))
    bounds = [s for s, _ in steps] + [r1]
    by: dict[str, list] = defaultdict(lambda: [0, [], set(), set(), [], []])
    idle, nsteps = [], 0
    for i, (s, tok) in enumerate(steps):
        if (tok <= 64) != (a.phase == "decode"):
            continue
        nsteps += 1
        ks = [k for k in kern if s <= k[3] < bounds[i + 1]]
        serve = sorted((k[0], k[1]) for k in ks if NS.cat(k[2]) == "serve")
        if not serve:
            continue
        first, last = serve[0][0], max(e for _, e in serve)
        idle.append((last - first - NS.union(serve)) / 1e6)
        seen: set[str] = set()
        for k in sorted(ks):
            b = by[k[2]]
            b[0] += 1
            b[1].append(k[1] - k[0])
            b[2].add(k[5])
            b[3].add(k[4])
            if k[2] not in seen:
                seen.add(k[2])
                b[4].append((k[0] - first) / 1e6)
                b[5].append((k[0] - last) / 1e6)
    print(f"{a.arm} {a.phase}: {nsteps} steps; GPU idle between the step's first and last model kernel {statistics.fmean(idle):.3f} ms "
          f"a step (median {statistics.median(idle):.3f})")
    print(f"  {'kernel':<44} {'cat':<15} {'n/step':>6} {'mean us':>8} {'ms/step':>7} {'grid/block':<18} {'stream':<8} "
          f"{'start-first':>11} {'start-last':>10}")
    rows = sorted(by.items(), key=lambda kv: -sum(kv[1][1]))
    shown = [kv for kv in rows if NS.cat(kv[0]) != "serve"][:a.top] + [kv for kv in rows if NS.cat(kv[0]) == "serve"][:5]
    for n, (cnt, durs, grids, sts, f0, f1) in shown:
        g = ",".join(f"{x}x{y}x{z}/{t}" for x, y, z, t in sorted(grids)[:2]) + ("…" if len(grids) > 2 else "")
        print(f"  {n[:44]:<44} {NS.cat(n):<15} {cnt / nsteps:>6.1f} {statistics.fmean(durs) / 1e3:>8.1f} "
              f"{sum(durs) / 1e6 / nsteps:>7.3f} {g:<18} {','.join(map(str, sorted(sts))):<8} "
              f"{statistics.median(f0):>11.3f} {statistics.median(f1):>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
