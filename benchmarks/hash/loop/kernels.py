"""Where a step's commit work sits, from a served_commit nsys report: for one arm's last profiled run and one phase (decode: steps
of at most 64 tokens, prefill: the rest), every kernel launched while a step ran, grouped by name, with its launches a step, mean
duration, grid and block, stream, and the median start of its first launch relative to the step's first and last model kernel
(ms; negative: before). Also, per step, the GPU's idle time between the step's model kernels (the gaps a hash could fill without
sharing SMs). `--gaps` instead groups those idle gaps by the kernels on either side, with the share whose next kernel was launched
after the gap opened (the host was late) and the runtime calls the launching thread made inside them. Measurement only.

    python3 benchmarks/hash/loop/kernels.py OUT/served.nsys-rep [--arm served-commit-rows] [--phase decode] [--top 20] [--gaps]
"""
from __future__ import annotations

import argparse
import bisect
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
    ap.add_argument("--gaps", action="store_true")
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
    api = q("SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid = ? AND start >= ? AND start <= ? "
            "ORDER BY start", gtid, r0, r1)
    launch = {c: s for s, c in q("SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ?", pid)}
    kern = [(s, e, strings.get(n, str(n)), launch.get(c, -1), st, (gx, gy, gz, bx)) for s, e, n, c, st, gx, gy, gz, bx in
            q("SELECT start, end, shortName, correlationId, streamId, gridX, gridY, gridZ, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL "
              "WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?", pid, r0, r1 + 5_000_000_000)]
    steps = sorted((s, int(t.split(":")[1])) for s, e, t, g in nvtx if g >> 24 == pid and r0 <= s <= r1 and t and t.startswith("exec:"))
    bounds = [s for s, _ in steps] + [r1]
    if a.gaps:
        return gaps(a, steps, bounds, kern, api, strings, nvtx, pid)
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


def gaps(a, steps, bounds, kern, api, strings, nvtx, pid) -> int:
    pairs: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0, 0, 0])   # count, ns, ns host-late, count host-late
    calls: dict[str, int] = defaultdict(int)
    ranges: dict[str, int] = defaultdict(int)
    api_starts = [s for s, _, _, _ in api]
    marks = [(s, e, t) for s, e, t, g in nvtx if g >> 24 == pid and t and not t.startswith(("run:", "exec:"))]
    total, nsteps = 0, 0
    for i, (s, tok) in enumerate(steps):
        if (tok <= 64) != (a.phase == "decode"):
            continue
        serve = sorted(k for k in kern if s <= k[3] < bounds[i + 1] and NS.cat(k[2]) == "serve")
        if not serve:
            continue
        nsteps += 1
        end, prev = serve[0][1], serve[0][2]
        for k in serve[1:]:
            if k[0] > end + 2_000:
                g0, g1 = end, k[0]
                p = pairs[(prev[:40], k[2][:40])]
                p[0] += 1
                p[1] += g1 - g0
                total += g1 - g0
                if k[3] > g0:
                    p[2] += g1 - g0
                    p[3] += 1
                j = bisect.bisect_left(api_starts, g0 - 1_000_000)
                while j < len(api) and api[j][0] < g1:
                    c0, c1 = max(api[j][0], g0), min(api[j][1], g1)
                    if c1 > c0:
                        calls[strings.get(api[j][3], str(api[j][3])).split("_v")[0]] += c1 - c0
                    j += 1
                for m0, m1, t in marks:
                    c0, c1 = max(m0, g0), min(m1, g1)
                    if c1 > c0:
                        ranges[t.split(":")[0]] += c1 - c0
            if k[1] > end:
                end, prev = k[1], k[2]
    if not nsteps:
        print(f"no {a.phase} steps", file=sys.stderr)
        return 1
    per = lambda ns: ns / 1e6 / nsteps  # noqa: E731
    print(f"{a.arm} {a.phase}: {nsteps} steps; GPU idle between model kernels {per(total):.3f} ms a step")
    print(f"  {'before the gap':<40} {'after the gap':<40} {'n/step':>6} {'ms/step':>7} {'host-late ms':>12} {'host-late n':>11}")
    for (b, n), (cnt, ns, late, nlate) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"  {b:<40} {n:<40} {cnt / nsteps:>6.1f} {per(ns):>7.3f} {per(late):>12.3f} {nlate / nsteps:>11.1f}")
    print("  runtime calls inside the gaps (ms/step): " + ", ".join(f"{c} {per(v):.3f}" for c, v in
                                                              sorted(calls.items(), key=lambda kv: -kv[1])[:10]))
    print("  NVTX ranges over the gaps (ms/step): " + ", ".join(f"{t} {per(v):.3f}" for t, v in
                                                          sorted(ranges.items(), key=lambda kv: -kv[1])[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
