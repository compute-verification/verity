"""How much the hash kernels slow the model's own kernels and the collector's gather when they run at once, from a served_commit nsys
report: for each arm's last profiled run, every kernel nsys_summary.py files as serve or collect.gather, grouped by name, its mean
duration while no hash kernel ran (clean) and while hash kernels covered more than half of it (beside), and their ratio.  A kernel's
work does not depend on what runs beside it, so the ratio is the contention.  Measurement only.

    python3 benchmarks/hash/loop/overlap.py OUT/served.nsys-rep [--arms served-commit-rows,...] [--top 15]   (nsys on PATH for a report)
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


def merged(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def covered(m: list[tuple[int, int]], starts: list[int], s: int, e: int) -> int:
    i = max(bisect.bisect_right(starts, s) - 1, 0)
    tot = 0
    while i < len(m) and m[i][0] < e:
        tot += max(0, min(e, m[i][1]) - max(s, m[i][0]))
        i += 1
    return tot


def arm_rows(q, strings: dict, run: tuple, top: int) -> list[tuple]:
    r0, r1, _, gtid = run
    pid = gtid >> 24
    launch = {c: s for s, c in q("SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ?", pid)}
    sel = "SELECT start, end, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?"
    kern = [(s, e, strings.get(n, str(n))) for s, e, n, c in q(sel, pid, r0, r1 + 5_000_000_000) if r0 <= launch.get(c, -1) <= r1]
    hm = merged([(s, e) for s, e, n in kern if NS.cat(n).startswith("hash.")])
    hs = [s for s, _ in hm]
    by: dict[str, list[list[int]]] = defaultdict(lambda: [[], []])
    for s, e, n in kern:
        if NS.cat(n) not in ("serve", "collect.gather") or e <= s:
            continue
        c = covered(hm, hs, s, e)
        if c == 0:
            by[n][0].append(e - s)
        elif 2 * c > e - s:
            by[n][1].append(e - s)
    rows = []
    for n, (clean, beside) in by.items():
        if clean and beside:
            mc, mb = statistics.fmean(clean), statistics.fmean(beside)
            rows.append((n, len(clean), mc / 1e3, len(beside), mb / 1e3, mb / mc, (mb - mc) * len(beside) / 1e6))
    return sorted(rows, key=lambda r: -r[6])[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--arms", default="served-commit-rows,served-commit-rows-blake3")
    ap.add_argument("--top", type=int, default=15)
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
    for arm in a.arms.split(","):
        runs = sorted((r for r in nvtx if r[2] == f"run:{arm}"), key=lambda r: r[0])
        if not runs:
            print(f"{arm}: no run range")
            continue
        print(f"{arm} (last run): kernel, n clean, mean clean us, n beside the hash, mean beside us, ratio, ms added a run")
        for n, nc, mc, nb, mb, ratio, added in arm_rows(q, strings, runs[-1], a.top):
            print(f"  {n[:60]:<60} {nc:>6} {mc:>9.1f} {nb:>6} {mb:>9.1f} {ratio:>6.2f} {added:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
