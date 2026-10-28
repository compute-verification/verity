"""Where a served_util run's steady steps leave the GPU idle: per step, its kernels' busy time and idle gaps, what the serving
thread (the engine loop's, `serving_thread`) was in during them, by CUDA call and, with PHASES=1's ranges
(`phase_nvtx.py`), by its innermost range, and the kernels before and after each gap of at least ``--min-gap-us``.

    python host_gaps.py --sqlite prof.sqlite --trace advice_trace.jsonl --timings main.timings.json --T 8192
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import served_util as su


#: a kernel only the forward launches: its launching thread is the serving thread when no ranges name it
FORWARD_KERNEL = "exactlin_scaled_mm%"


def serving_thread(db: sqlite3.Connection, lo: int, hi: int) -> tuple[int, str]:
    """The engine loop's thread: the one with the most NVTX ranges, else the one that launched the most `FORWARD_KERNEL`s."""
    if su._has(db, "NVTX_EVENTS"):
        row = db.execute("SELECT globalTid, COUNT(*) FROM NVTX_EVENTS WHERE eventType = 59 AND start < ? AND end > ? "
                         "GROUP BY 1 ORDER BY 2 DESC LIMIT 1", (hi, lo)).fetchone()
        if row:
            return int(row[0]), f"{row[1]} NVTX ranges"
    q = ("SELECT r.globalTid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
         "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId "
         "WHERE s.value LIKE ? AND k.start < ? AND k.end > ? GROUP BY 1 ORDER BY 2 DESC LIMIT 1")
    tid, c = db.execute(q, (FORWARD_KERNEL, hi, lo)).fetchone()
    return int(tid), f"{c} {FORWARD_KERNEL.rstrip('%')} launches"


def ranges(db: sqlite3.Connection, tid: int, lo: int, hi: int) -> list[tuple[int, int, str]]:
    """The closed NVTX push/pop ranges of thread ``tid`` that meet [lo, hi]."""
    if not su._has(db, "NVTX_EVENTS"):
        return []
    q = ("SELECT e.start, e.end, COALESCE(e.text, s.value) FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id = e.textId "
         "WHERE e.globalTid = ? AND e.end IS NOT NULL AND e.end > ? AND e.start < ?")
    return [(int(a), int(b), str(t)) for a, b, t in db.execute(q, (tid, lo, hi))]


def innermost(rs: list[tuple[int, int, str]], lo: int, hi: int) -> list[tuple[int, int, str]]:
    """[lo, hi] cut into segments, each labelled with the path of nested ranges open over it ("" outside every range)."""
    out: list[tuple[int, int, str]] = []
    stack: list[tuple[int, int, str]] = []
    t = lo

    def emit(a: int, b: int, path: list[tuple[int, int, str]]) -> None:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append((a, b, " > ".join(lab for _, _, lab in path)))

    for r in sorted(rs, key=lambda r: (r[0], -r[1])):
        while stack and stack[-1][1] <= r[0]:
            emit(t, stack[-1][1], stack)
            t = max(t, stack.pop()[1])
        emit(t, r[0], stack)
        t = max(t, r[0])
        stack.append(r)
    while stack:
        emit(t, stack[-1][1], stack)
        t = max(t, stack.pop()[1])
    emit(t, hi, [])
    return out


def off_cpu(db: sqlite3.Connection, tid: int, lo: int, hi: int) -> list[tuple[int, int]] | None:
    """The intervals in [lo, hi] thread ``tid`` was scheduled out (CTXSW=1's ``SCHED_EVENTS``), None without them."""
    if not su._has(db, "SCHED_EVENTS"):
        return None
    out, since = [], None
    for t, sched_in in db.execute("SELECT start, isSchedIn FROM SCHED_EVENTS WHERE globalTid = ? ORDER BY start", (tid,)):
        if not sched_in:
            since = int(t)
        elif since is not None:
            a, b = max(since, lo), min(int(t), hi)
            if b > a:
                out.append((a, b))
            since = None
    return out


def on_cpu(db: sqlite3.Connection, tid: int, lo: int, hi: int) -> dict[int, list[tuple[int, int]]]:
    """The intervals in [lo, hi] each other thread of ``tid``'s process was on a CPU (``globalTid``'s bits 24-47 are the pid)."""
    pid = (tid >> 24) & 0xFFFFFF
    since: dict[int, int] = {}
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    q = "SELECT start, isSchedIn, globalTid FROM SCHED_EVENTS WHERE start < ? AND ((globalTid >> 24) & 16777215) = ? ORDER BY start"
    for t, sched_in, g in db.execute(q, (hi, pid)):
        t, g = int(t), int(g)
        if sched_in:
            since[g] = t
        elif g in since:
            a, b = max(since.pop(g), lo), min(t, hi)
            if b > a:
                out[g].append((a, b))
    for g, t in since.items():
        if hi > max(t, lo):
            out[g].append((max(t, lo), hi))
    out.pop(tid, None)
    return out


def intersect(xs: list[tuple[int, int]], ys: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The intersection of two sorted lists of disjoint intervals."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if b > a:
            out.append((a, b))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap(segs: list[tuple[int, int, str]], ivs: list[tuple[int, int]]) -> Counter:
    """ns of the intervals ``ivs`` (sorted, disjoint) under each segment's label."""
    got: Counter = Counter()
    j = 0
    for a, b, lab in segs:
        while j < len(ivs) and ivs[j][1] <= a:
            j += 1
        k = j
        while k < len(ivs) and ivs[k][0] < b:
            got[lab] += min(b, ivs[k][1]) - max(a, ivs[k][0])
            k += 1
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sqlite", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--timings", type=Path)
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--min-gap-us", type=int, default=500)
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    st = su.steps(a.trace)
    z = su.utc0_ns(db)

    def at(s: float) -> int:
        return int(s * 1e9) - z

    ks = su.kernels(db)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    idx = su.steady(st, a.T, float("-inf"), float("inf"), after_s=su.last_send(a.timings))
    spans = [(at(st[i - 1]["t"]), at(st[i]["t"])) for i in idx]
    if not spans:
        print("HOSTGAP no steady steps")
        return
    med = statistics.median(b - x for x, b in spans)
    spans = [(x, b) for x, b in spans if b - x <= su.STEADY_X * med]
    lo, hi, n = spans[0][0], spans[-1][1], len(spans)
    calls = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if su._has(db, table):
            for x, y, nm, tid in db.execute(f"SELECT start, end, nameId, globalTid FROM {table} WHERE end > ? AND start < ?", (lo, hi)):
                calls[int(tid)].append((int(x), int(y), str(names.get(nm, nm))))
    launches = Counter({tid: sum(1 for _, _, nm in cs if nm.startswith("cudaLaunchKernel")) for tid, cs in calls.items()})
    serving, why = serving_thread(db, lo, hi)
    sc = sorted(calls[serving])
    starts = [x for x, _, _ in sc]
    rs = ranges(db, serving, lo, hi)
    offs = off_cpu(db, serving, lo, hi)
    print(f"HOSTGAP {n} steady steps (median wall {med / 1e6:.2f} ms), serving tid {serving} ({why}; {launches[serving]} "
          f"launches; the others {[c for t, c in launches.most_common(4) if t != serving]}), {len(rs)} NVTX ranges on it")
    per = defaultdict(list)
    by_call, pairs = Counter(), defaultdict(lambda: [0, 0, 0.0])
    wall_by, idle_by, gap_by, incl, off_by, idle_off_by = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    idle_off: list[tuple[int, int]] = []
    kstart = [x for x, _, _ in ks]
    min_gap = a.min_gap_us * 1000
    for x0, x1 in spans:
        busy = su.union([(x, y) for x, y, _ in ks if y > x0 and x < x1], x0, x1)
        edges = [x0] + [e for iv in busy for e in iv] + [x1]
        idle = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2) if edges[k + 1] > edges[k]]
        gaps = [(g0, g1) for g0, g1 in idle if g1 - g0 >= min_gap]
        per["wall"].append((x1 - x0) / 1e6)
        per["busy"].append(sum(y - x for x, y in busy) / 1e6)
        per["idle"].append(sum(y - x for x, y in idle) / 1e6)
        per["gaps"].append(sum(y - x for x, y in gaps) / 1e6)
        per["n_gaps"].append(len(gaps))
        for g0, g1 in gaps:
            j = bisect.bisect_left(kstart, g1)
            nxt = ks[j][2] if j < len(ks) else "-"
            prv = max((k for k in ks[max(0, j - 400):j] if k[1] <= g0 + 1), key=lambda k: k[1], default=(0, 0, "-"))[2]
            p = pairs[(prv[:40], nxt[:40])]
            p[0], p[1], p[2] = p[0] + 1, p[1] + g1 - g0, p[2] + (g0 - x0) / 1e6
            inside = 0
            for x, y, nm in sc[max(0, bisect.bisect_left(starts, g0) - 64):]:
                if x >= g1:
                    break
                o = min(y, g1) - max(x, g0)
                if o > 0:
                    by_call[nm] += o
                    inside += o
            by_call["(host, outside CUDA calls)"] += (g1 - g0) - inside
        if rs:
            mine = [r for r in rs if r[1] > x0 and r[0] < x1]
            segs = innermost(mine, x0, x1)
            for s0, s1, lab in segs:
                wall_by[lab] += s1 - s0
            idle_by.update(overlap(segs, idle))
            gap_by.update(overlap(segs, gaps))
            if offs is not None:
                mine_off = intersect(offs, [(x0, x1)])
                waits = intersect(idle, mine_off)
                idle_off.extend(waits)
                off_by.update(overlap(segs, mine_off))
                idle_off_by.update(overlap(segs, waits))
                per["off_cpu"].append(sum(y - x for x, y in mine_off) / 1e6)
                per["idle_off_cpu"].append(sum(y - x for x, y in waits) / 1e6)
            for r0, r1, lab in mine:
                incl[lab] += min(r1, x1) - max(r0, x0)
    if idle_off:
        tnames = {int(g): str(names.get(nm, nm)) for nm, g in db.execute("SELECT nameId, globalTid FROM ThreadNames")} \
            if su._has(db, "ThreadNames") else {}
        held = Counter({g: sum(y - x for x, y in intersect(idle_off, ivs)) for g, ivs in on_cpu(db, serving, lo, hi).items()})
        print("  with the GPU idle and the serving thread off the CPU, its process's other threads on a CPU (ms a step):")
        for g, ns in held.most_common(6):
            print(f"    {ns / 1e6 / n:8.2f}  tid {g & 0xFFFFFF} {tnames.get(g, '?')} ({launches.get(g, 0)} launches)")
    for k, v in per.items():
        print(f"  {k:7s} mean {statistics.fmean(v):8.2f}  median {statistics.median(v):8.2f}  (ms a step; n_gaps a count)")
    print(f"  gaps >= {a.min_gap_us} us, by what the serving thread was in (ms a step):")
    for nm, ns in by_call.most_common(6):
        print(f"    {ns / 1e6 / n:8.2f}  {nm[:90]}")
    print(f"  gaps >= {a.min_gap_us} us by the kernels before and after them (a step: count, ms; mean offset into the step, ms):")
    for (prv, nxt), (c, ns, off) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"    {c / n:5.2f} {ns / 1e6 / n:7.2f} ms  at {off / c:6.1f}  {prv} -> {nxt}")
    if not rs:
        return
    print("  the serving thread's ranges, inclusive (ms a step):")
    for lab, ns in incl.most_common(a.top):
        print(f"    {ns / 1e6 / n:8.2f}  {lab}")
    off = offs is not None
    print(f"  by the serving thread's innermost range (ms a step): its own time, the GPU idle under it, idle in gaps >= {a.min_gap_us} us"
          + (", the thread off the CPU, the GPU idle with the thread off the CPU" if off else ""))
    for lab, ns in sorted(wall_by.items(), key=lambda kv: -idle_by[kv[0]])[:a.top]:
        extra = f" {off_by[lab] / 1e6 / n:8.2f} {idle_off_by[lab] / 1e6 / n:8.2f}" if off else ""
        print(f"    {ns / 1e6 / n:8.2f} {idle_by[lab] / 1e6 / n:8.2f} {gap_by[lab] / 1e6 / n:8.2f}{extra}  {lab or '(no range)'}")


if __name__ == "__main__":
    main()
