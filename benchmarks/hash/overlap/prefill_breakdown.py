"""A served prefill step's GPU time by kernel class (GEMM, attention, other) from an nsys sqlite export of a served_commit
profiling run (benchmarks/served_commit/prof.sh): every `exec:<tokens>` step with more than 64 tokens inside the arm's `run:`
ranges, the kernels launched while it ran (as nsys_summary.py attributes them), their busy time per class (interval union), the
step's GPU-idle time, and the top kernels with their class. Measurement only.

    python prefill_breakdown.py served.sqlite [--run run:stock-prof] > prefill.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict

GEMM = ("gemm", "nvjet", "cutlass", "xmma", "cublas", "matmul", "splitk", "sm90_", "sm100_", "wgmma", "tcgen05")
ATTN = ("flash", "fmha", "attn", "attention", "paged")


def klass(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ATTN):
        return "attention"
    if any(k in low for k in GEMM):
        return "gemm"
    return "other"


def union(iv: list[tuple[int, int]]) -> int:
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ((ce - cs) if ce is not None else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--run", default="run:stock-prof")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    q = lambda s, *p: db.execute(s, p).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t is not None else strings.get(tid), g) for s, e, t, tid, g in
            q("SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    runs = sorted((r for r in nvtx if r[2] == a.run), key=lambda r: r[0])
    if not runs:
        raise SystemExit(f"no NVTX range {a.run}")
    kcols = {r[1] for r in q("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    dcol = "demangledName" if "demangledName" in kcols else "shortName"
    steps, top = [], defaultdict(lambda: [0, 0, ""])
    for r0, r1, _, gtid in runs:
        pid = gtid >> 24
        rt = dict(q("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ? AND start >= ? AND start <= ?",
                    pid, r0, r1))
        kern = [(s, e, strings.get(n, str(n)), strings.get(d, str(d)), c, gx * gy * gz, bx * by * bz, rg, ss + ds)
                for s, e, n, d, c, gx, gy, gz, bx, by, bz, rg, ss, ds in
                q(f"SELECT start, end, shortName, {dcol}, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread, "
                  "staticSharedMemory, dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?",
                  pid, r0, r1 + 5_000_000_000) if c in rt]
        ex = sorted((s, e, int(t.split(":")[1])) for s, e, t, g in nvtx
                    if g >> 24 == pid and r0 <= s <= r1 and t and t.startswith("exec:"))
        bounds = [s for s, _, _ in ex] + [r1]
        for i, (s, e, tok) in enumerate(ex):
            if tok <= 64:
                continue
            lo, hi = s, bounds[i + 1]
            ks = [k for k in kern if lo <= rt[k[4]] < hi]
            if not ks:
                continue
            by = defaultdict(list)
            for k in ks:
                c = klass(k[3])
                by[c].append((k[0], k[1]))
                t = top[k[2]]
                t[0] += k[1] - k[0]
                t[1] += 1
                t[2] = f"{c} grid {k[5]} block {k[6]} regs {k[7]} smem {k[8]}"
            g0, g1 = min(k[0] for k in ks), max(k[1] for k in ks)
            busy = union([(k[0], k[1]) for k in ks])
            steps.append({"tokens": tok, "gpu_span_ms": (g1 - g0) / 1e6, "busy_ms": busy / 1e6, "idle_ms": (g1 - g0 - busy) / 1e6,
                          "kernels": len(ks), **{f"{c}_ms": union(v) / 1e6 for c, v in by.items()},
                          **{f"{c}_n": len(v) for c, v in by.items()}})
    keys = sorted({k for p in steps for k in p if k != "tokens"})
    mean = {k: round(statistics.fmean(p.get(k, 0.0) for p in steps), 4) for k in keys}
    nst = max(len(steps), 1)
    out = {"run": a.run, "prefill_steps": len(steps), "tokens": sorted({p["tokens"] for p in steps}), "mean": mean,
           "top": [(n, round(v[0] / 1e6 / nst, 4), round(v[1] / nst, 2), v[2]) for n, v in sorted(top.items(), key=lambda x: -x[1][0])[:40]],
           "steps": steps}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
