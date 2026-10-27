"""The timed block's kernel timeline from an Nsight Systems export (`nsys export --type sqlite`): the longest burst of
kernels with no idle gap over `--gap-ms` (the period's timed block: the warm-up block and setup come before a host
pause), its span, the union of its busy time, and each kernel's time and union, GEMMs named apart: the plain arm's
cuBLASLt kernels and the protocol's `gemm_kernel`.

    timeline.py DB [--gap-ms 5]     prints one JSON object
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict


def union(iv: list[tuple[int, int]]) -> int:
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("db")
    ap.add_argument("--gap-ms", type=float, default=5.0)
    a = ap.parse_args(argv)
    con = sqlite3.connect(a.db)
    rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
                       "ORDER BY k.start").fetchall()
    bursts, cur, end = [], [], None
    for s, e, name in rows:
        if end is not None and s - end > a.gap_ms * 1e6:
            bursts.append(cur)
            cur = []
        cur.append((s, e, name))
        end = e if end is None else max(end, e)
    bursts.append(cur)
    b = max(bursts, key=lambda x: max(e for _, e, _ in x) - x[0][0])
    t0, t1 = b[0][0], max(e for _, e, _ in b)
    by = defaultdict(list)
    for s, e, name in b:
        by[name].append((s, e))
    own = ("form_", "wrap", "bits_kernel", "noise_kernel", "perm_kernel", "salt_", "leaf_", "level_kernel", "top_", "forge_kernel",
           "gemm_kernel", "fill_", "open_kernel")
    lt = [iv for n, ivs in by.items() if not n.startswith(own) for iv in ivs]
    pg = [iv for n, ivs in by.items() if n.startswith("gemm_kernel") for iv in ivs]
    span = t1 - t0
    out = {"span_ms": span / 1e6, "kernels": len(b), "busy_share": union([(s, e) for s, e, _ in b]) / span,
           "cublaslt_gemm_share": union(lt) / span, "protocol_gemm_share": union(pg) / span,
           "by_kernel": sorted(({"name": n, "launches": len(ivs), "sum_ms": sum(e - s for s, e in ivs) / 1e6,
                                 "union_share": union(ivs) / span} for n, ivs in by.items()), key=lambda r: -r["sum_ms"])}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
