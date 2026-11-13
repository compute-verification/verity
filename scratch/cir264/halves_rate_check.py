"""The sums of a kept `halves_rate.json` recomputed from its `halves_rate_positions.jsonl` (each draft row whose halves' sums
differ; only those can split), with what the report does not print: each rate's standard error over requests, how far the
two sums are apart, how many of the differing positions carry the rate, and the rate by the draft token's probability.

    python scratch/cir264/halves_rate_check.py DIR      (DIR holds halves_rate.json and halves_rate_positions.jsonl[.gz])
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

WORDS = 1 << 32
K = 5


def main() -> int:
    d = Path(sys.argv[1])
    record = json.loads((d / "halves_rate.json").read_text())
    p = d / "halves_rate_positions.jsonl"
    lines = (gzip.open(str(p) + ".gz", "rt") if not p.exists() else p.open()).read().splitlines()
    rows = [json.loads(x) for x in lines]
    f32 = lambda w: float(np.uint32(w).view(np.float32))  # noqa: E731
    for kind, k in record["kinds"].items():
        agg = collections.Counter()
        for v in k["agg"].values():
            agg.update(v)
        reqs = collections.Counter()
        for v in k["requests"].values():
            reqs.update(v)
        N = int(reqs["requests"])
        mine = [r for r in rows if r["kind"] == kind]
        elig = [r for r in mine if r["i"] < K - 1]
        q = lambda r: max(r["below_lp"] - r["below_lp_hi"], 0) / WORDS  # noqa: E731
        qrev = lambda r: max(r["below_lp_hi"] - r["below_lp"], 0) / WORDS  # noqa: E731
        sums = {"rows_differ": len(mine), "pos_differ": len(elig), "own_differ": sum(r["live"] for r in elig),
                "wq": sum(r["w"] * q(r) for r in elig), "wq_rev": sum(r["w"] * qrev(r) for r in elig),
                "rq": sum(r["r"] * q(r) for r in elig), "own_q": sum(q(r) for r in elig if r["live"]),
                "own_split": sum(r["live"] and r["test"] and not r["test_hi"] for r in elig)}
        print(f"{kind}: {N} requests; recomputed from {len(mine)} kept rows against the record's sums:")
        for name, v in sums.items():
            ok = abs(v - agg[name]) <= 1e-9 * max(abs(v), 1e-30) + 1e-18
            print(f"  {name:<12} {v:.6g} record {agg[name]:.6g} {'equal' if ok else 'DIFFERS'}")
        per = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in elig:
            t = per[r["request"]]
            t[0] += r["w"] * q(r)
            t[1] += q(r) * r["live"]
            t[2] += r["r"] * q(r)
        for j, (name, den) in enumerate((("rate a position, exact reach", agg["w"]), ("rate a position, own reach", agg["own_pos"]),
                                         ("rate a request (P)", N))):
            x = np.array([t[j] for t in per.values()])
            tot = x.sum()
            se = (max((x * x).sum() - tot * tot / N, 0.0)) ** 0.5
            print(f"  {name}: {tot / den:.3e} +- {se / den:.1e} (over {den:.6g}; {int((x > 0).sum())} requests carry it; the "
                  f"largest request {x.max() / tot:.1%} of it)")
        gap = collections.Counter()
        for r in elig:
            lo, hi = r["lse"], r["lse_hi"]
            gap[(hi - lo) if (lo < 1 << 31) == (hi < 1 << 31) else "signs"] += 1
        print(f"  lse_hi - lse in last places, positions i < K-1: {dict(sorted(gap.items(), key=str))}")
        qs = np.array([q(r) for r in elig if q(r) > 0])
        print(f"  positions with words that split: {qs.size}; their words: min {int(qs.min() * WORDS)}, median "
              f"{int(np.median(qs) * WORDS)}, max {int(qs.max() * WORDS)} of 2^32")
        by = collections.defaultdict(lambda: [0, 0.0])
        for r in elig:
            lse = f32(r["lse"])
            b = "below 4" if lse < 4 else "4 to 8" if lse < 8 else "8 to 16" if lse < 16 else "16 or above"
            by[b][0] += 1
            by[b][1] += r["w"] * q(r)
        print("  by lane 0's log-sum-exp (differing positions, their share of the exact-reach rate): "
              + "; ".join(f"{b}: {n}, {s / max(sums['wq'], 1e-300):.0%}" for b, (n, s) in sorted(by.items())))
        pairs = collections.Counter(tuple(r["pairs"]) for r in elig)
        print(f"  differing positions by which tile pairs hold mass: {dict(sorted(pairs.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
