#!/usr/bin/env python3
"""A run's report from its outputs (run.sh's $OUT: bench.jsonl, and trace.json when trace.py ran): per shape, the served
call's hashing (`ref_serial`) against pd's fastest schedule whose bytes are the served ones, both from the same interleaved
timing; per launch, the traced critical path (the main lane's hashing launches); and the decode step it projects to, given
that the served hashing adds STEP_SHARE of a stock step: (1 + STEP_SHARE r) / (1 + STEP_SHARE), r the layer's hashing
after over before (each linear once a layer).

  python3 report.py DIR [DIR...]
"""
from __future__ import annotations

import json
import os
import sys

STEP_SHARE = 0.51
SERVED = "ref_serial"
#: a launch's lane when the capture log predates lanes: the side lanes of the schedules that have them
SIDE = {"pd_tiles": "tile", "pd_tree": "tree", "hash_leaf:s512_tile_segs": "tile", "hash_leaf:s512_folds": "tile"}


def records(d: str) -> list[dict]:
    return [json.loads(x) for x in open(os.path.join(d, "bench.jsonl"))]


def lane(key: str, name: str, p: dict) -> str:
    if p.get("lane"):
        return p["lane"]
    return "main" if "serial" in key else SIDE.get(name, "main")


def best(hashing: dict) -> tuple[str, float]:
    """pd's fastest schedule with the served bytes (a `_proposal` schedule is timed only)."""
    keys = [k for k in hashing if k.startswith("pd") and not k.endswith("_proposal")]
    k = min(keys, key=hashing.get)
    return k, hashing[k]


def report(d: str) -> None:
    recs = records(d)
    run = os.path.basename(os.path.dirname(os.path.abspath(d)))
    timing = [r for r in recs if r["kind"] == "timing"]
    checks = {(r["shape"], r["m16"]): r["ok"] for r in recs if r["kind"] == "check"}
    graphs = {(r["shape"], r["m16"]): r["ok"] for r in recs if r["kind"] == "graph_check"}
    mhz = [c for r in timing for c in r["sm_mhz"] if c]
    env = next(r for r in recs if r["kind"] == "env")
    print(f"## {run}  pd cubin {str(env.get('pd_cubin_sha256'))[:12]}  SM {min(mhz)}-{max(mhz)} MHz over the timed reps")
    for r in recs:
        if r["kind"] == "probe" and r["compressions"] == 256:
            print(f"   probe {r.get('form', r['variant'])}: {r['cycles_per_compression']} cycles a compression")
        if r["kind"] == "probe_states":
            print(f"   probe states agree ({r['compressions']}): "
                  + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("kind", "compressions")))
    print("\n| shape | m16 | checks | served µs | ref_deferred µs | pd µs (schedule) | change | proposal µs |")
    print("|---|---|---|---|---|---|---|---|")
    sums: dict[int, list[float]] = {}
    for r in timing:
        h, key = r["hashing_us_per_call"], (r["shape"], r["m16"])
        k, v = best(h)
        prop = next((h[x] for x in h if x.endswith("_proposal")), None)
        ok = "ok" if checks.get(key) and graphs.get(key) else "FAIL"
        print(f"| {r['shape']} | {r['m16']} | {ok} | {h[SERVED]:.1f} | {h['ref_deferred']:.1f} | {v:.1f} ({k}) | "
              f"{100 * (v / h[SERVED] - 1):+.0f}% | {'' if prop is None else f'{prop:.1f}'} |")
        s = sums.setdefault(r["m16"], [0.0, 0.0])
        s[0] += h[SERVED]
        s[1] += v
    print()
    for m16, (before, after) in sums.items():
        rr = after / before
        step = (1 + STEP_SHARE * rr) / (1 + STEP_SHARE)
        print(f"m16 = {m16}: a layer's hashing {before:.0f} -> {after:.0f} µs (r = {rr:.3f}); decode step x{step:.3f} "
              f"({100 * (step - 1):+.1f}%) with the served hashing at {STEP_SHARE} of a stock step")
    tp = os.path.join(d, "trace.json")
    if not os.path.exists(tp):
        return
    tr = json.load(open(tp))
    print("\nper launch (trace, µs per call; * = on the main lane, the call's critical path):")
    for sk, sch in tr.items():
        if sk == "unlabeled":
            continue
        h = next((r["hashing_us_per_call"] for r in timing if f"{r['shape']}/m{r['m16']}" == sk), None)
        keys = [SERVED, best(h)[0]] if h else [x for x in sch if x != "ref_nohash"]
        for key in keys:
            if key not in sch:
                continue
            ls = [(n, p) for n, p in sch[key]["launches"].items() if p["hashing"]]
            crit = sum(p["us"] for n, p in ls if lane(key, n, p) == "main")
            body = ", ".join(f"{'*' if lane(key, n, p) == 'main' else ''}{n.split(':')[-1]} {p['us']:.0f} (d{p['depth']}, "
                             f"{p['grid']}x{p['block']})" for n, p in ls)
            print(f"  {sk} {key}: critical {crit:.0f}  [{body}]")


def main() -> int:
    for d in sys.argv[1:]:
        report(d)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
