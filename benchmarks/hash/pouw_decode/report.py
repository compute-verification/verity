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
import re
import statistics
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
    return "main" if "serial" in key else SIDE.get(re.sub(r"_\d+$", "", name), "main")


def best(hashing: dict) -> tuple[str, float]:
    """pd's fastest schedule with the served bytes (a `_proposal` schedule is timed only)."""
    keys = [k for k in hashing if k.startswith("pd") and not k.endswith("_proposal")]
    k = min(keys, key=hashing.get)
    return k, hashing[k]


def stamps(r: dict, shape: dict | None, mhz: float) -> None:
    """A PD_STAMP record: the tail CTA's chain, phase by phase from its own SM clock (the rows' phases, then the tail's), with
    each phase's serial compressions and µs a compression; and from the global timer, how far apart the rows start and
    finish their leaves."""
    rows, tail, (cta, sm) = r["rows"], r["tail"], r["tail_cta"]
    if not rows or not rows[cta]["PS_START"][0] or not tail["PT_START"][0]:
        print(f"\nstamps {r['shape']}/m{r['m16']} v{r['variant']}: none written")
        return
    m16, k = r["m16"], (shape or {}).get("k", 0)
    s, levels = k // 256, max(m16 - 1, 0).bit_length()
    me = rows[cta]
    chain = [("segments", me["PS_SEGS"], 9), ("the other warps (salt, frame leaf, pads)", me["PS_SYNC1"], 0),
             ("fold", me["PS_FOLD"], s // 2 + 1 if k else None), ("barrier", me["PS_SYNC2"], 0),
             ("leaf and counter", me["PS_LEAF"], 3), ("to the tail", tail["PT_START"], 0)]
    chain += [(f"level {lv + 1}", tail[f"PT_LEVEL{lv}"], 2) for lv in range(levels)]
    chain += [("seed_A and line_a", tail["PT_SEED_A"], 3), ("filler lines", tail["PT_FILLERS"], 1)]
    g0 = min(x["PS_START"][0] for x in rows if x["PS_START"][0])
    leaf = [x["PS_LEAF"][0] - g0 for x in rows if x["PS_LEAF"][0]]
    total = (chain[-1][1][1] - me["PS_START"][1]) / mhz
    print(f"\nstamps {r['shape']}/m{m16} {r['schedule']}: the tail on row {cta}'s CTA (SM {sm}), {total:.1f} µs from its start "
          f"(SM clock at {mhz:.0f} MHz); rows start within {(max(x['PS_START'][0] for x in rows) - g0) / 1e3:.2f} µs and "
          f"finish their leaves {min(leaf) / 1e3:.1f}-{max(leaf) / 1e3:.1f} µs after the first start (global timer)")
    print("| phase | µs | compressions | µs a compression |\n|---|---|---|---|")
    at, n_all = me["PS_START"][1], 0
    for what, (g, c), n in chain:
        if not c:
            continue
        us = (c - at) / mhz
        at, n_all = c, n_all + (n or 0)
        print(f"| {what} | {us:.1f} | {n if n is not None else '?'} | {us / n:.2f} |" if n else f"| {what} | {us:.1f} | 0 | |")
    print(f"| total | {total:.1f} | {n_all} | {total / n_all:.2f} |" if n_all else "")


def report(d: str) -> None:
    recs = records(d)
    run = os.path.basename(os.path.dirname(os.path.abspath(d)))
    timing = [r for r in recs if r["kind"] == "timing"]
    checks = {(r["shape"], r["m16"]): r["ok"] for r in recs if r["kind"] == "check"}
    graphs = {(r["shape"], r["m16"]): r["ok"] for r in recs if r["kind"] == "graph_check"}
    mhz = [c for r in timing for c in r["sm_mhz"] if c]
    env = next(r for r in recs if r["kind"] == "env")
    print(f"## {run}  pd cubin {str(env.get('pd_cubin_sha256'))[:12]}  SM {min(mhz)}-{max(mhz)} MHz over the timed reps")
    probes = [r for r in recs if r["kind"] == "probe"]
    for r in probes:
        if r["compressions"] == 256:
            cold = next((x.get("cycles_first") for x in probes if x["compressions"] == 1 and x["variant"] == r["variant"]
                         and x.get("form") == r.get("form")), None)
            print(f"   probe {r.get('form', '')} v{r['variant']}: {r['cycles_per_compression']} cycles a compression warm"
                  + ("" if cold is None else f", {cold} for one compression on its code's first fetch"))
    agree = [r.get("agree", all(v for k, v in r.items() if k not in ("kind", "compressions", "form")))
             for r in recs if r["kind"] == "probe_states"]
    if agree:
        print(f"   probe states agree across variants: {all(agree)} ({len(agree)} checks)")
    print("\n| shape | m16 | checks | served µs | ref_deferred µs | pd µs (schedule) | change | proposal µs |")
    print("|---|---|---|---|---|---|---|---|")
    sums: dict[int, list[float]] = {}
    for r in timing:
        h, key = r["hashing_us_per_call"], (r["shape"], r["m16"])
        k, v = best(h)
        prop = min((h[x] for x in h if x.endswith("_proposal")), default=None)
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
    serial = sorted({k for r in timing for k in r["hashing_us_per_call"] if re.fullmatch(r"pd_serial_\d+", k)},
                    key=lambda k: int(k.rsplit("_", 1)[1]))
    if serial:
        print("\npd's variants, serial (hashing µs per call):\n")
        print("| shape | m16 | " + " | ".join(serial) + " |")
        print("|---|---|" + "---|" * len(serial))
        for r in timing:
            h = r["hashing_us_per_call"]
            print(f"| {r['shape']} | {r['m16']} | " + " | ".join(f"{h[k]:.1f}" for k in serial) + " |")
    shapes = {(r["shape"], r["m16"]): r for r in recs if r["kind"] == "shape"}
    for r in (x for x in recs if x["kind"] == "stamps"):
        stamps(r, shapes.get((r["shape"], r["m16"])), statistics.median(mhz))
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
