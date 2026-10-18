#!/usr/bin/env python3
"""bench.py --trace's graphs read back from nsys (`nsys export --type sqlite`, --cuda-graph-trace=node): each cuGraphLaunch
matched in order to bench.trace.jsonl's launches (`graph_check`'s, skipped, then `trace_order`'s), its kernels by correlation
id, each kernel labelled with the call step
that launched it (the graph's nodes in id order against the capture log `graph_kernels`, checked kernel by kernel; a graph
whose nodes don't match keeps bare kernel names, and says so), and per schedule and shape, per call (CHAIN calls a graph):
the span, the kernels' busy time, and per launch its count, duration, the idle time before it (nothing running), the share of
it overlapped by another kernel, its grid and block, its depth (bench.depths) and its lane (`main` is the call's critical
path; the side lanes `tile` and `tree` feed nothing later in the call).

  python3 trace.py DIR    # DIR/trace.sqlite and DIR/bench.trace.jsonl in; DIR/trace.json out, a table on stdout
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys

CHAIN = 8
#: the kernels that hash (Pearl-C's and pd's); every other kernel is the call's forming, cleanup or GEMM
HASHING = ("s512_", "hm_leaves", "lines", "pd_")


def load(d: str):
    recs = [json.loads(x) for x in open(os.path.join(d, "bench.trace.jsonl"))]
    db = sqlite3.connect(os.path.join(d, "trace.sqlite"))
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    launches = db.execute("SELECT r.start, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
                          "WHERE s.value LIKE 'cuGraphLaunch%' ORDER BY r.start").fetchall()
    kern: dict[int, list] = {}
    for start, end, corr, short, gx, gy, gz, bx, node in db.execute(
            "SELECT start, end, correlationId, shortName, gridX, gridY, gridZ, blockX, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kern.setdefault(corr, []).append({"s": start, "e": end, "kernel": names[short], "grid": gx * gy * gz, "block": bx,
                                          "node": node})
    return recs, launches, kern


def label(ks: list[dict], seq: list) -> bool:
    """Each kernel's label: its step and kernel from the capture log, matched through the graph's node ids in order."""
    by_node = sorted({k["node"] for k in ks})
    ok = len(by_node) == len(ks) == len(seq)
    if ok:
        pos = {n: i for i, n in enumerate(by_node)}
        ok = all(seq[pos[k["node"]]][1] == k["kernel"] for k in ks)
    for k in ks:
        entry = seq[pos[k["node"]]] if ok else [k["kernel"], k["kernel"]]
        k["label"] = k["kernel"] if entry[0] == k["kernel"] else f"{entry[0]}:{k['kernel']}"
        k["lane"] = entry[2] if len(entry) > 2 else None
    return ok


def summarize(ks: list[dict], depth: dict[str, int]) -> dict:
    ks = sorted(ks, key=lambda k: k["s"])
    t0, t1 = ks[0]["s"], max(k["e"] for k in ks)
    busy, idle, last = 0, 0, t0
    per: dict[str, dict] = {}
    for i, k in enumerate(ks):
        s, e = k["s"], k["e"]
        gap = max(0, s - last)
        idle += gap
        busy += max(0, e - max(s, last))
        last = max(last, e)
        over = 0
        for j, k2 in enumerate(ks):
            if j != i and k2["s"] < e and k2["e"] > s:
                over = max(over, min(e, k2["e"]) - max(s, k2["s"]))
        p = per.setdefault(k["label"], {"n": 0, "ns": 0, "gap_ns": 0, "overlap_ns": 0, "grid": k["grid"], "block": k["block"],
                                        "depth": depth.get(k["label"]), "hashing": k["kernel"].startswith(HASHING),
                                        "lane": k["lane"]})
        p["n"] += 1
        p["ns"] += e - s
        p["gap_ns"] += gap
        p["overlap_ns"] += over
    us = lambda ns: round(ns / 1e3 / CHAIN, 2)  # noqa: E731 - per call
    return {"span_us": us(t1 - t0), "busy_us": us(busy), "idle_us": us(idle), "kernels_per_call": len(ks) / CHAIN,
            "hashing_us": us(sum(p["ns"] for p in per.values() if p["hashing"])),
            "launches": {name: {"per_call": p["n"] / CHAIN, "us": us(p["ns"]), "gap_us": us(p["gap_ns"]),
                                "overlapped": round(p["overlap_ns"] / p["ns"], 2) if p["ns"] else 0.0, "grid": p["grid"],
                                "block": p["block"], "depth": p["depth"], "hashing": p["hashing"], "lane": p["lane"]}
                         for name, p in per.items()}}


def median_of(runs: list[dict]) -> dict:
    """The median, over a schedule's traced launches, of every number in `summarize`'s output."""
    out = {k: statistics.median(r[k] for r in runs) for k in ("span_us", "busy_us", "idle_us", "kernels_per_call", "hashing_us")}
    out["launches"] = {}
    for name, first in runs[-1]["launches"].items():
        rows = [r["launches"][name] for r in runs if name in r["launches"]]
        out["launches"][name] = {**first, **{k: statistics.median(x[k] for x in rows) for k in ("us", "gap_us", "overlapped")}}
    return out


def depth_of(sh: dict, key: str) -> dict[str, int]:
    out = {name if step == "pd" else f"{step}:{name}": v for step, rows in sh["depths"].items() for name, v in rows}
    if key in ("pd_side_tree", "pd_side_tree_proposal"):
        out["pd_commit"] = out["pd_commit/rows"]
    return out


def main() -> int:
    d = sys.argv[1]
    recs, launches, kern = load(d)
    shapes = {(r["shape"], r["m16"]): r for r in recs if r["kind"] == "shape"}
    # every graph launch in order: graph_check's (one per graph it names, before the traced ones), then trace_order's
    order = [(r["shape"], r["m16"], key, r["kind"] == "graph_check") for r in recs if r["kind"] in ("graph_check", "trace_order")
             for key in (r["failed"] if r["kind"] == "graph_check" else r["launches"])]
    if len(order) != len(launches):
        print(f"{len(launches)} cuGraphLaunch calls traced, {len(order)} in graph_check and trace_order", file=sys.stderr)
        return 1
    runs: dict[tuple, list] = {}
    unlabeled = []
    for (shape, m16, key, checking), (_, corr) in zip(order, launches):
        if checking:
            continue
        sh, ks = shapes[(shape, m16)], kern.get(corr, [])
        if not label(ks, sh["graph_kernels"][key]):
            unlabeled.append(f"{shape}/m{m16}/{key}")
        runs.setdefault((shape, m16, key), []).append(summarize(ks, depth_of(sh, key)))
    out = {"unlabeled": sorted(set(unlabeled))}
    for (shape, m16, key), rs in runs.items():
        out.setdefault(f"{shape}/m{m16}", {})[key] = median_of(rs)
    json.dump(out, open(os.path.join(d, "trace.json"), "w"), indent=1)
    if unlabeled:
        print(f"graphs whose nodes didn't match the capture log (bare kernel names): {out['unlabeled']}")
    for sk, sch in out.items():
        if sk == "unlabeled":
            continue
        base = sch.get("ref_nohash", {}).get("span_us", 0.0)
        print(f"\n== {sk}  (us per call; exposed = span - ref_nohash's span {base})")
        for key, s in sch.items():
            print(f"  {key:24s} span {s['span_us']:8.2f}  exposed {s['span_us'] - base:7.2f}  busy {s['busy_us']:8.2f}  "
                  f"idle {s['idle_us']:6.2f}  hashing kernels {s['hashing_us']:7.2f}")
            for name, p in s["launches"].items():
                if p["hashing"]:
                    print(f"      {name:28s} x{p['per_call']:<4g} {p['us']:7.2f} us  gap {p['gap_us']:5.2f}  "
                          f"overlapped {p['overlapped']:4.2f}  grid {p['grid']:5d} x {p['block']:3d}  depth {p['depth']}  "
                          f"lane {p.get('lane')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
