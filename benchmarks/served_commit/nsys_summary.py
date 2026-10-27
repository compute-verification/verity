"""Summarize an nsys sqlite export of a served_commit profiling run (prof_arm.py): the last `run:` NVTX range of the profiled arm,
its GPU time by category, its host ranges, CUDA API time per thread, and per engine step (exec:<tokens>).  Measurement only.

    python nsys_summary.py OUT/served.sqlite [--run run:ARM] [--each] [--json OUT/summary.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict

COMMIT_KERNELS = (("chunk_leaves", "hash.leaves"), ("tree_level", "hash.tree"), ("gather", "collect.gather"), ("pos_leaves", "hash.leaves"),
                  ("rows_leaf", "hash.leaves"), ("rows_level", "hash.tree"))
COPY = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def cat(name: str) -> str:
    low = name.lower()
    return next((c for k, c in COMMIT_KERNELS if k in low), "serve")


def union(iv: list[tuple[int, int]]) -> int:
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json")
    ap.add_argument("--run", default="run:served-commit-rows")
    ap.add_argument("--each", action="store_true", help="every run range of the arm, not only the last: per run its step count, "
                    "prefill_mean and decode_mean on one line, and the full summaries as a JSON list")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    q = lambda s, *p: db.execute(s, p).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))
    nvtx = [(s, e, t if t is not None else strings.get(tid), g) for s, e, t, tid, g in
            q("SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    runs = sorted((r for r in nvtx if r[2] == a.run), key=lambda r: r[0])
    if not runs:
        raise SystemExit(f"no NVTX range {a.run}")
    if not a.each:
        out = summarize(q, strings, nvtx, runs[-1], a.run)
        js = json.dumps(out, indent=1)
        if a.json:
            open(a.json, "w").write(js)
        for k, v in out.items():
            if k != "steps":
                print(k, json.dumps(v)[:3000])
        return
    outs = []
    for i, r in enumerate(runs):
        out = summarize(q, strings, nvtx, r, a.run)
        outs.append(out)
        toks = [p["tokens"] for p in out["steps"]]
        print(json.dumps({"run": i, "steps": len(toks), "max_tokens": max(toks, default=0),
                          "run_wall_ms": round(out["run_wall_ms"], 3), "gpu_busy_serve_ms": round(out["gpu_busy_serve_ms"], 3),
                          "gpu_busy_hash_ms": round(out["gpu_busy_hash_ms"], 3), "prefill_mean": out.get("prefill_mean"),
                          "decode_mean": out.get("decode_mean")}))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(outs, f, indent=1)


def summarize(q, strings: dict, nvtx: list, run: tuple, run_label: str) -> dict:
    """One run range's summary: GPU time by category, host ranges, CUDA API per thread, and per engine step."""
    r0, r1, _, gtid = run
    pid = gtid >> 24
    names = {g: strings.get(n, str(n)) for n, g in q("SELECT nameId, globalTid FROM ThreadNames")}
    rt = {c: (s, e, strings.get(n, str(n)), g) for s, e, n, g, c in
          q("SELECT start, end, nameId, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid >> 24 = ?", pid)}
    kcols = {r[1] for r in q("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    gcol = "graphNodeId" if "graphNodeId" in kcols else "NULL"
    kern = [(s, e, strings.get(n, str(n)), c, st, gn) for s, e, n, c, st, gn in
            q(f"SELECT start, end, shortName, correlationId, streamId, {gcol} FROM CUPTI_ACTIVITY_KIND_KERNEL "
              "WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?", pid, r0, r1 + 5_000_000_000)]
    mcols = {r[1] for r in q("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
    mg = "graphNodeId" if "graphNodeId" in mcols else "NULL"
    mem = q(f"SELECT start, end, bytes, copyKind, correlationId, streamId, {mg} FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE globalPid >> 24 = ? AND end >= ? AND start <= ?", pid, r0, r1 + 5_000_000_000)
    # only work launched inside the run's host period
    inrun = lambda c: c in rt and r0 <= rt[c][0] <= r1  # noqa: E731
    kern = [k for k in kern if inrun(k[3])]
    mem = [m for m in mem if inrun(m[4])]
    gpu_end = max([k[1] for k in kern] + [m[1] for m in mem] + [r1])
    out: dict = {"run_wall_ms": (r1 - r0) / 1e6, "gpu_tail_ms": (gpu_end - r1) / 1e6}
    by = defaultdict(lambda: [0, 0])
    for s, e, n, c, st, gn in kern:
        k = cat(n) + (".graph" if gn else "")
        by[k][0] += e - s
        by[k][1] += 1
    for s, e, b, ck, c, st, gn in mem:
        k = f"memcpy.{COPY.get(ck, ck)}" + (".graph" if gn else "")
        by[k][0] += e - s
        by[k][1] += 1
        by[k + ".GB"][0] += b
    out["gpu_ms_by_category"] = {k: (round(v[0] / 1e6, 3) if not k.endswith(".GB") else round(v[0] / 1e9, 3), v[1])
                                 for k, v in sorted(by.items())}
    allv = [(k[0], k[1]) for k in kern] + [(m[0], m[1]) for m in mem]
    out["gpu_busy_ms"] = union(allv) / 1e6
    serve = [(k[0], k[1]) for k in kern if cat(k[2]) == "serve"]
    hashing = [(k[0], k[1]) for k in kern if cat(k[2]).startswith("hash.")]
    out["gpu_busy_serve_ms"] = union(serve) / 1e6
    out["gpu_busy_hash_ms"] = union(hashing) / 1e6
    # time a hash kernel and a serving kernel both ran (the side stream's overlap with serving)
    out["hash_overlap_serve_ms"] = (union(serve) + union(hashing) - union(serve + hashing)) / 1e6
    top = defaultdict(lambda: [0, 0])
    for s, e, n, c, st, gn in kern:
        top[n][0] += e - s
        top[n][1] += 1
    out["top_kernels"] = [(n, round(v[0] / 1e6, 3), v[1]) for n, v in sorted(top.items(), key=lambda x: -x[1][0])[:25]]
    streams = defaultdict(lambda: [0, 0])
    for s, e, n, c, st, gn in kern:
        streams[(st, cat(n))][0] += e - s
        streams[(st, cat(n))][1] += 1
    out["streams"] = {f"{st}:{c}": (round(v[0] / 1e6, 3), v[1]) for (st, c), v in sorted(streams.items())}
    # host: nvtx ranges in the run, CUDA API per thread
    rng = defaultdict(lambda: [0, 0])
    steps = []
    for s, e, t, g in nvtx:
        if g >> 24 == pid and r0 <= s <= r1 and t and t != run_label:
            key = t.split(":")[0]
            rng[key][0] += e - s
            rng[key][1] += 1
            if key == "exec":
                steps.append((s, e, int(t.split(":")[1])))
    out["host_ranges_ms"] = {k: (round(v[0] / 1e6, 3), v[1]) for k, v in sorted(rng.items())}
    api = defaultdict(lambda: [0, 0])
    for c, (s, e, n, g) in rt.items():
        if r0 <= s <= r1:
            api[(names.get(g, str(g & 0xFFFFFF)), n)][0] += e - s
            api[(names.get(g, str(g & 0xFFFFFF)), n)][1] += 1
    out["cuda_api_ms"] = [(t, n, round(v[0] / 1e6, 3), v[1]) for (t, n), v in sorted(api.items(), key=lambda x: -x[1][0])[:30]]
    # per step: host exec span and the GPU time of work launched while it ran (any thread)
    steps.sort()
    bounds = [s for s, _, _ in steps] + [r1]
    per = []
    for i, (s, e, tok) in enumerate(steps):
        lo, hi = s, bounds[i + 1]
        g = defaultdict(int)
        for ks, ke, n, c, st, gn in kern:
            if lo <= rt[c][0] < hi:
                g[cat(n)] += ke - ks
        for ms, me, b, ck, c, st, gn in mem:
            if lo <= rt[c][0] < hi:
                g["memcpy." + COPY.get(ck, str(ck)) + (".graph" if gn else "")] += me - ms
        per.append({"tokens": tok, "host_exec_ms": (e - s) / 1e6, "step_ms": (hi - lo) / 1e6, **{k: v / 1e6 for k, v in g.items()}})
    out["steps"] = per
    for label, sel in (("prefill", lambda p: p["tokens"] > 64), ("decode", lambda p: p["tokens"] <= 64)):
        ps = [p for p in per if sel(p)]
        if not ps:
            continue
        keys = sorted({k for p in ps for k in p if k != "tokens"})
        out[f"{label}_mean"] = {"n": len(ps), **{k: round(statistics.fmean(p.get(k, 0.0) for p in ps), 3) for k in keys}}
    return out


if __name__ == "__main__":
    main()
