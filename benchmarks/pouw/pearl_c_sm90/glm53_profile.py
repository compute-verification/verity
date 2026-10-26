"""Pearl-C's served call on one H100 or H200 under Nsight Systems: where a call's time goes, step by step, at GLM-5.3's expert
shapes, decode's m = 16 and prefill's m = 2,048 (by default).

  nsys profile -t cuda,nvtx -o DIR/prof python benchmarks/pouw/pearl_c_sm90/glm53_profile.py run [--shapes N:K ...] [--m M ...]
  nsys export -t sqlite -o DIR/prof.sqlite DIR/prof.nsys-rep
  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py report DIR/prof.sqlite --out DIR/profile.json

``run`` installs the run as glm53_timing.py does (`glm53_timing.device_run`) and, at each shape and m, with hashing on and
off, begins a fresh pass, makes `WARM` calls, then `CALLS` calls back to back as the timing's samples run them, then one call
alone between synchronizations (``solo``).  NVTX ranges name each call (``call N:K mM hash|nohash #i``) and, inside it, each
pipeline step (``step NAME``) and the call's host work around the steps: the keys and their upload (``host keys``,
``host upload``), and A's root, the screen and the records (``host after``, ``host screen``); the rest of the call's range
is x into A's words and Python.  The pass's end, its tile tree, is ``pass_end``.  The ranges are this script's alone: it wraps
the run's step functions and methods on its own instance, and every launch, byte and commitment is the run's.

``report`` reads nsys's SQLite export.  Each kernel, copy and memset is attributed, by its launch's correlation id, to the
innermost range around the launch on its thread.  Per call it averages, over the back-to-back calls, each step's GPU time (its
kernels' summed durations) and host time (its range), and gives the call's host wall, its GPU busy time, and the GPU's span a
call (the first call's first kernel to the last call's last, over `CALLS`); ``solo`` gives the GPU's span of one call alone,
its critical path.  It lists the kernels by name with their mean durations, and names any reference SHA-512 launch
(`FALLBACKS`) that ran where the fast ones (`HASH_FAST`, `HASH_DECODE`) should.  Writes profile.json, prints a table a call.

  nsys profile -t cuda,nvtx --cuda-graph-trace=node -o DIR/prof python benchmarks/pouw/pearl_c_sm90/glm53_profile.py run \
    --schedule deferred-a --graphs --calls 12 [--shapes N:K ...] [--m M ...]
  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py lanes DIR/prof.sqlite --out DIR/lanes.json

``run`` under a schedule with lanes, or graphed, names only the calls (their steps are lanes' launches or graph replays).
``lanes`` reads where back-to-back calls' GPU time goes by stream: each activity goes to the call whose range its launch (a
graph's nodes: the graph's launch) falls in, and each stream to its lane (`LANE_OF`).  Per label: the median period from one
call's first activity to the next's; for each lane over the calls between the first and the last, its first activity since
its call's first (``start_us``), span, busy time and the idle time before it since the call before's last activity there
(``idle_before_us``); each lane's kernels' mean durations; and the middle call's activities in order.  Writes lanes.json.

  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py stamps --out DIR/stamps.json [--shapes N:K ...] [--m M ...] [--schedule S ...]

``stamps`` (no nsys) reads where A's commitment spends a decode call, inside pd_commit and pd_tree: at each shape, m (default
`STAMP_M`) and schedule (default serial and deferred-a), after `WARM` calls, `CALLS` calls alone between synchronizations,
each on the stamp twins (`run.Pipeline.stamps`: each phase's end, hash_decode.cuh's PS_* and PT_*, as the global timer and
the SM's clock).  Per call: each phase's end since its CTA's start by the SM's clock (median and max over the rows' CTAs),
the CTAs' start and end on the global timer since the first one's start, and the tail's (A's levels, seed_A, the padding
rows' lines; pd_tree's entry when it runs apart).

  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py host --out DIR/host.json [--shapes N:K ...] [--m M ...] [--schedule S ...] [--graphs]

``host`` (no nsys) reads where a call's host time goes: at each shape (default `HOST_SHAPES`), m (default `STAMP_M`) and
schedule (default serial, deferred and deferred-a; with hashing, and serial's without), after `WARM` calls, `HOST_CALLS`
calls back to back, timed by the host's clock (until the last returns) and the GPU's events (until its work ends), then
as many again under cProfile: each function's own time a call, the `HOST_TOP` largest.

  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py grouped --out DIR/grouped.json [--shapes N:K ...] [--pattern P ...] [--schedule S ...] [--graphs]
  nsys profile -t cuda,nvtx --cuda-graph-trace=node -o DIR/prof python benchmarks/pouw/pearl_c_sm90/glm53_profile.py grouped --nvtx --calls 12 [--graphs]

``grouped`` is ``host`` for the grouped MoE call (`pouw_pearl_c_grouped.gemm_grouped`) as grouped_check.py makes it: its
experts at each of its shapes, its routed rows (``--pattern``, default decode), under each schedule (default grouped_check's)
with hashing, with ``--graphs`` each again with the run's graphs (after grouped_check's `WARM_GRAPHS` calls, so the layout's
lane graphs replay), and the first's without hashing, beside grouped_check's stock arms.  With ``--nvtx`` it makes `CALLS`
(``--calls``) NVTX-ranged calls an arm back to back instead, for ``lanes`` (labels ``grouped N:K PATTERN ARM``).
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm53  # noqa: E402

WARM, CALLS = 3, 5
SHAPES = ((4096, 6144), (6144, 2048))
MS = (16, 2048)
#: hash_fast.cuh's launches, hash_decode.cuh's and hash_prefill.cuh's (its code form past decode's rows), and the reference
#: ones (hash_h3.cuh) they replace
HASH_FAST = ("s512_row_segs", "s512_tile_segs", "s512_folds")
HASH_DECODE = ("pd_commit_", "pd_tiles_", "pd_tree_", "pl_")
FALLBACKS = ("s512_rows", "s512_tile_leaves")
STAMP_M = (16, 64)
HOST_SHAPES = ((512, 6144), (4096, 6144))
HOST_CALLS, HOST_TOP = 200, 25
#: hash_decode.cuh's stamp slots: a pd_commit CTA's phases (PS_*), and the tail's (PT_*: level l + 1 at PT_LEVEL + l)
PS = ("start", "segs", "salt", "frame", "sync1", "fold", "sync2", "leaf", "seed", "seg1", "seg8", "seed0", "line")
PT = {0: "start", **{1 + level: f"level{level + 1}" for level in range(8)}, 9: "seed_a", 10: "fillers", 11: "entry"}
PT_ROW = 15
#: a stream's lane in ``lanes``: the streams that run each kernel in turn, of those not named yet, numbered in their order
#: where there are several (a side lane a ring set); the others are named by their stream's id
LANE_OF = (("pd_commit_", "main"), ("pd_tiles_", "side"), ("pd_tree_", "tree"))


def _ranged(name: str, f):
    import torch

    def g(*a, **kw):
        torch.cuda.nvtx.range_push(name)
        try:
            return f(*a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
    return g


def _instrument(run, PPD) -> None:
    """NVTX ranges around each step of every call pipeline the run has built so far (the serial schedule's main lane), once."""
    for key, (pipe, main, side) in list(run.pipes.items()):
        if getattr(pipe, "_ranged", False):
            continue
        have = {n for n, _, _ in pipe.steps()}
        present = [x for x in run._call_names(pipe) if x in have]
        if side or len(present) != len(main):
            raise RuntimeError(f"{key}: steps {present} aren't the main lane's {len(main)} launches")
        run.pipes[key] = (pipe, [_ranged(f"step {x}", f) for x, f in zip(present, main, strict=True)], side)
        pipe._ranged = True


def run_mode(a) -> int:
    import torch
    import glm53_timing as T
    from verity_vllm.protocols.pouw import pearl_c as PPD

    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else list(SHAPES)
    schedule = (a.schedule or ["serial"])[0]
    run, weights, kernel, gen = T.device_run(shapes, a.scheme, a.cubin, schedule, a.graphs)
    steps = schedule == "serial" and not a.graphs
    run.keys = _ranged("host keys", run.keys)
    run.uploads.put = _ranged("host upload", run.uploads.put)
    run._after = _ranged("host after", run._after)
    PPD.screen_into = _ranged("host screen", PPD.screen_into)
    end_pass, calls = run.end_pass, a.calls or CALLS
    for w, W in weights.items():
        n, k = W.shape
        for m in a.m or MS:
            x = torch.randn(m, k, device=W.device, generator=gen).to(torch.bfloat16)
            for hashing in (True, False):
                end_pass()
                run.hashing = hashing
                for _ in range(WARM):
                    run.gemm(w, x, W, None)
                if steps:
                    _instrument(run, PPD)
                torch.cuda.synchronize()
                label = f"{n}:{k} m{m} {'hash' if hashing else 'nohash'}"
                for i in range(calls):
                    torch.cuda.nvtx.range_push(f"call {label} #{i}")
                    run.gemm(w, x, W, None)
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"solo {label}")
                run.gemm(w, x, W, None)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                if hashing:
                    torch.cuda.nvtx.range_push(f"pass_end {label}")
                    end_pass()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
    end_pass()
    print(json.dumps({"scheme": a.scheme, "device": torch.cuda.get_device_name(), "cubin_sha256": kernel.cubin_sha256,
                      "kernel_source_sha256": kernel.source_sha256, "shapes": shapes, "m": list(a.m or MS), "warm": WARM,
                      "calls": calls, "schedule": schedule, "graphs": a.graphs}), flush=True)
    return 0


def stamp_rows(raw: bytes, count: int, slots: int) -> tuple[list[dict], dict]:
    """The stamps' (timer ns, clock) pairs by phase: each pd_commit CTA's (rows 0 … count − 1, `PS`), and the tail's (row
    count, `PT`, with ``cta``: the (row, SM) of pd_commit's last CTA when it ran the tail)."""
    import struct
    words = struct.unpack(f"<{len(raw) // 8}Q", raw)

    def at(row: int, slot: int) -> tuple[int, int]:
        return words[2 * (slots * row + slot)], words[2 * (slots * row + slot) + 1]
    rows = [{nm: at(r, s) for s, nm in enumerate(PS) if at(r, s)[0]} for r in range(count)]
    tail = {nm: at(count, s) for s, nm in PT.items() if at(count, s)[0]}
    cta = at(count, PT_ROW)
    return rows, {**tail, **({"cta": cta} if cta != (0, 0) else {})}


def stamp_summary(rows: list[dict], tail: dict) -> dict:
    """One call's stamps (`stamp_rows`) as the module's docstring gives them, in µs."""
    rows = [r for r in rows if "start" in r]
    if not rows:
        return {"rows": 0}
    t0 = min(r["start"][0] for r in rows)
    spans = [(r[e][1] - r["start"][1], r[e][0] - r["start"][0]) for r in rows for e in [max(r, key=lambda nm: r[nm][0])]]
    spans = [(c, ns) for c, ns in spans if ns > 2000]
    ghz = statistics.median(c / ns for c, ns in spans) if spans else None
    phases = {}
    for nm in PS[1:]:
        d = [(r[nm][1] - r["start"][1]) / ghz / 1e3 for r in rows if nm in r] if ghz else []
        if d:
            phases[nm] = {"median_us": statistics.median(d), "max_us": max(d)}
    starts = [(r["start"][0] - t0) / 1e3 for r in rows]
    ends = [(max(v[0] for v in r.values()) - t0) / 1e3 for r in rows]
    return {"rows": len(rows), "clock_ghz": ghz, "since_cta_start_us": phases,
            "cta_start_us": {"median": statistics.median(starts), "max": max(starts)},
            "cta_end_us": {"median": statistics.median(ends), "max": max(ends)},
            "tail_us": {nm: (v[0] - t0) / 1e3 for nm, v in tail.items() if nm != "cta"},
            **({"tail_cta": list(tail["cta"])} if "cta" in tail else {})}


def stamps_mode(a) -> int:
    import torch
    import glm53_timing as T

    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else sorted(glm53.in_domain(a.scheme))
    schedules = a.schedule or ["serial", "deferred-a"]
    run, weights, kernel, gen = T.device_run(shapes, a.scheme, a.cubin, schedules[0])
    R = kernel.run
    buf = torch.zeros(R.PD_STAMP_BYTES, dtype=torch.uint8, device=run.device)
    rows = []
    for w, W in weights.items():
        n, k = W.shape
        for m in a.m or STAMP_M:
            x = torch.randn(m, k, device=W.device, generator=gen).to(torch.bfloat16)
            m16 = -(-m // 16) * 16
            if R.pd_variant(m16, n, k) is None:
                continue
            for sc in schedules:
                run.end_pass(schedule=sc)
                for _ in range(WARM):
                    run.gemm(w, x, W, None)
                pipes = [p for p, _, _ in run.pipes.values()]
                calls = []
                try:
                    for p in pipes:
                        p.stamps = buf.data_ptr()
                    for _ in range(CALLS):
                        buf.zero_()
                        torch.cuda.synchronize()
                        run.gemm(w, x, W, None)
                        torch.cuda.synchronize()
                        calls.append(stamp_summary(*stamp_rows(buf.cpu().numpy().tobytes(), m16, R.PD_SLOTS)))
                finally:
                    for p in pipes:
                        p.stamps = 0
                end = sorted(calls, key=lambda c: c["cta_end_us"]["max"])[len(calls) // 2]
                row = {"n": n, "k": k, "m": m, "schedule": sc, "variant": R.pd_variant(m16, n, k), "median_call": end,
                       "calls": calls}
                rows.append(row)
                ph = end["since_cta_start_us"]
                print(f"{n}:{k} m{m} {sc}: CTAs end by {end['cta_end_us']['max']:.1f} us (start skew "
                      f"{end['cta_start_us']['max']:.1f}); phases " + ", ".join(f"{nm} {v['median_us']:.1f}/{v['max_us']:.1f}"
                                                                                for nm, v in ph.items())
                      + "; tail " + ", ".join(f"{nm} {v:.1f}" for nm, v in end["tail_us"].items()), flush=True)
    run.end_pass(schedule="serial")
    out = {"scheme": a.scheme, "device": torch.cuda.get_device_name(), "cubin_sha256": kernel.cubin_sha256,
           "kernel_source_sha256": kernel.source_sha256, "warm": WARM, "calls": CALLS, "rows": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if rows else 1


def _calls(run, f, n: int) -> tuple[float, float]:
    """`n` calls of f back to back: µs a call by the host's clock (until the last returns) and by the GPU's events (until its
    work, the run's side lanes' too, ends)."""
    import time

    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    t = time.perf_counter()
    for _ in range(n):
        f()
    host = (time.perf_counter() - t) * 1e6 / n
    run.join_sides(torch.cuda.current_stream())
    e.record()
    e.synchronize()
    return host, s.elapsed_time(e) * 1e3 / n


def _profiled(run, f, n: int) -> tuple[float, list[dict]]:
    """`_calls`' host µs a call under cProfile, and the `HOST_TOP` functions by own time a call."""
    import cProfile
    import pstats
    prof = cProfile.Profile()
    prof.enable()
    host, _ = _calls(run, f, n)
    prof.disable()
    top = sorted(((v[2], v[3], fn) for fn, v in pstats.Stats(prof).stats.items()), reverse=True)[:HOST_TOP]
    return host, [{"fn": f"{fn[0].rsplit('/', 2)[-1]}:{fn[1]}({fn[2]})", "own": tt * 1e6 / n, "cum": ct * 1e6 / n}
                  for tt, ct, fn in top]


def host_mode(a) -> int:
    import torch
    import glm53_timing as T

    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else list(HOST_SHAPES)
    schedules = a.schedule or ["serial", "deferred", "deferred-a"]
    run, weights, kernel, gen = T.device_run(shapes, a.scheme, a.cubin, schedules[0], a.graphs)
    arms = [(sc, True) for sc in schedules] + [(schedules[0], False)]
    rows = []

    for w, W in weights.items():
        n, k = W.shape
        for m in a.m or STAMP_M:
            x = torch.randn(m, k, device=W.device, generator=gen).to(torch.bfloat16)
            for sc, hashing in arms:
                run.end_pass(schedule=sc)
                run.hashing = hashing

                def f(w=w, x=x, W=W):
                    run.gemm(w, x, W, None)
                for _ in range(WARM):
                    f()
                host_us, gpu_us = _calls(run, f, HOST_CALLS)
                prof_host_us, own = _profiled(run, f, HOST_CALLS)
                row = {"n": n, "k": k, "m": m, "schedule": sc, "hashing": hashing, "graphs": run._graphed(-(-m // 16) * 16),
                       "host_us": host_us, "events_us": gpu_us, "profiled_host_us": prof_host_us, "own_us": own}
                rows.append(row)
                print(f"{n}:{k} m{m} {sc} {'hash' if hashing else 'nohash'}{' graphs' if row['graphs'] else ''}: host "
                      f"{host_us:.1f} us a call, events {gpu_us:.1f} (profiled host {prof_host_us:.1f}); own: "
                      + ", ".join(f"{t['fn']} {t['own']:.1f}" for t in row["own_us"][:12]), flush=True)
            run.end_pass(schedule=schedules[0])
            run.hashing = True
    out = {"scheme": a.scheme, "device": torch.cuda.get_device_name(), "cubin_sha256": kernel.cubin_sha256,
           "kernel_source_sha256": kernel.source_sha256, "warm": WARM, "calls": HOST_CALLS, "rows": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if rows else 1


def grouped_mode(a) -> int:
    import glm53_timing as T
    import grouped_check as GC
    import torch
    from verity_vllm.protocols.pouw.pearl_c import grouped as PG

    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else list(GC.SHAPES)
    schedules = a.schedule or list(GC.SCHEDULES)
    rows, cubin = [], None
    for n, k in shapes:
        run, weights, kernel, gen = T.device_run([(n, k)] * GC.EXPERTS, a.scheme, a.cubin, graphs=a.graphs)
        cubin, ws = kernel.cubin_sha256, tuple(sorted(weights))
        dense_stock = [T.stock_arms(weights[w]) for w in ws]
        for pat in a.pattern:
            counts = GC.COUNTS[pat]
            x = torch.randn(sum(counts), k, device="cuda", generator=gen).to(torch.bfloat16)
            stock, errors = GC.stock_arms([weights[w] for w in ws], dense_stock, counts, x)

            def call(run=run, ws=ws, x=x, counts=counts):
                return PG.gemm_grouped(run, ws, x, counts)
            arms = [(sc, True, gr, call) for gr in ((False, True) if a.graphs else (False,)) for sc in schedules]
            arms += [(schedules[0], False, False, call), *((nm, None, False, g) for nm, g in stock.items())]
            for name, hashing, graphed, f in arms:
                if hashing is not None:
                    run.end_pass(schedule=name)
                    run.hashing, run.graphs = hashing, graphed
                label = f"grouped {n}:{k} {pat} {name}{' graph' if graphed else ''}{'' if hashing is not False else ' nohash'}"
                for _ in range(GC.WARM_GRAPHS if graphed else WARM):
                    f()
                if a.nvtx:
                    torch.cuda.synchronize()
                    for i in range(a.calls or CALLS):
                        torch.cuda.nvtx.range_push(f"call {label} #{i}")
                        f()
                        torch.cuda.nvtx.range_pop()
                    run.join_sides(torch.cuda.current_stream())
                    torch.cuda.synchronize()
                    continue
                host_us, gpu_us = _calls(run, f, HOST_CALLS)
                prof_host_us, own = _profiled(run, f, HOST_CALLS)
                rows.append({"n": n, "k": k, "pattern": pat, "counts": list(counts), "arm": label.split(" ", 3)[3],
                             "host_us": host_us, "events_us": gpu_us, "profiled_host_us": prof_host_us, "own_us": own})
                print(f"{label}: host {host_us:.1f} us a call, events {gpu_us:.1f} (profiled host {prof_host_us:.1f}); "
                      "own: " + ", ".join(f"{t['fn']} {t['own']:.1f}" for t in own[:12]), flush=True)
            run.end_pass(schedule="serial")
            run.hashing = True
            if errors:
                print(f"grouped {n}:{k} {pat}: stock errors {errors}", flush=True)
        run.finish()
    out = {"scheme": a.scheme, "device": torch.cuda.get_device_name(), "cubin_sha256": cubin, "experts": GC.EXPERTS,
           "warm": WARM, "calls": HOST_CALLS, "nvtx": a.nvtx, "rows": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if rows or a.nvtx else 1


def _columns(db, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _tables(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def load(path: str) -> tuple[list, list]:
    """The export's ranges, (thread, start, end, text) sorted by start, and its GPU activities, (thread, launch time, start,
    end, kind, name, stream), each joined to its launch's API call by correlation id (a graph's nodes to its launch, under
    nsys's ``--cuda-graph-trace=node``)."""
    db = sqlite3.connect(path)
    tables = _tables(db)
    strings = dict(db.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
    cols = _columns(db, "NVTX_EVENTS")
    text = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in cols else "text"
    ranges = sorted((t, s, e, x) for t, s, e, x in db.execute(
        f"SELECT globalTid, start, end, {text} FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start") if x)
    api = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for c, t, s in db.execute(f"SELECT correlationId, globalTid, start FROM {table}"):
                api[c] = (t, s)
    acts = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        name = "shortName" if "shortName" in _columns(db, "CUPTI_ACTIVITY_KIND_KERNEL") else "demangledName"
        for c, s, e, nm, st in db.execute(f"SELECT correlationId, start, end, {name}, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            if c in api:
                acts.append((*api[c], s, e, "kernel", strings.get(nm, str(nm)), st))
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table in tables:
            for c, s, e, st in db.execute(f"SELECT correlationId, start, end, streamId FROM {table}"):
                if c in api:
                    acts.append((*api[c], s, e, kind, kind, st))
    return ranges, acts


def report(ranges: list, acts: list, calls: int = CALLS) -> dict:
    """Each call's split (the module's docstring), by its label."""
    by_tid = defaultdict(list)
    for t, s, e, x in ranges:
        by_tid[t].append((s, e, x))
    starts = {t: [r[0] for r in rs] for t, rs in by_tid.items()}

    def around(t: int, at: int) -> list[tuple[int, int, str]]:
        """The ranges on thread t that hold time `at`, outermost first."""
        rs = by_tid.get(t, [])
        return [r for r in rs[:bisect.bisect_right(starts.get(t, []), at)] if r[1] >= at]

    calls_of = defaultdict(list)            # label -> [(start, end, {step: host ns}, [(step, kind, name, start, end)])]
    index = {}
    for t, rs in by_tid.items():
        for s, e, x in rs:
            if x.startswith(("call ", "solo ")):
                kind, label = x.split(" ", 1)
                label = label.rsplit(" #", 1)[0] if kind == "call" else label
                rec = {"start": s, "end": e, "host": defaultdict(int), "gpu": [], "top": 0}
                calls_of[(kind, label)].append(rec)
                index[(t, s)] = rec
    for t, rs in by_tid.items():
        for s, e, x in rs:
            if x.startswith(("step ", "host ")):
                held = around(t, s)
                outer = [r for r in held if r[2].startswith(("call ", "solo "))]
                if outer:
                    rec = index[(t, outer[-1][0])]
                    rec["host"][x] += e - s
                    if [r for r in held if r != (s, e, x)][-1] == outer[-1]:
                        rec["top"] += e - s
    for recs in calls_of.values():
        for rec in recs:
            rec["host"]["host other"] = rec["end"] - rec["start"] - rec["top"]
    for t, launched, s, e, kind, name, _ in acts:
        held = around(t, launched)
        outer = [r for r in held if r[2].startswith(("call ", "solo "))]
        if not outer:
            continue
        inner = [r for r in held if r[2].startswith(("step ", "host "))]
        index[(t, outer[-1][0])]["gpu"].append((inner[-1][2] if inner else "host other", kind, name, s, e))
    out = {}
    for (kind, label), recs in sorted(calls_of.items()):
        if kind != "call":
            continue
        steps = defaultdict(lambda: {"gpu_us": [], "host_us": [], "kernels": defaultdict(list)})
        for rec in recs:
            gpu = defaultdict(int)
            for step, _, name, s, e in rec["gpu"]:
                gpu[step] += e - s
                steps[step]["kernels"][name].append((e - s) / 1e3)
            for step in set(gpu) | set(rec["host"]):
                steps[step]["gpu_us"].append(gpu.get(step, 0) / 1e3)
                steps[step]["host_us"].append(rec["host"].get(step, 0) / 1e3)
        acts_all = [g for rec in recs for g in rec["gpu"]]
        span = (max(g[4] for g in acts_all) - min(g[3] for g in acts_all)) / 1e3 / len(recs) if acts_all else None
        solo = calls_of.get(("solo", label), [])
        solo_acts = [g for rec in solo for g in rec["gpu"]]
        names = sorted({n for st in steps.values() for n in st["kernels"]})
        out[label] = {
            "calls": len(recs),
            "host_wall_us": statistics.mean((r["end"] - r["start"]) / 1e3 for r in recs),
            "gpu_busy_us": sum((g[4] - g[3]) for g in acts_all) / 1e3 / len(recs),
            "gpu_span_us_per_call": span,
            "solo_gpu_span_us": (max(g[4] for g in solo_acts) - min(g[3] for g in solo_acts)) / 1e3 if solo_acts else None,
            "solo_host_wall_us": (solo[0]["end"] - solo[0]["start"]) / 1e3 if solo else None,
            "steps": {st: {"gpu_us": statistics.mean(v["gpu_us"]), "host_us": statistics.mean(v["host_us"]),
                           "kernels": {n: {"count": len(d) / len(recs), "mean_us": statistics.mean(d)} for n, d in v["kernels"].items()}}
                      for st, v in steps.items()},
            "hash_fast": [n for n in names if n in HASH_FAST],
            "hash_decode": [n for n in names if n.startswith(HASH_DECODE)],
            "fallbacks": [n for n in names if n in FALLBACKS],
        }
    ends = {}
    for t, rs in by_tid.items():
        for s, e, x in rs:
            if x.startswith("pass_end "):
                ends[x.split(" ", 1)[1]] = (e - s) / 1e3
    for label, us in ends.items():
        if label in out:
            out[label]["pass_end_host_us"] = us
    return out


def lanes(ranges: list, acts: list) -> dict:
    """Each label's back-to-back calls by stream (the module's docstring, ``lanes``)."""
    by_tid = defaultdict(list)
    for t, s, e, x in ranges:
        if x.startswith("call "):
            by_tid[t].append((s, e, x))
    starts = {t: [r[0] for r in rs] for t, rs in by_tid.items()}
    of = defaultdict(lambda: defaultdict(list))         # label -> call -> [(stream, name, start, end)]
    for t, launched, s, e, _, name, st in acts:
        i = bisect.bisect_right(starts.get(t, []), launched) - 1
        if i < 0 or by_tid[t][i][1] < launched:
            continue
        label, n = by_tid[t][i][2][len("call "):].rsplit(" #", 1)
        of[label][int(n)].append((st, name, s, e))
    out = {}
    for label, calls in of.items():
        ids = sorted(calls)
        role, ordered = {}, sorted((a for i in ids for a in calls[i]), key=lambda a: a[2])
        for prefix, lane in LANE_OF:
            got = list(dict.fromkeys(a[0] for a in ordered if a[1].startswith(prefix) and a[0] not in role))
            for n, st in enumerate(got):
                role[st] = lane if len(got) == 1 else f"{lane}{n}"
        lane_of = {st: role.get(st, f"stream {st}") for i in ids for st, _, _, _ in calls[i]}
        first = {i: min(a[2] for a in calls[i]) for i in ids}
        steady = ids[1:-1] or ids
        per = defaultdict(lambda: defaultdict(list))
        spans = {i: defaultdict(lambda: [None, None, 0]) for i in ids}    # call -> lane -> [first start, last end, busy]
        for i in ids:
            for st, _, s, e in calls[i]:
                v = spans[i][lane_of[st]]
                v[0] = s if v[0] is None else min(v[0], s)
                v[1] = e if v[1] is None else max(v[1], e)
                v[2] += e - s
        for i in steady:
            for ln, (s, e, busy) in spans[i].items():
                per[ln]["start_us"].append((s - first[i]) / 1e3)
                per[ln]["span_us"].append((e - s) / 1e3)
                per[ln]["busy_us"].append(busy / 1e3)
        for a, b in itertools.pairwise(ids):
            for ln in spans[b].keys() & spans[a].keys():
                per[ln]["idle_before_us"].append((spans[b][ln][0] - spans[a][ln][1]) / 1e3)
        kernels = defaultdict(list)
        for i in steady:
            for st, name, s, e in calls[i]:
                kernels[(lane_of[st], name)].append((e - s) / 1e3)
        mid = steady[len(steady) // 2]
        periods = [(first[b] - first[a]) / 1e3 for a, b in itertools.pairwise(ids)]
        out[label] = {
            "calls": len(ids), "period_us": statistics.median(periods) if periods else None, "periods_us": periods,
            "lanes": {ln: {k: statistics.median(v) for k, v in d.items()} for ln, d in per.items()},
            "kernels": {f"{ln} {nm}": {"count": len(v) / len(steady), "mean_us": statistics.mean(v)}
                        for (ln, nm), v in sorted(kernels.items())},
            "timeline": [{"at_us": (s - first[mid]) / 1e3, "lane": lane_of[st], "name": name, "us": (e - s) / 1e3}
                         for st, name, s, e in sorted(calls[mid], key=lambda a: a[2])],
        }
    return out


def print_lanes(out: dict) -> None:
    for label, r in out.items():
        print(f"\n{label}: {r['calls']} calls, period {r['period_us'] or 0:.1f} us a call (median)")
        for ln, v in sorted(r["lanes"].items()):
            print(f"  {ln:<14} " + ", ".join(f"{k} {x:.1f}" for k, x in v.items()))
        for nm, v in r["kernels"].items():
            print(f"    {nm:<40} {v['mean_us']:8.1f} us" + (f" x{v['count']:g}" if v["count"] != 1 else ""))
        print("  a steady call: " + "; ".join(f"{t['at_us']:.1f} {t['lane']} {t['name']} {t['us']:.1f}" for t in r["timeline"]))


def print_report(out: dict) -> None:
    for label, r in out.items():
        print(f"\n{label}: host wall {r['host_wall_us']:.1f} us a call, GPU busy {r['gpu_busy_us']:.1f} us, GPU span "
              f"{r['gpu_span_us_per_call'] or 0:.1f} us a call back to back, alone {r['solo_gpu_span_us'] or 0:.1f} us "
              f"(host {r['solo_host_wall_us'] or 0:.1f}); hash_fast {r['hash_fast']}, hash_decode {r['hash_decode']}, "
              f"fallbacks {r['fallbacks'] or 'none'}")
        for st, v in sorted(r["steps"].items(), key=lambda kv: -kv[1]["gpu_us"]):
            ks = ", ".join(f"{n} {d['mean_us']:.1f}" + (f" x{d['count']:g}" if d["count"] != 1 else "")
                           for n, d in sorted(v["kernels"].items(), key=lambda kv: -kv[1]["mean_us"] * kv[1]["count"]))
            print(f"  {st:<22} GPU {v['gpu_us']:8.1f} us  host {v['host_us']:7.1f} us  {ks}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("run", "report", "lanes", "stamps", "host", "grouped"))
    p.add_argument("sqlite", nargs="?", help="report, lanes: nsys's SQLite export")
    p.add_argument("--out", help="report, lanes, stamps, host, grouped: the JSON to write")
    p.add_argument("--calls", type=int, help=f"run, grouped --nvtx: back-to-back calls at each point (default {CALLS})")
    p.add_argument("--pattern", nargs="*", default=["decode"], help="grouped: grouped_check's routed-row patterns")
    p.add_argument("--nvtx", action="store_true", help="grouped: only the NVTX-ranged back-to-back calls (under nsys)")
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--shapes", nargs="*", help="N:K (default run: the two expert shapes; stamps: every in-domain linear)")
    p.add_argument("--m", nargs="*", type=int)
    p.add_argument("--schedule", nargs="*", help="stamps: the schedules (default serial and deferred-a); host: (default "
                                                  "serial, deferred and deferred-a); run: the first (default serial)")
    p.add_argument("--graphs", action="store_true", help="host, run: calls of up to 256 rows replayed as CUDA graphs; "
                                                         "grouped: each schedule's arm again with the run's graphs")
    a = p.parse_args(argv)
    if a.mode == "run":
        return run_mode(a)
    if a.mode == "stamps":
        return stamps_mode(a)
    if a.mode == "host":
        return host_mode(a)
    if a.mode == "grouped":
        return grouped_mode(a)
    out = (lanes if a.mode == "lanes" else report)(*load(a.sqlite))
    (print_lanes if a.mode == "lanes" else print_report)(out)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
