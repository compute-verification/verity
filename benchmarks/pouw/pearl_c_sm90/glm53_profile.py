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

  python benchmarks/pouw/pearl_c_sm90/glm53_profile.py stamps --out DIR/stamps.json [--shapes N:K ...] [--m M ...] [--schedule S ...]

``stamps`` (no nsys) reads where A's commitment spends a decode call, inside pd_commit and pd_tree: at each shape, m (default
`STAMP_M`) and schedule (default serial and deferred-a), after `WARM` calls, `CALLS` calls alone between synchronizations,
each on the stamp twins (`run.Pipeline.stamps`: each phase's end, hash_decode.cuh's PS_* and PT_*, as the global timer and
the SM's clock).  Per call: each phase's end since its CTA's start by the SM's clock (median and max over the rows' CTAs),
the CTAs' start and end on the global timer since the first one's start, and the tail's (A's levels, seed_A, the padding
rows' lines; pd_tree's entry when it runs apart).
"""

from __future__ import annotations

import argparse
import bisect
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
#: hash_decode.cuh's stamp slots: a pd_commit CTA's phases (PS_*), and the tail's (PT_*: level l + 1 at PT_LEVEL + l)
PS = ("start", "segs", "salt", "frame", "sync1", "fold", "sync2", "leaf", "seed")
PT = {0: "start", **{1 + level: f"level{level + 1}" for level in range(8)}, 9: "seed_a", 10: "fillers", 11: "entry"}
PT_ROW = 15


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
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

    shapes = [tuple(map(int, s.split(":"))) for s in a.shapes] if a.shapes else list(SHAPES)
    run, weights, kernel, gen = T.device_run(shapes, a.scheme, a.cubin)
    run.keys = _ranged("host keys", run.keys)
    run.uploads.put = _ranged("host upload", run.uploads.put)
    run._after = _ranged("host after", run._after)
    PPD.screen_into = _ranged("host screen", PPD.screen_into)
    end_pass = run.end_pass
    for w, W in weights.items():
        n, k = W.shape
        for m in a.m or MS:
            x = torch.randn(m, k, device=W.device, generator=gen).to(torch.bfloat16)
            for hashing in (True, False):
                end_pass()
                run.hashing = hashing
                for _ in range(WARM):
                    run.gemm(w, x, W, None)
                _instrument(run, PPD)
                torch.cuda.synchronize()
                label = f"{n}:{k} m{m} {'hash' if hashing else 'nohash'}"
                for i in range(CALLS):
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
                      "calls": CALLS}), flush=True)
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


def _columns(db, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _tables(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def load(path: str) -> tuple[list, list]:
    """The export's ranges, (thread, start, end, text) sorted by start, and its GPU activities, (thread, launch time, start,
    end, kind, name), each joined to its launch's API call by correlation id."""
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
        for c, s, e, nm in db.execute(f"SELECT correlationId, start, end, {name} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            if c in api:
                acts.append((*api[c], s, e, "kernel", strings.get(nm, str(nm))))
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table in tables:
            for c, s, e in db.execute(f"SELECT correlationId, start, end FROM {table}"):
                if c in api:
                    acts.append((*api[c], s, e, kind, kind))
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
    for t, launched, s, e, kind, name in acts:
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
    p.add_argument("mode", choices=("run", "report", "stamps"))
    p.add_argument("sqlite", nargs="?", help="report: nsys's SQLite export")
    p.add_argument("--out", help="report, stamps: the JSON to write")
    p.add_argument("--scheme", default=glm53.SCHEME)
    p.add_argument("--cubin")
    p.add_argument("--shapes", nargs="*", help="N:K (default run: the two expert shapes; stamps: every in-domain linear)")
    p.add_argument("--m", nargs="*", type=int)
    p.add_argument("--schedule", nargs="*", help="stamps: the schedules (default serial and deferred-a)")
    a = p.parse_args(argv)
    if a.mode == "run":
        return run_mode(a)
    if a.mode == "stamps":
        return stamps_mode(a)
    out = report(*load(a.sqlite))
    print_report(out)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
