"""The whole-step tensor utilization of one `served_util.sh` run: the mean of Nsight Systems' Tensor Active GPU
metric over the engine's steady state, the step wall time from the server's step trace, and where the rest of the step went.

    python benchmarks/one_stage/served_util.py --sqlite PROF.sqlite --trace ADVICE_TRACE.jsonl --T 8192 [--serve-dir D]
        [--timings MAIN.timings.json] [--arm A] [--N N] --out util.json

The steady state is the longest run of consecutive full steps (T tokens scheduled) inside the capture; a step spans the
previous step's end to its own (the trace's ``t``, taken once the engine has taken the step's output, so window opens and
closes, seals waited on and every other host gap between steps fall in the step they delay). ``queued`` is the same over
the steps that start once the last request has arrived. Tensor Active is the GPU's share of SM cycles with the tensor pipe
active, sampled every 1/f s, so its mean over the window is the tensor-busy share of the whole step, host gaps included.

Each sample is attributed to the kernels running at its time (CUPTI's kernel records on the same timeline): idle (no
kernel: split by whether a device-to-host or host-to-device copy is running) or the categories of the kernels running.
A label's loss is its share of the window times 100 less its mean Tensor Active, in points of Tensor Active; the losses sum
to 100 less the mean.
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

#: kernel categories, the first pattern a kernel's name matches (lower case)
CATEGORIES = (
    ("commit", r"sha512|sha_|hm96|merkle|frame|hash|leaf|commit|tree|digest"),
    ("pouw", r"(^|_)h1|pouw|noise|duo|pearl"),
    ("attention", r"flashattn|flash_fwd|flash_attn|fmha|attn|attention|mha|softmax"),
    ("tap", r"tap|collector|matreq|hidden|readback|snapshot"),
    ("gemm", r"gemm|matmul|cutlass|xmma|wgmma|imma|_mma|cublas|scaled_mm|gemv|linear"),
    ("quant", r"quant|int8|s8|scale|amax"),
    ("copy", r"copy|memcpy|cat_|concat|index|gather|scatter|fill|contiguous|transpose|arange|cache"),
    ("elementwise", r"elementwise|vectorized|reduce|norm|rms|silu|act|rotary|rope|add|mul|unrolled|triton|sum|max|where"),
)
D2H, H2D = 2, 1                                      # CUPTI_ActivityMemcpyKind


def category(name: str) -> str:
    low = name.lower()
    return next((c for c, pat in CATEGORIES if re.search(pat, low)), "other")


def _cols(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def _has(db: sqlite3.Connection, table: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def utc0_ns(db: sqlite3.Connection) -> int:
    """The UTC epoch time, in ns, of the report's timeline zero."""
    for table, col in (("TARGET_INFO_SESSION_START_TIME", "utcEpochNs"), ("TARGET_INFO_SESSION_START_TIME", "utcTime")):
        if _has(db, table) and col in _cols(db, table):
            return int(db.execute(f"SELECT {col} FROM {table}").fetchone()[0])
    raise SystemExit("the report has no session start time (TARGET_INFO_SESSION_START_TIME)")


def metrics(db: sqlite3.Connection) -> tuple[dict[str, int], dict[int, tuple[list[int], list[float]]]]:
    """{metric name: id} and {id: (timestamps, values)} of the GPU metrics, each sorted by time."""
    if not _has(db, "GPU_METRICS"):
        raise SystemExit("the report has no GPU_METRICS: was --gpu-metrics-devices given?")
    names = {str(n): int(i) for i, n in db.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")}
    series: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for ts, mid, v in db.execute("SELECT timestamp, metricId, value FROM GPU_METRICS ORDER BY timestamp"):
        s = series[int(mid)]
        s[0].append(int(ts))
        s[1].append(float(v))
    return names, dict(series)


#: short names that say nothing of the kernel (a template's entry point): such a kernel goes by its demangled name
GENERIC = re.compile(r"^(device_kernel|kernel|Kernel\d*|cutlass_kernel|generic_kernel)$")


def kernels(db: sqlite3.Connection) -> list[tuple[int, int, str]]:
    """(start, end, name) of every kernel, by start: its short name, or its demangled name's first 200 characters when the
    short name is generic."""
    if not _has(db, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    cols = _cols(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    short = "shortName" if "shortName" in cols else "demangledName"
    long = "demangledName" if "demangledName" in cols else short
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    out = []
    for a, b, s, d in db.execute(f"SELECT start, end, {short}, {long} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
        name = str(strings.get(s, s))
        out.append((int(a), int(b), str(strings.get(d, d))[:200] if GENERIC.match(name) else name))
    return out


def copies(db: sqlite3.Connection) -> list[tuple[int, int, int, int]]:
    """(start, end, copyKind, bytes) of every memcpy, by start."""
    if not _has(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    return [tuple(map(int, r)) for r in db.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")]


def steps(trace: Path) -> list[dict[str, Any]]:
    """The trace's step records (with ``t``) in step order."""
    recs = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    return sorted((r for r in recs if r.get("kind") == "step" and "t" in r), key=lambda r: r["step"])


def last_send(timings: Path | None) -> float | None:
    """The wall-clock second the load's last request was sent (its timings file: ``start_unix`` plus the latest ``send_s``)."""
    if timings is None or not timings.exists():
        return None
    doc = json.loads(timings.read_text())
    sends = [r["send_s"] for r in doc["requests"] if r.get("send_s") is not None]
    return doc["start_unix"] + max(sends) if sends else None


#: idle gaps between kernels by length, in µs: launch gaps, short host work, long host work, stalls
GAPS = ((0, 20), (20, 200), (200, 2_000), (2_000, 20_000), (20_000, 1 << 40))


def runtime_calls(db: sqlite3.Connection, lo: int, hi: int, top: int = 12) -> list[dict[str, Any]]:
    """The CUDA runtime and driver calls the host spent the most time in over [lo, hi]: name, calls, total ms."""
    got: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if not _has(db, table):
            continue
        q = (f"SELECT s.value, COUNT(*), SUM(MIN(r.end, ?) - MAX(r.start, ?)) FROM {table} r JOIN StringIds s ON s.id = r.nameId "
             "WHERE r.end > ? AND r.start < ? GROUP BY s.value")
        for name, n, ns in db.execute(q, (hi, lo, lo, hi)):
            got[str(name)][0] += int(n)
            got[str(name)][1] += float(ns or 0)
    return [{"name": n, "calls": int(c), "ms": round(ns / 1e6, 3)} for n, (c, ns) in sorted(got.items(), key=lambda kv: -kv[1][1])[:top]]


def steady(st: list[dict[str, Any]], T: int, lo_s: float, hi_s: float, after_s: float | None = None) -> list[int]:
    """Indices into ``st`` of the longest run of consecutive full steps whose span (the previous step's ``t`` to its own)
    lies in [lo_s, hi_s] and, with ``after_s``, starts at or after it."""
    best: list[int] = []
    run: list[int] = []
    for i in range(1, len(st)):
        a, b = st[i - 1]["t"], st[i]["t"]
        ok = (st[i]["T"] == T and st[i]["step"] == st[i - 1]["step"] + 1 and a >= lo_s and b <= hi_s
              and (after_s is None or a >= after_s))
        run = run + [i] if ok else []
        if len(run) > len(best):
            best = list(run)
    return best


def union(iv: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    """The union of the intervals ``iv``, clipped to [lo, hi]."""
    out: list[list[int]] = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def covered(u: list[tuple[int, int]], t: int) -> bool:
    i = bisect.bisect_right(u, (t, 1 << 62)) - 1
    return i >= 0 and u[i][0] <= t < u[i][1]


def mean_in(ts: list[int], vs: list[float], lo: int, hi: int) -> float | None:
    a, b = bisect.bisect_left(ts, lo), bisect.bisect_right(ts, hi)
    return statistics.fmean(vs[a:b]) if b > a else None


def analyse(sqlite: Path, trace: Path, T: int, sent_s: float | None = None) -> dict[str, Any]:
    """``sent_s``: when the load's last request was sent (`last_send`), from which the ``queued`` window starts."""
    db = sqlite3.connect(sqlite)
    z = utc0_ns(db)
    names, series = metrics(db)
    tensor = next((i for n, i in names.items() if n.lower().startswith("tensor active")), None) \
        or next((i for n, i in names.items() if "tensor" in n.lower()), None)
    if tensor is None:
        raise SystemExit(f"no Tensor Active metric among {sorted(names)}")
    ts, tv = series[tensor]
    period = statistics.median(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else 0
    ks, cs = kernels(db), copies(db)
    st = steps(trace)
    cap = ((ts[0] + z) / 1e9, (ts[-1] + z) / 1e9)
    at = lambda s: int(s * 1e9) - z                      # noqa: E731  (a wall-clock second on the report's timeline)
    picks = {"all": steady(st, T, *cap)}
    if sent_s is not None:
        picks["queued"] = steady(st, T, *cap, after_s=sent_s)
    out: dict[str, Any] = {"capture_s": round(cap[1] - cap[0], 3), "metric_period_us": period / 1e3, "metrics": sorted(names),
                           "tensor_metric": next(n for n, i in names.items() if i == tensor), "steps_traced": len(st),
                           "full_steps_traced": sum(1 for r in st if r["T"] == T), "kernels": len(ks), "memcpys": len(cs)}
    for key, idx in picks.items():
        if len(idx) < 2:
            out[key] = {"steps": len(idx), "note": "fewer than two full steps in the capture"}
            continue
        lo, hi = at(st[idx[0] - 1]["t"]), at(st[idx[-1]]["t"])
        walls = [st[i]["t"] - st[i - 1]["t"] for i in idx]
        per = [mean_in(ts, tv, at(st[i - 1]["t"]), at(st[i]["t"])) for i in idx]
        w = {"steps": len(idx), "first_step": st[idx[0]]["step"], "last_step": st[idx[-1]]["step"], "wall_s": round((hi - lo) / 1e9, 4),
             "step_wall_ms": {"mean": round(1e3 * statistics.fmean(walls), 3), "median": round(1e3 * statistics.median(walls), 3),
                              "min": round(1e3 * min(walls), 3), "max": round(1e3 * max(walls), 3)},
             "tensor_active_pct": round(mean_in(ts, tv, lo, hi) or 0.0, 3),
             "tensor_active_pct_per_step": {"median": round(statistics.median(p for p in per if p is not None), 3),
                                            "min": round(min(p for p in per if p is not None), 3),
                                            "max": round(max(p for p in per if p is not None), 3)},
             "tokens_per_s": round(T * len(idx) / ((hi - lo) / 1e9), 1)}
        for n, i in names.items():
            if i != tensor and i in series:
                m = mean_in(*series[i], lo, hi)
                if m is not None:
                    w.setdefault("other_metrics_mean", {})[n] = round(m, 3)
        kin = [(a, b, n) for a, b, n in ks if b > lo and a < hi]
        busy = union([(a, b) for a, b, _ in kin], lo, hi)
        busy_ns = sum(b - a for a, b in busy)
        d2h = union([(a, b) for a, b, k, _ in cs if k == D2H], lo, hi)
        h2d = union([(a, b) for a, b, k, _ in cs if k == H2D], lo, hi)
        w["gpu_busy_pct"] = round(100 * busy_ns / (hi - lo), 3)
        edges = [lo] + [x for iv in busy for x in iv] + [hi]
        gaps = [(edges[k + 1] - edges[k]) / 1e3 for k in range(0, len(edges), 2) if edges[k + 1] > edges[k]]
        w["idle_gaps"] = [{"us": f"{a}-{b}" if b < 1 << 40 else f">{a}", "gaps": sum(1 for g in gaps if a <= g < b),
                           "time_pct": round(100 * sum(g for g in gaps if a <= g < b) * 1e3 / (hi - lo), 3),
                           "per_step": round(sum(1 for g in gaps if a <= g < b) / len(idx), 2)} for a, b in GAPS]
        w["host_cuda_calls"] = runtime_calls(db, lo, hi)
        w["copies"] = {kind: {"GB": round(sum(n for a, b, k, n in cs if k == code and a >= lo and b <= hi) / 1e9, 3),
                              "busy_pct": round(100 * sum(b - a for a, b in u) / (hi - lo), 3)}
                       for kind, code, u in (("d2h", D2H, d2h), ("h2d", H2D, h2d))}
        # each sample's label: the categories of the kernels running at its time, or idle and the copy running
        a0, b0 = bisect.bisect_left(ts, lo), bisect.bisect_right(ts, hi)
        lab_n: dict[str, int] = defaultdict(int)
        lab_v: dict[str, float] = defaultdict(float)
        by_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])   # [ns, tensor·samples, samples]
        active: dict[int, str] = {}
        ends: list[tuple[int, int]] = []
        p = 0
        for j in range(a0, b0):
            t, v = ts[j], tv[j]
            while p < len(kin) and kin[p][0] <= t:
                if kin[p][1] > t:
                    heapq.heappush(ends, (kin[p][1], p))
                    active[p] = kin[p][2]
                p += 1
            while ends and ends[0][0] <= t:
                active.pop(heapq.heappop(ends)[1], None)
            run = set(active.values())
            if run:
                label = "+".join(sorted({category(n) for n in run}))
                for n in run:
                    by_kernel[n][1] += v
                    by_kernel[n][2] += 1
            else:
                label = "idle, D2H copy" if covered(d2h, t) else "idle, H2D copy" if covered(h2d, t) else "idle, no GPU work"
            lab_n[label] += 1
            lab_v[label] += v
        total = max(1, b0 - a0)
        w["losses"] = sorted(({"label": k, "share_pct": round(100 * n / total, 3), "tensor_pct": round(lab_v[k] / n, 3),
                               "loss_pts": round((100 * n - lab_v[k]) / total, 3)} for k, n in lab_n.items()),
                             key=lambda r: -r["loss_pts"])
        for a, b, n in kin:
            by_kernel[n][0] += min(b, hi) - max(a, lo)
        w["kernels_top"] = [{"name": n[:120], "category": category(n), "time_pct": round(100 * x[0] / (hi - lo), 3),
                             "tensor_pct": round(x[1] / x[2], 2) if x[2] else None}
                            for n, x in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:40]]
        cat_ns: dict[str, int] = defaultdict(int)
        for a, b, n in kin:
            cat_ns[category(n)] += min(b, hi) - max(a, lo)
        w["kernel_time_pct_by_category"] = {c: round(100 * v / (hi - lo), 3) for c, v in sorted(cat_ns.items(), key=lambda kv: -kv[1])}
        out[key] = w
    return out


def windows(serve_dir: Path) -> dict[str, Any]:
    """windows.jsonl's sizes and timings (no roots, requests or rows), and summary.json's counts."""
    got: dict[str, Any] = {}
    p = serve_dir / "windows.jsonl"
    if p.exists():
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        keep = ("window", "steps", "committed_bytes", "passed", "seconds", "close_phases_s", "on_loop_phases_s", "error")
        rows = [{k: r[k] for k in keep if k in r} | ({"pouw_h1_seconds": r["pouw_h1"].get("seconds"), "pouw_h1_tied": r["pouw_h1"].get("tied"),
                                                      "pouw_h1_error": (r["pouw_h1"].get("error") or None) and str(r["pouw_h1"]["error"])[:300]}
                                                     if isinstance(r.get("pouw_h1"), dict) else {})
                for r in recs]
        got["windows"] = rows
        sizes = [r["committed_bytes"] for r in rows if r.get("committed_bytes")]
        if sizes:
            got["committed_GB_per_window"] = {"mean": round(statistics.fmean(sizes) / 1e9, 3), "max": round(max(sizes) / 1e9, 3)}
        for k in ("open", "close", "on_loop", "seal_wait", "finalize_wait"):
            v = [float(r["seconds"][k]) for r in rows if isinstance(r.get("seconds"), dict) and k in r["seconds"]]
            if v:
                got.setdefault("window_seconds", {})[k] = {"mean": round(statistics.fmean(v), 4), "max": round(max(v), 4)}
    s = serve_dir / "summary.json"
    if s.exists():
        doc = json.loads(s.read_text())
        got["summary"] = {k: doc.get(k) for k in ("served", "engine_steps", "engine_error", "stopped_by", "pauses", "sealer")}
        if isinstance(doc.get("commit"), dict):
            got["summary"]["commit"] = {k: v for k, v in doc["commit"].items() if not isinstance(v, (list, dict)) or k in ("seals", "pouw_h1")}
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sqlite", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--serve-dir", type=Path, default=None)
    ap.add_argument("--timings", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    doc: dict[str, Any] = {"arm": a.arm, "T": a.T, "N": a.N, **analyse(a.sqlite, a.trace, a.T, last_send(a.timings))}
    if a.serve_dir is not None:
        doc["serve"] = windows(a.serve_dir)
    if a.timings is not None and a.timings.exists():
        reqs = json.loads(a.timings.read_text())["requests"]
        done = [r["done_s"] for r in reqs if r.get("done_s") is not None]
        sent = [r["send_s"] for r in reqs if r.get("send_s") is not None]
        doc["load"] = {"requests": len(reqs), "ok": sum(1 for r in reqs if r.get("ok")), "span_s": round(max(done), 3) if done else None,
                       "last_send_s": round(max(sent), 3) if sent else None}
    a.out.write_text(json.dumps(doc, indent=1))
    if rows := doc.get("serve", {}).get("windows"):
        print(f"WINDOWS {len(rows)} closed, {sum(1 for r in rows if r.get('passed'))} passed, committed GB a window "
              f"{doc['serve'].get('committed_GB_per_window')}, seconds {doc['serve'].get('window_seconds')}")
    if h1 := (doc.get("serve", {}).get("summary", {}).get("commit") or {}).get("pouw_h1"):
        why = next((r["pouw_h1_error"] for r in doc["serve"].get("windows", []) if r.get("pouw_h1_error")), None)
        print(f"H1 {h1.get('windows')} windows released, {h1.get('tied')} tied, untied {h1.get('untied')}, failed {h1.get('failed')}, "
              f"gate-passed rows {h1.get('gate_passed_rows')}" + (f"; first failure: {why}" if why else ""))
    for key in ("queued", "all"):
        w = doc.get(key)
        if not isinstance(w, dict) or "tensor_active_pct" not in w:
            continue
        print(f"UTIL {a.arm} T {a.T} [{key}]: tensor active {w['tensor_active_pct']:.2f}% over {w['steps']} full steps, "
              f"step wall {w['step_wall_ms']['mean']:.2f} ms (median {w['step_wall_ms']['median']:.2f}), GPU busy {w['gpu_busy_pct']:.1f}%, "
              f"D2H {w['copies']['d2h']['GB']:.1f} GB ({w['copies']['d2h']['busy_pct']:.1f}% of the time)")
        for r in w["losses"][:8]:
            print(f"  loss {r['loss_pts']:6.2f} pts  {r['label']:<40} {r['share_pct']:6.2f}% of the time at {r['tensor_pct']:.1f}% tensor")
        for r in w["kernels_top"][:12]:
            print(f"  kernel {r['time_pct']:6.2f}%  tensor {r['tensor_pct'] if r['tensor_pct'] is not None else '-':>6}  {r['category']:<11} {r['name'][:90]}")
        print("  idle gaps " + ", ".join(f"{g['us']} us: {g['per_step']}/step {g['time_pct']:.1f}%" for g in w["idle_gaps"]))
        print("  host CUDA calls " + ", ".join(f"{c['name']} {c['calls']}x {c['ms']:.0f} ms" for c in w["host_cuda_calls"][:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
