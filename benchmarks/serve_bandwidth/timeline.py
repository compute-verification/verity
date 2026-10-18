"""Where a served mix's DRAM reads go, from nsys's sqlite export of a `mix.py` run (GPU metrics, CUDA kernels, NVTX):
each GPU-metrics sample's DRAM read and write bytes, placed in a phase by the step whose span its midpoint falls in (a
step that schedules any prompt tokens is `prefill`, one that schedules only decodes `decode`; `idle` is the engine
waiting for an arrival; `gap` the rest of the trace, the GPU between steps) and split over kernel classes
(`kv_filler/step_profile.py`'s: `gemm` reads the weights, `attention` the KV) by how much of the sample's interval each
class's kernels cover on that GPU (`none`: no kernel running).

A step's span is, with `mix.py`'s `x` ranges (`--tp 1`), from the first to the last kernel on the GPU whose launch
(its CUDA runtime call, by correlation id) falls in the step's range (`method: launch`); without them, the scheduler's
range itself (`method: scheduler`, sound only with async scheduling off).

Per phase it writes the time, the read rate against `--R` (the card's measured read rate; nsys's percentages are of the
DRAM's peak, `TARGET_INFO_GPU`'s), the bandwidth left idle, and the modeled registered bytes of its steps (`mix.json`'s
weights a step and KV a context token) against the measured reads. Writes `timeline.json` and `timeline.csv` (10 ms
bins) under `--out`.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kv_filler"))

from step_profile import classify  # noqa: E402

CLASSES = ("gemm", "attention", "other")
PHASES = ("prefill", "decode", "gap", "idle")
STEP = re.compile(r"^s(\d+) p(\d+) d(\d+) kv(\d+)$")
XSTEP = re.compile(r"^x(\d+) p(\d+) d(\d+) kv(\d+)$")


def tables(db) -> set[str]:
    return {r[0] for r in db.execute("select name from sqlite_master where type in ('table', 'view')")}


def columns(db, table: str) -> list[str]:
    return [r[1] for r in db.execute(f"pragma table_info({table})")]


def union(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint sorted intervals covering the given ones, and the covered length before each."""
    if not len(starts):
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    reach = np.maximum.accumulate(e)
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] > reach[:-1]
    idx = np.flatnonzero(new)
    us = s[idx]
    ue = reach[np.append(idx[1:] - 1, len(s) - 1)]
    before = np.concatenate(([0], np.cumsum(ue - us)[:-1]))
    return us, ue, before


def covered(u: tuple, t: np.ndarray) -> np.ndarray:
    """Covered length in (-inf, t] by the union `u`."""
    us, ue, before = u
    if not len(us):
        return np.zeros(len(t))
    i = np.searchsorted(us, t, side="right") - 1
    ok = i >= 0
    ic = np.clip(i, 0, None)
    return np.where(ok, before[ic] + np.clip(np.minimum(t, ue[ic]) - us[ic], 0, None), 0).astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--mix", type=Path, required=True, help="mix.py's mix.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--R", type=float, default=1539.0, help="GB/s, the card's measured read rate")
    ap.add_argument("--bin-ms", type=float, default=10.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mix = json.loads(a.mix.read_text())
    db = sqlite3.connect(a.sqlite)
    have = tables(db)
    schema = {t: columns(db, t) for t in ("TARGET_INFO_GPU", "TARGET_INFO_GPU_METRICS", "GPU_METRICS", "NVTX_EVENTS",
                                          "CUPTI_ACTIVITY_KIND_KERNEL") if t in have}
    res: dict = {"format": "verity-serve-bandwidth/timeline/v0", "R_GBps": a.R, "schema": schema,
                 "mix": {k: v for k, v in mix.items() if k != "steps"}}

    gpus = [dict(zip(schema["TARGET_INFO_GPU"], r)) for r in db.execute("select * from TARGET_INFO_GPU")]
    res["gpus"] = gpus
    metrics = {(r[0], r[1]): r[2] for r in db.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS")}
    res["metrics"] = sorted({v for v in metrics.values()})
    type_names = {r[0]: r[1] for r in db.execute("select distinct typeId, typeName from TARGET_INFO_GPU_METRICS")}
    res["metric_types"] = {str(k): v for k, v in type_names.items()}

    def metric_id(type_id: int, rx: str):
        hits = [m for (t, m), n in metrics.items() if t == type_id and re.search(rx, n, re.IGNORECASE)]
        return hits[0] if hits else None

    # NVTX: the trace, steps and idle waits (start/end and push/pop ranges)
    rows = db.execute("select start, end, coalesce(text, (select value from StringIds where id = textId)) "
                      "from NVTX_EVENTS").fetchall()
    trace = [(s, e) for s, e, t in rows if t == "trace"]
    if not trace:
        print("no `trace` range in the export", file=sys.stderr)
        return 3
    T0, T1 = trace[0]
    if T1 is None:
        # the capture stopped on the range's pop before nsys recorded its end
        T1 = db.execute("select max(timestamp) from GPU_METRICS").fetchone()[0]
        print("the `trace` range has no end: the trace runs to the last GPU-metrics sample", file=sys.stderr)
    res["trace_end_recorded"] = trace[0][1] is not None
    rows = [r for r in rows if r[1] is not None]

    def ranges_of(rx) -> list[tuple]:
        return sorted((s, e, int(m[2]), int(m[3]), int(m[4])) for s, e, t in rows
                      if (m := rx.match(t or "")) and s >= T0 and e <= T1)

    xsteps = ranges_of(XSTEP)
    traced = "CUPTI_ACTIVITY_KIND_KERNEL" in have
    method = "launch" if xsteps and traced and "CUPTI_ACTIVITY_KIND_RUNTIME" in have else "scheduler"
    steps = xsteps if method == "launch" else ranges_of(STEP)
    res["method"] = method
    idles = sorted((s, e) for s, e, t in rows if t == "idle" and s >= T0)
    idle_s = np.array([s for s, _ in idles], dtype=np.int64)
    idle_e = np.array([e for _, e in idles], dtype=np.int64)

    # kernels by device and class, and (launch) by the step their launch falls in; an export without kernels (a
    # process nsys didn't trace) gets phases only
    names: dict = {}
    kn = np.zeros((0, 5), dtype=np.int64)
    if traced:
        names = {i: classify(v) for i, v in db.execute(
            "select id, value from StringIds where id in (select distinct demangledName from CUPTI_ACTIVITY_KIND_KERNEL)")}
        if method == "launch":
            q = ("select k.deviceId, k.start, k.end, k.demangledName, coalesce(r.start, -1) "
                 "from CUPTI_ACTIVITY_KIND_KERNEL k left join CUPTI_ACTIVITY_KIND_RUNTIME r "
                 "on r.correlationId = k.correlationId where k.end > ? and k.start < ?")
        else:
            q = ("select deviceId, start, end, demangledName, -1 from CUPTI_ACTIVITY_KIND_KERNEL "
                 "where end > ? and start < ?")
        kn = np.array(db.execute(q, (T0, T1)).fetchall(), dtype=np.int64).reshape(-1, 5)
    else:
        print("no kernels in the export: phases only, no class split", file=sys.stderr)
    kclass = np.array([CLASSES.index(names.get(int(n), "other")) for n in kn[:, 3]], dtype=np.int64)
    devices = sorted(int(d) for d in np.unique(kn[:, 0]))
    cov: dict = {}
    for d in devices:
        for ci, c in enumerate(CLASSES):
            m = (kn[:, 0] == d) & (kclass == ci)
            cov[d, c] = union(kn[m, 1], kn[m, 2])
    st_s = np.array([s[0] for s in steps], dtype=np.int64)
    st_e = np.array([s[1] for s in steps], dtype=np.int64)
    st_phase = np.array(["prefill" if s[2] else "decode" for s in steps], dtype=object)
    spans: dict = {}
    if method == "launch":
        # a step's span on a GPU: from its first kernel's start to its last's end
        j = np.searchsorted(st_s, kn[:, 4], side="right") - 1
        ok = (kn[:, 4] >= 0) & (j >= 0) & (kn[:, 4] < st_e[np.clip(j, 0, None)])
        res["kernels_outside_steps"] = int((~ok).sum())
        for d in devices:
            m = ok & (kn[:, 0] == d)
            jj = j[m]
            order = np.argsort(jj, kind="stable")
            jj, ks, ke = jj[order], kn[m, 1][order], kn[m, 2][order]
            uj, first = np.unique(jj, return_index=True)
            s0 = np.minimum.reduceat(ks, first) if len(ks) else ks
            e0 = np.maximum.reduceat(ke, first) if len(ke) else ke
            o = np.argsort(s0, kind="stable")
            uj, s0, e0 = uj[o], s0[o], e0[o]
            s0 = np.maximum(s0, np.concatenate(([s0[0]], np.maximum.accumulate(e0)[:-1]))) if len(s0) else s0
            spans[d] = (s0, e0, uj)
    else:
        spans[None] = (st_s, st_e, np.arange(len(steps)))
    res["kernels"] = {"traced": traced, "count": int(len(kn)), "devices": devices,
                      "names_by_class": {c: sum(1 for n in names.values() if n == c) for c in CLASSES},
                      "seconds_by_class": {c: round(float((kn[kclass == ci, 2] - kn[kclass == ci, 1]).sum()) / 1e9, 3)
                                           for ci, c in enumerate(CLASSES)}}

    peak = {}
    for g in gpus:
        bw = g.get("memoryBandwidth")
        if bw:
            peak[g.get("id")] = float(bw)
    types = sorted({t for t, _ in metrics})
    if devices and len(types) != len(devices) and len(devices) != 1:
        print(f"GPU-metrics types {types} against kernel devices {devices}: mapping by order", file=sys.stderr)
    per_dev = []
    bins: dict[int, dict] = {}
    for n, ty in enumerate(types):
        dev = devices[n] if n < len(devices) else None
        rid, wid = metric_id(ty, r"DRAM Read"), metric_id(ty, r"DRAM Write")
        if rid is None:
            continue
        smid, tid = metric_id(ty, r"^SMs? Active"), metric_id(ty, r"Tensor")
        q = db.execute("select timestamp, metricId, value from GPU_METRICS where typeId = ? and timestamp between ? and ? "
                       "and metricId in (?, ?, ?, ?) order by timestamp", (ty, T0, T1, rid, wid or -1, smid or -1,
                                                                            tid or -1)).fetchall()
        ts = np.array(sorted({r[0] for r in q}), dtype=np.int64)
        col = {m: np.full(len(ts), np.nan) for m in (rid, wid, smid, tid) if m is not None}
        pos = {t: i for i, t in enumerate(ts)}
        for t, m, v in q:
            col[m][pos[t]] = v
        gpu_id = ty & 0xFF
        pk = peak.get(gpu_id) or (next(iter(peak.values())) if peak else None)
        if pk is None:
            print("no memoryBandwidth in TARGET_INFO_GPU", file=sys.stderr)
            return 4
        t_prev = np.concatenate(([ts[0] - np.median(np.diff(ts))], ts[:-1]))
        dt = (ts - t_prev).astype(float)
        mid = (ts + t_prev) // 2
        rd = col[rid] / 100 * pk * dt / 1e9  # bytes in the sample
        wr = col[wid] / 100 * pk * dt / 1e9 if wid is not None else np.zeros(len(ts))
        sp_s, sp_e, sp_j = spans[dev if method == "launch" else None]
        i = np.searchsorted(sp_s, mid, side="right") - 1
        inr = (i >= 0) & (mid < sp_e[np.clip(i, 0, None)])
        phase = np.full(len(ts), "gap", dtype=object)
        step_of = np.full(len(ts), -1)
        step_of[inr] = sp_j[i[inr]]
        phase[inr] = st_phase[step_of[inr]]
        ii = np.searchsorted(idle_s, mid, side="right") - 1
        inidle = (ii >= 0) & (mid < idle_e[np.clip(ii, 0, None)]) if len(idle_s) else np.zeros(len(ts), dtype=bool)
        phase[~inr & inidle] = "idle"
        span_s = dict(zip(sp_j.tolist(), ((sp_e - sp_s) / 1e9).tolist()))
        frac = {}
        if dev is not None:
            for c in CLASSES:
                frac[c] = (covered(cov[dev, c], ts) - covered(cov[dev, c], t_prev)) / dt
            tot = sum(frac.values())
            scale = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
            for c in CLASSES:
                frac[c] = frac[c] * scale
            frac["none"] = 1 - sum(frac[c] for c in CLASSES)
        R = a.R * 1e9
        out = {"type_id": ty, "type_name": type_names.get(ty), "kernel_device": dev, "peak_GBps": pk / 1e9,
               "sample_us": float(np.median(dt)) / 1e3, "samples": len(ts), "phases": {}}
        for ph in PHASES:
            m = phase == ph
            secs = float(dt[m].sum()) / 1e9
            if not secs:
                continue
            rb, wb = float(rd[m].sum()), float(wr[m].sum())
            util = (col[rid][m] / 100 * pk / R)
            e = {"seconds": round(secs, 4), "read_GB": round(rb / 1e9, 3), "write_GB": round(wb / 1e9, 3),
                 "read_of_R": round(rb / (R * secs), 4), "write_of_R": round(wb / (R * secs), 4),
                 "idle_read_of_R": round(max(0.0, 1 - rb / (R * secs)), 4),
                 "read_of_R_p10_p50_p90": [round(float(x), 3) for x in np.nanpercentile(util, [10, 50, 90])]}
            for name, mid_ in (("sm_active", smid), ("tensor_active", tid)):
                if mid_ is not None:
                    e[f"{name}_pct"] = round(float(np.nansum(col[mid_][m] * dt[m]) / dt[m].sum()), 2)
            if frac:
                e["by_class"] = {}
                for c in (*CLASSES, "none"):
                    cs = float((frac[c][m] * dt[m]).sum()) / 1e9
                    cb = float((frac[c][m] * rd[m]).sum())
                    e["by_class"][c] = {"seconds": round(cs, 4), "read_GB": round(cb / 1e9, 3),
                                        "read_share": round(cb / rb, 4) if rb else None,
                                        "read_of_R_while_on": round(cb / (R * cs), 4) if cs else None}
            if ph in ("prefill", "decode"):
                ids = sorted({int(s) for s in step_of[m] if s >= 0})
                wbytes = len(ids) * mix["weight_bytes_per_step"]
                kvbytes = sum(steps[j][4] for j in ids) * mix["kv_bytes_per_token"]
                e["steps"] = len(ids)
                e["modeled_weight_GB"] = round(wbytes / 1e9, 3)
                e["modeled_kv_GB"] = round(kvbytes / 1e9, 3)
                e["weight_share_modeled"] = round(wbytes / rb, 4) if rb else None
                e["kv_share_modeled"] = round(kvbytes / rb, 4) if rb else None
            out["phases"][ph] = e
        tot_s = sum(v["seconds"] for v in out["phases"].values())
        out["trace_seconds"] = round(tot_s, 3)
        out["read_of_R"] = round(float(rd.sum()) / (R * tot_s), 4)
        # per-step: read of R by decode batch and by prefill tokens
        sr: dict = {}
        for j in sorted({int(s) for s in step_of if s >= 0}):
            m = step_of == j
            s, e_, p, d, kvt = steps[j]
            key = (f"prefill {'<=512' if p <= 512 else '<=2048' if p <= 2048 else '>2048'} tokens" if p
                   else f"decode {'1-4' if d <= 4 else '5-16' if d <= 16 else '17-64' if d <= 64 else '>64'} reqs")
            b = sr.setdefault(key, {"steps": 0, "seconds": 0.0, "read": 0.0})
            b["steps"] += 1
            b["seconds"] += span_s.get(j, (e_ - s) / 1e9)
            b["read"] += float(rd[m].sum())
        out["by_step_kind"] = {k: {"steps": v["steps"], "seconds": round(v["seconds"], 3),
                                   "read_of_R": round(v["read"] / (R * v["seconds"]), 4) if v["seconds"] else None}
                               for k, v in sorted(sr.items())}
        per_dev.append(out)
        # 10 ms bins of the first GPU, for the plot
        if n == 0:
            bw_ = int(a.bin_ms * 1e6)
            k = ((mid - T0) // bw_).astype(int)
            for j in range(k.max() + 1):
                m = k == j
                if not m.any():
                    continue
                vals, cnt = np.unique(phase[m], return_counts=True)
                bins[j] = {"t_s": round(j * a.bin_ms / 1e3, 3),
                           "read_of_R": round(float(rd[m].sum()) / (R * dt[m].sum() / 1e9), 4),
                           "write_of_R": round(float(wr[m].sum()) / (R * dt[m].sum() / 1e9), 4),
                           "phase": str(vals[np.argmax(cnt)]),
                           **({"gemm": round(float((frac["gemm"][m] * dt[m]).sum() / dt[m].sum()), 3),
                               "attention": round(float((frac["attention"][m] * dt[m]).sum() / dt[m].sum()), 3)}
                              if frac else {})}
    res["devices"] = per_dev
    (a.out / "timeline.json").write_text(json.dumps(res, indent=1))
    if bins:
        keys = list(next(iter(bins.values())).keys())
        with (a.out / "timeline.csv").open("w") as f:
            f.write(",".join(keys) + "\n")
            for j in sorted(bins):
                f.write(",".join(str(bins[j].get(k, "")) for k in keys) + "\n")
    for d in per_dev:
        print(f"GPU type {d['type_id']} ({d['type_name']}), kernels on device {d['kernel_device']}: "
              f"{d['trace_seconds']} s at {d['read_of_R']} R")
        for ph, e in d["phases"].items():
            cls = " ".join(f"{c} {v['read_share']}" for c, v in e.get("by_class", {}).items())
            print(f"  {ph:8s} {e['seconds']:8.3f} s read {e['read_of_R']:.3f} R (p10/50/90 "
                  f"{e['read_of_R_p10_p50_p90']}), write {e['write_of_R']:.3f} R; shares: {cls}; "
                  f"modeled weight {e.get('weight_share_modeled')} kv {e.get('kv_share_modeled')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
