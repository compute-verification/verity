"""Pool the erase-calibration runs into the percentile tables and the recommended secure-erasure deadline `T`.

Reads every driver JSON (`validate.json`, `sweep.json`, `host.json`, `soak-*.json`, `probe-*.json`) under each `--runs`
directory (run directories; any depth) and writes `calibration.json` and `calibration.md` to `--out`:

- HBM: each sweep point (size x generator x layout x order) pooled over GPUs: p50/p99/p99.9/max of the end-to-end
  fill+digest time (`wall`: fresh key in, digest value on the host) and of the device's fill and digest phases, with the
  bandwidths they imply;
- the soak: back-to-back max-size fill+digest under the node's co-tenant load, pooled and per GPU, back-to-back (`busy`)
  and after an idle gap (`after_idle`, the state an unannounced audit finds), its host-side share (`wall - gpu`), and each
  run past the p99.9 matched against the host-stall probe;
- the probe: the host's 1 ms sleep lateness, pooled;
- host DRAM: fill rates by generator and worker count (cold first touch, warm overwrite), the repeated-pass percentiles,
  the GPU's fill of pinned host memory over PCIe and its UVA digest of it.

The deadline rule: `T` is the soak's end-to-end p99.9 at the largest fill raised by `--margin`, and never under the
largest time observed times `1 + --max-guard`; it is stated as `T = M / BW` with `BW = M / T`, so it scales to a
device's whole memory `M` at the calibrated end-to-end rate. The host line states the measured rate and the cores it
used; the node's other socket is outside the research user's CPU set, so the whole-node figure is an extrapolation and
says so.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GIB = 1 << 30


def dist(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    q = np.percentile(a, [50, 99, 99.9])
    return {"n": int(a.size), "min": float(a.min()), "p50": float(q[0]), "p99": float(q[1]), "p999": float(q[2]),
            "max": float(a.max()), "mean": float(a.mean())}


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms / 1e3) / 1e9 if ms and ms > 0 else float("nan")


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_of(meta: dict) -> str:
    u = (meta.get("gpu_lease_uuid") or "").split(",")[0]
    return u[:12] if u else "?"


def load(runs: list[str]) -> dict:
    docs = defaultdict(list)
    for r in runs:
        for p in sorted(Path(r).rglob("*.json")):
            if p.name.startswith("."):
                continue
            try:
                d = json.loads(p.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(d, dict) and d.get("kind") in ("validate", "sweep", "host", "soak", "probe"):
                d["_path"] = str(p)
                d["_run"] = next((q.name for q in p.parents if q.name.startswith("r2")), "?")
                docs[d["kind"]].append(d)
    return docs


def sweep_table(sweeps: list[dict]) -> list[dict]:
    rows = defaultdict(lambda: {"wall": [], "fill": [], "digest": [], "bytes": set(), "gpus": set(), "runs": set(),
                                "repeat_ok": True, "distinct": True})
    for s in sweeps:
        n_max = s["hbm"]["n_max_blocks"]
        for p in s["points"]:
            label = "max" if p["n_blocks"] == n_max else f"{p['gib']:g} GiB"
            r = rows[(label, p["gen"], p["layout"], p["order"])]
            r["wall"] += p["runs"]["wall_ms"]
            r["fill"] += p["runs"]["fill_ms"]
            r["digest"] += p["runs"]["digest_ms"]
            r["bytes"].add(p["bytes"])
            r["gpus"].add(gpu_of(s["meta"]))
            r["runs"].add(s["_run"])
            r["repeat_ok"] &= p["repeat_digest_equal"]
            r["distinct"] &= p["distinct_digests"] == p["trials"]
    out = []
    size_key = lambda lab: math.inf if lab == "max" else float(lab.split()[0])  # noqa: E731
    for (label, gen, layout, order), r in sorted(rows.items(), key=lambda kv: (size_key(kv[0][0]), kv[0][1:])):
        nbytes = min(r["bytes"])
        w, f, d = dist(r["wall"]), dist(r["fill"]), dist(r["digest"])
        out.append({"size": label, "bytes": nbytes, "bytes_max": max(r["bytes"]), "gen": gen, "layout": layout,
                    "order": order, "gpus": sorted(r["gpus"]), "runs": len(r["wall"]), "wall_ms": w, "fill_ms": f,
                    "digest_ms": d, "fill_gbps_p50": gbps(nbytes, f["p50"]), "digest_gbps_p50": gbps(nbytes, d["p50"]),
                    "e2e_gbps_p999": gbps(nbytes, w["p999"]), "repeat_digest_equal": r["repeat_ok"],
                    "digests_distinct": r["distinct"]})
    return out


def probe_events(probes: list[dict]) -> np.ndarray:
    ev = [(p["t_start"] + t, ms) for p in probes for t, ms in p.get("events_over_2ms", [])]
    return np.asarray(sorted(ev), dtype=np.float64).reshape(-1, 2)


def host_windows(hosts: list[dict]) -> list[tuple[float, float, str]]:
    """When each host run loaded the host's memory (its sweep, repeated passes and GPU-to-pinned fill), as epoch spans."""
    out = []
    for h in hosts:
        t0, t1 = h["meta"].get("unix_time"), None
        for r in h.get("soak", {}).values():
            t1 = max(t1 or 0.0, r.get("t_end") or r["t_start"] + sum(r["seconds"]))
        if t0 and t1:
            d = h.get("d2h", {})
            t1 += sum(x["seconds"] for x in d.get("reps", [])) + sum(x["seconds"] for x in
                                                                   d.get("uva_digest", {}).get("reps", []))
            out.append((float(t0), float(t1), h["_run"]))
    return out


def soak_summary(soaks: list[dict], probes: list[dict], windows: list[tuple[float, float, str]] = ()) -> dict:
    cols = {k: [] for k in ("wall", "gpu", "fill", "digest", "idle", "t_epoch", "bytes", "order")}
    per_gpu = defaultdict(lambda: {"busy": [], "after_idle": [], "seconds": 0.0, "chunks": 0})
    seconds = 0.0
    for s in soaks:
        c = s["cols"]
        n = len(c["wall_ms"])
        t0 = s.get("t0_epoch", s["t_start"])
        cols["wall"] += c["wall_ms"]
        cols["gpu"] += c["gpu_ms"]
        cols["fill"] += c["fill_ms"]
        cols["digest"] += c["digest_ms"]
        if isinstance(s.get("idle_s"), (int, float)):  # a 0/1 flag beside one idle length
            cols["idle"] += [float(s["idle_s"]) if v else 0.0 for v in c["idle"]]
        else:
            cols["idle"] += [float(v) for v in c["idle"]]
        cols["order"] += [s["orders"][o] for o in c["order"]]
        cols["t_epoch"] += [t0 + t for t in c["t"]]
        cols["bytes"] += [s["bytes"]] * n
        dur = s.get("t_end", s["t_start"]) - s["t_start"]
        seconds += dur
        g = per_gpu[gpu_of(s["meta"])]
        g["busy"] += [w for w, i in zip(c["wall_ms"], c["idle"]) if not i]
        g["after_idle"] += [w for w, i in zip(c["wall_ms"], c["idle"]) if i]
        g["seconds"] += dur
        g["chunks"] += 1
    a = {k: np.asarray(v) for k, v in cols.items()}
    if not a["wall"].size:
        return {"runs": 0}
    idle = a["idle"].astype(bool)
    host = a["wall"] - a["gpu"]
    filling = np.zeros(a["wall"].size, dtype=bool)
    for t0, t1, _ in windows:
        filling |= (a["t_epoch"] >= t0) & (a["t_epoch"] <= t1)
    p999 = float(np.percentile(a["wall"], 99.9))
    ev = probe_events(probes)
    slow = np.flatnonzero(a["wall"] > p999)
    matched = []
    for i in slow:
        t0, t1 = a["t_epoch"][i], a["t_epoch"][i] + a["wall"][i] / 1e3
        hit = ev[(ev[:, 0] >= t0 - 0.002) & (ev[:, 0] <= t1 + 0.05)] if ev.size else ev
        matched.append({"utc": utc(t0), "wall_ms": float(a["wall"][i]), "gpu_ms": float(a["gpu"][i]),
                        "host_ms": float(host[i]), "idle": bool(idle[i]), "host_fill": bool(filling[i]),
                        "probe_stall_ms": float(hit[:, 1].max()) if hit.size else None})
    return {
        "runs": int(a["wall"].size), "hours": seconds / 3600, "chunks": len(soaks), "gpus": sorted(per_gpu),
        "bytes": int(a["bytes"].min()), "bytes_max": int(a["bytes"].max()),
        "t_first": utc(float(a["t_epoch"].min())), "t_last": utc(float(a["t_epoch"].max())),
        "span_hours": float(a["t_epoch"].max() - a["t_epoch"].min()) / 3600,
        "all": dist(a["wall"]), "busy": dist(a["wall"][~idle]), "after_idle": dist(a["wall"][idle]),
        "during_host_fill": dist(a["wall"][filling]), "outside_host_fill": dist(a["wall"][~filling]),
        "host_fill_windows": [{"run": r, "from": utc(t0), "to": utc(t1)} for t0, t1, r in windows],
        "gpu_ms": dist(a["gpu"]), "fill_ms": dist(a["fill"]), "digest_ms": dist(a["digest"]), "host_ms": dist(host),
        "by_order": {o: dist(a["wall"][a["order"] == o]) for o in sorted(set(cols["order"]))},
        "by_idle_s": {f"{x:g}": dist(a["wall"][a["idle"] == x]) for x in sorted(set(cols["idle"]))},
        "per_gpu": {g: {"chunks": v["chunks"], "hours": v["seconds"] / 3600, "busy": dist(v["busy"]),
                        "after_idle": dist(v["after_idle"])} for g, v in sorted(per_gpu.items())},
        "over_p999": {"count": len(matched), "with_probe_stall": sum(m["probe_stall_ms"] is not None for m in matched),
                      "host_share_over_1ms": sum(m["host_ms"] > 1.0 for m in matched),
                      "during_host_fill": sum(m["host_fill"] for m in matched),
                      "worst": sorted(matched, key=lambda m: -m["wall_ms"])[:10]},
        "_wall": a["wall"],
    }


def probe_summary(probes: list[dict]) -> dict:
    if not probes:
        return {"samples": 0}
    edges = probes[0]["edges_ms"]
    hist = np.sum([p["hist"] for p in probes], axis=0)
    n = int(hist.sum())
    hours = sum(p["t_end"] - p["t_start"] for p in probes) / 3600
    ev = probe_events(probes)
    over = {f">{e:g}ms": int(hist[i + 1:].sum()) for i, e in enumerate(edges) if e >= 1}
    return {"samples": n, "hours": hours, "files": len(probes), "edges_ms": edges, "hist": hist.tolist(),
            "max_ms": max(p["max_ms"] for p in probes), "over": over,
            "over_per_hour": {k: v / hours for k, v in over.items()} if hours else {},
            "worst": [{"utc": utc(t), "ms": ms} for t, ms in sorted(ev.tolist(), key=lambda e: -e[1])[:10]]}


def host_summary(hosts: list[dict]) -> dict:
    out = {"runs": [h["_run"] for h in hosts], "rates": [], "soak": [], "d2h": [], "uva_digest": [], "errors": []}
    rates = defaultdict(list)
    for h in hosts:
        for p in h.get("points", []):
            rates[(p["gen"], p["procs"], p["phase"], h["host_bytes"])].append(p["gbps"])
        out.setdefault("host_bytes", []).append(h.get("host_bytes"))
        if h["meta"].get("mem_total_bytes"):
            out["mem_total_bytes"] = h["meta"]["mem_total_bytes"]
        out["affinity"] = h["meta"].get("affinity")
        out["errors"] += h.get("errors", [])
        for spec, r in h.get("soak", {}).items():
            ms = [s * 1e3 for s in r["seconds"]]
            d = dist(ms)
            pr = [p for _, p in r.get("pressure", [])]
            out["soak"].append({"spec": spec, "run": h["_run"], "gen": r["gen"], "procs": r["procs"], "bytes": r["bytes"],
                                "pass_ms": d, "gbps_p50": gbps(r["bytes"], d.get("p50")),
                                "gbps_p999": gbps(r["bytes"], d.get("p999")), "gbps_min": gbps(r["bytes"], d.get("max")),
                                "load1_median": float(np.median([p["load1"] for p in pr])) if pr else None,
                                "cpu_psi_median": float(np.median([p["cpu"] for p in pr])) if pr else None})
        if "d2h" in h:
            d = h["d2h"]
            ms = [x["seconds"] * 1e3 for x in d["reps"]]
            out["d2h"].append({"bytes": d["bytes"], "pass_ms": dist(ms), "gbps_p50": gbps(d["bytes"], dist(ms).get("p50")),
                               "first_bytes_ok": d.get("first_bytes_ok")})
            u = d.get("uva_digest", {})
            if "reps" in u:
                ms = [x["seconds"] * 1e3 for x in u["reps"]]
                out["uva_digest"].append({"bytes": u["bytes"], "pass_ms": dist(ms),
                                          "gbps_p50": gbps(u["bytes"], dist(ms).get("p50")), "check_ok": u["check_ok"]})
            elif u:
                out["uva_digest"].append(u)
    for (gen, procs, phase, nbytes), v in sorted(rates.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3], kv[0][1])):
        out["rates"].append({"gen": gen, "procs": procs, "phase": phase, "bytes": nbytes, "n": len(v),
                             "gbps_median": float(np.median(v)),
                             "gbps_min": float(min(v)), "gbps_max": float(max(v))})
    return out


def recommend(soak: dict, sweep_rows: list[dict], gpu_total_bytes: int | None, margin: float, max_guard: float) -> dict:
    """The rule over the whole soak, or over its runs during a host-DRAM fill when that set binds: an erasure answer fills
    HBM and host DRAM at once, so the concurrent case is the one the deadline must cover."""
    if soak.get("runs"):
        sets = [("the whole soak (busy and after-idle)", soak["all"])]
        if soak.get("during_host_fill", {}).get("n", 0) >= 1000:
            sets.append(("the soak's runs during a concurrent host-DRAM fill", soak["during_host_fill"]))
        src, w = max(sets, key=lambda s: max(s[1]["p999"] * (1 + margin), s[1]["max"] * (1 + max_guard)))
        nbytes, n, walls = soak["bytes"], w["n"], soak["_wall"]
    else:
        rows = [r for r in sweep_rows if r["size"] == "max" and r["layout"] == "seq"]
        if not rows:
            return {}
        r = max(rows, key=lambda r: r["wall_ms"]["p999"])
        w, nbytes, src, n = r["wall_ms"], r["bytes"], f"sweep {r['gen']}/{r['layout']}/{r['order']}", r["runs"]
        walls = None
    t = max(w["p999"] * (1 + margin), w["max"] * (1 + max_guard))
    t = math.ceil(t)
    bw_t = nbytes / (t / 1e3) / 1e9
    rec = {"source": src, "runs": n, "bytes": nbytes, "p50_ms": w["p50"], "p99_ms": w["p99"], "p999_ms": w["p999"],
           "max_ms": w["max"], "T_ms": t, "margin_over_p999": t / w["p999"] - 1, "margin_over_max": t / w["max"] - 1,
           "bw_e2e_p999_gbps": gbps(nbytes, w["p999"]), "bw_T_gbps": bw_t,
           "runs_over_T": int((walls > t).sum()) if walls is not None else None}
    if gpu_total_bytes:
        rec["gpu_total_bytes"] = gpu_total_bytes
        rec["T_full_device_ms"] = gpu_total_bytes / (bw_t * 1e9) * 1e3
        rec["filled_fraction"] = nbytes / gpu_total_bytes
    return rec


def to_markdown(a: dict) -> str:
    L = []
    rec, soak, probe, host = a["recommendation"], a["soak"], a["probe"], a["host"]
    L.append("# Secure-erasure deadline calibration (volatile memory: HBM + host DRAM), node 2 RTX PRO 6000\n")
    L.append(f"**Recommended deadline:** {a['one_line']}\n")
    if a.get("host_line"):
        L.append(f"**Host DRAM:** {a['host_line']}\n")
    L.append(f"Runs: {a['counts']['fill_digest_runs']} fill+digest runs ({a['counts']['sweep_runs']} sweep, "
             f"{a['counts']['soak_runs']} soak) on {len(a['counts']['gpus'])} GPUs; validation "
             f"{a['counts']['validate_ok']}/{a['counts']['validate']} bit-exact.\n")
    L.append("## HBM fill+digest, soak under co-tenant load (max fill)\n")
    if soak.get("runs"):
        L.append(f"{soak['runs']} runs in {soak['hours']:.2f} GPU-hours of soak over {soak['span_hours']:.2f} h of wall "
                 f"time ({soak['t_first']} to {soak['t_last']}), {soak['bytes'] / GIB:.2f} GiB per fill, GPUs "
                 f"{', '.join(soak['gpus'])}. `during_host_fill` is the runs while this lane's host-DRAM runs filled "
                 "host memory on the same cores (" + ", ".join(f"{w['run']} {w['from'][11:]}-{w['to'][11:]}"
                                                                for w in soak["host_fill_windows"]) + ").\n")
        L.append("| set | n | p50 ms | p99 ms | p99.9 ms | max ms |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k in ("all", "busy", "after_idle", "during_host_fill", "outside_host_fill", "gpu_ms", "fill_ms",
                  "digest_ms", "host_ms"):
            d = soak[k]
            if d.get("n"):
                L.append(f"| {k} | {d['n']} | {d['p50']:.2f} | {d['p99']:.2f} | {d['p999']:.2f} | {d['max']:.2f} |")
        for g, v in soak["per_gpu"].items():
            for k in ("busy", "after_idle"):
                d = v[k]
                if d.get("n"):
                    L.append(f"| {g} {k} | {d['n']} | {d['p50']:.2f} | {d['p99']:.2f} | {d['p999']:.2f} | "
                             f"{d['max']:.2f} |")
        L.append("\n| idle before the run (s) | n | p50 ms | p99 ms | p99.9 ms | max ms |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        for x, d in soak["by_idle_s"].items():
            L.append(f"| {x} | {d['n']} | {d['p50']:.2f} | {d['p99']:.2f} | {d['p999']:.2f} | {d['max']:.2f} |")
        o = soak["over_p999"]
        L.append(f"\nRuns over the p99.9: {o['count']}; with a probe stall > 2 ms inside them: {o['with_probe_stall']}; "
                 f"with a host share over 1 ms: {o['host_share_over_1ms']}; during a host-DRAM fill: "
                 f"{o['during_host_fill']}.\n")
    L.append("## HBM sweep (pooled over GPUs)\n")
    L.append("| size | gen | layout | order | n | wall p50 | p99 | p99.9 | max | fill p50 (GB/s) | digest p50 (GB/s) |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in a["sweep"]:
        w = r["wall_ms"]
        L.append(f"| {r['size']} | {r['gen']} | {r['layout']} | {r['order']} | {r['runs']} | {w['p50']:.2f} | "
                 f"{w['p99']:.2f} | {w['p999']:.2f} | {w['max']:.2f} | {r['fill_ms']['p50']:.2f} "
                 f"({r['fill_gbps_p50']:.0f}) | {r['digest_ms']['p50']:.2f} ({r['digest_gbps_p50']:.0f}) |")
    L.append("\n## Host stalls (1 ms sleep probe, its own process)\n")
    if probe.get("samples"):
        L.append(f"{probe['samples']} samples over {probe['hours']:.2f} h; max lateness {probe['max_ms']:.2f} ms; "
                 + ", ".join(f"{k}: {v} ({probe['over_per_hour'].get(k, 0):.1f}/h)" for k, v in probe["over"].items())
                 + ".\n")
    L.append("## Host DRAM\n")
    if host.get("rates"):
        L.append(f"Cores {_cpus(host.get('affinity'))}.\n")
        L.append("| gen | phase | buffer | procs | n | GB/s median | min | max |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in host["rates"]:
            L.append(f"| {r['gen']} | {r['phase']} | {r['bytes'] / GIB:.0f} GiB | {r['procs']} | {r['n']} | "
                     f"{r['gbps_median']:.1f} | "
                     f"{r['gbps_min']:.1f} | {r['gbps_max']:.1f} |")
    if host.get("soak"):
        L.append("\n| repeated passes (gen:procs:reps) | run | node load1 / CPU PSI % (median) | bytes | n | p50 ms | "
                 "p99 ms | p99.9 ms | max ms | GB/s p50 | GB/s at p99.9 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in host["soak"]:
            d = r["pass_ms"]
            if d.get("n"):
                load = (f"{r['load1_median']:.0f} / {r['cpu_psi_median']:.0f}" if r.get("load1_median") is not None
                        else "")
                L.append(f"| {r['spec']} | {r['run']} | {load} | {r['bytes'] / GIB:.0f} GiB | {d['n']} | "
                         f"{d['p50']:.1f} | {d['p99']:.1f} | {d['p999']:.1f} | {d['max']:.1f} | {r['gbps_p50']:.1f} | "
                         f"{r['gbps_p999']:.1f} |")
        for name in ("d2h", "uva_digest"):
            for r in host.get(name, []):
                d = r.get("pass_ms", {})
                if d.get("n"):
                    L.append(f"| {name} (1 GPU, PCIe) | | | {r['bytes'] / GIB:.0f} GiB | {d['n']} | {d['p50']:.1f} | "
                             f"{d['p99']:.1f} | {d['p999']:.1f} | {d['max']:.1f} | {r['gbps_p50']:.1f} | "
                             f"{gbps(r['bytes'], d['p999']):.1f} |")
    if host.get("errors"):
        L.append(f"\nHost errors: {host['errors']}")
    L.append("\n## Rule\n")
    L.append(f"T = max(p99.9 x (1 + {a['params']['margin']}), max x (1 + {a['params']['max_guard']})), rounded up "
             "to 1 ms, over the soak's end-to-end times (fresh key in, digest value on the host); BW = M / T.")
    return "\n".join(L) + "\n"


def _cpus(aff) -> str:
    if not aff:
        return "?"
    return f"{min(aff)}-{max(aff)} ({len(aff)})"


def host_deadline(r: dict, m: int, margin: float, max_guard: float) -> float:
    """The same rule as the HBM deadline, over one spec's repeated passes, scaled from the pass buffer to `m` bytes (s)."""
    d = r["pass_ms"]
    return math.ceil(m / r["bytes"] * max(d["p999"] * (1 + margin), d["max"] * (1 + max_guard)) / 100) / 10


def host_line(host: dict, margin: float, max_guard: float) -> str | None:
    """T_host from the keyed fill whose p99.9 rate is best (a memset is not keyed and only shows the write ceiling), and
    the same rule on the slowest keyed spec (on this node, the one that ran under the heaviest co-tenant load)."""
    keyed = [r for r in host.get("soak", []) if r["pass_ms"].get("n", 0) >= 100 and r["gen"] != "memset"]
    if not keyed or not host.get("mem_total_bytes"):
        return None
    m = host["mem_total_bytes"]
    best, worst = max(keyed, key=lambda r: r["gbps_p999"]), min(keyed, key=lambda r: r["gbps_p999"])

    def say(r):
        t, d, k = host_deadline(r, m, margin, max_guard), r["pass_ms"], m / r["bytes"] / 1e3
        return (f"{m / 1e9:.0f} GB ({m / GIB:.0f} GiB) / {m / 1e9 / t:.1f} GB/s = {t:.1f} s ({r['gen']} on "
                f"{r['procs']} processes, {d['n']} passes of {r['bytes'] / GIB:.0f} GiB"
                + (f" at node load1 {r['load1_median']:.0f}" if r.get("load1_median") is not None else "") + ": "
                f"p50 {r['gbps_p50']:.1f} GB/s; p99.9 {r['gbps_p999']:.1f} GB/s, +{t / (k * d['p999']) - 1:.0%}; "
                f"slowest pass {gbps(r['bytes'], d['max']):.1f} GB/s, +{t / (k * d['max']) - 1:.0%})")

    line = f"T_host = M_host / BW = {say(best)}"
    if worst is not best:
        line += f"; the slowest keyed fill measured gives {say(worst)}"
    return line + (f". Measured on cores {_cpus(host.get('affinity'))} of NUMA node 0 only: a prover using every core "
                   "of both sockets fills faster, so these bound the honest T_host from above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--max-guard", type=float, default=0.02)
    ap.add_argument("--host-mem-bytes", type=int, default=0, help="the node's MemTotal where host.json lacks it")
    args = ap.parse_args()
    docs = load(args.runs)
    sweep_rows = sweep_table(docs["sweep"])
    soak = soak_summary(docs["soak"], docs["probe"], host_windows(docs["host"]))
    totals = [s["meta"]["gpu"]["total_mem_bytes"] for s in docs["sweep"] + docs["soak"] if "gpu" in s["meta"]]
    rec = recommend(soak, sweep_rows, max(totals) if totals else None, args.margin, args.max_guard)
    host = host_summary(docs["host"])
    if not host.get("mem_total_bytes") and args.host_mem_bytes:
        host["mem_total_bytes"] = args.host_mem_bytes
    gpus = sorted({gpu_of(d["meta"]) for d in docs["sweep"] + docs["soak"]})
    counts = {"validate": len(docs["validate"]), "validate_ok": sum(bool(v.get("all_ok")) for v in docs["validate"]),
              "sweep_runs": sum(r["runs"] for r in sweep_rows), "soak_runs": soak.get("runs", 0), "gpus": gpus,
              "runs": sorted({d["_run"] for k in docs for d in docs[k]})}
    counts["fill_digest_runs"] = counts["sweep_runs"] + counts["soak_runs"]
    one = "no HBM fill+digest runs"
    if rec:
        one = (f"T_HBM = M / BW = {rec['bytes'] / 1e9:.2f} GB ({rec['bytes'] / GIB:.2f} GiB) / {rec['bw_T_gbps']:.0f} GB/s "
               f"= {rec['T_ms']} ms per GPU (fill+digest end to end, from {rec['source']}: p99.9 {rec['p999_ms']:.2f} ms "
               f"over {rec['runs']} runs, +{rec['margin_over_p999']:.0%}; max {rec['max_ms']:.2f} ms, "
               f"+{rec['margin_over_max']:.0%}"
               + (f"; {rec['runs_over_T']} of {soak['runs']} soak runs over T" if rec.get("runs_over_T") is not None
                  else "")
               + (f"; {rec['T_full_device_ms']:.0f} ms for the whole {rec['gpu_total_bytes'] / 1e9:.1f} GB device at "
                  "that rate" if rec.get("T_full_device_ms") else "") + ")")
    soak.pop("_wall", None)
    a = {"schema": "verity/erase-calib/v0", "params": {"margin": args.margin, "max_guard": args.max_guard},
         "one_line": one, "host_line": host_line(host, args.margin, args.max_guard), "recommendation": rec, "counts": counts, "soak": soak,
         "probe": probe_summary(docs["probe"]), "sweep": sweep_rows, "host": host}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(a, indent=1, default=float))
    (out / "calibration.md").write_text(to_markdown(a))
    print(one)
    if a["host_line"]:
        print(a["host_line"])


if __name__ == "__main__":
    main()
