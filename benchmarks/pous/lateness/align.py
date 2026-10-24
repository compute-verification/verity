"""Align concurrent ping series in time and ask which machine's side the slow episodes follow.

    python align.py RUN_DIR --json align.json

Each record's wall time is rebuilt as the probe's start plus the cumulative (elapsed or timeout) + gap, rescaled to
the series' measured end. Per bin: median RTT and the share over 300 µs and 150 µs; then the correlation of the
paths' per-bin shares, and the share of slow bins (median over 150 µs) each pair has in common."""
import argparse
import json
import os
import re

import numpy as np

REC = np.dtype([("elapsed", "<u4"), ("server", "<u4"), ("device", "<u4"), ("op", "<u2"), ("flags", "<u2")])
LOST = 0xFFFFFFFF


def series(path):
    r = np.fromfile(path + ".bin", dtype=REC)
    m = json.load(open(path + ".json"))
    e = r["elapsed"].astype(np.float64)
    lost = r["elapsed"] == LOST
    dur = np.where(lost, m["timeout_ms"] * 1e6, e) + m["gap_us"] * 1e3
    t = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
    t = m["start"] + t * (m["end"] - m["start"]) / (t[-1] + dur[-1])
    return t, np.where(lost, np.inf, e / 1000), m


def binned(t, us, t0, t1, width):
    edges = np.arange(t0, t1 + width, width)
    idx = np.digitize(t, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        x = us[idx == b]
        if len(x) < 50:
            out.append(None)
            continue
        out.append({"t": float(edges[b] - t0), "n": int(len(x)), "p50": float(np.median(x)),
                    "over300": float((x > 300).mean()), "over150": float((x > 150).mean())})
    return out


def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else None


def machines(run, paths):
    """Per sampler interval and machine: where the NIC's interrupts went and how busy those CPUs were; then, for each
    path, those against the path's share over 300 µs in the same interval (Spearman), and slow seconds against fast."""
    out = {}
    for fn in sorted(os.listdir(run)):
        if not fn.endswith("-cpus.jsonl"):
            continue
        name = fn[: -len("-cpus.jsonl")]
        rows = [json.loads(x) for x in open(os.path.join(run, fn)) if x.strip()]
        if len(rows) < 3:
            out[name] = {"samples": len(rows)}
            continue
        ncpu = len(rows[0]["busy"])
        t = np.array([r["t"] for r in rows])
        busy = np.array([r["busy"] for r in rows], dtype=float)
        soft = np.array([r["softirq"] for r in rows], dtype=float)
        rx = np.array([r["net_rx"][:ncpu] for r in rows], dtype=float)
        kso = np.array([[r["ksoftirqd"].get(str(c), 0) for c in range(ncpu)] for r in rows], dtype=float)
        nic = np.zeros((len(rows), ncpu))
        nic_names = set()
        for i, r in enumerate(rows):
            for k, v in r["irqs"].items():
                if "mlx5_comp" in k or re.search(r"virtio\d+-(input|output)\.", k):
                    nic_names.add(k)
                    for c, n in v.items():
                        nic[i, int(c)] += n
        dt = np.diff(t)[:, None]
        d_busy = np.diff(busy, axis=0) / (100 * dt)       # USER_HZ = 100
        d_soft = np.diff(soft, axis=0) / (100 * dt)
        d_rx = np.diff(rx, axis=0) / dt
        d_kso = np.diff(kso, axis=0) / (100 * dt)
        d_nic = np.diff(nic, axis=0) / dt
        w = d_nic / np.maximum(d_nic.sum(axis=1, keepdims=True), 1)
        irq_busy = (w * d_busy).sum(axis=1)
        irq_soft = (w * d_soft).sum(axis=1)
        irq_kso = (w * d_kso).sum(axis=1)
        node_busy = d_busy.mean(axis=1)
        running = np.array([r["procs_running"] for r in rows[1:]], dtype=float)
        top = np.argsort(-d_nic.sum(axis=0))[:6]
        m = {"samples": len(rows), "nic_irqs": sorted(nic_names)[:16],
             "nic_irq_cpus": {int(c): round(float(d_nic[:, c].mean()), 1) for c in top if d_nic[:, c].sum()},
             "net_rx_per_s_top_cpus": {int(c): round(float(d_rx[:, c].mean()), 1) for c in np.argsort(-d_rx.sum(axis=0))[:6]},
             "irq_cpu_busy": {"p50": float(np.median(irq_busy)), "max": float(irq_busy.max())},
             "irq_cpu_softirq": {"p50": float(np.median(irq_soft)), "max": float(irq_soft.max())},
             "irq_cpu_ksoftirqd": {"p50": float(np.median(irq_kso)), "max": float(irq_kso.max())},
             "node_busy": {"p50": float(np.median(node_busy)), "min": float(node_busy.min()), "max": float(node_busy.max())},
             "procs_running": {"p50": float(np.median(running)), "max": float(running.max())},
             "paths": {}}
        for pname, (pt, us, _meta) in paths.items():
            idx = np.digitize(pt, t) - 1
            over, med, keep = [], [], []
            for i in range(len(t) - 1):
                x = us[idx == i]
                if len(x) >= 200:
                    keep.append(i)
                    over.append(float((x > 300).mean()))
                    med.append(float(np.median(x)))
            if len(keep) < 10:
                continue
            keep = np.array(keep)
            over, med = np.array(over), np.array(med)
            slow, fast = med > 150, med < 120
            feats = {"irq_cpu_busy": irq_busy, "irq_cpu_softirq": irq_soft, "irq_cpu_ksoftirqd": irq_kso,
                     "node_busy": node_busy, "procs_running": running}
            m["paths"][pname] = {
                "seconds": int(len(keep)), "slow_s": int(slow.sum()), "fast_s": int(fast.sum()),
                "spearman_over300": {k: rank_corr(over, v[keep]) for k, v in feats.items()},
                "slow_vs_fast_mean": {k: [float(v[keep][slow].mean()) if slow.any() else None,
                                          float(v[keep][fast].mean()) if fast.any() else None] for k, v in feats.items()}}
        out[name] = m
        print(f"{name}: NIC irq CPUs {m['nic_irq_cpus']} irq-CPU busy p50 {m['irq_cpu_busy']['p50']:.2f} "
              f"max {m['irq_cpu_busy']['max']:.2f}; ksoftirqd p50 {m['irq_cpu_ksoftirqd']['p50']:.3f}; node busy "
              f"{m['node_busy']['p50']:.2f} ({m['node_busy']['min']:.2f}-{m['node_busy']['max']:.2f}); running {m['procs_running']}")
        for pname, v in m["paths"].items():
            sc = " ".join(f"{k}={x:+.2f}" if x is not None else f"{k}=n/a" for k, x in v["spearman_over300"].items())
            sf = " ".join(f"{k}={a:.2f}/{b:.2f}" if a is not None and b is not None else f"{k}=n/a"
                          for k, (a, b) in v["slow_vs_fast_mean"].items())
            print(f"  {pname}: {v['seconds']} s ({v['slow_s']} slow, {v['fast_s']} fast); rho {sc}\n    slow/fast {sf}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", required=True)
    ns = ap.parse_args()
    paths = {}
    for d in sorted(os.listdir(ns.run)):
        p = os.path.join(ns.run, d, "ping-series-concurrent")
        if os.path.exists(p + ".bin"):
            paths[d] = series(p)
    t0 = max(v[2]["start"] for v in paths.values())
    t1 = min(v[2]["end"] for v in paths.values())
    report = {"window_s": t1 - t0, "bins": {}}
    for width in (10.0, 1.0):
        bins = {k: binned(t, us, t0, t1, width) for k, (t, us, _m) in paths.items()}
        keep = [i for i in range(len(next(iter(bins.values())))) if all(b[i] for b in bins.values())]
        names = list(bins)
        corr, common = {}, {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                xa = np.array([bins[a][j]["over300"] for j in keep])
                xb = np.array([bins[b][j]["over300"] for j in keep])
                ma = np.array([bins[a][j]["p50"] > 150 for j in keep])
                mb = np.array([bins[b][j]["p50"] > 150 for j in keep])
                corr[f"{a}~{b}"] = float(np.corrcoef(xa, xb)[0, 1])
                common[f"{a}~{b}"] = {"slow_a": int(ma.sum()), "slow_b": int(mb.sum()), "both": int((ma & mb).sum()),
                                      "either": int((ma | mb).sum())}
        slow = {k: int(sum(bins[k][j]["p50"] > 150 for j in keep)) for k in names}
        report["bins"][str(width)] = {"n_bins": len(keep), "corr_over300": corr, "slow_bins_p50_over150": slow,
                                      "slow_bins_common": common, "series": bins}
        print(f"bin {width:g} s: {len(keep)} bins; slow bins (p50 > 150 µs) {slow}")
        for k, v in corr.items():
            print(f"  corr(share over 300 µs) {k}: {v:+.2f}  slow-bin overlap {common[k]}")
    report["machines"] = machines(ns.run, paths)
    w = report["bins"]["10.0"]["series"]
    print("t(s)  " + "  ".join(f"{k:>24s}" for k in w))
    for j in range(len(next(iter(w.values())))):
        row = [w[k][j] for k in w]
        if all(row):
            print(f"{row[0]['t']:5.0f} " + "  ".join(f"p50 {x['p50']:6.1f} >300 {x['over300']:.3f}".rjust(24) for x in row))
    json.dump(report, open(ns.json, "w"), indent=1)


if __name__ == "__main__":
    main()
