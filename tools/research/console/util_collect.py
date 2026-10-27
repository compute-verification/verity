"""Utilization of vy-nebius-1 and vy-nebius-2 from the telemetry already on them; no sampler of our own.

    python3 util_collect.py --since YYYY-MM-DDTHH:MMZ [--out DIR] [--put]

- vy-nebius-1: Prometheus (DCGM GPU utilization and framebuffer, node-exporter CPU, memory and /workspace), 60 s steps, read
  through the k3s API server's service proxy over ssh (`n1`). DCGM is scraped twice there, so every GPU series is `max by (gpu)`.
  `gpu-lease status` and the Kueue sampler's last line are recorded too.
- vy-nebius-2: POUS's own 10 s sampler (`/workspace/pouw/infra/util/*.jsonl`), read over ssh (`n2`), never written.
Writes DIR/{n1-prom.json,n2-util.jsonl,n1-queues.jsonl,summary.json} and prints the summary; --put stores DIR as evidence.
Definitions, per server and per UTC hour:
- a GPU is *busy* in a step when its utilization > 0, and *held* when a process holds >= 1 GiB of its memory or a lease names it;
- busy GPU-hours = steps busy x step; effective GPU-hours = sum(util/100) x step; idle GPU-hours = available - held-or-busy;
- node 1's Kueue-allocated GPU-hours come from vy-usage's 5-minute queue samples (admitted, whether or not the GPU is used);
- CPU core-hours busy = mean busy cores x hours (non-idle, non-iowait); RAM = MemTotal - MemAvailable (page cache is not used).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

PROM = "/api/v1/namespaces/skypilot/services/skypilot-prometheus-server:80/proxy/api/v1/query_range"
Q = {
    "gpu_util": "max by (gpu) (DCGM_FI_DEV_GPU_UTIL)",
    "gpu_fb_mib": "max by (gpu) (DCGM_FI_DEV_FB_USED)",
    "cpu_busy": 'sum(rate(node_cpu_seconds_total{mode!="idle",mode!="iowait"}[2m]))',
    "mem_used": "sum(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)",
    "ws_avail": 'max(node_filesystem_avail_bytes{mountpoint="/workspace"})',
}
STEP = 60


def ssh(host: str, cmd: str) -> str:
    return subprocess.run(["ssh", host, cmd], check=True, capture_output=True, text=True, timeout=300).stdout


def ts(s: str) -> float:
    return dt.datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc).timestamp()


def prom_range(query: str, start: float, end: float) -> list[dict]:
    out: list[dict] = []
    t = start
    while t < end:  # Prometheus caps a range at 11,000 points
        e = min(end, t + 10000 * STEP)
        qs = urllib.parse.urlencode({"query": query, "start": int(t), "end": int(e), "step": STEP})
        doc = json.loads(ssh("n1", f"sudo k3s kubectl get --raw '{PROM}?{qs}'"))
        out.extend(doc["data"]["result"])
        t = e + STEP
    merged: dict[str, dict] = {}
    for r in out:
        k = json.dumps(r["metric"], sort_keys=True)
        merged.setdefault(k, {"metric": r["metric"], "values": []})["values"].extend(r["values"])
    return list(merged.values())


def hour(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:00Z")


def summarize_n1(prom: dict, start: float, end: float) -> dict:
    util = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_util"]}
    fb = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_fb_mib"]}
    cpu = {int(float(t)): float(v) for r in prom["cpu_busy"] for t, v in r["values"]}
    mem = {int(float(t)): float(v) for r in prom["mem_used"] for t, v in r["values"]}
    ws = {int(float(t)): float(v) for r in prom["ws_avail"] for t, v in r["values"]}
    h = STEP / 3600
    per: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for g, series in util.items():
        for t, u in series.items():
            b = per[hour(t)]
            held = fb.get(g, {}).get(t, 0.0) >= 1024
            b["gpu_h_observed"] += h
            b["gpu_h_busy"] += h * (u > 0)
            b["gpu_h_effective"] += h * u / 100
            b["gpu_h_held_or_busy"] += h * (u > 0 or held)
    for t, c in cpu.items():
        per[hour(t)]["cpu_core_h_busy"] += c * h
        per[hour(t)]["cpu_h_observed"] += h
    for t, m in mem.items():
        per[hour(t)]["ram_peak_gib"] = max(per[hour(t)]["ram_peak_gib"], m / 2**30)
    for t, a in ws.items():
        per[hour(t)]["ws_free_min_tib"] = min(per[hour(t)].get("ws_free_min_tib", 1e9), a / 2**40)
    return finish(per, gpus=8, cores=192)


def kueue_allocated(queue_lines: list[dict], start: float) -> dict[str, float]:
    """GPU-hours Kueue had admitted per UTC hour, from vy-usage's 5-minute samples (each sample stands for the 5 min before it)."""
    per: dict[str, float] = defaultdict(float)
    for d in queue_lines:
        if d.get("t", 0) < start:
            continue
        per[hour(d["t"])] += sum(q.get("usage", {}).get("nvidia.com/gpu", 0.0) for q in d.get("queues", {}).values()) * 300 / 3600
    return per


def summarize_n2(lines: list[dict]) -> dict:
    per: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    lines = sorted(lines, key=lambda d: d["t"])
    prev = None
    for d in lines:
        t = dt.datetime.strptime(d["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
        dh = min((t - prev) if prev else 10.0, 60.0) / 3600
        prev = t
        b = per[hour(t)]
        for g in d.get("gpus", []):
            u = g.get("util") or 0
            held = (g.get("mem_mib") or 0) >= 1024 or bool(g.get("lease")) or bool(g.get("procs"))
            b["gpu_h_observed"] += dh
            b["gpu_h_busy"] += dh * (u > 0)
            b["gpu_h_effective"] += dh * u / 100
            b["gpu_h_held_or_busy"] += dh * (u > 0 or held)
        b["cpu_core_h_busy"] += dh * 192 * (d.get("cpu_busy") or 0)
        b["cpu_h_observed"] += dh
        b["ram_peak_gib"] = max(b["ram_peak_gib"], d.get("mem_used_gib") or 0)
    return finish(per, gpus=8, cores=192)


def finish(per: dict, gpus: int, cores: int) -> dict:
    hours = {}
    tot: dict[str, float] = defaultdict(float)
    for k in sorted(per):
        b = dict(per[k])
        b["gpu_h_idle"] = b.get("gpu_h_observed", 0) - b.get("gpu_h_held_or_busy", 0)
        b["cpu_core_h_idle"] = b.get("cpu_h_observed", 0) * cores - b.get("cpu_core_h_busy", 0)
        hours[k] = {x: round(v, 3) for x, v in b.items()}
        for x in ("gpu_h_observed", "gpu_h_busy", "gpu_h_effective", "gpu_h_held_or_busy", "gpu_h_idle", "cpu_core_h_busy",
                  "cpu_core_h_idle", "cpu_h_observed"):
            tot[x] += b.get(x, 0)
    tot["ram_peak_gib"] = max((b.get("ram_peak_gib", 0) for b in hours.values()), default=0)
    tot["gpu_busy_frac"] = tot["gpu_h_busy"] / tot["gpu_h_observed"] if tot["gpu_h_observed"] else 0
    tot["cpu_busy_frac"] = tot["cpu_core_h_busy"] / (tot["cpu_h_observed"] * cores) if tot["cpu_h_observed"] else 0
    return {"total": {k: round(v, 3) for k, v in tot.items()}, "hours": hours}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="UTC, e.g. 2026-09-30T05:16Z")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--put", action="store_true")
    ns = ap.parse_args(argv)
    start, end = ts(ns.since), dt.datetime.now(dt.timezone.utc).timestamp()
    out = ns.out or Path(f"/tmp/nebius-util-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%MZ}")
    out.mkdir(parents=True, exist_ok=True)
    prom = {k: prom_range(q, start, end) for k, q in Q.items()}
    (out / "n1-prom.json").write_text(json.dumps(prom))
    (out / "n1-queues.jsonl").write_text(ssh("n1", "cat /workspace/usage/queues.jsonl"))
    (out / "n1-gpu-lease.txt").write_text(ssh("n1", "gpu-lease status"))
    n2raw = ssh("n2", "cat /workspace/pouw/infra/util/*.jsonl")
    (out / "n2-util.jsonl").write_text(n2raw)
    n2 = [json.loads(ln) for ln in n2raw.splitlines() if ln.strip()]
    n2 = [d for d in n2 if dt.datetime.strptime(d["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp() >= start]
    summary = {"since": ns.since, "until": dt.datetime.fromtimestamp(end, dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
               "definitions": __doc__.split("Definitions, per server and per UTC hour:\n", 1)[1],
               "vy-nebius-1": summarize_n1(prom, start, end), "vy-nebius-2": summarize_n2(n2)}
    qlines = [json.loads(ln) for ln in (out / "n1-queues.jsonl").read_text().splitlines() if ln.strip().startswith("{")]
    alloc = kueue_allocated(qlines, start)
    for h, v in alloc.items():
        summary["vy-nebius-1"]["hours"].setdefault(h, {})["gpu_h_kueue_allocated"] = round(v, 3)
    summary["vy-nebius-1"]["total"]["gpu_h_kueue_allocated"] = round(sum(alloc.values()), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for s in ("vy-nebius-1", "vy-nebius-2"):
        print(s, json.dumps(summary[s]["total"]))
    if ns.put:
        meta = json.dumps({"what": f"utilization of vy-nebius-1/2 {ns.since}..{summary['until']} (Prometheus + POUS sampler)"})
        src = Path(__file__).resolve().parents[1] / "src"
        env = dict(os.environ, PYTHONPATH=str(src)) if (src / "research").is_dir() else None
        r = subprocess.run([sys.executable, "-m", "research", "data", "put", "--kind", "evidence/v1", "--meta", meta, "--tree",
                            str(out), "--preserve", "--json"], capture_output=True, text=True, env=env)
        try:
            print("stored", json.loads(r.stdout)["id"])
        except (ValueError, KeyError):
            print("put failed:", (r.stderr or r.stdout).strip()[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
