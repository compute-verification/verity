#!/usr/bin/env python3
"""Node 2's utilization and queue as `infra-pool/v1`, pushed to vy-nebius-1's `/workspace/usage/infra-pool.json` for the console.

Run by `infra-pool-publish.timer` (every 5 min, as `research`, nice 19). Reads only files: the tail of the sampler's
`util/*.jsonl` (the last hour), `fill/status.txt` and the `# fill:` headers of the fill folders; no NVML, no flock. While a
timed window is on (`status.txt` says `timed True`) it neither computes nor pushes. The push is `scp` to a temp file on
`vy-n1` (the `vy-cluster` key) then `mv`, so a reader sees the old file or the new one.

Busy is the hourly report's rule: a GPU sample is busy when util >= 1% or SM active >= 1%. A busy sample is filler when the
lease's fill job carries `filler=` in its header, else useful. `ready_gpu_h` is the queued GPU jobs' `max_min` summed (the
most their next runs may take), against the 12 GPU-h watermark.

    publish_pool.py [--out FILE] [--no-push]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTIL = Path(os.environ.get("UTIL_OUT_DIR", "/workspace/pouw/infra/util"))
FILL = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill"))
OUT = Path(os.environ.get("POOL_OUT", "/workspace/pouw/infra/infra-pool.json"))
DEST_HOST = os.environ.get("POOL_DEST_HOST", "vy-n1")
DEST = os.environ.get("POOL_DEST", "/workspace/usage/infra-pool.json")
NODE, HOST, CPUS, WATERMARK_GPU_H = "n2", "vy-nebius-2", 192, 12
FOLDERS = ("queue", "running", "done", "failed", "withdrawn")


def ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def tail_records(files: list[Path], since: datetime) -> list[dict]:
    """The records at or after `since`, reading each file backwards in blocks until one is older."""
    out: list[dict] = []
    for f in files:
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            pos, buf, done = fh.tell(), b"", False
            while pos > 0 and not done:
                step = min(1 << 20, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                first, _, rest = buf.partition(b"\n") if pos > 0 else (b"", b"", buf)
                m = re.search(rb'"t":"([^"]+)"', rest)
                done = bool(m) and ts(m.group(1).decode()) < since
            if pos > 0:
                buf = buf.partition(b"\n")[2]
        for line in buf.splitlines():
            try:
                r = json.loads(line)
                if ts(r["t"]) >= since:
                    out.append(r)
            except (ValueError, KeyError):
                continue
    return sorted(out, key=lambda r: r["t"])


def header(p: Path) -> dict:
    h = {"gpus": 1, "max_min": 25, "project": "pous"}
    try:
        with open(p, errors="replace") as fh:
            lines = [next(fh, "") for _ in range(20)]
    except OSError:
        return h
    for line in lines:
        m = re.match(r"#\s*fill:\s*(.*)", line)
        if m:
            for k, v in re.findall(r"(\w+)=(\S+)", m.group(1)):
                h[k] = int(v) if v.isdigit() else v
    h["gpus"] = 1 if int(h["gpus"]) >= 1 else 0
    h["max_min"] = min(int(h["max_min"]), 360 if h["project"] == "verity" and not h["gpus"] else 30)
    return h


def jobs() -> dict[str, dict[str, dict]]:
    return {d: {p.name: header(p) for p in (FILL / d).glob("*.sh")} for d in FOLDERS}


def fill_job(lease: str, known: dict[str, dict]) -> dict | None:
    """The fill job a lease runs; the sampler truncates the lease's cmd, so its name matches as a prefix."""
    m = re.search(r"/fill/running/(\S+)", lease or "")
    if not m:
        return None
    frag = m.group(1)
    hits = [h for name, h in known.items() if name.startswith(frag) or frag.startswith(name)]
    return next((h for h in hits if "filler" in h), hits[0] if hits else None)


def busy(g: dict) -> bool:
    return (g.get("util") or 0) >= 1 or (g.get("smact") or 0) >= 0.01


def timed_window() -> bool:
    try:
        return " timed True," in (FILL / "status.txt").read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return False


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def build(now: datetime) -> dict:
    day = now.strftime("%Y-%m-%d")
    files = [UTIL / f"{(now - timedelta(hours=1)).strftime('%Y-%m-%d')}.jsonl"] if (now - timedelta(hours=1)).strftime("%Y-%m-%d") != day else []
    recs = tail_records(files + [UTIL / f"{day}.jsonl"], now - timedelta(hours=1))
    r5 = [r for r in recs if ts(r["t"]) >= now - timedelta(minutes=5)]
    js = jobs()
    known = {k: v for d in js.values() for k, v in d.items()}
    b1 = [busy(g) for r in recs for g in r.get("gpus", [])]
    b5 = [busy(g) for r in r5 for g in r.get("gpus", [])]
    useful = [not (fill_job(g.get("lease"), known) or {}).get("filler") for r in recs for g in r.get("gpus", []) if busy(g)]
    last = recs[-1] if recs else {"gpus": []}
    gpus = []
    for g in last.get("gpus", []):
        lease = g.get("lease") or ""
        who = re.search(r"\bwho=(\S+)", lease)
        job = fill_job(lease, known)
        cls = "free" if not lease else "filler" if job and job.get("filler") else job["project"] if job else "direct"
        per = [busy(x) for r in r5 for x in r.get("gpus", []) if x.get("i") == g["i"]]
        gpus.append({"index": g["i"], "busy_5m": mean([float(x) for x in per]), "holder": who.group(1) if who else None,
                     "class": cls})
    queued = js["queue"]
    return {"schema": "infra-pool/v1", "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "watermark_gpu_h": WATERMARK_GPU_H,
            "nodes": {NODE: {
                "host": HOST, "sampled_at": last.get("t"), "samples_5m": len(r5), "samples_1h": len(recs),
                "gpu_busy_5m": mean([float(x) for x in b5]), "gpu_busy_1h": mean([float(x) for x in b1]),
                "useful_share_1h": mean([float(x) for x in useful]),
                "cpu_busy_5m": mean([r["cpu_busy"] for r in r5 if r.get("cpu_busy") is not None]),
                "gpus": gpus,
                "ready_gpu_h": round(sum(h["max_min"] for h in queued.values() if h["gpus"]) / 60, 2),
                "waiting": len(queued),
                "waiting_gpu": sum(1 for h in queued.values() if h["gpus"]),
                "waiting_cpu": sum(1 for h in queued.values() if not h["gpus"]),
                "lease_waiters": len(last.get("lease_waiters") or []),
                "timed_window": False}}}


def push(path: Path) -> None:
    tmp = f"{DEST}.tmp-{NODE}"
    ssh = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    subprocess.run(["scp", "-q", *ssh, str(path), f"{DEST_HOST}:{tmp}"], check=True, timeout=60)
    subprocess.run(["ssh", *ssh, DEST_HOST, f"mv -f {tmp} {DEST}"], check=True, timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if timed_window():
        return 0
    doc = build(datetime.now(timezone.utc).replace(microsecond=0))
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(a.out)
    if not a.no_push:
        push(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
