#!/usr/bin/env python3
"""Node 2's utilization and queue as `infra-pool/v1`, pushed to vy-nebius-1's `/workspace/usage/infra-pool.json` for the console.

Run by `infra-pool-publish.timer` (every 5 min, as `research`, nice 19). Reads only files: the tail of the sampler's
`util/*.jsonl` (the last hour), `fill/status.txt` and the `# fill:` headers of the fill folders; no NVML, no flock. While a
timed window is on (`status.txt` says `timed True`) it neither computes nor pushes. The push is `scp` to a temp file on
`vy-n1` (the `vy-cluster` key) then `mv`, so a reader sees the old file or the new one.

Busy is the hourly report's rule: a GPU sample is busy when util >= 1% or SM active >= 1%. A busy sample is filler when the
lease's fill job carries `filler=` in its header, else useful. `ready_gpu_h` is the queued GPU jobs' `max_min` summed (the
most their next runs may take), against the 12 GPU-h watermark.

Monitors, appended to node_ops' `alerts.jsonl` once per lease or process (again after 30 min while it lasts):
- `gpu-idle-in-lease`: a lease held for the whole last 5 min whose GPU's mean util over them is under 10%.
- `gpu-unleased`: a process on a GPU that no lease holds, or outside the holding lease's scope and process tree.
`kinds` is each kind's leases that ended in the last 24 h (`gpu-lease/usage/v1`): leased and useful GPU-h (a filler job's busy
time counts as leased only). A job's kind is its header's `kind=`, else `unlabeled:<first two words of its name>`, or
`timed:<who>` or `direct:<who>` outside the fill queue (a timed lease idles the node's other GPUs by design, and is never flagged).

Node 1's own file (`/workspace/usage/infra-pool-n1.json`, kueue-fold's) is merged in as `nodes.n1` while it is under 15 min old.

`delivered_by_hour` (kueue-fold's names, node 1's shape): the last 4 UTC hours, oldest first, each lease's GPU-seconds split across
the hours it spans. Leases still held count as leased up to now. A lease delivered when it ended with rc 0, or 99 (a fill chunk done,
more to come), and isn't filler; a preempted or stopped lease (143, 137, 124) didn't. `guest_gpu_s` is the leased part held by Verity
guests (`project=verity` fill jobs and `verity:` holders). `provisional` stays true until an hour after the hour ends.

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
OTHER = os.environ.get("POOL_OTHER", "/workspace/usage/infra-pool-n1.json")
OTHER_MAX_AGE_S = 900
USAGE = Path(os.environ.get("LEASE_USAGE", "/workspace/research/lease-usage.jsonl"))
ALERTS = Path(os.environ.get("POOL_ALERTS", "/workspace/pouw/infra/alerts.jsonl"))
STATE = Path(os.environ.get("POOL_STATE", "/workspace/pouw/infra/pool-monitor-state.json"))
PROC = Path(os.environ.get("POOL_PROC", "/proc"))
NODE, HOST, CPUS, WATERMARK_GPU_H = "n2", "vy-nebius-2", 192, 12
IDLE_UTIL, IDLE_S, REALERT_S = 10, 300, 1800
FOLDERS = ("queue", "running", "done", "failed", "withdrawn")


def ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def ts_or_none(s: str | None) -> datetime | None:
    try:
        return ts(s) if s else None
    except ValueError:
        return None


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


def field(lease: str, k: str) -> str | None:
    m = re.search(rf"\b{k}=(\S+)", lease or "")
    return m.group(1) if m else None


def kind_of(lease: str, known: dict[str, dict]) -> str:
    job = fill_job(lease, known)
    if job and job.get("kind"):
        return str(job["kind"])
    m = re.search(r"/fill/running/(\S+)", lease)
    if m:
        return "unlabeled:" + "-".join(m.group(1).removesuffix(".sh").split("-")[:2])
    return f"{'timed' if field(lease, 'timed') == '1' else 'direct'}:{field(lease, 'who') or '?'}"


def owner_of(lease: str) -> str | None:
    who = field(lease, "who")
    return who.removeprefix("fill:") if who else None


def leased_pid(pid: int, lease: str) -> bool | None:
    """Whether a GPU process belongs to the lease: in its scope or under its pid. None when the process is gone."""
    try:
        cg = (PROC / str(pid) / "cgroup").read_text()
    except OSError:
        return None
    scope, top = field(lease, "scope"), field(lease, "pid")
    if scope and scope != "-" and f"/{scope}.scope" in cg:
        return True
    for _ in range(64):
        if top and str(pid) == top:
            return True
        try:
            pid = int((PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            return False
        if pid <= 1:
            return False
    return False


def monitors(r5: list[dict], now: datetime, known: dict[str, dict]) -> list[dict]:
    if not r5:
        return []
    found, last = [], r5[-1]
    for g in last.get("gpus", []):
        lease = g.get("lease") or ""
        pid, since = field(lease, "pid"), field(lease, "since")
        start = ts_or_none(since)
        if lease and pid and start and start <= now - timedelta(seconds=IDLE_S) and field(lease, "timed") != "1":
            utils = [x.get("util") or 0 for r in r5 for x in r.get("gpus", []) if x.get("i") == g["i"] and field(x.get("lease"), "pid") == pid]
            if utils and sum(utils) / len(utils) < IDLE_UTIL:
                found.append({"kind": "gpu-idle-in-lease", "key": f"idle:{pid}:{g['i']}", "gpu": g["i"], "owner": owner_of(lease),
                              "job_kind": kind_of(lease, known), "mean_util_5m": round(sum(utils) / len(utils), 1), "lease_since": since,
                              "msg": f"GPU {g['i']} leased since {since} by {field(lease, 'who')} ({kind_of(lease, known)}) averaged "
                                     f"{sum(utils) / len(utils):.1f}% util over the last 5 min"})
        for p in g.get("procs") or []:
            if leased_pid(int(p["pid"]), lease) is False:
                found.append({"kind": "gpu-unleased", "key": f"unleased:{p['pid']}:{g['i']}", "gpu": g["i"], "pid": p["pid"],
                              "cmd": (p.get("cmd") or "")[:120], "lease": field(lease, "who"),
                              "msg": f"pid {p['pid']} uses GPU {g['i']} ({p.get('mib')} MiB) outside any lease of it"
                                     f"{' (held by ' + field(lease, 'who') + ')' if lease else ''}: {(p.get('cmd') or '')[:80]}"})
    return found


def alert(found: list[dict], now: datetime) -> None:
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    t = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    new = [a for a in found if a["key"] not in state or ts(state[a["key"]]) <= now - timedelta(seconds=REALERT_S)]
    if new:
        with open(ALERTS, "a") as f:
            f.write("".join(json.dumps({"t": t, **a}) + "\n" for a in new))
    state = {k: v for k, v in state.items() if ts(v) > now - timedelta(days=1)} | {a["key"]: t for a in new}
    tmp = STATE.with_name(STATE.name + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)


def kinds(now: datetime, known: dict[str, dict]) -> list[dict]:
    """Each kind's leases that ended in the last 24 h, the most idle first."""
    since, out = now - timedelta(days=1), {}
    try:
        lines = USAGE.read_text(errors="replace").splitlines()
    except OSError:
        return []
    for line in lines[-20000:]:
        try:
            r = json.loads(line)
            if (ts_or_none(r.get("end")) or since) < since:
                continue
        except (ValueError, KeyError, TypeError):
            continue
        lease = f"who={r.get('who')} {'timed=1 ' if r.get('timed') else ''}cmd={r.get('cmd') or ''}"
        k = kind_of(lease, known)
        filler = bool((fill_job(lease, known) or {}).get("filler"))
        e = out.setdefault(k, {"kind": k, "owners": set(), "leases": 0, "leased_gpu_h": 0.0, "useful_gpu_h": 0.0})
        e["owners"].add(owner_of(lease) or "?")
        e["leases"] += 1
        for g in r.get("gpus") or []:
            e["leased_gpu_h"] += (g.get("held_s") or 0) / 3600
            e["useful_gpu_h"] += 0 if filler else (g.get("busy_s") or 0) / 3600
    rows = []
    for e in out.values():
        rows.append({**e, "owners": sorted(e["owners"]), "leased_gpu_h": round(e["leased_gpu_h"], 2),
                     "useful_gpu_h": round(e["useful_gpu_h"], 2), "idle_gpu_h": round(e["leased_gpu_h"] - e["useful_gpu_h"], 2),
                     "efficiency": round(e["useful_gpu_h"] / e["leased_gpu_h"], 4) if e["leased_gpu_h"] else None})
    return sorted(rows, key=lambda x: -x["idle_gpu_h"])


def delivered_by_hour(now: datetime, known: dict[str, dict], last: dict) -> list[dict]:
    h0 = now.replace(minute=0, second=0) - timedelta(hours=3)
    hours = [h0 + timedelta(hours=i) for i in range(4)]
    rows = {h: {"leased_gpu_s": 0.0, "delivered_gpu_s": 0.0, "guest_gpu_s": 0.0} for h in hours}

    def add(start: datetime, end: datetime, gpus: int, delivered: bool, guest: bool) -> None:
        for h in hours:
            s = (min(end, h + timedelta(hours=1)) - max(start, h)).total_seconds()
            if s > 0:
                rows[h]["leased_gpu_s"] += s * gpus
                rows[h]["delivered_gpu_s"] += s * gpus if delivered else 0
                rows[h]["guest_gpu_s"] += s * gpus if guest else 0

    try:
        lines = USAGE.read_text(errors="replace").splitlines()[-20000:]
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
            start, end = ts(r["since"]), ts(r["end"])
        except (ValueError, KeyError, TypeError):
            continue
        if end < h0:
            continue
        lease = f"who={r.get('who')} cmd={r.get('cmd') or ''}"
        job = fill_job(lease, known) or {}
        guest = job.get("project") == "verity" or str(r.get("who") or "").startswith("verity:")
        add(start, end, len(r.get("gpus") or []), r.get("rc") in (0, 99) and not job.get("filler"), guest)
    for g in last.get("gpus", []):
        lease = g.get("lease") or ""
        start = ts_or_none(field(lease, "since"))
        if start:
            job = fill_job(lease, known) or {}
            add(start, now, 1, False, job.get("project") == "verity" or (field(lease, "who") or "").startswith("verity:"))
    out = []
    for h in hours:
        r = rows[h]
        out.append({"hour": h.strftime("%Y-%m-%dT%H:%M:%SZ"), "leased_gpu_s": round(r["leased_gpu_s"]),
                    "delivered_gpu_s": round(r["delivered_gpu_s"]),
                    "delivered_share": round(r["delivered_gpu_s"] / r["leased_gpu_s"], 4) if r["leased_gpu_s"] else None,
                    "guest_gpu_s": round(r["guest_gpu_s"]), "provisional": now < h + timedelta(hours=2)})
    return out


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
    found = monitors(r5, now, known)
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
                "unlabeled_waiting": sum(1 for h in queued.values() if "kind" not in h),
                "monitors": found,
                "kinds": kinds(now, known),
                "delivered_by_hour": delivered_by_hour(now, known, last),
                "timed_window": False}}}


def merge(doc: dict, other: dict | None, now: datetime) -> dict:
    """Node 1's own file (kueue-fold's `pool_n1.py`) joins `nodes` while it is under 15 min old; node 2's entry always wins."""
    t = ts_or_none((other or {}).get("generated_at"))
    if t and now - t <= timedelta(seconds=OTHER_MAX_AGE_S):
        for name, node in (other.get("nodes") or {}).items():
            if name != NODE:
                doc["nodes"][name] = {**node, "generated_at": other["generated_at"]}
    return doc


def fetch_other() -> dict | None:
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", DEST_HOST, f"cat {OTHER}"],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (subprocess.SubprocessError, ValueError):
        return None


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
    now = datetime.now(timezone.utc).replace(microsecond=0)
    doc = build(now)
    if not a.no_push:
        alert(doc["nodes"][NODE]["monitors"], now)
        doc = merge(doc, fetch_other(), now)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(a.out)
    if not a.no_push:
        push(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
