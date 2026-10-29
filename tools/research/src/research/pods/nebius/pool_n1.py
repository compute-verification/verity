#!/usr/bin/env python3
"""vy-nebius-1's half of `infra-pool/v1` (stdlib only): `/workspace/usage/infra-pool-n1.json`, which node 2's publisher merges in
as `nodes.n1` (node2-ops 21:40Z). Report-only: it flags, it never stops anything.

    pool_n1.py [--out PATH] [--loop [--every 300]]     as research on node 1

From Prometheus (DCGM's `nvidia-dcgm-exporter` series, whose `pod` label names the pod holding each GPU, and node-exporter) and
Kueue's Workloads it writes the n2 fields it can compute here: GPU busy (share of 15 s samples with util >= 1%) over 5 min and 1 h,
`useful_share_1h` (the busy samples a pod holds), `cpu_busy_5m`, per GPU {index, busy_5m, holder, class}, and `waiting` split into
GPU and CPU Workloads. `ready_gpu_h`, `lease_waiters` and `kinds` are left out: node 1's queue has no durations yet. `monitors`:
  - `gpu-idle-in-lease`: a pod has held a GPU for 5 min at a mean util under 10%;
  - `gpu-unleased`: a GPU with memory in use (over 1 GiB for all of the last 5 min) or busy that no admitted Workload's
    running pod holds (a GPU whose DCGM pod has just finished is a handover, not flagged);
  - `template-drift`: the dispatcher's templates (`$DISPATCH_SKY`) differ from `infra/nebius` as last pushed into node 1's bare repo
    (`refs/kueue-fold/infra-nebius`; neither node can read GitHub, so an agent pushes it and writes `ref.json` with the time).

`delivered_by_hour` is the delivered-output metric, one entry per UTC hour for the last 4, oldest first:
{hour, leased_gpu_s, delivered_gpu_s, delivered_share, provisional}. Leased is each GPU Workload's admitted-to-finished time (or to
now) × its GPUs, split across the hours it spans. It is delivered when its Job is not labelled `verity.dev/filler` and one of the
Attempts its pods launched (the store's `attempts.host` is the pod) is `done` with validation `passed`, or with no validation and
rc 0 and declared outputs. A verdict lands up to ~30 min after the lease ends, so an hour is provisional until an hour after it
ends. `gpu-pool-*` holder Jobs count as leased; the leases inside them are not yet attributed, so they deliver nothing.

Node 2's publisher neither computes nor pushes the merged `/workspace/usage/infra-pool.json` while a timed window is on. From
10 minutes to 3 hours after its last push, each pass here rewrites that file with this pass's `nodes.n1` and an empty
`nodes.n2` that names the pause (`paused_since`, also kept as `n2_generated_at`), so the console shows node 1 now and no node-2
numbers rather than both going stale. After 3 hours it leaves the file alone, and the console's staleness error shows the outage.
Node 2's next push replaces the file whole.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROMETHEUS = os.environ.get("VY_PROMETHEUS", "http://10.43.136.178")
DCGM = 'service="nvidia-dcgm-exporter"'
OUT = Path(os.environ.get("POOL_N1_OUT", "/workspace/usage/infra-pool-n1.json"))
MERGED = Path(os.environ.get("POOL_MERGED", "/workspace/usage/infra-pool.json"))   # node 2's publisher writes it, by scp
PAUSE_COVER_S = (600, 3 * 3600)
GIT = os.environ.get("POOL_N1_GIT", "/workspace/research/git/verity.git")
REF = "refs/kueue-fold/infra-nebius"
REF_JSON = Path(os.environ.get("POOL_N1_REF_JSON", "/workspace/verity-guest/drift/ref.json"))
DISPATCH_SKY = Path(os.environ.get("DISPATCH_SKY", "/workspace/jobs/dispatch/infra/nebius/sky"))
REPO_SKY = "tools/research/src/research/pods/nebius/sky"
IDLE_UTIL, IDLE_S, UNLEASED_FB_MIB = 10.0, 300, 1024
CLASS_OF_QUEUE = {"pous-overflow": "pous"}
LEASE_STATE = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease")) / "n1-lease.json"
STORE_DB = Path(os.environ.get("POOL_N1_STORE_DB", "/workspace/jobs/store/catalog.sqlite"))
HOURS = 4                                                   # Jobs are deleted 6 h after they finish (ttlSecondsAfterFinished)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def prom(q: str) -> list[dict]:
    url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": q})
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)["data"]["result"]


def by_gpu(q: str) -> dict[int, float]:
    return {int(r["metric"]["gpu"]): float(r["value"][1]) for r in prom(q)}


def scalar(q: str) -> float | None:
    r = prom(q)
    return float(r[0]["value"][1]) if r else None


def kubectl(*args: str) -> dict:
    out = subprocess.run(["sudo", "-n", "k3s", "kubectl", *args, "-o", "json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def observe() -> dict:
    util = f"max by (gpu) (DCGM_FI_DEV_GPU_UTIL{{{DCGM}}})"      # a GPU has one series per pod it has held
    busy = f"({util} >= bool 1)"
    obs = {
        "t": time.time(),
        "sampled_at": scalar(f"max(timestamp(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}))"),
        "samples_5m": scalar(f"min(count_over_time(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}[5m]))"),
        "samples_1h": scalar(f"min(count_over_time(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}[1h]))"),
        "busy_5m": by_gpu(f"avg_over_time({busy}[5m:15s])"),
        "busy_1h": by_gpu(f"avg_over_time({busy}[1h:15s])"),
        "util_5m": by_gpu(f"max by (gpu) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}[5m]))"),
        "held_busy_1h": scalar(f'sum(sum_over_time((max by (gpu) (DCGM_FI_DEV_GPU_UTIL{{{DCGM},pod!=""}}) >= bool 1)[1h:15s]))'),
        "all_busy_1h": scalar(f"sum(sum_over_time({busy}[1h:15s]))"),
        "cpu_busy_5m": scalar('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))'),
        "fb_mib": by_gpu(f"min by (gpu) (min_over_time(DCGM_FI_DEV_FB_USED{{{DCGM}}}[5m]))"),
        "pod_of_gpu": {int(r["metric"]["gpu"]): r["metric"].get("pod") or None for r in prom(f"DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}")},
        "pods": kubectl("get", "pods")["items"],
        "jobs": kubectl("get", "jobs")["items"],
        "workloads": kubectl("get", "workloads")["items"],
        "lease_pool": lease_pool(),
    }
    obs["attempts"] = attempts_by_job(iso(obs["t"] - (HOURS + 4) * 3600))
    return obs


def attempts_by_job(since: str) -> dict[str, list[tuple]]:
    """Job name -> (state, rc, validation, outputs) of each Attempt its pods launched since `since`, from node 1's store."""
    try:
        with sqlite3.connect(f"file:{STORE_DB}?mode=ro", uri=True) as c:
            rows = c.execute("select a.host, a.state, a.rc, a.validation, (select count(*) from attempt_outputs o where o.attempt = a.id)"
                             " from attempts a where a.start_utc >= ?", (since,)).fetchall()
    except sqlite3.Error:
        return {}
    out: dict[str, list[tuple]] = {}
    for host, *rest in rows:
        if host:
            out.setdefault(host.rsplit("-", 1)[0], []).append(tuple(rest))   # a Job's pod is <job>-<5 chars>
    return out


def _delivered(a: tuple) -> bool:
    state, rc, validation, outputs = a
    return state == "done" and (validation == "passed" or (validation in (None, "not_run") and rc == 0 and outputs > 0))


def _at(w: dict, kind: str) -> float | None:
    for c in (w.get("status") or {}).get("conditions") or []:
        if c.get("type") == kind and c.get("status") == "True" and c.get("lastTransitionTime"):
            return datetime.strptime(c["lastTransitionTime"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    return None


def delivery(obs: dict) -> list[dict]:
    """`delivered_by_hour` (see the module docstring) from the Workloads, the Jobs' labels and the store's Attempts."""
    t = obs["t"]
    now_h = int(t // 3600) * 3600
    hours = {now_h - 3600 * k: [0.0, 0.0] for k in range(HOURS)}
    labels = {j["metadata"]["name"]: j["metadata"].get("labels") or {} for j in obs["jobs"]}
    attempts = obs.get("attempts") or {}
    for w in obs["workloads"]:
        n, start = _gpu_request(w), _at(w, "QuotaReserved") or _at(w, "Admitted")
        if not n or start is None:
            continue
        end = _at(w, "Finished") or t
        job = next((o["name"] for o in w["metadata"].get("ownerReferences") or []), "")
        ok = "verity.dev/filler" not in labels.get(job, {}) and any(_delivered(a) for a in attempts.get(job, []))
        for h, acc in hours.items():
            s = max(0.0, min(end, h + 3600) - max(start, h)) * n
            acc[0] += s
            acc[1] += s if ok else 0.0
    return [{"hour": iso(h), "leased_gpu_s": round(l), "delivered_gpu_s": round(d),
             "delivered_share": round(d / l, 4) if l else None, "provisional": t < h + 7200}
            for h, (l, d) in sorted(hours.items())]


def lease_pool() -> dict[int, dict]:
    """n1_lease.py's GPUs in the lease pool (index -> its record), when its state is fresh; else none."""
    try:
        s = json.loads(LEASE_STATE.read_text())
    except (OSError, ValueError):
        return {}
    if time.time() - datetime.strptime(s["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() > 120:
        return {}
    return {g["index"]: g for g in s.get("gpus") or [] if g.get("holder")}


def _cond(w: dict, kind: str) -> bool:
    return any(c.get("type") == kind and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or [])


def _gpu_request(w: dict) -> int:
    n = 0
    for ps in w["spec"].get("podSets") or []:
        for c in ps["template"]["spec"].get("containers") or []:
            n += int(((c.get("resources") or {}).get("requests") or {}).get("nvidia.com/gpu", 0) or 0) * int(ps.get("count", 1))
    return n


def build(obs: dict, drift: dict | None = None) -> dict:
    """The `nodes.n1` entry from one observation (see observe()); `drift` is template_drift()'s monitor, if any.  A GPU in
    n1_lease.py's pool is attributed to its lease (the lease files), not to the holder pod DCGM names."""
    t = obs["t"]
    admitted: dict[str, dict] = {}                           # Job or Pod name -> its admitted, unfinished Workload
    waiting_gpu = waiting_cpu = 0
    for w in obs["workloads"]:
        if _cond(w, "Finished"):
            continue
        if _cond(w, "QuotaReserved"):
            for o in w["metadata"].get("ownerReferences") or []:
                admitted[o["name"]] = w
        elif _gpu_request(w):
            waiting_gpu += 1
        else:
            waiting_cpu += 1
    pods = {p["metadata"]["name"]: p for p in obs["pods"]}
    jobs = {j["metadata"]["name"]: j for j in obs["jobs"]}
    gpus, monitors = [], []
    for i in sorted(obs["pod_of_gpu"]):
        pod_name = obs["pod_of_gpu"][i]
        pod = pods.get(pod_name or "")
        job = (pod or {}).get("metadata", {}).get("labels", {}).get("job-name")
        wl = admitted.get(job or "") or admitted.get(pod_name or "")
        busy5, util5, fb = obs["busy_5m"].get(i), obs["util_5m"].get(i, 0.0), obs["fb_mib"].get(i, 0.0)
        holder, cls = None, "free"
        pooled = (obs.get("lease_pool") or {}).get(i)
        if pooled:
            lease = pooled.get("lease")
            holder, cls = (f"lease:{lease.get('who')}", "verity") if lease else ("pool:free", "free")
            since = (lease or {}).get("since")
            held_s = t - datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if since else 0
            if lease and held_s >= IDLE_S and util5 < IDLE_UTIL:
                name = lease.get("pod") or lease.get("run") or "-"
                meta = (jobs.get((pods.get(name) or {}).get("metadata", {}).get("labels", {}).get("job-name", "")) or {}).get("metadata", {})
                kind = (meta.get("annotations") or {}).get("verity.dev/template") or "unlabeled:" + "-".join(name.split("-")[:3])
                monitors.append({"kind": "gpu-idle-in-lease", "gpu": i, "owner": lease.get("who"), "job_kind": kind,
                                 "mean_util_5m": round(util5, 1), "lease_since": since, "key": f"n1:{name}:gpu{i}",
                                 "msg": f"{name} ({kind}, gpu-lease for {lease.get('who')}) holds GPU {i} at {util5:.1f}% over 5 min"})
        elif wl:
            holder = f"kueue:{job or pod_name}"
            cls = CLASS_OF_QUEUE.get(wl["spec"].get("queueName", ""), "verity")
            since = (pod or {}).get("status", {}).get("startTime")
            held_s = t - datetime.strptime(since, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if since else 0
            if held_s >= IDLE_S and util5 < IDLE_UTIL:
                meta = (jobs.get(job or "") or pod or {}).get("metadata", {})
                name = job or pod_name
                kind = (meta.get("annotations") or {}).get("verity.dev/template") or "unlabeled:" + "-".join(name.split("-")[:3])
                owner = (meta.get("labels") or {}).get("verity.dev/lane") or wl["spec"].get("queueName")
                monitors.append({"kind": "gpu-idle-in-lease", "gpu": i, "owner": owner, "job_kind": kind,
                                 "mean_util_5m": round(util5, 1), "lease_since": since, "key": f"n1:{name}:gpu{i}",
                                 "msg": f"{name} ({kind}, queue {wl['spec'].get('queueName')}) holds GPU {i} at {util5:.1f}% over 5 min"})
        elif (pod or {}).get("status", {}).get("phase") in ("Succeeded", "Failed"):
            pass                                         # DCGM still names the pod that just finished: a handover
        elif fb > UNLEASED_FB_MIB or util5 >= 1:
            holder, cls = pod_name, "direct"
            monitors.append({"kind": "gpu-unleased", "gpu": i, "pid": None, "cmd": pod_name, "lease": None,
                             "key": f"n1:unleased:gpu{i}:{pod_name or '-'}",
                             "msg": f"GPU {i}: {fb:.0f} MiB in use, {util5:.1f}% util, and no admitted Workload's pod holds it"
                                    + (f" (pod {pod_name} runs outside Kueue)" if pod_name else "")})
        gpus.append({"index": i, "busy_5m": None if busy5 is None else round(busy5, 4), "holder": holder, "class": cls})
    if drift:
        monitors.append(drift)
    b5, b1 = obs["busy_5m"].values(), obs["busy_1h"].values()
    held, all_busy = obs.get("held_busy_1h"), obs.get("all_busy_1h")
    return {"host": "vy-nebius-1",
            "sampled_at": iso(obs["sampled_at"]) if obs.get("sampled_at") else None,
            "samples_5m": int(obs["samples_5m"]) if obs.get("samples_5m") is not None else 0,
            "samples_1h": int(obs["samples_1h"]) if obs.get("samples_1h") is not None else 0,
            "gpu_busy_5m": round(sum(b5) / len(b5), 4) if b5 else None,
            "gpu_busy_1h": round(sum(b1) / len(b1), 4) if b1 else None,
            "useful_share_1h": round(held / all_busy, 4) if held is not None and all_busy else None,
            "cpu_busy_5m": None if obs.get("cpu_busy_5m") is None else round(obs["cpu_busy_5m"], 4),
            "gpus": gpus, "waiting": waiting_gpu + waiting_cpu, "waiting_gpu": waiting_gpu, "waiting_cpu": waiting_cpu,
            "delivered_by_hour": delivery(obs), "monitors": monitors}


def git(*args: str) -> str:
    return subprocess.run(["git", f"--git-dir={GIT}", *args], capture_output=True, text=True, check=True).stdout


def template_drift() -> dict:
    """Compare the dispatcher's sky/ with infra/nebius's (as pushed into node 1's bare repo): a `template-drift` monitor, or {}."""
    try:
        ref = json.loads(REF_JSON.read_text())
        want = {}
        for line in git("ls-tree", "-r", REF, "--", REPO_SKY).splitlines():
            meta, path = line.split("\t", 1)
            want[path[len(REPO_SKY) + 1:]] = meta.split()[2]
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        return {"kind": "template-drift", "key": "n1:template-drift:no-ref", "files": [], "ref": None,
                "msg": f"no infra/nebius reference to compare the templates with ({str(e)[:120]})"}
    have = {}
    for f in DISPATCH_SKY.rglob("*"):
        rel = str(f.relative_to(DISPATCH_SKY))
        if f.is_file() and "__pycache__" not in rel and ".bak" not in rel:
            have[rel] = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True).stdout.strip()
    files = sorted([f"changed {p}" for p in want if p in have and have[p] != want[p]]
                   + [f"missing {p}" for p in want if p not in have] + [f"extra {p}" for p in have if p not in want])
    if not files:
        return {}
    return {"kind": "template-drift", "key": f"n1:template-drift:{ref.get('commit', '')[:12]}", "files": files,
            "ref": ref.get("commit"), "ref_pushed_at": ref.get("pushed_at"),
            "msg": f"{len(files)} dispatcher template file(s) differ from infra/nebius {ref.get('commit', '')[:9]} "
                   f"(pushed {ref.get('pushed_at')}): {', '.join(files[:4])}{' …' if len(files) > 4 else ''}"}


def write(out: Path, n1: dict) -> None:
    doc = {"generated_at": iso(time.time()), "nodes": {"n1": n1}}
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(out)


def cover_pause(merged: Path, n1: dict, now: float) -> bool:
    """Rewrite the merged file while node 2's publisher is paused (the module docstring); True when it did."""
    try:
        doc = json.loads(merged.read_text())
        last = doc.get("n2_generated_at") or doc["generated_at"]
        age = now - datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    if not PAUSE_COVER_S[0] < age < PAUSE_COVER_S[1]:
        return False
    cover = {"generated_at": iso(now), "n2_generated_at": last,
             "nodes": {"n1": n1, "n2": {"paused_since": last, "note": "node 2's publisher pauses while a timed window is on"}}}
    tmp = merged.with_name(f".{merged.name}.n1-tmp")
    tmp.write_text(json.dumps(cover, indent=1) + "\n")
    tmp.replace(merged)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    ns = ap.parse_args()
    drift, n = None, 0
    while True:
        try:
            if n % 3 == 0:
                drift = template_drift()
            n1 = build(observe(), drift)
            write(ns.out, n1)
            if cover_pause(MERGED, n1, time.time()):
                print(f"{iso(time.time())} pool_n1: covered node 2's paused {MERGED}", flush=True)
        except Exception as e:  # noqa: BLE001  (a failed pass is retried next time; the file's age shows it)
            print(f"{iso(time.time())} pool_n1 failed: {str(e)[:300]}", flush=True)
        if not ns.loop:
            return 0
        n += 1
        time.sleep(ns.every)


if __name__ == "__main__":
    raise SystemExit(main())
