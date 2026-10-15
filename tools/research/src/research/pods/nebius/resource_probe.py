#!/usr/bin/env python3
"""The resource steward's probe (stdlib only): one cheap reading of a Nebius node's disk, inodes, RAM, replay-RAM reservations,
idle GPU memory and custody backlog against the steward's watermarks. It reads; it never deletes, stops or touches NVML.

    nice -n 19 python3 resource_probe.py --node n1|n2 [--json] [--state PATH]     as research on the node
    nice -n 19 python3 resource_probe.py --node n1|n2 --reading                   the reading alone, for `research monitors`

Exit 0: every metric is under its watermark (the steward's tick ends silently). Exit 1: one or more breaches, one line each on
stdout (`--json` prints the whole reading instead). Exit 2: the probe, or one of its sources, failed: never read as all clear.
`--reading` prints read()'s JSON and exits 0, writing nothing but the directory walk's output: its caller keeps the rate base
and evaluates (`research monitors`).

Sources, all cheap: statvfs of `/` and `/workspace`; `/proc/meminfo`; the ramlock dir's live reservations (research.mem's
convention); replay RAM to come (node 1: Kueue's pending `*-replay-*` Workloads' memory requests, floored at a bundle's size;
node 2: queued `*replay*` fill jobs' `mem_gb=` and running `*commit*` jobs, see replay_queue_fill); GPU memory held at 0% for 15 min (node 1: Prometheus'
DCGM series; node 2: the tail of the utilisation sampler's JSONL, so a timed window is never disturbed); finished run dirs
without `.custody`, `.fetched` or `preserved.json`. Inode growth per hour compares with the oldest of the last hour's readings,
kept in `--state`: one at least 15 min old warns (so one burst between two close ticks isn't an hourly rate), and only one at
least 45 min old makes the HARD line that pages. The inode line on /workspace names the directories two levels down that gained
most between the last two `du --inodes` walks, which the probe starts in the background (idle I/O class) at most hourly, never
in a timed window: a walk of node 1's 11M inodes takes about 100 s, so no reading waits on one.
The watermarks are the resource steward's policy (research-notes `lanes/resource-steward/`).
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GB = 10**9
GIB = 2**30
LIMITS: dict[str, dict[str, float]] = {
    "n1": {"workspace_pct": 80.0, "workspace_hard_pct": 85.0},
    "n2": {"workspace_pct": 55.0, "workspace_hard_pct": 55.0},
}
COMMON = {"root_free_gb": 60.0, "root_hard_free_gb": 45.0, "inode_pct": 80.0, "mem_avail_pct": 10.0, "idle_gpu_gib": 20.0,
          "idle_gpu_s": 900.0, "unpreserved_runs": 10, "unpreserved_age_s": 3600.0, "inode_growth_per_h": 100_000,
          "inode_rate_min_s": 900.0, "inode_page_base_s": 2700.0, "inode_base_max_s": 3600.0, "inode_hard_h": 12.0,
          "replay_share": 0.5, "replay_floor_gb": 90.0}
PROMETHEUS = os.environ.get("VY_PROMETHEUS", "http://10.43.136.178")
DCGM = 'service="nvidia-dcgm-exporter"'


def disk(path: str) -> dict[str, Any]:
    s = os.statvfs(path)
    size, free = s.f_blocks * s.f_frsize, s.f_bavail * s.f_frsize
    used = size - s.f_bfree * s.f_frsize
    iused = s.f_files - s.f_ffree
    return {"path": path, "size_b": size, "used_b": used, "free_b": free,
            "pct": round(100.0 * used / (used + free), 2) if used + free else 0.0,
            "inodes": s.f_files, "iused": iused, "ipct": round(100.0 * iused / s.f_files, 2) if s.f_files else 0.0}


def meminfo(text: str) -> dict[str, float]:
    kb = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\w+):\s+(\d+) kB", text, re.M)}
    total, avail = kb.get("MemTotal", 0) * 1024, kb.get("MemAvailable", 0) * 1024
    return {"total_b": total, "avail_b": avail, "avail_pct": round(100.0 * avail / total, 2) if total else 0.0}


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def ramlock(dirpath: Path) -> dict[str, Any]:
    """Live reservations (a file whose pid is gone is stale and left out, as research.mem does)."""
    live, stale = [], 0
    for f in sorted(Path(dirpath).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if "pid" in r and not _alive(r["pid"]):
            stale += 1
            continue
        live.append({"file": f.name, "owner": r.get("owner"), "gb": float(r.get("gb") or 0)})
    return {"reserved_gb": round(sum(r["gb"] for r in live), 1), "live": live, "stale": stale}


def _mem_gb(q: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]i?)?", str(q).strip())
    if not m:
        return 0.0
    mult = {"K": 1e3, "Ki": 2**10, "M": 1e6, "Mi": 2**20, "G": 1e9, "Gi": 2**30, "T": 1e12, "Ti": 2**40}.get(m.group(2) or "", 1)
    return float(m.group(1)) * mult / GB


def replay_queue_kueue(workloads: list[dict], floor_gb: float) -> dict[str, Any]:
    """Pending (no quota reserved, not finished) `*-replay-*` Workloads and the RAM each will need: max(request, floor)."""
    items = []
    for w in workloads:
        name = w.get("metadata", {}).get("name", "")
        conds = {c.get("type"): c.get("status") for c in (w.get("status") or {}).get("conditions") or []}
        if "replay" not in name or conds.get("QuotaReserved") == "True" or conds.get("Finished") == "True":
            continue
        req = sum(_mem_gb(((c.get("resources") or {}).get("requests") or {}).get("memory", 0)) * int(ps.get("count", 1))
                  for ps in w.get("spec", {}).get("podSets") or [] for c in ps["template"]["spec"].get("containers") or [])
        items.append({"name": name, "request_gb": round(req, 1), "gb": round(max(req, floor_gb), 1)})
    return {"queued": len(items), "gb": round(sum(i["gb"] for i in items), 1), "items": items}


def replay_queue_fill(fill_dir: Path, floor_gb: float) -> dict[str, Any]:
    """Node 2's replay RAM to come: queued `*replay*` fill jobs (max(mem_gb, floor)) and running `*commit*` jobs, each of which
    queues a replay of its bundle when it ends (at the floor). A queued Commit reserves nothing until it runs. The runner's
    dotfiles beside a running job (`.<job>.pgid`, `.<job>.unit`) are not jobs."""
    items = []
    for sub, pat in (("queue", "replay"), ("running", "commit")):
        for f in sorted((Path(fill_dir) / sub).glob(f"*{pat}*")):
            if f.name.startswith("."):
                continue
            try:
                head = f.read_text(errors="replace")[:2000]
            except OSError:
                continue
            m = re.search(r"^# fill:.*\bmem_gb=([\d.]+)", head, re.M)
            req = float(m.group(1)) if m and pat == "replay" else 0.0
            items.append({"name": f"{sub}/{f.name}", "request_gb": req, "gb": max(req, floor_gb)})
    return {"queued": len(items), "gb": round(sum(i["gb"] for i in items), 1), "items": items}


def tail_lines(path: Path, nbytes: int = 4 * 2**20) -> list[str]:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        data = f.read().decode(errors="replace")
    lines = data.splitlines()
    return lines[1:] if size > nbytes else lines


def idle_gpus_sampler(lines: list[str], now: float, window_s: float, gib: float) -> list[dict[str, Any]]:
    """GPUs whose every sample in the last `window_s` shows util 0 and memory over `gib` (the node-2 sampler's JSONL)."""
    per: dict[int, list[tuple[float, float]]] = {}
    first = now
    for ln in lines:
        try:
            r = json.loads(ln)
            t = calendar.timegm(time.strptime(r["t"], "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, KeyError):
            continue
        if t < now - window_s:
            continue
        first = min(first, t)
        for g in r.get("gpus") or []:
            per.setdefault(int(g["i"]), []).append((float(g.get("util") or 0), float(g.get("mem_mib") or 0) / 1024))
    if now - first < 0.8 * window_s:                         # too little history to call anything idle for the window
        return []
    return [{"gpu": i, "min_gib": round(min(m for _, m in s), 1), "samples": len(s)}
            for i, s in sorted(per.items()) if s and max(u for u, _ in s) == 0 and min(m for _, m in s) > gib]


def idle_from_series(util: dict, fb: dict, n: dict, gib: float, cover: float = 0.8) -> list[dict[str, Any]]:
    """Node 1's DCGM series keyed by (gpu, pod): a GPU is idle when one pod held it for the whole window (its sample count at least
    `cover` of the fullest series'), at 0% util, with over `gib` GiB in use throughout. A window that spans a pod handover isn't."""
    full = max(n.values(), default=0)
    out = {}
    for k, u in util.items():
        gpu, pod = k
        if pod and u == 0 and fb.get(k, 0) / 1024 > gib and n.get(k, 0) >= cover * full:
            out[gpu] = {"gpu": gpu, "min_gib": round(fb[k] / 1024, 1), "pod": pod}
    return [out[g] for g in sorted(out)]


def pool_flagged(path: Path, now: float, max_age_s: float = 900.0) -> set[int]:
    """GPUs that pool_n1's fresh `gpu-idle-in-lease` monitor already flags (job norms own those; the steward doesn't re-alert)."""
    try:
        doc = json.loads(Path(path).read_text())
        if now - calendar.timegm(time.strptime(doc["generated_at"], "%Y-%m-%dT%H:%M:%SZ")) > max_age_s:
            return set()
        return {int(m["gpu"]) for n in doc["nodes"].values() for m in n.get("monitors") or [] if m.get("kind") == "gpu-idle-in-lease"}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def idle_gpus_prometheus(window_s: float, gib: float) -> list[dict[str, Any]]:
    def by_series(q: str) -> dict[tuple[int, str], float]:
        url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": q})
        with urllib.request.urlopen(url, timeout=10) as r:
            return {(int(x["metric"]["gpu"]), x["metric"].get("pod", "")): float(x["value"][1]) for x in json.load(r)["data"]["result"]}
    w = f"{int(window_s)}s"
    return idle_from_series(by_series(f"max by (gpu, pod) (max_over_time(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}[{w}]))"),
                            by_series(f"min by (gpu, pod) (min_over_time(DCGM_FI_DEV_FB_USED{{{DCGM}}}[{w}]))"),
                            by_series(f"max by (gpu, pod) (count_over_time(DCGM_FI_DEV_GPU_UTIL{{{DCGM}}}[{w}]))"), gib)


def unpreserved(runs_dir: Path, now: float, age_s: float) -> dict[str, Any]:
    old, recent = [], 0
    for d in Path(runs_dir).glob("r2*"):
        if any((d / m).exists() for m in (".custody", ".fetched", "preserved.json")):
            continue
        try:
            st = d / "status.json"
            state = json.loads(st.read_text())["transitions"][-1]["state"]
            mtime = st.stat().st_mtime
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if state in ("running", "queued", "starting"):
            continue
        if now - mtime > age_s:
            old.append(d.name)
        else:
            recent += 1
    return {"old": len(old), "recent": recent, "oldest": sorted(old)[:5]}


def scratch(cache: Path, now: float) -> list[dict[str, Any]]:
    """Check scratch directories (candidates for the steward's cleanup, which checks open files itself)."""
    out = []
    for d in sorted(Path(cache).glob("lean-audit-scratch-*")):
        try:
            out.append({"path": str(d), "age_h": round((now - d.stat().st_mtime) / 3600, 1)})
        except OSError:
            continue
    return out


def _du(f: Path) -> dict[str, int]:
    out = {}
    for ln in f.read_text(errors="replace").splitlines():
        n, _, p = ln.partition("\t")
        if n.isdigit():
            out[p] = int(n)
    return out


def inode_dirs(state: Path, ws: str, now: float, every_s: float, *, walk: bool) -> dict[str, Any] | None:
    """The three directories under `ws`, two levels down, that gained the most inodes between the last two walks (None before
    there are two). With `walk`, a new walk starts in the background once the last is `every_s` old and none is running (its
    output, which du writes as it goes, changed in the last 10 min); it replaces the last walk only once it ends."""
    cur, prev, tmp = (state / f"inode-dirs{s}" for s in (".txt", ".prev.txt", ".tmp"))

    def age(f: Path) -> float | None:
        try:
            return now - f.stat().st_mtime
        except OSError:
            return None

    last, running = age(cur), age(tmp)
    if walk and (last is None or last >= every_s) and (running is None or running > 600):
        state.mkdir(parents=True, exist_ok=True)
        c, p, t = (shlex.quote(str(x)) for x in (cur, prev, tmp))
        sh = (f"nice -n 19 ionice -c3 du --inodes -x -d 2 {shlex.quote(ws)} > {t} 2>/dev/null; "
              f"[ -s {t} ] && {{ [ -e {c} ] && mv -f {c} {p}; mv -f {t} {c}; }}")
        subprocess.Popen(["sh", "-c", sh], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    if last is None or age(prev) is None:
        return None
    c, p = _du(cur), _du(prev)
    leaves = [d for d in c if d != ws and not any(o.startswith(d + "/") for o in c)]
    top = sorted(((c[d] - p.get(d, 0), d) for d in leaves), reverse=True)[:3]
    return {"over_s": round(age(prev) - last), "top": [{"dir": os.path.relpath(d, ws), "gained": n} for n, d in top if n > 0]}


def _bases(prev: Any) -> list[dict[str, Any]]:
    """The rate's base readings: a list, or one reading as an older state kept it."""
    return prev if isinstance(prev, list) else [prev] if prev else []


def evaluate(r: dict[str, Any], prev: dict[str, Any] | None, lim: dict[str, float]) -> list[str]:
    """The breaches of one reading (`r`, from read()) against the limits; `prev` is the rate's base readings (next_base())."""
    b = []
    root, ws = r["disk"]["root"], r["disk"]["workspace"]
    if root["free_b"] < lim["root_hard_free_gb"] * GB:
        b.append(f"HARD root free {root['free_b'] / GB:.0f} GB < {lim['root_hard_free_gb']:.0f} GB: no new check starts")
    elif root["free_b"] < lim["root_free_gb"] * GB:
        b.append(f"root free {root['free_b'] / GB:.0f} GB < {lim['root_free_gb']:.0f} GB")
    if ws["pct"] >= lim["workspace_hard_pct"]:
        b.append(f"HARD /workspace {ws['pct']:.1f}% >= {lim['workspace_hard_pct']:.0f}%")
    elif ws["pct"] >= lim["workspace_pct"]:
        b.append(f"/workspace {ws['pct']:.1f}% >= {lim['workspace_pct']:.0f}%")
    for k, d in r["disk"].items():
        if d["ipct"] >= lim["inode_pct"]:
            b.append(f"{d['path']} inodes {d['ipct']:.1f}% >= {lim['inode_pct']:.0f}%")
        olds = [x["disk"][k] for x in _bases(prev) if k in (x.get("disk") or {})
                and r["t"] >= x["disk"][k].get("t", r["t"]) + lim["inode_rate_min_s"]]
        if olds:
            p = min(olds, key=lambda x: x["t"])
            span = r["t"] - p["t"]
            rate = (d["iused"] - p["iused"]) / span * 3600
            if rate > lim["inode_growth_per_h"]:
                left_h = (lim["inode_pct"] / 100 * d.get("inodes", 0) - d["iused"]) / rate
                g = r.get("inode_dirs") if k == "workspace" else None
                tops = ", ".join(f"{x['dir']} +{x['gained']:,}" for x in (g or {}).get("top") or [])
                grew = f" (growing {tops} in {g['over_s'] / 60:.0f} min)" if tops else ""
                if left_h < lim["inode_hard_h"] and span >= lim["inode_page_base_s"]:
                    b.append(f"HARD {d['path']} gaining {rate:,.0f} inodes/h over {span / 60:.0f} min: "
                             f"{lim['inode_pct']:.0f}% of inodes in {max(left_h, 0):.1f} h{grew}")
                elif left_h < lim["inode_hard_h"]:
                    b.append(f"{d['path']} gaining {rate:,.0f} inodes/h over {span / 60:.0f} min: {lim['inode_pct']:.0f}% of "
                             f"inodes in {max(left_h, 0):.1f} h if it holds; pages once it holds for "
                             f"{lim['inode_page_base_s'] / 60:.0f} min{grew}")
                else:
                    b.append(f"{d['path']} gaining {rate:,.0f} inodes/h > {lim['inode_growth_per_h']:,.0f}{grew}")
    m = r["mem"]
    if m["avail_pct"] < lim["mem_avail_pct"]:
        b.append(f"RAM available {m['avail_pct']:.1f}% < {lim['mem_avail_pct']:.0f}%")
    rq = r.get("replay_queue") or {}
    avail_gb = m["avail_b"] / GB - (r.get("ramlock") or {}).get("reserved_gb", 0.0)
    if rq.get("gb", 0) > lim["replay_share"] * avail_gb:
        b.append(f"queued replay {rq['gb']:.0f} GB ({rq['queued']} Commits) > half of available RAM "
                 f"{avail_gb:.0f} GB (after ramlock reservations)")
    for g in r.get("idle_gpus") or []:
        b.append(f"GPU {g['gpu']} holds {g['min_gib']:.0f} GiB at 0% for {lim['idle_gpu_s'] / 60:.0f} min")
    u = r.get("unpreserved") or {}
    if u.get("old", 0) > lim["unpreserved_runs"]:
        b.append(f"{u['old']} finished runs older than {lim['unpreserved_age_s'] / 3600:.0f} h lack custody "
                 f"(oldest {', '.join(u['oldest'][:3])})")
    return b


def next_base(r: dict[str, Any], base: Any, lim: dict[str, float]) -> list[dict[str, Any]]:
    """The base readings the next inode rate compares with: those of `base` under `inode_base_max_s` old, and this one."""
    kept = [x for x in _bases(base) if r["t"] - x.get("t", 0) < lim["inode_base_max_s"]]
    return kept + [{"t": r["t"], "disk": {k: {"t": d.get("t", r["t"]), "iused": d["iused"]} for k, d in r["disk"].items()}}]


def read(node: str, *, now: float, home: Path, ws: str = "/workspace") -> dict[str, Any]:
    lim = {**COMMON, **LIMITS[node]}
    r: dict[str, Any] = {"schema": "resource-probe/v1", "node": node, "t": now, "errors": []}
    r["disk"] = {"root": {**disk("/"), "t": now}, "workspace": {**disk(ws), "t": now}}
    r["mem"] = meminfo(Path("/proc/meminfo").read_text())
    r["ramlock"] = ramlock(Path(ws) / "ramlock")
    status = Path(ws) / "pouw/fill/status.txt"
    r["timed"] = status.exists() and "timed True" in status.read_text(errors="replace")[:4000]
    try:
        if node == "n1":
            out = subprocess.run(["sudo", "-n", "k3s", "kubectl", "get", "workloads", "-A", "-o", "json"],
                                 capture_output=True, text=True, check=True, timeout=30).stdout
            r["replay_queue"] = replay_queue_kueue(json.loads(out)["items"], lim["replay_floor_gb"])
        else:
            r["replay_queue"] = replay_queue_fill(Path(ws) / "pouw/fill", lim["replay_floor_gb"])
    except Exception as e:  # noqa: BLE001
        r["errors"].append(f"replay_queue: {type(e).__name__}: {str(e)[:200]}")
    try:
        if node == "n1":
            idle = idle_gpus_prometheus(lim["idle_gpu_s"], lim["idle_gpu_gib"])
            flagged = pool_flagged(Path(ws) / "usage/infra-pool-n1.json", now)
            r["idle_gpus_pool_flagged"] = [g for g in idle if g["gpu"] in flagged]
            r["idle_gpus"] = [g for g in idle if g["gpu"] not in flagged]
        else:
            util = Path(ws) / "pouw/infra/util" / time.strftime("%Y-%m-%d.jsonl", time.gmtime(now))
            r["idle_gpus"] = idle_gpus_sampler(tail_lines(util), now, lim["idle_gpu_s"], lim["idle_gpu_gib"]) if util.exists() else []
    except Exception as e:  # noqa: BLE001
        r["errors"].append(f"idle_gpus: {type(e).__name__}: {str(e)[:200]}")
    r["unpreserved"] = unpreserved(Path(ws) / "research/runs", now, lim["unpreserved_age_s"])
    r["scratch"] = scratch(home / ".cache/verity-check", now)
    try:
        r["inode_dirs"] = inode_dirs(home / ".local/state/resource-steward", ws, now, lim["inode_base_max_s"], walk=not r["timed"])
    except Exception as e:  # noqa: BLE001
        r["errors"].append(f"inode_dirs: {type(e).__name__}: {str(e)[:200]}")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--node", choices=sorted(LIMITS), required=True)
    ap.add_argument("--state", type=Path, default=Path.home() / ".local/state/resource-steward/probe.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--reading", action="store_true", help="print the reading as JSON; no state, no history, no evaluation")
    ns = ap.parse_args(argv)
    if ns.reading:
        try:
            print(json.dumps(read(ns.node, now=time.time(), home=Path.home())))
        except Exception as e:  # noqa: BLE001
            print(f"probe failed: {type(e).__name__}: {e}")
            return 2
        return 0
    try:
        prev = json.loads(ns.state.read_text()) if ns.state.exists() else None
    except (OSError, ValueError):
        prev = None
    try:
        lim = {**COMMON, **LIMITS[ns.node]}
        r = read(ns.node, now=time.time(), home=Path.home())
        base = (prev or {}).get("rate_base")
        r["breaches"] = evaluate(r, base, lim)
        r["rate_base"] = next_base(r, base, lim)
    except Exception as e:  # noqa: BLE001
        print(f"probe failed: {type(e).__name__}: {e}")
        return 2
    ns.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.state.with_name(ns.state.name + ".tmp")
    tmp.write_text(json.dumps(r) + "\n")
    tmp.replace(ns.state)
    with open(ns.state.with_name("history.jsonl"), "a") as h:
        h.write(json.dumps({"t": r["t"], "root_free_gb": round(r["disk"]["root"]["free_b"] / GB, 1),
                            "ws_pct": r["disk"]["workspace"]["pct"], "ws_ipct": r["disk"]["workspace"]["ipct"],
                            "mem_avail_pct": r["mem"]["avail_pct"], "replay_gb": (r.get("replay_queue") or {}).get("gb"),
                            "unpreserved_old": r["unpreserved"]["old"], "breaches": len(r["breaches"])}) + "\n")
    if ns.json:
        print(json.dumps(r, indent=1))
    else:
        for line in r["breaches"] + [f"probe error: {e}" for e in r["errors"]]:
            print(f"{ns.node}: {line}")
    return 1 if r["breaches"] else 2 if r["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
