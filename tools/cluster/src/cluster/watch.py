"""`cluster watch`: the jobs that stopped making progress, each with the Slack handle that owns it. Read-only.

A **finding** is a job that has held its allocation for the window (`WINDOW_S`, 30 min) and made no progress over it, or a
goal job (`Job.goal`) that has waited the window for a grant:

- **running** (node 1: a Workload whose pod runs, or a bare pod; node 2: a lease) for the window, while no file of its run
  changed, its processes used no CPU and its GPUs did no work. A run's files are everything under its run dir
  (`<root>/<run>/`, `ROOTS`) but the harness's own periodic writes, which go on while a workload hangs (`QUIET`:
  `resources*.jsonl`, `sampler.*`, custody markers). Its processes' CPU is the harness's record of the workload's process
  tree (`resources.jsonl`'s `process` records, each process's `cpu_s`), the record at the window's start against the last:
  busy at `CPU_BUSY` of a core on average (a build that writes only at its end shows here, and the harness's own sampler
  isn't in the tree). Its GPUs did work when one read above `GPU_BUSY` % utilization in the window: node 1's from DCGM
  through Prometheus (`DCGM_FI_DEV_GPU_UTIL` by pod, `max_over_time`), node 2's from its sampler's 10 s lines
  (`/workspace/pouw/infra/util/<day>.jsonl`, each GPU's utilization beside its lease's owner line, so its pid). Node 2's
  run is its lease's `run=`; node 1's are the run dirs created since its pod started whose `events.jsonl` names the pod as
  its `host`. These are what every job leaves without being asked. A job with no run found and no GPU read is listed as
  unwatched, never taken for progress or for a stall; a signal that can't be read is named in the finding. A hung
  collective that spins its GPUs reads as progress.
- **admitted** (node 1): Kueue reserved its quota the window ago and no pod runs.
- **queued with a goal** (node 1's `verity.dev/goal` label; node 2's ledger, from gpu-lease's `goal=`): it has waited the
  window for a grant.

Not watched: held Workloads (someone held them), frozen leases (a timed window froze them), SkyPilot's controllers (idle
by design), and node 2's leases during a timed window, when the watch reads only the ledger state, so it never loads a
measured window.

A job's owner is `status`'s (node 1's `verity.dev/owner` label, else its workstream; node 2's submitter), routed to a
handle by the Slack registry (`.agents/skills/using-slack/registry.json`): a handle may list the `lanes` it owns (names,
or prefixes ending in `*`) and the `agents` (agent id prefixes). The owner's `:`-separated parts are tried last first
(`fill:bc-2aa33ad8`, `lean-audit:proofs-w`): an agent prefix, else the longest lane pattern; an owner nobody claims is
@infra's. Its lane, where its note goes, is the owner's own when the registry names it as a lane, else the handle's.

On node 1 a timer runs it every 5 minutes (`--local nebius-1 --ssh nebius-2=vy-n2 --out FILE`), and the exporter
(`vy_exporter.py --watch FILE`, infra/nebius `monitoring/`) turns each finding into `vy_job_stalled{...} 1`, which a
Grafana rule sends to the handle's user group. `candidates` and `judge` are the pure half: a snapshot, the node's
progress reads and `now` in, findings out.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import ledger, nebius1, nebius2, status

WINDOW_S = 30 * 60
GPU_BUSY = 5
CPU_BUSY = 0.02
ROOTS = ("/workspace/research/runs", "/workspace/jobs/runs")
UTIL_DIR = "/workspace/pouw/infra/util"
QUIET = r"resources(\..+)?\.jsonl|sampler\..+|\.fetched.*|\.custody.*|preserved\.json"
CONTROLLERS = ("sky-jobs-controller-", "sky-serve-controller-")
GOAL_LABEL = "verity.dev/goal"
REGISTRY = Path(__file__).resolve().parents[4] / ".agents" / "skills" / "using-slack" / "registry.json"
DEFAULT_HANDLE = "infra"
SCHEMA = "cluster/watch/v1"


def why(window_s: int, *, state: str = "running", run: bool = False, cpu: bool = False, gpus: bool = False,
        gpu: bool = False) -> str:
    """A finding's reason: which signals were read and idle, and which couldn't be read. A fixed text for each case (it is a
    label of the exported series, so it never carries a count)."""
    m = f"{window_s // 60} min"
    if state == "admitted":
        return f"admitted {m} ago and no pod is running"
    if state == "queued":
        return f"a goal job waiting {m} for a grant"
    if not run:
        return f"no run found for it, and its GPUs were idle for {m}"
    idle = " and ".join(x for x, ok in (("processes", cpu), ("GPUs", gpus and gpu)) if ok)
    unread = " and ".join(x for x, ok in (("CPU use", not cpu), ("GPUs", gpus and not gpu)) if ok)
    return (f"no run file changed, and its {idle} were idle for {m}" if idle else f"no run file changed for {m}") \
        + (f"; its {unread} couldn't be read" if unread else "")


# --- routing --------------------------------------------------------------------------------------------------------------

def load_registry(path: Path) -> list[dict[str, Any]]:
    """The registry's handles, as routing reads them (`research slack` validates the file)."""
    return json.loads(Path(path).read_text())["handles"]


def route(owner: str, handles: list[dict[str, Any]]) -> tuple[str, str]:
    """(handle, lane) for an owner, by the module's rule."""
    for part in reversed(owner.split(":")):
        best: tuple[str, str] | None = None
        for h in handles:
            if any(part.startswith(a) for a in h.get("agents", ())):
                return h["handle"], h["handle"]
            for pat in h.get("lanes", ()):
                if (part == pat or pat.endswith("*") and part.startswith(pat[:-1])) and (not best or len(pat) > len(best[0])):
                    best = (pat, h["handle"])
        if best:
            return best[1], part
    return DEFAULT_HANDLE, DEFAULT_HANDLE


def usergroup(handle: str, handles: list[dict[str, Any]]) -> str:
    return next((h.get("usergroup") or "" for h in handles if h["handle"] == handle), "")


# --- the pure half --------------------------------------------------------------------------------------------------------

def _node1_extra(snap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per status row name (a Workload's, or a bare pod job's first pod's): its running pods with their start, its goal."""
    by: dict[str, list[dict[str, Any]]] = {}
    for it in snap["k8s"]:
        by.setdefault(it["kind"], []).append(it)
    jobs = {j["metadata"]["name"]: j for j in by.get("Job", ())}
    pods: dict[str, list[tuple[str, int]]] = {}
    for p in by.get("Pod", ()):
        if (p.get("status") or {}).get("phase") == "Running":
            pods.setdefault(nebius1._owner(p["metadata"]), []).append((p["metadata"]["name"], ledger.epoch(p["status"]["startTime"])))
    out: dict[str, dict[str, Any]] = {}
    kueued = set()
    for w in by.get("Workload", ()):
        owner = nebius1._owner(w["metadata"])
        kueued.add(owner)
        ps = w["spec"]["podSets"][0]
        labels = {**((ps["template"].get("metadata") or {}).get("labels") or {}), **(w["metadata"].get("labels") or {}),
                  **(((jobs.get(owner) or {}).get("metadata") or {}).get("labels") or {})}
        out[w["metadata"]["name"]] = {"pods": pods.get(owner, []), "goal": labels.get(GOAL_LABEL)}
    for owner, ps_ in pods.items():
        if owner not in kueued:
            out[ps_[0][0]] = {"pods": ps_, "goal": None}
    return out


def _node2_extra(snap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per status row name (an allocation id, or a waiter's handle): its lease's pid, run and goal."""
    st = ledger.state_from_dict(snap["state"])
    allocs, _ = st.on(nebius2.NODE)
    out = {a.id: {"pid": a.id.partition("@")[0].removeprefix("gl-"), "run": a.job.run, "goal": a.job.goal} for a in allocs}
    out.update({q.handle or q.job.id: {"goal": q.job.goal} for q in st.queue.values() if q.node == nebius2.NODE})
    return out


def candidates(snap: dict[str, Any], now: int, window_s: int = WINDOW_S) -> dict[str, Any]:
    """From a node's snapshot (`status.read`'s), the jobs to judge: those running or admitted for the window, and goal jobs
    queued for it; and whether node 2 is in a timed window, when its running leases aren't judged."""
    rep = status.report(snap, now)
    node = snap["node"]
    extra = _node1_extra(snap) if node == nebius1.NODE else _node2_extra(snap)
    window = bool((rep.get("ledger") or {}).get("window"))
    out = []
    for r in rep["jobs"]:
        x = extra.get(r["name"] or "", {})
        c = {"node": node, "owner": r["owner"], "job": r["key"], "name": r["name"], "state": r["state"], "gpus": r["gpus"],
             "for_s": r["for_s"], "goal": x.get("goal"), "pods": [p for p, _ in x.get("pods", ())],
             "started": {p: t for p, t in x.get("pods", ())}, "pid": x.get("pid"), "run": x.get("run")}
        if r["for_s"] is None or r["for_s"] < window_s:
            continue
        if r["state"] == "running" and not window and not (r["name"] or "").startswith(CONTROLLERS):
            out.append(c)
        elif r["state"] == "admitted" or (r["state"] == "queued" and c["goal"]):
            out.append(c)
    return {"node": node, "window": window, "stale": bool((rep.get("ledger") or {}).get("stale")), "jobs": out}


def probe_args(cands: dict[str, Any], window_s: int = WINDOW_S) -> dict[str, Any] | None:
    """What `PROBE` reads on the node for its running candidates; None when there are none."""
    running = [c for c in cands["jobs"] if c["state"] == "running"]
    if not running:
        return None
    a: dict[str, Any] = {"window_s": window_s, "quiet": QUIET, "roots": list(ROOTS)}
    if cands["node"] == nebius1.NODE:
        a.update(pods=sorted({(p, c["started"][p]) for c in running for p in c["pods"]}), prom=nebius1.PROM)
    else:
        a.update(runs=sorted({c["run"] for c in running if c["run"]}), pids=sorted({c["pid"] for c in running if c["pid"]}),
                 util_dir=UTIL_DIR, tail_bytes=max(1 << 20, window_s // 10 * 8192))
    return a


def judge(cands: dict[str, Any], progress: dict[str, Any] | None, handles: list[dict[str, Any]],
          window_s: int = WINDOW_S) -> dict[str, Any]:
    """The node's findings and the running jobs it couldn't watch, from its candidates and `PROBE`'s output (its `t` is the
    node's clock, against which its files' and GPUs' times are read)."""
    findings, unwatched = [], []
    p = progress or {"runs": {}, "pods": {}, "gpu": {}, "gpu_error": None, "t": None}

    def find(c: dict[str, Any], reason: str, run: str | None = None, quiet_s: int | None = None) -> None:
        handle, lane = route(c["owner"], handles)
        findings.append({"node": c["node"], "owner": c["owner"], "handle": handle, "usergroup": usergroup(handle, handles),
                         "lane": lane, "job": c["job"], "name": c["name"], "run": run or c["run"], "goal": c["goal"],
                         "state": c["state"], "gpus": c["gpus"], "for_s": c["for_s"], "quiet_s": quiet_s, "why": reason})

    for c in cands["jobs"]:
        if c["state"] in ("admitted", "queued"):
            find(c, why(window_s, state=c["state"]))
            continue
        runs = [c["run"]] if c["run"] else sorted({r for pod in c["pods"] for r in p["pods"].get(pod, ())})
        found = {r: p["runs"][r] for r in runs if (p["runs"].get(r) or {}).get("dir")}
        newest = max((f["newest"] for f in found.values() if f["newest"] is not None), default=None)
        cpus = [f["cpu"] for f in found.values() if f.get("cpu")]
        cpu_busy = any(x["cpu_s"] >= CPU_BUSY * max(x["over_s"], 60) for x in cpus)
        keys = c["pods"] if c["node"] == nebius1.NODE else [c["pid"]] if c["pid"] else []
        util = [p["gpu"][k] for k in keys if k in p["gpu"]]
        gpu_read = bool(c["gpus"]) and p["gpu_error"] is None and bool(util)
        if (newest is not None and newest >= p["t"] - window_s) or cpu_busy or (gpu_read and max(util) >= GPU_BUSY):
            continue
        last_run = max(found, key=lambda r: found[r]["newest"] or 0) if found else None
        quiet_s = p["t"] - newest if newest is not None else None
        if found and not all(f["complete"] for f in found.values()):
            unwatched.append({**c, "why": "its run dir has too many files to scan for a change"})
        elif found:
            find(c, why(window_s, run=True, cpu=bool(cpus), gpus=bool(c["gpus"]), gpu=gpu_read), last_run, quiet_s)
        elif gpu_read:
            find(c, why(window_s, gpus=True, gpu=True))
        else:
            unwatched.append({**c, "why": "no run found for it, and no GPU reading"
                              + (f" ({p['gpu_error']})" if c["gpus"] and p["gpu_error"] else "")})
    for f in findings:
        f["run"] = f["run"] or "-"
    return {"node": cands["node"], "window": cands["window"], "watched": len(cands["jobs"]), "findings": findings,
            "unwatched": [{k: u[k] for k in ("owner", "job", "name", "state", "gpus", "for_s", "why")} for u in unwatched]}


# --- reading the nodes ----------------------------------------------------------------------------------------------------

# Runs on the node, read-only: each run dir's newest file outside QUIET (walking until one is newer than the window), the CPU
# seconds its workload's process tree used since the window began (resources.jsonl, binary-searched to the record at the
# window's start; None when its last record is over 2 minutes old), node 1's pods' run dirs, and the GPUs' highest
# utilization in the window. `now` in its argument stands in for the node's clock.
PROBE = r'''
import json, os, re, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone
a = json.loads(sys.argv[1])
t = int(a.get("now") or time.time())
since = t - a["window_s"]
quiet = re.compile(a["quiet"])
out = {"t": t, "runs": {}, "pods": {}, "gpu": {}, "gpu_error": None}
CHUNK = 1 << 20

def stamp(r):
    return datetime.fromisoformat(r["t_utc"].replace("Z", "+00:00")).timestamp()

def procs(f, off):
    f.seek(off)
    lines = f.read(CHUNK).split(b"\n")[1 if off else 0:]
    out = []
    for line in lines:
        if b'"process"' in line:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "process" and r.get("t_utc"):
                out.append(r)
    return out

def cpu(d):
    try:
        f = open(os.path.join(d, "resources.jsonl"), "rb")
    except OSError:
        return None
    with f:
        size = f.seek(0, 2)
        tail = procs(f, max(0, size - CHUNK))
        if not tail or stamp(tail[-1]) < t - 120:
            return None
        last, lo, hi = tail[-1], 0, size
        while hi - lo > CHUNK:
            mid = (lo + hi) // 2
            rs = procs(f, mid)
            if rs and stamp(rs[0]) < since:
                lo = mid
            else:
                hi = mid
        rs = procs(f, lo) or tail
        before = [r for r in rs if stamp(r) <= since]
        first = before[-1] if before else rs[0]
    tree = lambda r: {(p.get("pid"), p.get("starttime_ticks")): p.get("cpu_s") or 0.0 for p in r.get("processes") or ()}
    base = tree(first)
    used = sum(max(0.0, v - base.get(k, 0.0)) for k, v in tree(last).items())
    return {"cpu_s": round(used, 2), "over_s": int(stamp(last) - stamp(first))}

def newest(d, cap=200000):
    best, n = None, 0
    for root, _, files in os.walk(d):
        for f in files:
            if quiet.fullmatch(f):
                continue
            n += 1
            try:
                m = int(os.lstat(os.path.join(root, f)).st_mtime)
            except OSError:
                continue
            best = m if best is None or m > best else best
            if best >= since:
                return best, True
            if n >= cap:
                return best, False
    return best, True

def created(name):
    m = re.fullmatch(r"r(\d{8}-\d{6})-[0-9a-f]+", name)
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).timestamp() if m else None

runs = set(a.get("runs", ()))
pods = dict(a.get("pods", ()))
if pods:
    first = min(pods.values()) - 120
    for root in a["roots"]:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            c = created(name)
            if c is None or c < first:
                continue
            try:
                with open(os.path.join(root, name, "events.jsonl"), "rb") as f:
                    head = f.read(8192).decode("utf-8", "replace")
            except OSError:
                continue
            for pod, start in pods.items():
                if c >= start - 120 and re.search(r'"host":\s*"%s"' % re.escape(pod), head):
                    out["pods"].setdefault(pod, []).append(name)
                    runs.add(name)
for run in sorted(runs):
    d = next((os.path.join(r, run) for r in a["roots"] if os.path.isdir(os.path.join(r, run))), None)
    n, complete = newest(d) if d else (None, True)
    out["runs"][run] = {"dir": d, "newest": n, "complete": complete, "cpu": cpu(d) if d and (n is None or n < since) else None}
try:
    if a.get("prom"):
        q = ('max by (exported_pod, pod) (max_over_time(DCGM_FI_DEV_GPU_UTIL{service="nvidia-dcgm-exporter"}[%ds]))'
             % a["window_s"])
        with urllib.request.urlopen(a["prom"] + "/api/v1/query?" + urllib.parse.urlencode({"query": q}), timeout=10) as r:
            for s in json.load(r)["data"]["result"]:
                pod = s["metric"].get("exported_pod") or s["metric"].get("pod")
                if pod:
                    out["gpu"][pod] = max(out["gpu"].get(pod, 0.0), float(s["value"][1]))
    if a.get("util_dir"):
        pids, last = set(a.get("pids", ())), None
        days = sorted({datetime.fromtimestamp(x, timezone.utc).date().isoformat() for x in (since, t)})
        for day in days:
            try:
                with open(os.path.join(a["util_dir"], day + ".jsonl"), "rb") as f:
                    f.seek(0, 2)
                    start = max(0, f.tell() - a["tail_bytes"])
                    f.seek(start)
                    lines = f.read().splitlines()[1 if start else 0:]
            except OSError:
                continue
            for line in lines:
                try:
                    rec = json.loads(line)
                    rt = rec["t"]
                    rt = datetime.fromisoformat(rt.replace("Z", "+00:00")).timestamp() if isinstance(rt, str) else float(rt)
                except (ValueError, KeyError, TypeError):
                    continue
                if rt < since or rt > t:
                    continue
                last = rt if last is None else max(last, rt)
                for g in rec.get("gpus") or ():
                    m = re.match(r"pid=(\d+) ", g.get("lease") or "")
                    if m and m.group(1) in pids and g.get("util") is not None:
                        out["gpu"][m.group(1)] = max(out["gpu"].get(m.group(1), 0.0), float(g["util"]))
        if last is None or t - last > 120:
            out["gpu_error"] = "the sampler wrote no line in the last 120 s"
except Exception as e:
    out["gpu_error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
'''


def probe_command(args: dict[str, Any]) -> str:
    return f"python3 -c {shlex.quote(PROBE)} {shlex.quote(json.dumps(args, sort_keys=True))}"


def local() -> Callable[[str], str]:
    """A runner of one shell command here, as `status.ssh`'s is there."""
    def run(cmd: str) -> str:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise status.ReadError(f"local: `{cmd[:60]}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout
    return run


def _without_bundles(run: Callable[[str], str]) -> Callable[[str], str]:
    """`status.read`'s commands less the bundle count, which the watch doesn't use and which walks every bundle."""
    return lambda cmd: "" if cmd == status.COMMANDS["bundles"] else run(cmd)


def watch(node: str, run: Callable[[str], str], now: int, handles: list[dict[str, Any]],
          window_s: int = WINDOW_S) -> dict[str, Any]:
    """One node: its snapshot, its progress reads when anything has run the window, its findings. Node 2's ledger state
    gone stale (its agent down) leaves its leases unseen, so it is an error, as an unreadable node is."""
    cands = candidates(status.read(node, _without_bundles(run), now), now, window_s)
    if cands["stale"]:
        raise status.ReadError(f"{node}'s ledger state is stale (is the agent running?), so its leases can't be watched")
    args = probe_args(cands, window_s)
    progress = json.loads(run(probe_command(args))) if args else None
    return judge(cands, progress, handles, window_s)


def render(doc: dict[str, Any]) -> str:
    out = []
    for n in doc["nodes"]:
        if "error" in n:
            out.append(f"{n['node']}: could not watch: {n['error']}")
            continue
        note = " (a timed window: its leases aren't read)" if n["window"] else ""
        out.append(f"{n['node']}: {len(n['findings']) or 'none'} stalled of {n['watched']} watched{note}")
        for f in n["findings"]:
            what = ", ".join(x for x in (f"run {f['run']}" if f["run"] != "-" else "", f"goal {f['goal']}" if f["goal"] else "",
                                         f"{f['gpus']} GPU" + ("s" if f["gpus"] != 1 else ""), f"{f['state']} {status.span(f['for_s'])}",
                                         f"quiet {status.span(f['quiet_s'])}" if f["quiet_s"] is not None else "") if x)
            out.append(f"  STALLED @{f['handle']} {f['job']} (owner {f['owner']}; {what}): {f['why']}")
        for u in n["unwatched"]:
            out.append(f"  unwatched {u['job']} (owner {u['owner']}; {u['state']} {status.span(u['for_s'])}): {u['why']}")
    return "\n".join(out) + "\n"


clock = time.time


def write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(nodes: list[str], as_json: bool, *, window_s: int = WINDOW_S, registry: Path = REGISTRY, out: Path | None = None,
         runner: Callable[[str], Callable[[str], str]] | None = None, key: Path | None = status.SSH_KEY) -> int:
    """Watch `nodes` at once and print their findings (`--out`: also the JSON the exporter reads, written whole); 1 when a
    node or the registry couldn't be read."""
    now = int(clock())
    runner = runner or (lambda node: status.ssh(status.HOSTS[node], key))
    doc: dict[str, Any] = {"schema": SCHEMA, "t": now, "window_s": window_s, "nodes": []}
    try:
        handles = load_registry(registry)
    except (OSError, ValueError, KeyError) as e:
        doc["error"] = f"registry {registry}: {type(e).__name__}: {e}"
        handles = None
    if handles is not None:
        def one(node: str) -> dict[str, Any]:
            try:
                return watch(node, runner(node), now, handles, window_s)
            except (status.ReadError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError, TypeError) as e:
                return {"node": node, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(len(nodes)) as ex:
            doc["nodes"] = list(ex.map(one, nodes))
    if out is not None:
        write(out, doc)
    if as_json:
        print(json.dumps(doc, indent=1, sort_keys=True))
    else:
        print((f"could not watch: {doc['error']}\n" if "error" in doc else "") + (render(doc) if doc["nodes"] else ""), end="")
    return 1 if "error" in doc or any("error" in n for n in doc["nodes"]) else 0


__all__ = ["WINDOW_S", "GPU_BUSY", "CPU_BUSY", "ROOTS", "QUIET", "REGISTRY", "PROBE", "why", "load_registry", "route", "usergroup",
           "candidates", "probe_args", "judge", "probe_command", "local", "watch", "render", "main"]
