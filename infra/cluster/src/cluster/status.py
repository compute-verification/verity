"""`cluster status`: each node's running, queued and held work by owner and key, its GPU-hours ready, its holds, its disk,
and node 1's unreplayed Commit bundles. Read-only: every command it sends a node (`COMMANDS`) only reads, through
`kubectl get`, `df`, `find` and `du`, or `ls` and `cat`.

`read` takes a node's snapshot over SSH (`HOSTS`, as `machines.d/<node>.toml` registers them); `node1` and `node2` are the
pure half, snapshot and `now` in, report out, and `render` prints reports as text.

**Node 1** is Kueue's (`nebius1`'s mapping, from the same `kubectl get -o json`, plus the Jobs and queues). A Workload
that isn't finished is one job: held when `spec.active` is false, running once Kueue reserved its quota (`admitted` until
its pod runs), queued otherwise. Its key is the dispatcher's item (the Job's `verity.dev/item`, `<workstream>/<id>`), its
phase the task in the Job's name (`nd-<workstream>-<hash>-<task>-<try>`), its batch the `__b<N>__` in its `ROW`. Its owner
is the `verity.dev/owner` label (on the Job, the Workload or its pods) where there is one, else its workstream. A running
pod no Workload admitted (SkyPilot's controller) is a running job too. A job's length is its `activeDeadlineSeconds`,
which the dispatcher's Jobs don't set, so their GPU-hours are unknown. Holds are queues (ClusterQueues and LocalQueues)
whose `stopPolicy` isn't `None`, and the held Workloads.

**Unreplayed Commit bundles** are counted by the steward's pacer's rule (`~research/commit-release/release.py`,
`bundles_gb`): the apparent size (`du -sb`) of every `replay_bundle_p*` directory under `jobs/cov` and `jobs/probe-jit`,
in 10^9 bytes. A replay deletes its bundle once it passes, so every bundle still there is unreplayed (or its Commit
failed, or it is being written). Copies inside runs' outputs (`jobs/runs/*/outputs/`) are not counted, as the pacer
doesn't count them.

**Node 2** is its agent's ledger state (`state.json`, `ledger.state`): leases are running jobs and waiters queued, owned by
their submitter and keyed by their run (else the lease's job id). A job's length is the model's (`Job.length_s`: its
`max_s`, else a session's `expected_s`). `gpu-lease` doesn't manage CPUs, so a lease's CPUs are 0. Node 2 has no Kueue,
so no holds. The agent writes no state during a timed window (`agent.py`), so the state is stale only when it is older
than `STALE_S` outside one, or than the window's end plus `STALE_S`.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import ledger, model, nebius1, nebius2

NODES = {"nebius-1": nebius1.NODE, "nebius-2": nebius2.NODE}
HOSTS = {nebius1.NODE: "research@81.85.2.165", nebius2.NODE: "research@81.85.2.121"}
SSH_KEY = Path.home() / ".ssh" / "research_key"
LIVE = "/workspace/pouw/infra/cluster/live"
BUNDLE_ROOTS = ("/workspace/jobs/cov", "/workspace/jobs/probe-jit")
VY_SUDO = "/usr/local/sbin/vy-sudo"
STALE_S = 60

COMMANDS = {
    "kueue": shlex.join([*nebius1.KUBECTL, "get", "pods,workloads,jobs,clusterqueues,localqueues", "-n", "default", "-o", "json"]),
    "df": "df -P /workspace",
    # release.py's bundles_gb, one line per bundle; `sudo -n true` first, so a refused sudo is an error and not 0 GB. vy-sudo's
    # du-bundles is the same find, the form the node's sudoers allows once research has no blanket root (pods/nebius/lockout)
    "bundles": f"sudo -n true && {{ if [ -x {VY_SUDO} ]; then sudo -n {VY_SUDO} du-bundles 2>/dev/null; else"
               f" sudo -n find {' '.join(BUNDLE_ROOTS)} -xdev -name 'replay_bundle_p*' -prune -print0 2>/dev/null"
               " | sudo -n xargs -0r du -sb 2>/dev/null; fi; true; }",
    # the agent's state: <live>/state.json with --roll, else its run's <live>/<stamp>/state.json; the newest
    "state": f'cd {LIVE} && f=$(ls -t state.json */state.json 2>/dev/null | head -1) && test -n "$f" && echo "{LIVE}/$f"'
             ' && cat "$f"',
}
READS = {nebius1.NODE: ("kueue", "df", "bundles"), nebius2.NODE: ("state", "df")}


class ReadError(RuntimeError):
    pass


def ssh(dest: str, key: Path | None = SSH_KEY) -> Callable[[str], str]:
    """A runner of one shell command on `dest`, returning its stdout."""
    def run(cmd: str) -> str:
        argv = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                *(["-i", str(key)] if key and key.exists() else []), dest, cmd]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        if r.returncode:
            raise ReadError(f"{dest}: `{cmd[:60]}` exited {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout
    return run


def read(node: str, run: Callable[[str], str], t: int) -> dict[str, Any]:
    """The node's snapshot at `t`: each of its `READS`' raw output, the commands run at once."""
    with ThreadPoolExecutor(len(READS[node])) as ex:
        out = {k: ex.submit(run, COMMANDS[k]) for k in READS[node]}
        raw = {k: f.result() for k, f in out.items()}
    snap: dict[str, Any] = {"node": node, "t": t, "df": raw["df"]}
    if "kueue" in raw:
        snap["k8s"] = json.loads(raw["kueue"])["items"]
        snap["bundles"] = raw["bundles"]
    if "state" in raw:
        path, _, body = raw["state"].partition("\n")
        snap["state_path"], snap["state"] = path, json.loads(body)
    return snap


def disk_pct(df: str) -> int:
    """`df -P`'s Capacity of its one filesystem: used / (used + available), rounded up, as `df` reports it."""
    return int(df.strip().splitlines()[-1].split()[4].rstrip("%"))


def bundles(du: str) -> dict[str, Any]:
    """`du -sb` lines (bytes, tab, path) as release.py's total in GB (10^9 bytes), with each bundle."""
    items = [{"path": p, "bytes": int(b)} for b, _, p in (line.partition("\t") for line in du.splitlines() if line.strip())]
    return {"rule": "release.py bundles_gb", "roots": list(BUNDLE_ROOTS), "count": len(items),
            "gb": round(sum(i["bytes"] for i in items) / 1e9, 1), "items": sorted(items, key=lambda i: -i["bytes"])}


def batch(row: str) -> int | None:
    m = re.search(r"__b(\d+)__", row)
    return int(m.group(1)) if m else None


TASK = re.compile(r"-[0-9a-f]{10}-(?P<task>.+)-\d+$")


def _conds(w: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["type"]: c for c in (w.get("status") or {}).get("conditions") or ()}


def _at(c: dict[str, Any] | None) -> int | None:
    return ledger.epoch(c["lastTransitionTime"]) if c and c.get("lastTransitionTime") else None


def _row(owner: str, key: str, **f: Any) -> dict[str, Any]:
    return {"owner": owner, "key": key, "phase": None, "state": None, "queue": None, "cls": None, "gpus": 0, "cpus": 0,
            "memory_gib": 0, "since": None, "for_s": None, "length_s": None, "length_is": None, "batch": None, "row": None,
            "name": None, "owner_label": None, **f}


def node1(snap: dict[str, Any], now: int, owner: str | None = None) -> dict[str, Any]:
    """Node 1's report from a snapshot of its Kueue objects, `df` and bundles (the module's mapping)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for it in snap["k8s"]:
        by.setdefault(it["kind"], []).append(it)
    jobs = {j["metadata"]["name"]: j for j in by.get("Job", ())}
    pods: dict[str, list[dict[str, Any]]] = {}
    for p in by.get("Pod", ()):
        if (p.get("status") or {}).get("phase") == "Running":
            pods.setdefault(nebius1._owner(p["metadata"]), []).append(p)
    rows, held, kueued = [], [], set()
    for w in by.get("Workload", ()):
        conds = _conds(w)
        if (conds.get("Finished") or {}).get("status") == "True":
            continue
        name = nebius1._owner(w["metadata"])
        kueued.add(name)
        job = jobs.get(name) or {}
        jmeta = job.get("metadata") or {}
        ps = w["spec"]["podSets"][0]
        tspec = ps["template"]["spec"]
        labels = {**((ps["template"].get("metadata") or {}).get("labels") or {}), **(w["metadata"].get("labels") or {}),
                  **(jmeta.get("labels") or {})}
        item = (jmeta.get("annotations") or {}).get("verity.dev/item", "")
        req = nebius1._requests(tspec)
        n = int(ps.get("count", 1))
        env = {e["name"]: e.get("value") or "" for c in tspec.get("containers") or () for e in c.get("env") or ()}
        task = TASK.search(name)
        length = tspec.get("activeDeadlineSeconds") or (job.get("spec") or {}).get("activeDeadlineSeconds")
        r = _row(labels.get("verity.dev/owner") or labels.get("verity.dev/workstream") or item.partition("/")[0] or "-",
                 item or name, phase=task.group("task") if task else None, queue=w["spec"].get("queueName"),
                 cls=nebius1._prio(w["spec"]) or labels.get("kueue.x-k8s.io/priority-class"),
                 gpus=n * int(req.get("nvidia.com/gpu", 0)), cpus=n * nebius1._cpus(req.get("cpu")),
                 memory_gib=n * nebius1._gib(req.get("memory")), length_s=length, length_is="max" if length else None,
                 row=env.get("ROW") or None, batch=batch(env.get("ROW", "")), name=w["metadata"]["name"],
                 owner_label=labels.get("verity.dev/owner"))
        created = ledger.epoch(w["metadata"]["creationTimestamp"])
        if not w["spec"].get("active", True):
            ev = conds.get("Evicted")
            r.update(state="held", since=_at(ev) if ev and ev.get("reason") == "Deactivated" else created)
            held.append(r)
            continue
        if (conds.get("QuotaReserved") or {}).get("status") == "True":
            running = pods.get(name)
            r.update(state="running" if running else "admitted",
                     since=min(ledger.epoch(p["status"]["startTime"]) for p in running) if running
                     else _at(conds["QuotaReserved"]) or created)
        else:
            rq = conds.get("Requeued")
            r.update(state="queued", since=max(created, _at(rq) or 0) if rq and rq.get("status") == "True" else created)
        rows.append(r)
    for name, ps_ in sorted(pods.items()):
        if name in kueued:
            continue
        labels = ps_[0]["metadata"].get("labels") or {}
        reqs = [nebius1._requests(p["spec"]) for p in ps_]
        rows.append(_row(labels.get("verity.dev/owner") or labels.get("verity.dev/workstream")
                         or ("skypilot" if labels.get("skypilot-cluster") else "-"), name, state="running",
                         gpus=sum(int(q.get("nvidia.com/gpu", 0)) for q in reqs), cpus=sum(nebius1._cpus(q.get("cpu")) for q in reqs),
                         memory_gib=sum(nebius1._gib(q.get("memory")) for q in reqs),
                         since=min(ledger.epoch(p["status"]["startTime"]) for p in ps_), name=ps_[0]["metadata"]["name"],
                         owner_label=labels.get("verity.dev/owner")))
    queues = [{"kind": q["kind"], "name": q["metadata"]["name"], "stop_policy": q["spec"]["stopPolicy"]}
              for kind in ("ClusterQueue", "LocalQueue") for q in by.get(kind, ())
              if (q.get("spec") or {}).get("stopPolicy") not in (None, "None")]
    return _report(snap, now, owner, rows, held, queues, bundles(snap["bundles"]))


def node2(snap: dict[str, Any], now: int, owner: str | None = None) -> dict[str, Any]:
    """Node 2's report from its agent's `state.json` and `df` (the module's mapping)."""
    st = ledger.state_from_dict(snap["state"])
    allocs, _ = st.on(nebius2.NODE)

    def length(j: model.Job) -> dict[str, Any]:
        return {"length_s": j.length_s, "length_is": "max" if j.max_s is not None else "expected" if j.length_s else None}

    rows = []
    for a in allocs:
        j = a.job
        rows.append(_row(j.submitter, j.run or j.id, state="frozen" if a.state == "frozen" else "running", cls=j.priority,
                         gpus=len(a.gpus), cpus=len(a.cpus) or j.cpus, memory_gib=a.memory_gib, since=a.started,
                         name=a.id, **length(j)))
    for q in st.queue.values():
        if q.node != nebius2.NODE:
            continue
        j = q.job
        rows.append(_row(j.submitter, j.run or j.id, state="queued", cls=j.priority, gpus=j.gpus, cpus=j.cpus,
                         memory_gib=j.memory_gib, since=q.since, name=q.handle or j.id, **length(j)))
    rep = _report(snap, now, owner, rows, [], [], None)
    window = [a for a in allocs if a.job.quiet]
    ends = [a.ends_by for a in window]
    age = now - st.t if st.t is not None else None
    held_back = bool(window) and (None in ends or now <= max(ends) + STALE_S)
    rep["ledger"] = {"path": snap.get("state_path"), "seq": st.seq, "t": st.t, "age_s": age, "window": bool(window),
                     "stale": age is None or (age > STALE_S and not held_back)}
    return rep


ORDER = {"running": 0, "admitted": 1, "frozen": 2, "queued": 3, "held": 4}


def _report(snap, now, owner, rows, held, queues, bund) -> dict[str, Any]:
    if owner is not None:
        rows, held = [r for r in rows if r["owner"] == owner], [r for r in held if r["owner"] == owner]
    for r in rows + held:
        r["for_s"] = now - r["since"] if r["since"] is not None else None
    rows.sort(key=lambda r: (r["owner"], r["key"], ORDER[r["state"]], r["phase"] or "", r["name"] or ""))
    held.sort(key=lambda r: (r["owner"], r["key"], r["name"] or ""))
    queued = [r for r in rows if r["state"] == "queued"]
    known = [r for r in queued if r["length_s"]]
    return {"node": snap["node"], "t": now, "snapshot_t": snap["t"], "owner": owner, "jobs": rows,
            "ready": {"jobs": len(queued), "gpus": sum(r["gpus"] for r in queued),
                      "gpu_jobs": sum(1 for r in queued if r["gpus"]), "known": sum(1 for r in known if r["gpus"]),
                      "gpu_h": round(sum(r["gpus"] * r["length_s"] for r in known) / 3600, 2)},
            "holds": {"queues": queues, "workloads": held},
            "disk": {"path": "/workspace", "use_pct": disk_pct(snap["df"])},
            **({"bundles": bund} if bund is not None else {})}


def report(snap: dict[str, Any], now: int, owner: str | None = None) -> dict[str, Any]:
    return (node1 if snap["node"] == nebius1.NODE else node2)(snap, now, owner)


def span(s: int | None) -> str:
    if s is None:
        return "-"
    s = max(0, s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m" if s >= 60 else f"{s}s"


def _table(rows: list[dict[str, Any]], indent: str = "  ") -> list[str]:
    cols = ("STATE", "OWNER", "KEY", "PHASE", "MODEL", "B", "GPU", "CPU", "FOR")
    cells = [cols] + [(r["state"], r["owner"], r["key"].removeprefix(f"{r['owner']}/"), r["phase"] or "-",
                       (r["row"] or "").split("__")[0] or "-", "-" if r["batch"] is None else str(r["batch"]),
                       str(r["gpus"]), str(r["cpus"] or "-"), span(r["for_s"])) for r in rows]
    keep = [i for i in range(len(cols)) if cols[i] in ("GPU", "CPU") or any(c[i] != "-" for c in cells[1:])]
    w = {i: max(len(c[i]) for c in cells) for i in keep}
    return [indent + "  ".join(c[i].ljust(w[i]) for i in keep).rstrip() for c in cells]


def render(reports: list[dict[str, Any]]) -> str:
    """The reports as text: per node a header (disk, bundles, ready, holds), then its jobs by owner and key, held last."""
    out = []
    for rep in reports:
        if "error" in rep:
            out += [f"{rep['node']}: could not read: {rep['error']}", ""]
            continue
        when = datetime.fromtimestamp(rep["t"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(f"{rep['node']} at {when}" + (f", owner {rep['owner']}" if rep["owner"] else ""))
        out.append(f"  disk: {rep['disk']['path']} {rep['disk']['use_pct']}% used (df)")
        if "bundles" in rep:
            b = rep["bundles"]
            out.append(f"  unreplayed Commit bundles: {b['gb']:.1f} GB in {b['count']} (release.py's rule: "
                       f"{', '.join(p.removeprefix('/workspace/') for p in b['roots'])})")
        if "ledger" in rep:
            lg = rep["ledger"]
            note = (" (STALE: is the agent running?)" if lg["stale"]
                    else " (a timed window: the agent writes no state until it ends)" if lg["window"] else "")
            out.append(f"  ledger state: seq {lg['seq']}, {span(lg['age_s'])} old{note} ({lg['path']})")
        rd = rep["ready"]
        gh = ("" if not rd["gpu_jobs"] else f"; {rd['gpu_h']:.2f} GPU-h" if rd["known"] == rd["gpu_jobs"]
              else f"; GPU-h unknown (none of {rd['gpu_jobs']} GPU jobs declares a length)" if not rd["known"]
              else f"; at least {rd['gpu_h']:.2f} GPU-h ({rd['known']} of {rd['gpu_jobs']} GPU jobs declare a length)")
        out.append(f"  ready: {rd['jobs']} queued, {rd['gpus']} GPUs asked{gh}")
        qs = rep["holds"]["queues"]
        hw = rep["holds"]["workloads"]
        if rep["node"] == nebius2.NODE:
            out.append("  holds: none (no Kueue on node 2)")
        else:
            out.append("  holds: " + (", ".join(f"{q['kind']} {q['name']} stopPolicy={q['stop_policy']}" for q in qs)
                                      or "no queue stopped")
                       + f"; {len(hw)} Workloads deactivated (spec.active=false)"
                       + (f", {sum(1 for r in hw if r['owner_label'])} with a verity.dev/owner label" if hw else ""))
        rows = rep["jobs"] + hw
        out += _table(rows) if rows else ["  no jobs"]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


clock = time.time


def main(nodes: list[str], owner: str | None, as_json: bool, key: Path | None = SSH_KEY,
         runner: Callable[[str], Callable[[str], str]] | None = None) -> int:
    """Read `nodes` at once, print their reports; 1 when a node couldn't be read."""
    now = int(clock())
    runner = runner or (lambda node: ssh(HOSTS[node], key))

    def one(node: str) -> dict[str, Any]:
        try:
            return report(read(node, runner(node), now), now, owner)
        except (ReadError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as e:
            return {"node": node, "t": now, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(len(nodes)) as ex:
        reps = list(ex.map(one, nodes))
    if as_json:
        print(json.dumps({"schema": "cluster/status/v1", "t": now, "owner": owner, "nodes": reps}, indent=1, sort_keys=True))
    else:
        print(render(reps), end="")
    return 1 if any("error" in r for r in reps) else 0


__all__ = ["NODES", "HOSTS", "COMMANDS", "READS", "BUNDLE_ROOTS", "ReadError", "ssh", "read", "disk_pct", "bundles", "batch",
           "node1", "node2", "report", "render", "span", "main"]
