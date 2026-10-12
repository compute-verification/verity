"""vy-nebius-1's Kueue as the model sees it: the adapter between the node's Kubernetes state and the planner.

Node 1 runs its work as pods that Kueue admits (SkyPilot's head pods and the dispatcher's `batch/v1` Jobs). `read` takes a
snapshot without changing anything: the pods and Workloads through `kubectl get -o json`, and which GPU each pod holds from
DCGM's per-GPU series, whose `pod` label (`exported_pod` without honor_labels) is the device plugin's assignment.
`from_k8s` is the pure half, which the tests drive.

The mapping: a running pod is an allocation of the GPUs it requests and no pinned CPUs (Kueue doesn't pin; the dispatcher's
tasks all share `taskset` 96-127), so the planner doesn't yet account node 1's CPU work. A GPU job is CPU-unmanaged
(`cpus = 0`); a CPU job declares its request, rounded up. Job id `k8s-<owner>` (the Job, or the SkyPilot cluster for a bare
pod), allocation id `k8s-<owner>@<start>`. A pod or Workload with `activeDeadlineSeconds` is a batch job with that `max_s`;
without it, a session expected to take an hour. A pending Workload is a queued job under the same id, so a start takes it out of the
queue. The Kueue priority class maps onto the description's priority (`PRIORITY`); its queue, when it is one of the
description's workstreams (`provers`, `deployments-gpu`), is the job's workstream. `provers` never evicts (root 14:34Z);
anything else Kueue may reclaim is `requeue`. `pous-overflow` is PoUW's.

DCGM labels a GPU with its pod when the device plugin assigned it, and not always at once. A GPU-holding pod that no GPU
names gets the unnamed busy GPUs in index order when their count matches (the planner needs which GPUs are held, not
by whom); otherwise those GPUs are `unowned`: held, by nothing the planner may move.
"""
from __future__ import annotations

import json
import math
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .ledger import epoch
from .model import Allocation, Cluster, Job

NODE = "vy-nebius-1"
PROM = "http://10.43.136.178"
KUBECTL = ("sudo", "-n", "k3s", "kubectl")
GPU_SERIES = 'DCGM_FI_DEV_FB_USED{service="nvidia-dcgm-exporter"}'

# kueue.yaml's WorkloadPriorityClasses onto descriptions/nebius.toml's [priorities]
PRIORITY = {"capture": "capture", "sweep-night": "work", "circuits-gpu": "work", "circuits": "work",
            "prover-bench": "bench", "dev": "dev", "backfill": "fill"}
NEVER_EVICTED = frozenset({"provers"})
PROJECT_OF_QUEUE = {"pous-overflow": "pous"}
EXPECTED_S = 60 * 60


@dataclass(frozen=True)
class Observation:
    """The node at `t`: its pods and Workloads (`kubectl get -o json` items), and each GPU index's DCGM pod label ("" when
    DCGM names no pod) with its framebuffer use in MiB."""
    t: int
    pods: tuple[dict[str, Any], ...]
    workloads: tuple[dict[str, Any], ...]
    gpus: dict[int, tuple[str, int]]


def _gib(q: str | None) -> int:
    """A Kubernetes memory quantity in whole GiB, at least 1."""
    if not q:
        return 1
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
    for u in sorted(units, key=len, reverse=True):
        if q.endswith(u):
            return max(1, round(float(q[:-len(u)]) * units[u] / 2**30))
    return max(1, round(float(q) / 2**30))


def _owner(meta: dict[str, Any]) -> str:
    """What Kueue admits: a pod's Job, or the pod itself (SkyPilot's bare pods); for a Workload, the object it owns."""
    refs = [r for r in meta.get("ownerReferences") or () if r.get("kind") in ("Job", "Pod")]
    return refs[0]["name"] if refs else meta["name"]


def _cpus(q: str | None) -> int:
    if not q:
        return 1
    return max(1, -(-int(q[:-1]) // 1000) if q.endswith("m") else math.ceil(float(q)))


def _job(cluster: Cluster, name: str, labels: dict[str, str], requests: dict[str, str], queue: str, prio_class: str,
         submitted: int, gpu_ids: tuple[int, ...], node: str, deadline_s: int | None) -> Job:
    ws = {w.name for w in cluster.workstreams if w.node == node}
    gpus = int(requests.get("nvidia.com/gpu", 0))
    return Job(id=f"k8s-{name}", project=PROJECT_OF_QUEUE.get(queue, cluster.nodes[node].owner),
               submitter=labels.get("verity.dev/workstream") or ("skypilot" if labels.get("skypilot-cluster") else "-"),
               priority=PRIORITY.get(prio_class, "work"), kind="batch" if deadline_s else "session", gpus=gpus,
               cpus=0 if gpus else _cpus(requests.get("cpu")), memory_gib=_gib(requests.get("memory")),
               max_s=deadline_s or None, expected_s=None if deadline_s else EXPECTED_S,
               preempt="never" if queue in NEVER_EVICTED else "requeue",
               node=node, gpu_ids=gpu_ids, workstream=queue if queue in ws else None, submitted=submitted)


def _prio(spec: dict[str, Any]) -> str:
    """A Workload's priority class: `priorityClassRef.name` in v1beta2, `priorityClassName` before."""
    return (spec.get("priorityClassRef") or {}).get("name") or spec.get("priorityClassName", "")


def _requests(spec: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in spec.get("containers") or ():
        for k, v in ((c.get("resources") or {}).get("requests") or {}).items():
            out[k] = str(int(out.get(k, 0)) + int(v)) if k == "nvidia.com/gpu" else out.get(k, v)
    return out


def from_k8s(cluster: Cluster, obs: Observation, node: str = NODE) -> tuple[list[Allocation], list[Job]]:
    """The node's allocations (running pods on it) and queue (Workloads without quota), by the module's mapping."""
    running = [p for p in obs.pods if (p.get("status") or {}).get("phase") == "Running"]
    live = {p["metadata"]["name"] for p in running}
    named: dict[str, list[int]] = {}
    for i, (pod, _) in sorted(obs.gpus.items()):
        if pod in live:
            named.setdefault(pod, []).append(i)
    unnamed = [i for i, (pod, used) in sorted(obs.gpus.items()) if pod not in live and used > 0]
    wanting = [p for p in running if int(_requests(p["spec"]).get("nvidia.com/gpu", 0)) > 0 and p["metadata"]["name"] not in named]
    if sum(int(_requests(p["spec"])["nvidia.com/gpu"]) for p in wanting) == len(unnamed):
        for p in wanting:
            n = int(_requests(p["spec"])["nvidia.com/gpu"])
            named[p["metadata"]["name"]], unnamed = unnamed[:n], unnamed[n:]
    admitted = {_owner(w["metadata"]): w["spec"] for w in sorted(obs.workloads, key=lambda w: w["metadata"]["creationTimestamp"])}
    allocs: list[Allocation] = []
    for p in running:
        meta, labels = p["metadata"], p["metadata"].get("labels") or {}
        req = _requests(p["spec"])
        gpus = tuple(named.get(meta["name"], ()))
        if int(req.get("nvidia.com/gpu", 0)) != len(gpus):
            continue
        started = epoch(p["status"]["startTime"])
        ws = admitted.get(_owner(meta), {})
        job = _job(cluster, _owner(meta), labels, req, ws.get("queueName") or labels.get("kueue.x-k8s.io/local-queue-name", ""),
                   _prio(ws) or labels.get("kueue.x-k8s.io/priority-class", ""), started, gpus, node,
                   p["spec"].get("activeDeadlineSeconds"))
        allocs.append(Allocation(id=f"{job.id}@{started}", job=job, node=node, gpus=gpus, cpus=frozenset(),
                                 memory_gib=job.memory_gib, started=started))
    if unnamed:
        allocs.append(unowned(cluster, tuple(unnamed), node))
    queue: list[Job] = []
    for w in obs.workloads:
        conds = {c["type"]: c["status"] for c in (w.get("status") or {}).get("conditions") or ()}
        if conds.get("QuotaReserved") == "True" or conds.get("Finished") == "True" or not (w.get("spec") or {}).get("active", True):
            continue
        ps = w["spec"]["podSets"][0]["template"]
        name = _owner(w["metadata"])
        queue.append(_job(cluster, name, {**(ps.get("metadata") or {}).get("labels", {}), **(w["metadata"].get("labels") or {})},
                          _requests(ps["spec"]), w["spec"].get("queueName", ""), _prio(w["spec"]),
                          epoch(w["metadata"]["creationTimestamp"]), (), node, ps["spec"].get("activeDeadlineSeconds")))
    return allocs, queue


def unowned(cluster: Cluster, gpus: tuple[int, ...], node: str = NODE) -> Allocation:
    """GPUs in use that neither DCGM nor the pods' requests account for: held, by nothing the planner may move."""
    job = Job(id=f"k8s-unowned-{'-'.join(map(str, gpus))}", project=cluster.nodes[node].owner, submitter="-",
              priority="session", kind="session", gpus=len(gpus), cpus=0, expected_s=60, preempt="never", node=node,
              gpu_ids=gpus)
    return Allocation(id=f"{job.id}@0", job=job, node=node, gpus=gpus, cpus=frozenset(), memory_gib=1, started=0)


def read(t: int, kubectl: tuple[str, ...] = KUBECTL, prom: str = PROM) -> Observation:
    """The live node, read-only: `kubectl get pods,workloads -o json` in the default namespace and DCGM's framebuffer series."""
    def items(kind: str) -> tuple[dict[str, Any], ...]:
        out = subprocess.run([*kubectl, "get", kind, "-n", "default", "-o", "json"], capture_output=True, text=True, check=True)
        return tuple(json.loads(out.stdout)["items"])
    url = f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": GPU_SERIES})
    with urllib.request.urlopen(url, timeout=10) as r:
        series = json.load(r)["data"]["result"]
    gpus = {int(s["metric"]["gpu"]): (s["metric"].get("exported_pod") or s["metric"].get("pod", ""), int(float(s["value"][1])))
            for s in series}
    return Observation(t, items("pods"), items("workloads"), gpus)


def save(obs: Observation) -> dict[str, Any]:
    return {"t": obs.t, "pods": list(obs.pods), "workloads": list(obs.workloads), "gpus": {str(i): list(v) for i, v in obs.gpus.items()}}


def load(d: dict[str, Any]) -> Observation:
    return Observation(d["t"], tuple(d["pods"]), tuple(d["workloads"]), {int(i): (v[0], v[1]) for i, v in d["gpus"].items()})


def main(argv: list[str] | None = None) -> int:
    """`python -m cluster.nebius1 DESCRIPTION [--save SNAPSHOT.json | --from SNAPSHOT.json]`: one JSON line per GPU
    allocation, a count of the rest, and one per planned action, as the planner would decide beside Kueue."""
    import argparse
    import sys
    import time
    from dataclasses import asdict

    from . import model
    from .plan import plan
    ap = argparse.ArgumentParser(prog="cluster.nebius1")
    ap.add_argument("description")
    ap.add_argument("--save")
    ap.add_argument("--from", dest="source")
    ap.add_argument("--report", action="store_true", help="one JSON object: the node's allocations and queue as model dicts")
    a = ap.parse_args(argv)
    cluster = model.load(a.description)
    obs = load(json.loads(open(a.source).read())) if a.source else read(int(time.time()))
    if a.save:
        with open(a.save, "w") as f:
            json.dump(save(obs), f)
    allocs, queue = from_k8s(cluster, obs)
    out = sys.stdout
    if a.report:
        out.write(json.dumps({"t": obs.t, "node": NODE, "allocations": [model.allocation_to_dict(al) for al in allocs],
                              "queue": [model.job_to_dict(j) for j in queue]}) + "\n")
        return 0
    for al in allocs:
        if al.gpus:
            out.write(json.dumps({"alloc": al.id, "gpus": al.gpus, "priority": al.job.priority, "workstream": al.job.workstream,
                                  "preempt": al.job.preempt, "memory_gib": al.memory_gib}) + "\n")
    out.write(json.dumps({"t": obs.t, "cpu_allocations": sum(1 for al in allocs if not al.gpus), "queued": len(queue),
                          "memory_gib_requested": sum(al.memory_gib for al in allocs)}) + "\n")
    for act in plan(cluster, NODE, allocs, queue, obs.t):
        d = {"action": type(act).__name__, **asdict(act)}
        if "alloc" in d and isinstance(d["alloc"], dict):
            d["alloc"] = {"id": act.alloc.id, "gpus": act.alloc.gpus, "cpus": len(act.alloc.cpus)}
        out.write(json.dumps(d, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["NODE", "PRIORITY", "Observation", "from_k8s", "unowned", "read", "save", "load", "main"]
