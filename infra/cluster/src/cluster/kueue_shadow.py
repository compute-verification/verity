"""vy-nebius-1 with Kueue as its only scheduler, in shadow: what Kueue alone would admit, beside what the node did.

Four things decide what runs on node 1 today: Kueue (the dispatcher's Jobs, SkyPilot's pods, the lease pools' holders), the
lease pools (`n1_lease.py`: which `gpu-lease` waiter gets a pool GPU, under its own `max`, `borrow` and `borrow-until`), the
merge-check slots (`tools/verity/check/slot.py`) and the Lean slots (`tools/verity/check/lean_slot.py`). Each tick the shadow mirrors the
demand of all four as bare Kueue Workloads, which Kueue admits against quota but which run no pods, and records where Kueue's
admission differs from what the node did. Kueue is asked, not simulated, so the comparison is with the real admission logic
(priority, FIFO, borrowing, preemption) at the node's own quotas.

**What it writes.** Only objects labelled `vy.dev/shadow=true`: namespace `vy-shadow`, its LocalQueues, ClusterQueues named
`shadow-*` in cohort `vy-shadow` (so they neither borrow from nor lend to the node's cohort `nebius`), two ResourceFlavors
for the check slots, and the Workloads. `cluster kueue-shadow down` deletes all of them. It reads Kueue, the lease
directories and the slot files and never changes any of them.

**The queues** (`queues`): each of the node's ClusterQueues as it is now (quotas, preemption, stop policy) as `shadow-<name>`;
`shadow-checks`, whose resource `vy.dev/check-slot` counts the check slots, the `only=trains` ones in a flavor tainted
`vy.dev/trains` that only a `--priority` (train) check tolerates; and `shadow-lean`, one resource `vy.dev/lean-<pool>` per
Lean pool, as many as the pool's N.

**The demand** (`demand`), one `Item` per allocation or waiter, its key stable from waiting to running:
- a Kueue Workload in `default` that isn't finished, under its own podSets and priority, except the lease pools' holders
  (their pod template's `verity.dev/gpu-pool` label), whose GPUs are the leases below;
- a `gpu-lease` lease or waiter of a pool (`POOLS`): its GPUs, on the pool's ClusterQueue at the pool's priority class.
  Fences (`who=kubernetes`) are not leases;
- a check slot's holder or waiter: one `vy.dev/check-slot`, a train check's at priority 1;
- a Lean slot's holder or waiter: one `vy.dev/lean-<pool>`.

**The comparison** (`compare`), on two ticks in a row: an item the node runs that Kueue holds pending is `node-ran` (the node let it past Kueue's
quota, or Kueue would have preempted it); an item the node keeps waiting that Kueue admitted is `kueue-ran` (a rule outside
Kueue held it back: a pool's `max` or expired `borrow-until`, a slot reserved for trains). Each tick appends one record to
`--out` (`cluster/kueue-shadow/v1`): the time, the item and divergence counts, and each divergence's key and reason.

Not covered yet, and needed before Kueue can be the only scheduler: `research run --on vy-nebius-1` direct runs on
`/etc/vy/direct-cpus`, `research run --queue` jobs' systemd scopes, and slot windows (node 1's slots have none).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import nebius1, nebius2

SCHEMA = "cluster/kueue-shadow/v1"
NS = "vy-shadow"
COHORT = "vy-shadow"
LABEL = "vy.dev/shadow"
KEY = "vy.dev/shadow-key"
API = "kueue.x-k8s.io/v1beta2"
WORKLOADS, CLUSTER_QUEUES, LOCAL_QUEUES = (f"{r}.v1beta2.kueue.x-k8s.io" for r in ("workloads", "clusterqueues", "localqueues"))
FLAVOR = "vy-nebius-1"
SLOT_FLAVOR = "vy-shadow-slots"
TRAIN_FLAVOR = "vy-shadow-train-slots"
TRAINS_TAINT = {"key": "vy.dev/trains", "value": "only", "effect": "NoSchedule"}
#: which slot a holder has, as its flavor's node label: Kueue would otherwise put a train holding a trains-only slot in the
#: first flavor it tolerates, and count it against the slots everyone may take
SLOT_KIND = "vy.dev/slot-kind"
CHECK_SLOT = "vy.dev/check-slot"
POOL_LABEL = "verity.dev/gpu-pool"
# n1_lease.py's pools as node 1 runs them: lease directory, ClusterQueue, WorkloadPriorityClass
POOLS = {"node1": ("/run/gpu-lease", "provers", "dev"),
         "circuits": ("/run/gpu-lease-circuits", "deployments-gpu", "circuits-gpu")}
TREE = Path(__file__).resolve().parents[4]
SLOT_PY = TREE / "tools/verity/check/slot.py"
LEAN_SLOT_PY = TREE / "tools/verity/check/lean_slot.py"
TICK_S = 60


@dataclass(frozen=True)
class Item:
    """One allocation or waiter as a Workload: `key` names it from waiting to running, `queue` is its shadow ClusterQueue,
    `pod_sets` the Workload's podSets, `priority` its value (with `priority_class` the WorkloadPriorityClass it came from),
    `running` what the node did, `since` when it arrived (ISO), and `trains` whether it may take a trains-only check slot."""
    key: str
    queue: str
    pod_sets: tuple[dict[str, Any], ...]
    priority: int
    running: bool
    since: str
    priority_class: str | None = None
    trains: bool = False
    active: bool = True


def _true(w: dict, kind: str) -> bool:
    return any(c.get("type") == kind and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or ())


def _pod_set(requests: dict[str, str], tolerations: tuple[dict, ...] = (), node_selector: dict[str, str] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"restartPolicy": "Never",
                            "containers": [{"name": "shadow", "image": "none", "resources": {"requests": requests}}]}
    if tolerations:
        spec["tolerations"] = list(tolerations)
    if node_selector:
        spec["nodeSelector"] = dict(node_selector)
    return {"name": "main", "count": 1, "template": {"spec": spec}}


def shadow_name(cq: str) -> str:
    return f"shadow-{cq}"


def workload_name(key: str) -> str:
    """A Kubernetes name for `key` (which may hold `/`, `@` or capitals): `s-` and 16 hex digits of its SHA-256."""
    return "s-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def queues(cqs: list[dict], slots: dict | None, lean: dict | None) -> list[dict]:
    """The shadow's objects but its Workloads, from the node's ClusterQueues (`kubectl get clusterqueues -o json` items) and
    the slots' `--status --json` documents (None: no such file on the node, so no queue)."""
    meta = {"labels": {LABEL: "true"}}
    out: list[dict] = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NS, **meta}}]

    def cq(name: str, groups: list[dict], extra: dict | None = None) -> None:
        spec = {**(extra or {}), "cohortName": COHORT, "namespaceSelector": {"matchLabels": {LABEL: "true"}},
                "resourceGroups": groups}
        out.append({"apiVersion": API, "kind": "ClusterQueue", "metadata": {"name": name, **meta}, "spec": spec})
        out.append({"apiVersion": API, "kind": "LocalQueue", "metadata": {"name": name, "namespace": NS, **meta},
                    "spec": {"clusterQueue": name}})

    for real in sorted(cqs, key=lambda c: c["metadata"]["name"]):
        spec = {k: v for k, v in real["spec"].items() if k not in ("cohortName", "cohort", "namespaceSelector", "resourceGroups")}
        cq(shadow_name(real["metadata"]["name"]), real["spec"]["resourceGroups"], spec)
    if slots and slots.get("slots"):
        n_train = sum(1 for s in slots["slots"] if s.get("trains"))
        flavors = [(SLOT_FLAVOR, len(slots["slots"]) - n_train, "any", None),
                   (TRAIN_FLAVOR, n_train, "trains", [TRAINS_TAINT])]
        for name, _, kind, taints in flavors:
            out.append({"apiVersion": API, "kind": "ResourceFlavor", "metadata": {"name": name, **meta},
                        "spec": {"nodeLabels": {SLOT_KIND: kind}, **({"nodeTaints": taints} if taints else {})}})
        cq(shadow_name("checks"), [{"coveredResources": [CHECK_SLOT], "flavors": [
            {"name": name, "resources": [{"name": CHECK_SLOT, "nominalQuota": n}]} for name, n, _, _ in flavors]}])
    if lean and lean.get("pools"):
        res = [f"vy.dev/lean-{p}" for p in sorted(lean["pools"])]
        cq(shadow_name("lean"), [{"coveredResources": res, "flavors": [{"name": FLAVOR, "resources": [
            {"name": f"vy.dev/lean-{p}", "nominalQuota": lean["pools"][p]["n"]} for p in sorted(lean["pools"])]}]}])
    return out


def demand(workloads: list[dict], local_queues: dict[str, str], leases: dict[str, nebius2.Observation],
           priorities: dict[str, int], slots: dict | None, lean: dict | None) -> list[Item]:
    """Every allocation and waiter on the node as an Item. `workloads` are Kueue's in `default`, `local_queues` maps each
    LocalQueue there to its ClusterQueue, `leases` is each pool's (`POOLS`) `nebius2.read`, `priorities` the
    WorkloadPriorityClasses' values, and `slots` and `lean` the slots' `--status --json` documents."""
    out: list[Item] = []
    for w in workloads:
        meta, spec = w["metadata"], w["spec"]
        if _true(w, "Finished") or any(POOL_LABEL in ((ps.get("template") or {}).get("metadata") or {}).get("labels", {})
                                       for ps in spec["podSets"]):
            continue
        cq = local_queues.get(spec.get("queueName", ""))
        if cq is None:
            continue
        out.append(Item(key=f"k8s/{meta['name']}", queue=shadow_name(cq), pod_sets=tuple(spec["podSets"]),
                        priority=int(spec.get("priority") or 0), priority_class=(spec.get("priorityClassRef") or {}).get("name"),
                        running=_true(w, "QuotaReserved"), since=meta.get("creationTimestamp", ""),
                        active=spec.get("active", True) is not False))
    for pool, obs in sorted(leases.items()):
        _, cq, prio = POOLS[pool]
        held: dict[str, list[int]] = {}
        since: dict[str, str] = {}
        for gpu, line in obs.leases.items():
            f = nebius2.fields(line)
            if f.get("who") == "kubernetes":
                continue
            held.setdefault(f["pid"], []).append(gpu)
            since[f["pid"]] = f.get("since", "")
        wanted = {}
        for name, line in obs.waits:
            f = nebius2.fields(line)
            pid = f.get("pid") or name.rsplit(".", 1)[-1]
            if pid not in held:
                wanted[pid] = int(f.get("n", "1"))
                since[pid] = f.get("since", "")
        for pid, n, running in [*((p, len(g), True) for p, g in held.items()), *((p, n, False) for p, n in wanted.items())]:
            out.append(Item(key=f"lease/{pool}/gl-{pid}", queue=shadow_name(cq),
                            pod_sets=(_pod_set({"nvidia.com/gpu": str(n)}),), priority=priorities.get(prio, 0),
                            priority_class=prio, running=running, since=since[pid]))
    toleration = ({"key": TRAINS_TAINT["key"], "operator": "Equal", "value": TRAINS_TAINT["value"],
                   "effect": TRAINS_TAINT["effect"]},)
    if slots and slots.get("slots"):
        for s in slots["slots"]:
            h = s.get("holder")
            if h:
                # a holder's ticket said whether it was a train; a held trains-only slot can only be one
                out.append(Item(key=f"slot/{h.get('run') or h['pid']}", queue=shadow_name("checks"),
                                pod_sets=(_pod_set({CHECK_SLOT: "1"}, toleration if s.get("trains") else (),
                                                   {SLOT_KIND: "trains" if s.get("trains") else "any"}),),
                                priority=1 if s.get("trains") else 0, running=True, since=h.get("since") or "",
                                trains=bool(s.get("trains"))))
        for wt in slots.get("waiters") or ():
            pr = bool(wt.get("priority"))
            out.append(Item(key=f"slot/{wt.get('run') or wt['pid']}", queue=shadow_name("checks"),
                            pod_sets=(_pod_set({CHECK_SLOT: "1"}, toleration if pr else ()),), priority=int(pr),
                            running=False, since=wt.get("since", ""), trains=pr))
    if lean and lean.get("pools"):
        for pool, p in sorted(lean["pools"].items()):
            res = {f"vy.dev/lean-{pool}": "1"}
            for s in p["slots"]:
                if s.get("holder"):
                    h = s["holder"]
                    out.append(Item(key=f"lean/{pool}/{h.get('run') or h['pid']}", queue=shadow_name("lean"),
                                    pod_sets=(_pod_set(res),), priority=0, running=True, since=h.get("since") or ""))
            for wt in p.get("waiters") or ():
                out.append(Item(key=f"lean/{pool}/{wt.get('run') or wt['pid']}", queue=shadow_name("lean"),
                                pod_sets=(_pod_set(res),), priority=0, running=False, since=wt.get("since", "")))
    return list({i.key: i for i in out}.values())


def workload(item: Item) -> dict:
    """`item` as a bare Workload in the shadow's namespace."""
    spec: dict[str, Any] = {"queueName": item.queue, "podSets": list(item.pod_sets), "priority": item.priority,
                            "active": item.active}
    if item.priority_class:
        spec["priorityClassRef"] = {"group": "kueue.x-k8s.io", "kind": "WorkloadPriorityClass", "name": item.priority_class}
    return {"apiVersion": API, "kind": "Workload",
            "metadata": {"name": workload_name(item.key), "namespace": NS, "labels": {LABEL: "true"},
                         "annotations": {KEY: item.key, "vy.dev/shadow-since": item.since}},
            "spec": spec}


def reconcile(items: list[Item], shadow: list[dict]) -> tuple[list[dict], list[str]]:
    """The Workloads to create, in arrival order (Kueue's FIFO is by creation time), and the names to delete (Workloads whose
    item is gone). While any new item is one the node runs, only those are created, and new waiters wait a tick, so Kueue
    has admitted what already runs before it orders the waiters behind it. A waiter whose request is newer than its Workload's
    (a holder that let go and asked again) is a new request: its Workload is deleted, and made again next tick."""
    have = {w["metadata"]["name"]: w for w in shadow}
    want = {workload_name(i.key): i for i in items}
    stale = {n for n, i in want.items() if n in have and _stale(have[n], i)}
    new = sorted((i for n, i in want.items() if n not in have), key=lambda i: (i.since, i.key))
    if any(i.running for i in new):
        new = [i for i in new if i.running]
    return [workload(i) for i in new], sorted((set(have) - set(want)) | stale)


def _stale(w: dict, item: Item) -> bool:
    """Whether `w` is no longer `item`'s request: being deleted, or a waiter's request newer than the one `w` was made for."""
    meta = w["metadata"]
    return bool(meta.get("deletionTimestamp")) or (
        not item.running and bool(item.since) and (meta.get("annotations") or {}).get("vy.dev/shadow-since") != item.since)


def compare(t: str, items: list[Item], shadow: list[dict], before: frozenset[str] = frozenset()) -> dict:
    """One record: each item the node and Kueue disagree on. An item whose Workload is new this tick isn't compared, and a
    stale Workload isn't compared either, and a
    disagreement is a divergence only when the last tick had it too (`before`, its record's `disagree`): a slot handed over
    between two ticks reads as one for a tick, until the shadow deletes the old holder's Workload."""
    by_name = {w["metadata"]["name"]: w for w in shadow}
    div = []
    disagree = []
    compared = 0
    for i in sorted(items, key=lambda i: i.key):
        w = by_name.get(workload_name(i.key))
        if w is None or not i.active or _stale(w, i):
            continue
        compared += 1
        admitted = _true(w, "QuotaReserved")
        if i.running == admitted:
            continue
        disagree.append(i.key)
        if i.key not in before:
            continue
        if i.running:
            div.append({"key": i.key, "queue": i.queue, "kind": "node-ran", "kueue": _reason(w)})
        else:
            div.append({"key": i.key, "queue": i.queue, "kind": "kueue-ran", "since": i.since})
    return {"schema": SCHEMA, "t": t, "items": len(items), "compared": compared,
            "running": sum(i.running for i in items), "waiting": sum(not i.running for i in items),
            "divergent": len(div), "divergences": div, "disagree": disagree}


def last_disagree(out: Path) -> frozenset[str]:
    """The keys the record's last tick disagreed on (none without a record, or after a failed tick)."""
    try:
        with out.open("rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - (1 << 20)))
            lines = f.read().decode(errors="replace").splitlines()
    except FileNotFoundError:
        return frozenset()
    for line in reversed(lines):
        try:
            return frozenset(json.loads(line).get("disagree") or ())
        except ValueError:
            continue
    return frozenset()


def _reason(w: dict) -> str:
    for c in (w.get("status") or {}).get("conditions") or ():
        if c.get("type") == "QuotaReserved":
            return (c.get("message") or c.get("reason") or "")[:200]
    return "not yet considered"


def kubectl(args: list[str], stdin: str | None = None) -> str:
    r = subprocess.run([*nebius1.KUBECTL, *args], input=stdin, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: exit {r.returncode}: {r.stderr.strip()[-600:]}")
    return r.stdout


def _status(script: Path) -> dict | None:
    r = subprocess.run([sys.executable, str(script), "--status", "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{script.name} --status --json: exit {r.returncode}: {r.stderr.strip()[-300:]}")
    doc = json.loads(r.stdout)
    return doc if doc.get("slots") is not None or doc.get("pools") is not None else None


def observe(run: Callable[[list[str]], str] = kubectl) -> dict[str, Any]:
    """The node now, read-only: Kueue's queues, LocalQueues, priority classes and Workloads in `default`, each pool's
    lease directory, the slots' status and the shadow's Workloads."""
    cqs = json.loads(run(["get", CLUSTER_QUEUES, "-l", f"!{LABEL}", "-o", "json"]))["items"]
    lqs = json.loads(run(["get", LOCAL_QUEUES, "-n", "default", "-o", "json"]))["items"]
    wpcs = json.loads(run(["get", "workloadpriorityclasses", "-o", "json"]))["items"]
    wls = json.loads(run(["get", WORKLOADS, "-n", "default", "-o", "json"]))["items"]
    t = int(time.time())
    return {"cqs": cqs, "local_queues": {q["metadata"]["name"]: q["spec"]["clusterQueue"] for q in lqs},
            "priorities": {p["metadata"]["name"]: int(p["value"]) for p in wpcs}, "workloads": wls,
            "leases": {pool: nebius2.read(t, Path(d)) for pool, (d, _, _) in POOLS.items()},
            "slots": _status(SLOT_PY), "lean": _status(LEAN_SLOT_PY)}


def shadow_workloads(run: Callable[[list[str]], str] = kubectl) -> list[dict]:
    return json.loads(run(["get", WORKLOADS, "-n", NS, "-l", f"{LABEL}=true", "-o", "json"]))["items"]


def tick(out: Path, run: Callable[[list[str]], str] = kubectl,
         apply: Callable[[list[dict]], None] | None = None, delete: Callable[[list[str]], None] | None = None) -> dict:
    """One round: read the node, bring the shadow's queues and Workloads to its demand, compare, append the record."""
    apply = apply or (lambda objs: run(["apply", "-f", "-"], json.dumps({"apiVersion": "v1", "kind": "List", "items": objs}))
                      if objs else None)
    delete = delete or (lambda names: run(["delete", WORKLOADS, "-n", NS, "--ignore-not-found", "--wait=false", *names])
                        if names else None)
    obs = observe(run)
    items = demand(obs["workloads"], obs["local_queues"], obs["leases"], obs["priorities"], obs["slots"], obs["lean"])
    apply(queues(obs["cqs"], obs["slots"], obs["lean"]))
    shadow = shadow_workloads(run)
    rec = compare(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), items, shadow, last_disagree(out))
    create, gone = reconcile(items, shadow)
    for w in create:
        apply([w])
    delete(gone)
    rec |= {"created": len(create), "deleted": len(gone)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def down(run: Callable[[list[str]], str] = kubectl) -> None:
    """Delete everything the shadow wrote: its namespace (and so its Workloads and LocalQueues), ClusterQueues and flavors."""
    run(["delete", "namespace", NS, "--ignore-not-found"])
    run(["delete", "clusterqueues,resourceflavors", "-l", f"{LABEL}=true", "--ignore-not-found"])


def add_parser(sub: Any) -> None:
    p = sub.add_parser("kueue-shadow", help="node 1 with Kueue as its only scheduler, in shadow (kueue_shadow.py); on node 1")
    ps = p.add_subparsers(dest="kcmd", required=True)
    for name in ("tick", "loop"):
        q = ps.add_parser(name)
        q.add_argument("--out", required=True, help="the JSONL record to append to")
        if name == "loop":
            q.add_argument("--every", type=int, default=TICK_S)
            q.add_argument("--hours", type=float, required=True)
    ps.add_parser("down", help="delete every object the shadow wrote")


def main(a: argparse.Namespace) -> int:
    """`cluster kueue-shadow tick|loop|down`; a loop's failed tick is recorded and the loop goes on."""
    if a.kcmd == "down":
        down()
        return 0
    out = Path(a.out)
    if a.kcmd == "tick":
        print(json.dumps(tick(out), sort_keys=True))
        return 0
    end = time.time() + 3600 * a.hours
    while time.time() < end:
        try:
            rec = tick(out)
            print(f"{rec['t']} items {rec['items']} divergent {rec['divergent']}", flush=True)
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            with out.open("a") as f:
                f.write(json.dumps({"schema": SCHEMA, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "error": f"{type(e).__name__}: {e}"[:800]}) + "\n")
        time.sleep(a.every)
    return 0
