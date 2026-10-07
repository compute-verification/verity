"""The submit path: a job declared on the command line becomes the model's `Job`, is placed (`route.place`), and gets the
wrappers its node's executor runs it under. `research run --queue` asks for this and runs its machine-side runner inside the
job's CPU allocation, and a GPU phase's workload alone inside its GPU lease: the run's custody finishes before the CPU
allocation is released, and the GPU is booked only while the workload runs.

The executors, as they stand:
- **vy-nebius-2, GPUs:** `gpu-lease N --wait`, with the job's flags, around the workload; the runner is in the CPU scope
  below. Under the node agent's live mode, gpu-lease's wait file is the brain's queue and its grant is the brain's
  decision; with the agent down, gpu-lease's own rules decide.
- **vy-nebius-2, CPUs only:** a systemd scope with the memory cap, pinned to the project's CPUs, CUDA hidden. A guest runs at
  nice 19 on its share's CPUs, and node 2's ops daemon pauses it through a timed window. The scope holds the CPUs itself
  (`AllowedCPUs=`), so a workload's own `taskset` can't leave them; systemd drops that without a word where user@.service
  doesn't delegate cpuset (research's pods/nebius/user-cpuset.conf), so the wrapper starts nothing (exit 70) unless the
  scope's cpuset is the pin.
- **vy-nebius-1, CPUs only:** the same scope on the CPUs outside the reserved pools, idle (`CPUWeight=idle`): GPU jobs' host
  threads share those CPUs, and network's timing chunks among them stalled for seconds behind Lean audits' replays
  (network-accounting, art:894c0447). An idle scope runs only on CPU time its siblings leave. A job that takes its GPUs
  with gpu-lease itself, around each GPU step (`gpu_host`, `research run --queue --gpu-host`), is a GPU job's host: its scope
  runs at normal weight, as a GPU job's runner does, so its verifier and replays between leases don't wait on idle CPU time.
- **vy-nebius-1, GPUs:** Kueue's, lent per process by `n1_lease.py` (infra/nebius): node 1's `gpu-lease` copy waits on
  `/run/gpu-lease`, the controller grows its pool of 1-GPU holder Jobs in `provers` for the waiter, and the lease takes a holder's
  GPU. Only with `gpu_executor = "n1-lease"` on the node in the description (the cutover's one-line switch); without it such
  a job goes to node 2 or is refused. A timed job never runs there: node 1 has no window protocol. A job that isn't
  preemptible gets at most `provers`' own quota there, so a larger one is refused, not left waiting forever. Only a job
  pinned there (`--on vy-nebius-1`) is sent: the submit path sees neither node's live state, so the router, which prefers the owner's
  node, would queue every Verity GPU job on node 1's pool (`provers`' unborrowed quota) while node 2 had GPUs free.

A job asks for `max_s` of its own. A CPU job's allocation also covers `custody_s` (the runner's publish), so its `max_s` in the
model is their sum; a GPU job's lease covers the workload alone, and custody runs in its CPU allocation after the lease ends.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from . import cpuset
from .kinds import DISK_GB, Kind, warnings
from .model import OWNER, Cluster, Job, job_to_dict
from .plan import allowed_cpus, static_refusal
from .route import place

GPU_LEASE = "/workspace/pouw/infra/bin/gpu-lease"
GPU_LEASE_N1 = "/workspace/verity-guest/bin/gpu-lease"
N1_LEASE_ENV = ["GPU_LEASE_DIR=/run/gpu-lease", "GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus"]
NODE1, NODE2 = "vy-nebius-1", "vy-nebius-2"
N1_POOL_QUEUE = "provers"
UNSEEN_POOL = f"{NODE1} lends GPUs only to a job pinned there (--on {NODE1}): the submit path can't see its lease pool"
CPUSET_CHECK = ('c=$(cut -d: -f3 /proc/self/cgroup); '
                '[ "$(cat "/sys/fs/cgroup$c/cpuset.cpus" 2>/dev/null)" = "$1" ] && shift && exec "$@"; '
                'echo "cluster: scope $c does not hold CPUs $1 (is cpuset delegated to user@.service?)" >&2; exit 70')


class Refused(Exception):
    pass


def job(*, run_id: str, project: str, submitter: str, priority: str = "work", gpus: int = 0, gpu_model: str | None = None,
        cpus: int = 1, memory_gib: int = 8, max_s: int, custody_s: int = 0, preempt: str = "requeue", quiet: bool = False,
        node: str | None = None, gpu_ids: tuple[int, ...] = (), submitted: int = 0) -> Job:
    return Job(id=f"rr-{run_id}", project=project, submitter=submitter, priority=priority, kind="batch", gpus=gpus,
               gpu_model=gpu_model, cpus=cpus, memory_gib=memory_gib, max_s=max_s + (0 if gpus else custody_s), preempt=preempt, quiet=quiet,
               node=node, gpu_ids=gpu_ids, submitted=submitted)


def wrappers(cluster: Cluster, j: Job, node: str, run_id: str, kind: str | None = None,
             gpu_host: bool = False) -> tuple[list[str], list[str]]:
    """The runner's wrapper (the job's CPU allocation, which holds the run through custody) and the workload's (a GPU phase's
    lease, around the workload alone, so the GPU is booked for the GPU phase only); Refused when the node has no executor.
    A kind is named in the lease's holder, `who=<kind>:<submitter>`, as the fill runner's leases are `fill:<owner>`."""
    if node not in (NODE1, NODE2):
        raise Refused(f"no executor on {node}")
    if j.gpus and node == NODE1 and cluster.nodes[node].gpu_executor != "n1-lease":
        raise Refused(f"{node}'s GPUs are Kueue's, and its lease executor isn't switched on (gpu_executor = \"n1-lease\")")
    if j.gpus and node == NODE1 and j.quiet:
        raise Refused(f"{node} has no timed windows")
    own = cluster.workstream(N1_POOL_QUEUE, node) if j.gpus and node == NODE1 and j.preempt != "requeue" else None
    if own and j.gpus > own.gpus:
        raise Refused(f"{node}'s lease pool grants a lease that isn't --preemptible at most {N1_POOL_QUEUE}' own quota, {own.gpus} "
                      f"GPUs (n1_lease.py borrows beyond it only while every waiter is --preemptible), and this asks for {j.gpus}")
    guest = cluster.standing(j.project, node) != OWNER
    pin = cpuset.format(allowed_cpus(cluster, cluster.nodes[node], j))
    runner = ["env", f"GPU_LEASE_WHO={f'{kind}:' if kind else ''}{j.submitter}", f"RESEARCH_RUN_ID={run_id}", "CUDA_VISIBLE_DEVICES=",
              "systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", f"cluster-{run_id}",
              "-p", f"MemoryMax={j.memory_gib}G", "-p", "MemorySwapMax=0", "-p", f"AllowedCPUs={pin}",
              *(["-p", "CPUWeight=idle"] if node == NODE1 and not j.gpus and not gpu_host else []), "--",
              "sh", "-c", CPUSET_CHECK, "cpuset-check", pin, "taskset", "-c", pin, *(["nice", "-n", "19"] if guest else [])]
    if not j.gpus:
        return runner, []
    lease = ["env", *N1_LEASE_ENV, GPU_LEASE_N1] if node == NODE1 else [GPU_LEASE]
    lease += [str(j.gpus), "--wait", "--max-min", str(math.ceil(j.max_s / 60)), "--mem-gb", str(j.memory_gib)]
    lease += ["--preemptible"] if j.preempt == "requeue" else []
    lease += ["--timed"] if j.quiet else []
    lease += ["--on", ",".join(map(str, j.gpu_ids))] if j.gpu_ids else []
    return runner, [*lease, "--"]


def resolve(cluster: Cluster, j: Job, run_id: str, now: int, kind: str | None = None, gpu_host: bool = False) -> dict[str, Any]:
    """Where the job runs and under what: `{node, machine, why, job, wrap, workload_wrap}`. The router's node first, then any other node
    that could run it (only the pinned one when `j.node` is set); Refused with each node's reason when none can."""
    first = place(cluster, j, {}, now)
    if first.node is None:
        raise Refused(first.why)
    reasons = []
    for name in [first.node, *(n for n in sorted(cluster.nodes) if n != first.node and not j.node)]:
        pinned = replace(j, node=name)
        why = static_refusal(cluster, name, pinned, now) or (UNSEEN_POOL if j.gpus and name == NODE1 and not j.node else None)
        if why:
            reasons.append(f"{name}: {why}")
            continue
        try:
            wrap, workload_wrap = wrappers(cluster, pinned, name, run_id, kind, gpu_host)
        except Refused as e:
            reasons.append(f"{name}: {e}")
            continue
        return {"node": name, "machine": name, "why": first.why if name == first.node else f"{reasons[-1]}; so {name}",
                "job": job_to_dict(pinned), "wrap": wrap, "workload_wrap": workload_wrap}
    raise Refused("; ".join(reasons))


def submit(cluster: Cluster, registry: dict[str, Kind], *, run_id: str, now: int, kind: str = "adhoc", question: str = "",
           gpus: int | None = None, cpus: int | None = None, memory_gib: int | None = None, max_s: int | None = None,
           quiet: bool | None = None, preempt: str = "requeue", disk_gb: float | None = None, gpu_host: bool = False,
           **declared: Any) -> dict[str, Any]:
    """`resolve` for a job of `kind`: what the job leaves unsaid comes from its registered kind (else 0 GPUs, 1 CPU), and the
    placement also names the kind, its question, the registry file it came from, and the held rules it breaks. A job whose
    kind isn't registered names its memory: the scope's `MemoryMax=` kills the job at that cap."""
    k = registry.get(kind)
    shape = k.shape if k else {}
    gpus = gpus if gpus is not None else int(shape.get("gpus", 0))
    cpus = cpus if cpus is not None else int(shape.get("cpus", 1))
    if memory_gib is None and "memory_gib" not in shape:
        raise Refused(f"give the job's memory (--mem-gb G): kind {kind!r} isn't registered, so it has no shape's memory, and the job's "
                      "scope is killed at its cap (MemoryMax=G)")
    memory_gib = memory_gib if memory_gib is not None else int(shape["memory_gib"])
    quiet = quiet if quiet is not None else bool(shape.get("quiet", False))
    if max_s is None:
        if k is None:
            raise Refused(f"give the job's longest run: kind {kind!r} isn't registered, so it has no max_wall_min")
        max_s = k.max_wall_min * 60
    question = question or (k.question if k else "")
    if gpu_host and gpus:
        raise Refused(f"--gpu-host is for a job that takes its GPUs with gpu-lease itself; this one asks the queue for {gpus}")
    j = job(run_id=run_id, gpus=gpus, cpus=cpus, memory_gib=memory_gib, max_s=max_s, quiet=quiet, preempt=preempt,
            submitted=now, **declared)
    out = resolve(cluster, j, run_id, now, kind, gpu_host)
    disk_gb = disk_gb if disk_gb is not None else k.disk_gb if k and k.disk_gb is not None else DISK_GB
    out.update(kind=kind, question=question, max_s=max_s, disk_gb=disk_gb, kind_source=k.source if k else None, gpu_host=gpu_host,
               warnings=warnings(k, kind, question=question, gpus=gpus, cpus=cpus, memory_gib=memory_gib, max_s=max_s,
                                 preempt=preempt))
    return out


__all__ = ["CPUSET_CHECK", "GPU_LEASE", "GPU_LEASE_N1", "N1_POOL_QUEUE", "UNSEEN_POOL", "Refused", "job", "wrappers", "resolve", "submit"]
