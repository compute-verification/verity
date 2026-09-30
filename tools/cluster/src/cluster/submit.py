"""The submit path: a job declared on the command line becomes the model's `Job`, is placed (`route.place`), and gets the
wrapper its node's executor runs it under. `research run --queue` asks for this and runs its machine-side runner inside the
wrapper, so the run's custody finishes before the allocation is released.

The executors, as they stand:
- **vy-nebius-2, GPUs:** `gpu-lease N --wait`, with the job's flags. Under the node agent's live mode, gpu-lease's wait file is
  the brain's queue and its grant is the brain's decision; with the agent down, gpu-lease's own rules decide.
- **vy-nebius-2, CPUs only:** a systemd scope with the memory cap, pinned to the project's CPUs, CUDA hidden. A guest runs at
  nice 19 on its share's CPUs, and node 2's ops daemon pauses it through a timed window.
- **vy-nebius-1, CPUs only:** the same scope on the CPUs outside the reserved pools.
- **vy-nebius-1, GPUs:** Kueue's, through kueue-fold's executor, which isn't built yet; such a job goes to node 2 or is refused.

A job asks for `max_s` of its own; the allocation also covers `custody_s` (the runner's publish), so the job's `max_s` in the
model is their sum.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from . import cpuset
from .model import OWNER, Cluster, Job, job_to_dict
from .plan import allowed_cpus, static_refusal
from .route import place

GPU_LEASE = "/workspace/pouw/infra/bin/gpu-lease"
NODE1, NODE2 = "vy-nebius-1", "vy-nebius-2"


class Refused(Exception):
    pass


def job(*, run_id: str, project: str, submitter: str, priority: str = "work", gpus: int = 0, gpu_model: str | None = None,
        cpus: int = 1, memory_gib: int = 8, max_s: int, custody_s: int = 0, preempt: str = "requeue", quiet: bool = False,
        node: str | None = None, gpu_ids: tuple[int, ...] = (), submitted: int = 0) -> Job:
    return Job(id=f"rr-{run_id}", project=project, submitter=submitter, priority=priority, kind="batch", gpus=gpus,
               gpu_model=gpu_model, cpus=cpus, memory_gib=memory_gib, max_s=max_s + custody_s, preempt=preempt, quiet=quiet,
               node=node, gpu_ids=gpu_ids, submitted=submitted)


def wrapper(cluster: Cluster, j: Job, node: str, run_id: str) -> list[str]:
    """The argv the node's executor runs the job's runner under; Refused when the node has no executor for it yet."""
    n = cluster.nodes[node]
    guest = cluster.standing(j.project, node) != OWNER
    env = ["env", f"GPU_LEASE_WHO={j.submitter}", f"RESEARCH_RUN_ID={run_id}"]
    pin = cpuset.format(allowed_cpus(cluster, n, j))
    if j.gpus:
        if node != NODE2:
            raise Refused(f"{node}'s GPUs are Kueue's, and its executor (kueue-fold's) isn't built yet")
        lease = [GPU_LEASE, str(j.gpus), "--wait", "--max-min", str(math.ceil(j.max_s / 60)), "--mem-gb", str(j.memory_gib)]
        lease += ["--preemptible"] if j.preempt == "requeue" else []
        lease += ["--timed"] if j.quiet else []
        lease += ["--on", ",".join(map(str, j.gpu_ids))] if j.gpu_ids else []
        return [*env, *lease, "--", *(["taskset", "-c", pin, "nice", "-n", "19"] if guest else [])]
    if node not in (NODE1, NODE2):
        raise Refused(f"no executor on {node}")
    scope = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", f"cluster-{run_id}",
             "-p", f"MemoryMax={j.memory_gib}G", "-p", "MemorySwapMax=0", "--"]
    return [*env, "CUDA_VISIBLE_DEVICES=", *scope, "taskset", "-c", pin, *(["nice", "-n", "19"] if guest else [])]


def resolve(cluster: Cluster, j: Job, run_id: str, now: int) -> dict[str, Any]:
    """Where the job runs and under what: `{node, machine, why, job, wrap}`. The router's node first, then any other node
    that could run it (only the pinned one when `j.node` is set); Refused with each node's reason when none can."""
    first = place(cluster, j, {}, now)
    if first.node is None:
        raise Refused(first.why)
    reasons = []
    for name in [first.node, *(n for n in sorted(cluster.nodes) if n != first.node and not j.node)]:
        pinned = replace(j, node=name)
        why = static_refusal(cluster, name, pinned, now)
        if why:
            reasons.append(f"{name}: {why}")
            continue
        try:
            wrap = wrapper(cluster, pinned, name, run_id)
        except Refused as e:
            reasons.append(f"{name}: {e}")
            continue
        return {"node": name, "machine": name, "why": first.why if name == first.node else f"{reasons[-1]}; so {name}",
                "job": job_to_dict(pinned), "wrap": wrap}
    raise Refused("; ".join(reasons))


__all__ = ["GPU_LEASE", "Refused", "job", "wrapper", "resolve"]
