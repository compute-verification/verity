"""The node-1 adapter: Kueue's pods and Workloads, with DCGM's GPU-to-pod labels, as the model's allocations and queue."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster import model, nebius1
from cluster.ledger import epoch
from cluster.plan import Start, plan

DESC = Path(__file__).parents[1] / "descriptions" / "nebius.toml"
T = epoch("2026-09-30T20:20:00Z")


@pytest.fixture(scope="module")
def c() -> model.Cluster:
    return model.load(DESC)


def pod(name: str, *, gpus: int = 0, cpu: str = "4", memory: str = "60G", job: str | None = None,
        start: str = "2026-09-30T20:00:00Z", labels: dict | None = None, phase: str = "Running") -> dict:
    req = {"cpu": cpu, "memory": memory, **({"nvidia.com/gpu": str(gpus)} if gpus else {})}
    meta = {"name": name, "labels": labels or {}, "creationTimestamp": start}
    if job:
        meta["ownerReferences"] = [{"kind": "Job", "name": job}]
    return {"metadata": meta, "spec": {"containers": [{"resources": {"requests": req}}]},
            "status": {"phase": phase, "startTime": start}}


def workload(owner: str, queue: str, prio: str, *, kind: str = "Job", gpus: int = 0, cpu: str = "4", memory: str = "60G",
             admitted: bool = True, created: str = "2026-09-30T19:59:00Z") -> dict:
    req = {"cpu": cpu, "memory": memory, **({"nvidia.com/gpu": str(gpus)} if gpus else {})}
    return {"metadata": {"name": f"wl-{owner}", "creationTimestamp": created, "ownerReferences": [{"kind": kind, "name": owner}]},
            "spec": {"active": True, "queueName": queue, "priorityClassRef": {"name": prio},
                     "podSets": [{"template": {"metadata": {"labels": {}}, "spec": {"containers": [{"resources": {"requests": req}}]}}}]},
            "status": {"conditions": [{"type": "QuotaReserved", "status": "True" if admitted else "False"}]}}


def test_running_pods_are_allocations_of_the_gpus_dcgm_names_with_their_workloads_queue_and_priority(c):
    obs = nebius1.Observation(T, (
        pod("gpu-329-head", gpus=1, memory="170G"),
        pod("nd-sweep-a-0-xyz", gpus=1, job="nd-sweep-a-0", memory="96Gi"),
        pod("nd-build-b-0-abc", job="nd-build-b-0", cpu="3500m"),
    ), (
        workload("gpu-329-head", "deployments-gpu", "circuits-gpu", kind="Pod", gpus=1),
        workload("nd-sweep-a-0", "provers", "dev", gpus=1),
        workload("nd-build-b-0", "deployments-cpu", "circuits"),
    ), {3: ("gpu-329-head", 50000), 0: ("nd-sweep-a-0-xyz", 10), 1: ("", 0)})
    allocs, queue = nebius1.from_k8s(c, obs)
    by = {a.job.id: a for a in allocs}
    commit, sweep, build = by["k8s-gpu-329-head"], by["k8s-nd-sweep-a-0"], by["k8s-nd-build-b-0"]
    assert (commit.gpus, commit.job.priority, commit.job.workstream, commit.job.preempt, commit.memory_gib) == \
        ((3,), "work", "deployments-gpu", "requeue", 158)
    assert (sweep.gpus, sweep.job.priority, sweep.job.workstream, sweep.job.preempt, sweep.memory_gib) == \
        ((0,), "dev", "provers", "never", 96)
    assert (build.gpus, build.cpus, build.job.cpus, build.job.workstream) == ((), frozenset(), 4, None)
    assert commit.id == f"k8s-gpu-329-head@{epoch('2026-09-30T20:00:00Z')}" and commit.job.kind == "session"
    assert queue == []
    assert all(not a.job.problems(c) for a in allocs)


def test_a_gpu_pod_dcgm_hasnt_named_yet_takes_the_unnamed_busy_gpus_when_the_counts_match(c):
    pods = (pod("p1", gpus=1), pod("p2", gpus=1))
    wls = (workload("p1", "deployments-gpu", "circuits-gpu", kind="Pod", gpus=1),
           workload("p2", "deployments-gpu", "circuits-gpu", kind="Pod", gpus=1))
    allocs, _ = nebius1.from_k8s(c, nebius1.Observation(T, pods, wls, {2: ("p1", 9), 5: ("", 40000), 6: ("", 0)}))
    assert {a.job.id: a.gpus for a in allocs} == {"k8s-p1": (2,), "k8s-p2": (5,)}
    allocs, _ = nebius1.from_k8s(c, nebius1.Observation(T, pods, wls, {2: ("p1", 9), 5: ("", 40000), 6: ("gone", 30000)}))
    assert {a.job.id: a.gpus for a in allocs} == {"k8s-p1": (2,), "k8s-unowned-5-6": (5, 6)}


def test_a_workload_without_quota_is_queued_under_the_id_its_allocation_will_have(c):
    obs = nebius1.Observation(T, (pod("old", phase="Succeeded", gpus=1),), (
        workload("nd-epoch-gpu-0", "deployments-gpu", "circuits-gpu", gpus=1, memory="170G", admitted=False),
        workload("replay-336-head", "deployments-cpu", "circuits", kind="Pod", cpu="8", admitted=False),
        workload("done-one", "backfill", "backfill", admitted=True),
    ), {i: ("", 0) for i in range(8)})
    allocs, queue = nebius1.from_k8s(c, obs)
    assert allocs == []
    gpu, replay = sorted(queue, key=lambda j: -j.gpus)
    assert (gpu.id, gpu.gpus, gpu.cpus, gpu.memory_gib, gpu.workstream, gpu.priority) == \
        ("k8s-nd-epoch-gpu-0", 1, 0, 158, "deployments-gpu", "work")
    assert (replay.id, replay.cpus, replay.workstream, replay.kind, replay.expected_s) == \
        ("k8s-replay-336-head", 8, None, "session", nebius1.EXPECTED_S)
    acts = plan(c, nebius1.NODE, allocs, queue, T)
    starts = {a.alloc.job.id: a.alloc for a in acts if isinstance(a, Start)}
    assert set(starts) == {"k8s-nd-epoch-gpu-0", "k8s-replay-336-head"}
    assert len(starts["k8s-replay-336-head"].cpus) == 8 and not starts["k8s-replay-336-head"].cpus & c.nodes[nebius1.NODE].pooled_cpus


def test_pous_overflow_is_pouss_and_backfill_is_fill(c):
    obs = nebius1.Observation(T, (), (workload("pous-x", "pous-overflow", "backfill", gpus=1, admitted=False),), {})
    (job,) = nebius1.from_k8s(c, obs)[1]
    assert (job.project, job.priority, job.preempt) == ("pous", "fill", "requeue")


def test_a_snapshot_round_trips_and_the_command_plans_from_it(c, tmp_path, capsys):
    obs = nebius1.Observation(T, (pod("p1", gpus=1),), (workload("p1", "deployments-gpu", "circuits-gpu", kind="Pod", gpus=1),),
                              {4: ("p1", 100)})
    snap = tmp_path / "s.json"
    snap.write_text(json.dumps(nebius1.save(obs)))
    assert nebius1.load(json.loads(snap.read_text())) == obs
    assert nebius1.main([str(DESC), "--from", str(snap)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert lines[0]["alloc"].startswith("k8s-p1@") and lines[0]["gpus"] == [4]
    assert lines[1]["cpu_allocations"] == 0 and lines[1]["queued"] == 0
    assert nebius1.main([str(DESC), "--from", str(snap), "--report"]) == 0
    (rep,) = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    (al,) = [model.allocation_from_dict(d) for d in rep["allocations"]]
    assert (rep["node"], al.gpus, al.job.id, rep["queue"]) == (nebius1.NODE, (4,), "k8s-p1", [])


@pytest.mark.parametrize("q, gib", [("170G", 158), ("96Gi", 96), ("512Mi", 1), ("1Ti", 1024), (None, 1)])
def test_memory_quantities_are_whole_gib(q, gib):
    assert nebius1._gib(q) == gib


@pytest.mark.parametrize("q, n", [("4", 4), ("3500m", 4), ("250m", 1), ("0.5", 1), (None, 1)])
def test_cpu_quantities_round_up(q, n):
    assert nebius1._cpus(q) == n
