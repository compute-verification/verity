from pathlib import Path

import pytest
from _build import T0, alloc, job, one

from cluster import cpuset, model

NEBIUS = Path(__file__).parents[1] / "descriptions" / "nebius.toml"


def test_cpuset_round_trip_and_canonical_form():
    assert cpuset.parse("0-3,8, 9,10-11") == frozenset([0, 1, 2, 3, 8, 9, 10, 11])
    assert cpuset.format(cpuset.parse("10-11,0-3,9,8")) == "0-3,8-11"
    assert cpuset.parse("") == frozenset() and cpuset.format([]) == ""


@pytest.mark.parametrize("bad, why", [("3-1", "descending"), ("a", "bad CPU list item"), ("1-", "bad CPU list item")])
def test_cpuset_rejects_by_name(bad, why):
    with pytest.raises(cpuset.CpuSetError, match=why):
        cpuset.parse(bad)


def test_the_nebius_description_loads():
    c = model.load(NEBIUS)
    n1, n2 = c.nodes["vy-nebius-1"], c.nodes["vy-nebius-2"]
    assert (n1.owner, n2.owner) == ("verity", "pous")
    assert len(n1.cpus) == 184 and len(n2.cpus) == 192 and len(n1.gpus) == len(n2.gpus) == 8
    assert n2.admission_ends == model.instant("2026-10-14T13:00:00Z")
    assert c.standing("verity", "vy-nebius-2") == c.standing("pous", "vy-nebius-1") == model.GUEST
    share = c.share("verity", "vy-nebius-2")
    # Daniel's defaults (decision 1): Verity's node-2 share gets GPUs; PoUW overflows onto node 1 off its reserved CPUs
    assert cpuset.format(share.cpus) == "48-95" and share.max_job_memory_gib == 256 and share.max_gpus is None
    assert share.until == model.instant("2026-10-14T15:00:00Z")
    assert cpuset.format(c.share("pous", "vy-nebius-1").cpus) == "96-191"
    assert cpuset.format(n1.pools["checks"]) == "8-95,128-159" and cpuset.format(n1.pools["provers"]) == "160-191"
    assert n2.pools == {}
    assert [(w.name, w.gpus, w.borrow_gpus) for w in c.workstreams] == [("provers", 2, 6), ("deployments-gpu", 6, 2)]
    order = sorted(c.priorities, key=c.priorities.get, reverse=True)
    # kueue.yaml's order is policy (the steward, 30 Sep 19:40Z)
    assert order.index("capture") < order.index("work") < order.index("bench") < order.index("dev") < order.index("fill")


def test_validation_names_each_problem():
    d = {"priorities": {"work": 1},
         "nodes": {"a": {"owner": "p", "memory_gib": 8, "numa": {"0": "0-3", "1": "3-5"}, "system_cpus": "9",
                         "gpus": [{"index": 0, "numa": 2}, {"index": 0, "numa": 0}],
                         "pools": {"x": "0-1", "y": "1-2", "z": "40"}}},
         "shares": [{"project": "p", "node": "a"}, {"project": "q", "node": "b"}, {"project": "r", "node": "a", "cpus": "7"}],
         "workstreams": [{"name": "u", "node": "a", "gpus": 3}, {"name": "w", "node": "b", "gpus": 1}]}
    problems = model.validate(model.Cluster("x", {}, {}, ())) + [str(e) for e in _errors(d)]
    text = " | ".join(problems)
    for want in ("no priority classes", "overlaps another", "system CPUs 9", "duplicate GPU index", "unknown NUMA node 2",
                 "the project owns that node", "unknown node 'b'", "CPUs 7 aren't the node's",
                 "pool y overlaps another pool (1)", "pool z holds CPUs 40", "workstreams own 3 GPUs of 2",
                 "workstream w names unknown node 'b'"):
        assert want in text


def _errors(d):
    try:
        model.from_dict(d)
    except model.ModelError as e:
        return [e]
    return []


def test_job_problems():
    c = one()
    assert job("a-1").problems(c) == []
    bad = [job("b-1", priority="urgent"), job("b-2", max_s=None), job("b-3", kind="service", priority="service", max_s=60),
           job("b-4", quiet_safe=True), job("b-5", kind="stage", quiet=True), job("b-6", gpus=2, gpu_ids=(1,)),
           job("b-7", cpus=0), job("b-8", node="elsewhere"), job("b-9", kind="session", max_s=None),
           job("b-10", expected_s=60), job("b-11", workstream="nobody"), job("b-12", gpus=1, cpus=0, pool="checks")]
    wants = ["unknown priority class", "declares max_s", "a service has no max_s", "only a service may be quiet_safe",
             "only a batch job or a session may be quiet", "gpu_ids names 1 GPUs", "at least one CPU", "unknown node",
             "declares expected_s", "only a session has an expected_s", "unknown workstream", "names no CPU pool"]
    assert job("s-1", kind="session", max_s=None, expected_s=1800, gpus=1, cpus=0).problems(c) == []
    for j, want in zip(bad, wants):
        assert any(want in p for p in j.problems(c)), (j.id, j.problems(c))


def test_jobs_and_allocations_round_trip_through_dicts():
    j = job("a-1", gpus=2, locks=frozenset({"lean-store"}), inputs=frozenset({"hf:x@1"}), after=("z-1",), gpu_ids=(0, 1))
    assert model.job_from_dict(model.job_to_dict(j)) == j
    s = job("s-1", kind="session", max_s=None, expected_s=900, gpus=1, cpus=0, workstream="w")
    assert model.job_from_dict(model.job_to_dict(s)) == s
    a = alloc(j, (0, 1), "0-3", state="frozen")
    assert model.allocation_from_dict(model.allocation_to_dict(a)) == a
    with pytest.raises(model.ModelError, match="unknown job fields"):
        model.job_from_dict({"id": "x", "project": "p", "submitter": "s", "priority": "work", "gpu": 1})
    assert model.instant("2026-10-07T15:00:00Z") - model.instant("2026-10-07T14:00:00Z") == 3600
    with pytest.raises(model.ModelError, match="no time zone"):
        model.instant("2026-10-07T15:00:00")
    assert T0 == model.instant(T0)
