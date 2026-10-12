"""The rule checks the simulation and the agent share: each violation is named, and a clean plan reports none."""
from _build import T0, alloc, cluster, job, node, one

from cluster import invariants
from cluster.model import Share
from cluster.plan import Evict, Start, apply, plan


def test_a_clean_plan_reports_nothing():
    c = one()
    before = [alloc(job("f-1", gpus=2, priority="fill"), (0, 1), "0-1")]
    acts = plan(c, "n1", before, [job("w-1", gpus=8, priority="timed", quiet=True)], T0)
    after = apply(before, acts, T0)
    assert invariants.check(c, "n1", after, T0, actions=acts, before=before) == []


def test_each_violation_is_named():
    c = cluster(node("n1", "p", memory=16, stops_at=T0 + 1000), shares=(Share("g", "n1", max_jobs=1),))
    a = alloc(job("a-1", gpus=1, memory_gib=10), (0,), "0-1")
    b = alloc(job("b-1", gpus=1, memory_gib=10, preempt="never"), (0,), "1-2")
    g = alloc(job("g-1", "g", preempt="never", memory_gib=1), (), "3")
    g2 = alloc(job("g-2", "g", memory_gib=1), (), "4")
    q = alloc(job("q-1", gpus=1, quiet=True, priority="timed", memory_gib=1), (5,), "5")
    late = alloc(job("l-1", max_s=2000, memory_gib=1), (), "6", started=T0)
    got = " | ".join(invariants.check(c, "n1", [a, b, g, g2, q, late], T0,
                                      actions=[Start(late), Evict(b.id, "l-1")], before=[a, b, g, g2, q]))
    for want in ("GPU 0 is in 2 allocations", "CPU 1 is in 2 allocations", "memory overcommitted", "runs beside quiet q-1",
                 "admitted past the admission end", "started inside a quiet window", "guest g-1@1790000000 is quiet or not preemptible",
                 "g's share of jobs exceeded", "evicted though it isn't preemptible", "evicted by an equal or lower rank"):
        assert want in got, want
    assert invariants.check(c, "n1", [a], T0 + 1000) == [f"n1: {a.id} outlives the admission end"]
    session = alloc(job("s-1", gpus=1, preempt="never", memory_gib=1), (1,), "7")
    assert invariants.check(c, "n1", [session, q], T0) == [], "a 1-GPU window may run beside what it may not displace"
    assert "runs beside quiet q-1" in " ".join(invariants.check(c, "n1", [a, session, q], T0)), "but not beside fill"
