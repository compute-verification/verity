"""Random workloads on the Nebius description and on a small cluster: the planner's rules hold on every tick, and every job
ends, expires or is refused (nothing waits forever). Seeded, and time is simulated, never read."""
import random
from pathlib import Path

import pytest
from _build import PRIO, T0, cluster, node

from cluster import ledger, model
from cluster.model import Job, Share
from cluster.plan import Evict, Expire, Freeze, Refuse, Start, Thaw, apply, plan
from cluster.route import View, place

NEBIUS = model.load(Path(__file__).parents[1] / "descriptions" / "nebius.toml")
SMALL = cluster(node("a", "p", memory=96), node("b", "q", memory=96, stops_at=T0 + 20_000, drain_s=1_000),
                shares=(Share("q", "a", max_gpus=2, max_jobs=3), Share("p", "b", max_memory_gib=48)))


def workload(c, rng, n, t0):
    projects = sorted({nd.owner for nd in c.nodes.values()})
    jobs = []
    for i in range(n):
        project = rng.choice(projects)
        shape = rng.random()
        kw = dict(id=f"j{i:03d}", project=project, submitter=f"{project}-{rng.randrange(3)}", submitted=t0 + rng.randrange(4000))
        if shape < 0.08:
            kw.update(priority="timed", quiet=True, gpus=rng.choice([1, 8]), cpus=8, memory_gib=64, max_s=rng.choice([600, 1200]),
                      node=next(nd.name for nd in c.nodes.values() if nd.owner == project))
        elif shape < 0.18:
            kw.update(kind="session", priority="session", preempt="never", gpus=1, cpus=4, memory_gib=32,
                      max_s=rng.choice([600, 1800]))
        elif shape < 0.5:
            kw.update(priority=rng.choice(["work", "fill"]), gpus=rng.choice([1, 1, 2]), cpus=rng.choice([2, 8]),
                      memory_gib=rng.choice([16, 64]), max_s=rng.choice([300, 900, 1500]), preempt="requeue")
        elif shape < 0.6:
            kw.update(priority="work", cpus=16, memory_gib=48, max_s=1800, preempt="freeze", locks=frozenset({"lean-store"}))
        elif shape < 0.66 and any(nd.pools for nd in c.nodes.values() if nd.owner == project):
            home = next(nd for nd in c.nodes.values() if nd.owner == project and nd.pools)
            kw.update(priority="work", cpus=32, memory_gib=64, max_s=1800, preempt="never", node=home.name,
                      pool=sorted(home.pools)[0])
        else:
            kw.update(priority=rng.choice(["work", "dev", "fill"]), cpus=rng.choice([4, 8, 32]),
                      memory_gib=rng.choice([8, 32, 128]), max_s=rng.choice([300, 1200, 3600]),
                      preempt=rng.choice(["freeze", "requeue"]))
        jobs.append(Job(**kw))
    return jobs


def check(c, live, acts_by_node, now):
    for name, nd in c.nodes.items():
        mine = [a for a in live if a.node == name]
        gpus = [g for a in mine for g in a.gpus]
        cpus = [x for a in mine for x in a.cpus]
        assert len(gpus) == len(set(gpus)), "a GPU is in two allocations"
        assert len(cpus) == len(set(cpus)), "a CPU is in two allocations"
        assert set(cpus) <= nd.cpus and sum(a.memory_gib for a in mine) <= nd.memory_gib
        for a in mine:
            assert a.cpus <= nd.pools[a.job.pool] if a.job.pool else not a.cpus & nd.pooled_cpus, "a pool's CPUs went astray"
        quiet = [a for a in mine if a.job.quiet and a.state == "running"]
        assert len(quiet) <= 1
        if quiet:
            assert all(a.state == "frozen" or a.job.quiet_safe for a in mine if a is not quiet[0]), "a quiet node isn't quiet"
        else:
            assert all(a.state == "running" for a in mine), "frozen with no quiet job"
        if nd.admission_ends is not None and now >= nd.admission_ends:
            assert all(a.job.kind == "service" for a in mine), "a job outlives its node's admission end"
        for a in (x.alloc for x in acts_by_node[name] if isinstance(x, Start)):
            assert nd.admission_ends is None or a.started + a.job.max_s <= nd.admission_ends, "admitted past the admission end"
        for a in mine:
            share = c.share(a.job.project, name) if nd.owner != a.job.project else None
            if share:
                assert share.cpus is None or a.cpus <= share.cpus
                assert not a.job.quiet and a.job.preempt != "never"
        for s in (share for share in c.shares if share.node == name):
            theirs = [a for a in mine if a.job.project == s.project]
            for cap, used in ((s.max_gpus, sum(len(a.gpus) for a in theirs)), (s.max_jobs, len(theirs)),
                              (s.max_memory_gib, sum(a.memory_gib for a in theirs))):
                assert cap is None or used <= cap, "a share's cap is exceeded"
        locks = [lk for a in live for lk in a.job.locks]
        assert len(locks) == len(set(locks)), "a lock is held twice"
        starts = {a.alloc.job.id: a.alloc for a in acts_by_node[name] if isinstance(a, Start)}
        before = {a.id: a for a in acts_by_node.get(f"{name}:before", [])}
        for e in (a for a in acts_by_node[name] if isinstance(a, Evict)):
            assert c.rank(starts[e.by].job, name) > c.rank(before[e.alloc].job, name), "evicted by an equal or lower rank"


def simulate(c, seed, n, tmp_path):
    rng = random.Random(seed)
    jobs = sorted(workload(c, rng, n, T0), key=lambda j: j.submitted)
    need = {j.id: rng.randrange(1, j.max_s) for j in jobs}
    left: dict[str, int] = {}
    live, queue, routed, done, finished = [], [], {}, set(), {}
    lg = ledger.Ledger(tmp_path / f"ledger-{seed}.jsonl")
    now = T0
    for _ in range(4000):
        while jobs and jobs[0].submitted <= now:
            queue.append(jobs.pop(0))
        views = {name: View(allocations=tuple(a for a in live if a.node == name)) for name in c.nodes}
        for j in queue:
            if j.id not in routed:
                held = frozenset(lk for a in live for lk in a.job.locks)
                routed[j.id] = place(c, j, views, now, held).node
        for j in [j for j in queue if routed[j.id] is None]:
            queue.remove(j)
            finished[j.id] = "refused by routing"
        acts_by_node = {}
        for name in sorted(c.nodes):
            mine = [j for j in queue if routed.get(j.id) == name]
            held = frozenset(lk for a in live if a.node != name for lk in a.job.locks)
            acts = plan(c, name, live, mine, now, done=frozenset(done), held_locks=held)
            acts_by_node[name], acts_by_node[f"{name}:before"] = acts, list(live)
            live = apply(live, acts, now)
            lg.actions(now, acts)
            for act in acts:
                if isinstance(act, Start):
                    queue.remove(act.alloc.job)
                    left[act.alloc.id] = need[act.alloc.job.id]
                elif isinstance(act, Evict):
                    job = next(a.job for a in acts_by_node[f"{name}:before"] if a.id == act.alloc)
                    queue.append(job)
                    routed.pop(job.id)
                elif isinstance(act, Expire):
                    finished[act.alloc.split("@")[0]] = "expired"
                elif isinstance(act, Refuse):
                    queue.remove(next(j for j in queue if j.id == act.job))
                    finished[act.job] = "refused"
        check(c, live, acts_by_node, now)
        if not jobs and not queue and not live:
            break
        step = rng.choice([30, 60, 120])
        now += step
        for a in list(live):
            if a.state == "running":
                left[a.id] -= step
                if left[a.id] <= 0:
                    live.remove(a)
                    done.add(a.job.id)
                    finished[a.job.id] = "done"
                    lg.append(now, "end", alloc=a.id, rc=0)
    assert not jobs and not queue and not live, f"still waiting at the end: {[j.id for j in queue]}"
    records = ledger.read(tmp_path / f"ledger-{seed}.jsonl")
    assert ledger.verify(records) == []
    for r in records:
        if r["kind"] == "start" and r["alloc"]["job"].get("quiet"):
            assert ledger.quiet_report(records, r["alloc"]["id"])["quiet"], "the ledger can't certify a quiet window"
    return finished, records


def exercised(records):
    """Every rule had something to do: a quiet window, a freeze and its thaw, an eviction, a guest."""
    quiet = [r for r in records if r["kind"] == "start" and r["alloc"]["job"].get("quiet")]
    return bool(quiet) and {"freeze", "thaw", "evict"} <= {r["kind"] for r in records}


def test_small_cluster(tmp_path):
    records = []
    for seed in range(6):
        finished, recs = simulate(SMALL, seed, 60, tmp_path)
        assert len(finished) == 60 and sum(v == "done" for v in finished.values()) >= 45
        records += recs
    assert exercised(records)


@pytest.mark.parametrize("seed", range(3))
def test_nebius(seed, tmp_path):
    finished, records = simulate(NEBIUS, seed, 120, tmp_path)
    assert list(finished.values()).count("done") == 120 and exercised(records)
    guests = [r for r in records if r["kind"] == "start" and r["alloc"]["job"]["project"] == "verity"
              and r["alloc"]["node"] == "vy-nebius-2"]
    assert guests


def test_priorities_match_the_description():
    assert NEBIUS.priorities == PRIO
