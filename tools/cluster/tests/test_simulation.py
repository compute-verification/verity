"""Random workloads on the Nebius description and on a small cluster: the planner's rules hold on every tick
(`invariants.check`), the ledger alone replays into the simulator's allocations and queue on every tick (`ledger.state`),
and every job ends, expires or is refused (nothing waits forever). Seeded, and time is simulated, never read."""
import random
from pathlib import Path

import pytest
from _build import PRIO, T0, alloc, cluster, job, node

from cluster import invariants, ledger, model
from cluster.model import Job, Share, Workstream
from cluster.plan import Evict, Expire, Refuse, Start, Wait, apply, plan, whole_quiet
from cluster.route import View, place

NEBIUS = model.load(Path(__file__).parents[1] / "descriptions" / "nebius.toml")
SMALL = cluster(node("a", "p", memory=96), node("b", "q", memory=96, stops_at=T0 + 20_000, drain_s=1_000),
                shares=(Share("q", "a", max_gpus=2, max_jobs=3), Share("p", "b", max_memory_gib=48)),
                workstreams=(Workstream("benches", "a", 2, borrow_gpus=0), Workstream("deploy", "a", 4, borrow_gpus=2)))


def workload(c, rng, n, t0):
    projects = sorted({nd.owner for nd in c.nodes.values()})
    jobs = []
    for i in range(n):
        project = rng.choice(projects)
        shape = rng.random()
        kw = dict(id=f"j{i:03d}", project=project, submitter=f"{project}-{rng.randrange(3)}", submitted=t0 + rng.randrange(4000))
        streams = [w for w in c.workstreams if c.nodes[w.node].owner == project]
        if shape < 0.08:
            kw.update(priority="timed", quiet=True, gpus=rng.choice([1, 8]), cpus=8, memory_gib=64, max_s=rng.choice([600, 1200]),
                      node=next(nd.name for nd in c.nodes.values() if nd.owner == project))
        elif shape < 0.18:
            # a lease by hand: half with a hard limit, half with only an expected length (gpu-lease without --max-min), some
            # CPU-unmanaged as gpu-lease's are
            kw.update(kind="session", priority="session", preempt="never", gpus=1, cpus=rng.choice([0, 4]), memory_gib=32)
            kw.update(max_s=rng.choice([600, 1800])) if rng.random() < 0.5 else kw.update(expected_s=rng.choice([600, 1800]))
        elif shape < 0.3 and streams:
            w = rng.choice(streams)
            kw.update(priority=rng.choice(["work", "fill"]), gpus=rng.choice([1, 2]), cpus=2, memory_gib=16, workstream=w.name,
                      node=w.node, max_s=rng.choice([300, 900]), preempt=rng.choice(["never", "requeue"]))
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
    for name in c.nodes:
        assert invariants.check(c, name, live, now, actions=acts_by_node[name], before=acts_by_node[f"{name}:before"]) == []
    locks = [lk for a in live for lk in a.job.locks]
    assert len(locks) == len(set(locks)), "a lock is held twice across the cluster"


def replayed_matches(c, st, live, queue, routed):
    """The ledger's state is the simulator's: the same allocations, and exactly the routed jobs not yet started queued."""
    assert st.allocations == {a.id: a for a in live}
    assert {j: (q.node, q.job) for j, q in st.queue.items()} == {j.id: (routed[j.id], j) for j in queue if routed.get(j.id)}
    assert not {a.job.id for a in st.allocations.values()} & set(st.queue), "a holder is also a waiter"


def simulate(c, seed, n, tmp_path):
    rng = random.Random(seed)
    jobs = sorted(workload(c, rng, n, T0), key=lambda j: j.submitted)
    need = {j.id: rng.randrange(1, j.length_s) for j in jobs}
    left: dict[str, int] = {}
    live, queue, routed, done, finished = [], [], {}, set(), {}
    path = tmp_path / f"ledger-{seed}.jsonl"
    lg = ledger.Ledger(path)
    st, seen = ledger.state([]), 0
    now = T0
    for _ in range(4000):
        while jobs and jobs[0].submitted <= now:
            queue.append(jobs.pop(0))
        views = {name: View(allocations=tuple(a for a in live if a.node == name)) for name in c.nodes}
        for j in queue:
            if j.id not in routed:
                held = frozenset(lk for a in live for lk in a.job.locks)
                routed[j.id] = place(c, j, views, now, held).node
                if routed[j.id] is not None:
                    lg.submit(now, j, routed[j.id])
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
        records = ledger.read(path) if path.exists() else []
        st, seen = ledger.state(records[seen:], st), len(records)
        replayed_matches(c, st, live, queue, routed)
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
                    lg.end(now, a.id, 0)
    assert not jobs and not queue and not live, f"still waiting at the end: {[j.id for j in queue]}"
    records = ledger.read(path)
    assert ledger.verify(records) == []
    assert ledger.state(records).allocations == {} and ledger.state(records).queue == {}
    for r in records:
        if r["kind"] == "start" and r["alloc"]["job"].get("quiet"):
            rep = ledger.quiet_report(records, r["alloc"]["id"])
            if whole_quiet(c, r["alloc"]["node"], model.job_from_dict(r["alloc"]["job"])):
                assert rep["quiet"], "the ledger can't certify a quiet window"
            else:
                beside = [o for o in rep["overlaps"] if o["state"] == "running" and not o["quiet_safe"]]
                assert all(o["t0"] == rep["running"][0][0] for o in beside), "something started inside a partial window"
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


def test_nebius_workloads_run_verity_as_a_guest_on_node_2(tmp_path):
    guests = [r for seed in range(3) for r in simulate(NEBIUS, seed, 120, tmp_path)[1]
              if r["kind"] == "start" and r["alloc"]["job"]["project"] == "verity" and r["alloc"]["node"] == "vy-nebius-2"]
    assert guests


def test_workstreams_reclaim_what_they_lend_and_never_exceed_what_they_may_borrow(tmp_path):
    finished, records = simulate(SMALL, 11, 80, tmp_path)
    assert len(finished) == 80
    starts = [r["alloc"] for r in records if r["kind"] == "start"]
    assert {a["job"].get("workstream") for a in starts} >= {"benches", "deploy"}


def test_a_lease_holder_is_never_a_waiter_and_a_dead_waiter_leaves_the_queue(tmp_path):
    """pous infra's 18:07Z failure mode, in the model: a `gpu-lease 1 --wait` that holds its GPU kept `--wait` in its argv and
    was counted as a waiter. Here held versus waiting is only the ledger's: a start takes the job out of the queue, and a
    waiter's queue entry is bound to its live handle, so when the handle drops (the waiter died) the agent withdraws it and
    nothing still counts it."""
    c = cluster(node("n1", "p", gpus=2))
    lg = ledger.Ledger(tmp_path / "ledger.jsonl")
    holder = job("h-1", gpus=1, cpus=0, kind="session", priority="session", preempt="never", max_s=None, expected_s=1800)
    waiter = job("w-1", gpus=2, cpus=0, kind="session", priority="session", preempt="never", max_s=None, expected_s=600, submitted=T0 + 5)
    late = job("x-1", gpus=1, cpus=0, kind="session", priority="session", preempt="never", max_s=None, expected_s=300, submitted=T0 + 9)
    lg.submit(T0, holder, "n1", handle="wait.1.101")
    lg.actions(T0, plan(c, "n1", [], [holder], T0))
    for j, t in ((waiter, T0 + 5), (late, T0 + 9)):
        lg.submit(t, j, "n1", handle=f"wait.{t}.{j.id}")
    st = ledger.state(ledger.read(tmp_path / "ledger.jsonl"))
    allocs, queue = st.on("n1")
    assert [a.job.id for a in allocs] == ["h-1"] and [j.id for j in queue] == ["w-1", "x-1"]
    acts = plan(c, "n1", allocs, queue, T0 + 10)
    # the 2-GPU waiter can't start while h-1 holds a GPU; the late 1-GPU lease has only an expected length, so it may not
    # backfill ahead of w-1's reservation
    assert {type(a).__name__ for a in acts} == {"Wait"} and next(a for a in acts if a.job == "w-1").start_by == T0 + 1800
    lg.withdraw(T0 + 20, "w-1", "handle wait.T.w-1 dropped: the waiter is gone")
    st = ledger.state(ledger.read(tmp_path / "ledger.jsonl"))
    allocs, queue = st.on("n1")
    assert [j.id for j in queue] == ["x-1"]
    (start,) = plan(c, "n1", allocs, queue, T0 + 25)
    assert isinstance(start, Start) and start.alloc.gpus == (1,) and start.alloc.cpus == frozenset()


def test_priorities_match_the_description():
    assert NEBIUS.priorities == PRIO
