"""The node planner's rules, one scenario each (the random-workload check is test_simulation.py)."""
from _build import T0, alloc, cluster, job, node, one

from cluster import cpuset
from cluster.model import Share
from cluster.plan import Evict, Expire, Freeze, Idle, Refuse, Start, Thaw, Wait, apply, idle, plan


def kinds(actions):
    return [type(a).__name__ for a in actions]


def by_job(actions, cls):
    return {(a.alloc.job.id if cls is Start else a.job): a for a in actions if isinstance(a, cls)}


def test_gpu_job_gets_one_numa_node_and_whole_cores_there():
    acts = plan(one(), "n1", [], [job("a-1", gpus=2, cpus=4)], T0)
    (s,) = acts
    assert s.alloc.gpus == (0, 1) and s.alloc.cpus == cpuset.parse("0-3") and s.alloc.id == f"a-1@{T0}"


def test_gpus_follow_the_free_numa_node():
    busy = alloc(job("x-1", gpus=4, cpus=2), (0, 1, 2, 3), "0-1")
    (s,) = plan(one(), "n1", [busy], [job("a-1", gpus=1, cpus=2)], T0)
    assert s.alloc.gpus == (4,) and s.alloc.cpus == cpuset.parse("16-17")


def test_no_gpu_cpu_or_memory_is_given_twice():
    c = one(memory=20)
    acts = plan(c, "n1", [], [job("a-1", gpus=4, memory_gib=8), job("b-1", gpus=4, memory_gib=8),
                              job("c-1", gpus=0, memory_gib=8)], T0)
    starts = [a.alloc for a in acts if isinstance(a, Start)]
    assert len(starts) == 2 and not set(starts[0].gpus) & set(starts[1].gpus) and not starts[0].cpus & starts[1].cpus
    assert by_job(acts, Wait)["c-1"].reason == "no memory free"


def test_owner_reclaims_its_node_from_a_guest():
    c = cluster(node("n1", "p"), shares=(Share("g", "n1"),))
    guest = alloc(job("g-1", "g", gpus=8, priority="work"), tuple(range(8)), "0-1")
    acts = plan(c, "n1", [guest], [job("o-1", "p", gpus=1, priority="fill")], T0)
    assert acts[0] == Evict(guest.id, "o-1") and isinstance(acts[1], Start)


def test_a_guest_never_displaces_the_owner():
    c = cluster(node("n1", "p"), shares=(Share("g", "n1"),))
    owner = alloc(job("o-1", "p", gpus=8, priority="fill"), tuple(range(8)), "0-1")
    acts = plan(c, "n1", [owner], [job("g-1", "g", gpus=1, priority="timed")], T0)
    assert kinds(acts) == ["Wait"]


def test_guests_must_be_preemptible_and_never_quiet_and_need_a_share():
    c = cluster(node("n1", "p"), shares=(Share("g", "n1", max_job_memory_gib=64),))
    acts = plan(c, "n1", [], [job("g-1", "g", preempt="never"), job("g-2", "g", quiet=True, priority="timed"),
                              job("h-1", "h"), job("g-3", "g", memory_gib=128)], T0)
    r = by_job(acts, Refuse)
    assert "must be preemptible" in r["g-1"].reason and "only p runs quiet jobs" in r["g-2"].reason
    assert "h has no share" in r["h-1"].reason and "at most 64 GiB each" in r["g-3"].reason


def test_quiet_job_evicts_gpu_work_and_freezes_cpu_work_then_everything_thaws():
    c = cluster(node("n1", "p"), shares=(Share("g", "n1"),))
    cpu_own = alloc(job("o-1", preempt="freeze", max_s=3600), (), "0-1")
    cpu_guest = alloc(job("g-1", "g", max_s=3600), (), "2-3")
    gpu_fill = alloc(job("o-2", gpus=2, priority="fill"), (0, 1), "4-5")
    window = job("w-1", gpus=8, cpus=8, priority="timed", quiet=True, max_s=1200)
    acts = plan(c, "n1", [cpu_own, cpu_guest, gpu_fill], [window, job("o-3")], T0)
    assert Evict(gpu_fill.id, "w-1") in acts
    assert {a.alloc for a in acts if isinstance(a, Freeze)} == {cpu_own.id, cpu_guest.id}
    assert by_job(acts, Start)["w-1"].alloc.gpus == tuple(range(8))
    assert by_job(acts, Wait)["o-3"].reason == "node is quiet"
    during = apply([cpu_own, cpu_guest, gpu_fill], acts, T0)
    assert kinds(plan(c, "n1", during, [job("o-4")], T0 + 60)) == ["Wait"]
    after = [a for a in during if not a.job.quiet]
    assert sorted(a.alloc for a in plan(c, "n1", after, [], T0 + 1200) if isinstance(a, Thaw)) == sorted([cpu_own.id, cpu_guest.id])


def test_quiet_safe_service_keeps_running_through_a_window():
    svc = alloc(job("s-1", kind="service", priority="service", preempt="never", quiet_safe=True, cpus=1), (), "0")
    acts = plan(one(), "n1", [svc], [job("w-1", gpus=8, priority="timed", quiet=True)], T0)
    assert kinds(acts) == ["Start"]


def test_quiet_job_waits_for_a_session_it_may_not_stop_and_the_node_backfills_meanwhile():
    session = alloc(job("s-1", gpus=1, kind="session", priority="session", preempt="never", max_s=900), (0,), "0-1")
    window = job("w-1", gpus=8, priority="timed", quiet=True)
    short = job("f-1", gpus=1, priority="fill", preempt="freeze", max_s=300)
    long_kill = job("f-2", gpus=1, priority="fill", max_s=3000)
    long_keep = job("o-1", gpus=1, priority="work", preempt="freeze", max_s=3000)
    acts = plan(one(), "n1", [session], [window, short, long_kill, long_keep], T0)
    w = by_job(acts, Wait)
    assert w["w-1"].start_by == T0 + 900 and "waits for s-1" in w["w-1"].reason
    assert set(by_job(acts, Start)) == {"f-1", "f-2"}
    assert "reserved for w-1" in w["o-1"].reason


def test_a_one_gpu_window_clears_the_node_of_fill_but_starts_beside_a_session_it_may_not_stop():
    """Rule 3's partial window, as `gpu-lease --timed` runs today (infra's and PoUW's rulings until the 7 Oct clamp)."""
    session = alloc(job("s-1", gpus=1, kind="session", priority="session", preempt="never", max_s=900), (0,), "0-1")
    fill = alloc(job("f-1", gpus=1, priority="fill", max_s=3000), (1,), "2-3")
    acts = plan(one(), "n1", [session, fill], [job("w-1", gpus=1, priority="timed", quiet=True, max_s=600)], T0)
    assert by_job(acts, Start)["w-1"].alloc.gpus == (1,) and [a.alloc for a in acts if isinstance(a, Evict)] == [fill.id]
    full = [alloc(job(f"s-{i}", gpus=1, kind="session", priority="session", preempt="never", max_s=900), (i,), str(i))
            for i in range(8)]
    assert "waits for" in by_job(plan(one(), "n1", full, [job("w-1", gpus=1, priority="timed", quiet=True)], T0), Wait)["w-1"].reason


def test_whole_node_request_is_not_starved_by_short_leases():
    lease = alloc(job("a-1", gpus=1, preempt="never", max_s=900), (0,), "0-1")
    big = job("b-1", gpus=8, priority="work", preempt="never")
    fits_before = job("c-1", gpus=1, priority="work", preempt="never", max_s=600)
    would_delay = job("c-2", gpus=1, priority="work", preempt="never", max_s=1200)
    acts = plan(one(), "n1", [lease], [big, fits_before, would_delay], T0)
    assert by_job(acts, Wait)["b-1"].start_by == T0 + 900
    assert set(by_job(acts, Start)) == {"c-1"} and "reserved for b-1" in by_job(acts, Wait)["c-2"].reason


def test_no_job_runs_past_its_nodes_admission_end_or_its_share():
    c = cluster(node("n1", "p", stops_at=T0 + 3600, drain_s=600), shares=(Share("g", "n1", until=T0 + 1000),))
    acts = plan(c, "n1", [], [job("a-1", max_s=3000), job("a-2", max_s=3001), job("g-1", "g", max_s=1001)], T0)
    assert set(by_job(acts, Start)) == {"a-1"}
    r = by_job(acts, Refuse)
    assert "admission end" in r["a-2"].reason and "share on n1 ends" in r["g-1"].reason
    late = plan(c, "n1", [], [job("s-1", kind="service", priority="service", preempt="never")], T0 + 3000)
    assert "admission end" in late[0].reason


def test_frozen_time_does_not_count_against_max_s_but_the_admission_end_is_a_hard_cut():
    c = one(stops_at=T0 + 5000, drain_s=1000)
    a = alloc(job("o-1", preempt="freeze", max_s=1000), (), "0-1")
    frozen = apply([a], [Freeze(a.id, "w-1")], T0 + 400)
    thawed = apply(frozen, [Thaw(a.id)], T0 + 1000)
    assert thawed[0].frozen_s == 600 and thawed[0].running_s(T0 + 1300) == 700 and thawed[0].ends_by == T0 + 1600
    assert plan(c, "n1", thawed, [], T0 + 1599) == [] and plan(c, "n1", thawed, [], T0 + 1600) == [Expire(a.id)]
    held = apply([a], [Freeze(a.id, "w-1")], T0 + 100)
    assert Thaw(a.id) in plan(c, "n1", held, [], T0 + 3999) and plan(c, "n1", held, [], T0 + 4000) == [Expire(a.id)]


def test_guest_share_caps_and_cpus():
    c = cluster(node("n1", "p"), shares=(Share("g", "n1", cpus=cpuset.parse("20-23"), max_jobs=1, max_gpus=0),))
    acts = plan(c, "n1", [], [job("g-1", "g", cpus=4), job("g-2", "g", cpus=1), job("g-3", "g", gpus=1, submitted=T0 + 1)], T0)
    assert by_job(acts, Start)["g-1"].alloc.cpus == cpuset.parse("20-23")
    w = by_job(acts, Wait)
    assert "share of jobs" in w["g-2"].reason and "share of" in w["g-3"].reason


def test_reserved_pools_go_only_to_jobs_naming_them_even_while_idle():
    c = one(pools={"checks": "8-15"})
    acts = plan(c, "n1", [], [job("t-1", cpus=8, pool="checks"), job("o-1", cpus=24), job("o-2", cpus=1, submitted=T0 + 1)], T0)
    got = by_job(acts, Start)
    assert got["t-1"].alloc.cpus == cpuset.parse("8-15")
    assert not got["o-1"].alloc.cpus & cpuset.parse("8-15") and len(got["o-1"].alloc.cpus) == 24
    assert by_job(acts, Wait)["o-2"].reason == "no CPUs free"
    assert "has no CPU pool 'lean'" in plan(c, "n1", [], [job("x-1", pool="lean")], T0)[0].reason


def test_named_locks_are_exclusive_here_and_across_the_cluster():
    lean = dict(locks=frozenset({"lean-store"}))
    acts = plan(one(), "n1", [], [job("a-1", **lean), job("b-1", **lean, submitted=T0 + 1)], T0)
    assert set(by_job(acts, Start)) == {"a-1"} and by_job(acts, Wait)["b-1"].reason == "lock lean-store is held"
    elsewhere = plan(one(), "n1", [], [job("c-1", **lean)], T0, held_locks=frozenset({"lean-store"}))
    assert kinds(elsewhere) == ["Wait"]


def test_equal_jobs_take_turns_by_submitter():
    running = alloc(job("a-1", gpus=7), tuple(range(7)), "0-1")
    acts = plan(one(), "n1", [running], [job("a-2", gpus=1), job("b-1", gpus=1, submitted=T0 + 5)], T0)
    assert set(by_job(acts, Start)) == {"b-1"}


def test_expiry_dependencies_and_staged_inputs():
    old = alloc(job("a-1", max_s=600), (), "0-1", started=T0 - 600)
    acts = plan(one(), "n1", [old], [job("b-1", after=("a-0",)), job("c-1", inputs=frozenset({"hf:x@1"}))], T0)
    assert acts[0] == Expire(old.id)
    w = by_job(acts, Wait)
    assert w["b-1"].reason == "after a-0" and w["c-1"].reason == "inputs not staged: hf:x@1"
    assert set(by_job(plan(one(), "n1", [], [job("c-1", inputs=frozenset({"hf:x@1"}))], T0, has=frozenset({"hf:x@1"})),
                      Start)) == {"c-1"}


def test_idle_leases_are_named_and_never_acted_on():
    fill = alloc(job("f-1", gpus=4, priority="work"), (0, 1, 2, 3), "0-1")
    session = alloc(job("s-1", gpus=4, kind="session", priority="session", preempt="never"), (4, 5, 6, 7), "16-17")
    acts = idle(one(), "n1", [fill, session], {fill.id: T0 + 10}, T0 + 700, idle_s={"batch": 600, "session": 600})
    assert acts == [Idle(fill.id, "f", 690), Idle(session.id, "s", 700)]
    assert idle(one(), "n1", [fill], {fill.id: T0 + 200}, T0 + 700, idle_s={"batch": 600}) == []
    assert idle(one(), "n1", [fill], {}, T0 + 700, idle_s={"session": 600}) == []


def test_a_workstream_reclaims_its_own_gpus_from_a_borrower_whatever_the_borrowers_priority():
    from cluster.model import Workstream
    c = cluster(node("n1", "p", gpus=4), workstreams=(Workstream("benches", "n1", 1, borrow_gpus=0),
                                                       Workstream("deploy", "n1", 3, borrow_gpus=1)))
    borrowed = alloc(job("d-1", gpus=4, priority="work", workstream="deploy", max_s=3000), (0, 1, 2, 3), "0-1")
    bench = job("b-1", gpus=1, priority="fill", workstream="benches")
    acts = plan(c, "n1", [borrowed], [bench], T0)
    assert acts[0] == Evict(borrowed.id, "b-1") and by_job(acts, Start)["b-1"].alloc.gpus == (0,)
    # within its own GPUs a deployment keeps them against a higher-priority job that has no workstream
    own = alloc(job("d-2", gpus=3, priority="fill", workstream="deploy", max_s=3000), (1, 2, 3), "2-3")
    assert kinds(plan(c, "n1", [own], [job("x-1", gpus=4, priority="work")], T0)) == ["Wait"]
    # benches never borrow: a second bench waits even with GPUs free, and a 2-GPU bench is refused outright
    one_bench = alloc(job("b-0", gpus=1, priority="bench", workstream="benches"), (0,), "4-5")
    w = plan(c, "n1", [one_bench], [job("b-2", gpus=1, priority="bench", workstream="benches"),
                                    job("b-3", gpus=2, priority="bench", workstream="benches")], T0)
    assert by_job(w, Wait)["b-2"].reason == "at workstream benches's GPUs on n1"
    assert "benches may hold" in by_job(w, Refuse)["b-3"].reason


def test_a_workstreams_share_is_gpus_so_a_job_asking_none_reclaims_nothing_with_it():
    """kueue-fold's divergence on vy-nebius-1: two CPU-only sweeps evicted borrowing GPU Builds for memory."""
    from cluster.model import Workstream
    c = cluster(node("n1", "p", gpus=4, memory=64), workstreams=(Workstream("benches", "n1", 1, borrow_gpus=0),
                                                                  Workstream("deploy", "n1", 3, borrow_gpus=1)))
    borrowed = alloc(job("d-1", gpus=4, priority="work", workstream="deploy", max_s=3000, memory_gib=60), (0, 1, 2, 3), "0-1")
    sweep = job("b-1", gpus=0, priority="fill", workstream="benches", memory_gib=8)
    assert by_job(plan(c, "n1", [borrowed], [sweep], T0), Wait)["b-1"].reason == "no memory free"


def test_a_session_with_only_an_expected_length_plans_backfill_but_is_never_expired():
    session = alloc(job("s-1", gpus=1, cpus=0, kind="session", priority="session", preempt="never", max_s=None,
                        expected_s=900), (0,), "")
    assert session.ends_by == T0 + 900 and plan(one(), "n1", [session], [], T0 + 5000) == []
    window = job("w-1", gpus=8, priority="timed", quiet=True)
    w = by_job(plan(one(), "n1", [session], [window], T0 + 60), Wait)["w-1"]
    assert w.start_by == T0 + 900
    # overrun: the window expects it gone now, as gpu-lease counts a lease past its rule
    assert by_job(plan(one(), "n1", [session], [window], T0 + 1000), Wait)["w-1"].start_by == T0 + 1000
    # it still ends at the admission end, and it isn't admitted when its expected length crosses it
    c = one(stops_at=T0 + 3600, drain_s=600)
    assert plan(c, "n1", [session], [], T0 + 3000) == [Expire(session.id)]
    late = job("s-2", kind="session", priority="session", preempt="never", max_s=None, expected_s=1200, gpus=1, cpus=0)
    assert "admission end" in plan(c, "n1", [], [late], T0 + 2000)[0].reason


def test_a_cpu_unmanaged_lease_gets_gpus_and_memory_but_no_cpus():
    (s,) = plan(one(), "n1", [], [job("g-1", gpus=2, cpus=0, memory_gib=190)], T0)
    assert s.alloc.gpus == (0, 1) and s.alloc.cpus == frozenset() and s.alloc.memory_gib == 190


def test_a_gpu_pool_bounds_which_gpus_a_job_gets_and_what_it_evicts():
    fill = [alloc(job(f"f-{i}", gpus=1, priority="fill", max_s=3000), (i,), str(i)) for i in range(8)]
    j = job("w-1", gpus=1, priority="work", gpu_pool=(6, 7), max_s=600)
    acts = plan(one(), "n1", fill, [j], T0)
    [s] = [a for a in acts if isinstance(a, Start)]
    assert s.alloc.gpus in ((6,), (7,)) and [a.alloc for a in acts if isinstance(a, Evict)] == [f"f-{s.alloc.gpus[0]}@{T0}"]
    assert "gpu_pool offers 2 GPUs" in plan(one(), "n1", [], [job("w-2", gpus=3, gpu_pool=(6, 7))], T0)[0].reason
