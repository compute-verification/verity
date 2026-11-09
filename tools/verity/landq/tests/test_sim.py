"""The simulator's world on small traces whose outcome is known: timing, the cache by key, capacity, ejects and faults.  The
queue is the contract's small fake (`fake_vqueue`) through `snapshots.Driver`, or a scripted one where a test reads the
world alone."""
import types

import fake_vqueue as vq
import pytest

from verity_landq import model as lq
from verity_landq import sim, snapshots

A = sim.Task("unit:a", frozenset({"a/"}), 10.0)
B = sim.Task("unit:b", frozenset({"b/"}), 30.0)
W = sim.World(tasks=(A, B), nodes=2, build=2.0, listing=3.0)
ONE = lq.Policy(batch=1, depth=2)
Q = snapshots.Driver(vq)


def simulate(trace: list[sim.Arrival], world: sim.World = W, policy: lq.Policy = ONE, q: snapshots.Driver = Q,
             **kw) -> sim.Report:
    return sim.simulate(trace, world, policy, step=q.step, state=q.start("main"), **kw)


def arrive(pr: int, t: float, *paths: str, breaks: tuple[str, ...] = (), reviewed: float | None = None) -> sim.Arrival:
    return sim.Arrival(pr, f"h{pr}", t, frozenset(paths), reviewed, frozenset(breaks))


def mains(r: sim.Report) -> dict[int, float | None]:
    return {x.pr: x.main for x in r.landings}


def test_a_pr_lands_after_its_build_its_listing_and_the_tasks_it_touches() -> None:
    r = simulate([arrive(1, 0.0, "a/x.py")])
    assert mains(r) == {1: 15.0}
    assert (r.runs, r.cached, r.candidates, r.advances) == (1, 1, 1, 1)
    assert sim.summary(r)["p50_h"] == pytest.approx(0.25)


def test_ready_to_main_counts_from_the_ready_mark_and_from_the_review() -> None:
    s = sim.summary(simulate([arrive(1, 0.0, "a/x.py", reviewed=60.0)]))
    assert (s["p50_h"], s["review_p50_h"]) == pytest.approx((1.25, 0.25))


def test_a_new_head_ends_the_hold_on_the_head_it_replaced() -> None:
    trace = [sim.Arrival(1, "h1", 0.0, frozenset({"a/x.py"})), sim.Arrival(1, "h2", 5.0, frozenset({"a/x.py"}))]
    r = simulate(trace)
    assert [(x.head, x.main) for x in r.landings] == [("h1", 15.0), ("h2", 30.0)]
    assert sorted(r.held) == [(0.0, 5.0), (5.0, 30.0)]


def test_a_red_task_of_the_start_fails_until_a_fix_and_results_carry_their_keys() -> None:
    seen: list[lq.Result] = []

    def step(s, e, p):
        if isinstance(e, lq.Result):
            seen.append(e)
        return Q.step(s, e, p)

    fix = sim.Arrival(1, "h1", 0.0, frozenset({"b/x.py"}), fixes=frozenset({"unit:b"}))
    r = sim.simulate([fix], sim.World(tasks=(A, B), red=frozenset({"unit:b"})), ONE, step=step, state=Q.start("main"))
    assert mains(r) == {1: 35.0} and not r.unsound
    got = {e.task: e for e in seen}
    assert got["unit:b"].status == lq.PASS and got["unit:b"].key != got["unit:b"].base_key
    assert got["unit:a"].status == lq.PASS and got["unit:a"].key == got["unit:a"].base_key


def test_a_report_can_end_at_a_minute_before_the_queue_is_done() -> None:
    r = simulate([arrive(1, 0.0, "a/x.py"), arrive(2, 100.0, "a/y.py")], until=50.0)
    assert (mains(r), r.end) == ({1: 15.0, 2: None}, 50.0)


def test_an_advance_is_unsound_only_when_its_tree_fails_a_test_main_did_not() -> None:
    s = sim._Sim((), sim.World(tasks=(A, B), red=frozenset({"unit:b"})), ONE, Q.step, "main", 10, Q.start("main"))
    s.trees["T1"] = ((0, 0), frozenset({(1, sim.RED)}))
    s.trees["T2"] = ((1, 0), frozenset({(0, "#5"), (1, sim.RED)}))
    s.trees["T3"] = ((1, 1), frozenset({(1, sim.RED), (1, "#7")}))
    for tree in ("T1", "T2", "T1", "T3"):
        s.do_Advance(lq.Advance(tree, ()))
    assert s.unsound == ["T2", "T3"]


def built_once(seen: list[lq.Result], rerun: bool = False):
    """A queue that builds the first PR alone on `main` as candidate 0, records each result, and reruns the first."""

    def step(s, e, p):
        if isinstance(e, lq.Ready):
            return s, (lq.Build(0, "main", (lq.Member(e.pr, e.head, e.paths),)),)
        if isinstance(e, lq.Result):
            seen.append(e)
            return s, (lq.Rerun(e.tree, e.task),) if rerun and len(seen) == 1 else ()
        return s, ()

    return step, types.SimpleNamespace(stack=[types.SimpleNamespace(tree="T0")])


def stacked(seen: list[tuple[str, str, float]]):
    """A queue that builds the first PR on `main` as candidate 0 and the second on its tree once it is built, and records
    when each result arrives."""

    def step(s, e, p):
        if isinstance(e, lq.Ready):
            s.ready.append(lq.Member(e.pr, e.head, e.paths))
            return s, (lq.Build(0, "main", (s.ready[0],)),) if len(s.ready) == 1 else ()
        if isinstance(e, lq.Built) and e.cid == 0:
            return s, (lq.Build(1, e.tree, (s.ready[1],)),)
        if isinstance(e, lq.Result):
            seen.append((e.tree, e.task, s.now))
        return s, ()

    return step, types.SimpleNamespace(stack=[types.SimpleNamespace(tree="T0"), types.SimpleNamespace(tree="T1")],
                                       ready=[], now=0.0)


def test_each_fail_names_its_failing_tests_and_the_base_s() -> None:
    seen: list[lq.Result] = []
    step, state = built_once(seen)
    sim.simulate([arrive(1, 0.0, "a/x.py", "b/x.py", breaks=("unit:a",))],
                 sim.World(tasks=(A, B), red=frozenset({"unit:b"})), ONE, step=step, state=state)
    assert [(e.task, e.status, e.tests, e.base_tests) for e in seen] == [
        ("unit:a", lq.FAIL, {"#1"}, None), ("unit:b", lq.FAIL, {sim.RED}, {sim.RED})]


def test_a_rerun_runs_the_task_afresh_past_its_cached_fail() -> None:
    seen: list[lq.Result] = []
    step, state = built_once(seen, rerun=True)
    r = sim.simulate([arrive(1, 0.0, "a/x.py")], sim.World(tasks=(A,), red=frozenset({"unit:a"})), ONE, step=step,
                     state=state)
    assert [(e.status, e.tests) for e in seen] == [(lq.FAIL, {sim.RED})] * 2
    assert (r.runs, r.cached, r.reruns) == (2, 0, 1)


def test_a_cold_start_runs_every_task_once_and_later_candidates_read_the_cache() -> None:
    r = simulate([arrive(1, 0.0, "a/x.py"), arrive(2, 100.0, "a/y.py")], sim.World(tasks=(A, B), warm=False))
    assert mains(r) == {1: 35.0, 2: 115.0}
    assert r.runs == 3


def test_a_second_candidate_waits_for_a_node_and_shares_the_run_below() -> None:
    trace = [arrive(1, 0.0, "a/x.py"), arrive(2, 0.0, "b/x.py")]
    got = {}
    for nodes in (1, 2):
        seen: list[tuple[str, str, float]] = []
        step, state = stacked(seen)
        r = sim.simulate(trace, sim.World(tasks=(A, B), nodes=nodes), ONE, step=step, state=state)
        got[nodes] = (sorted(seen), r.runs)
    assert got[1] == ([("T0", "unit:a", 15.0), ("T0", "unit:b", 5.0), ("T1", "unit:a", 15.0), ("T1", "unit:b", 45.0)], 2)
    assert got[2] == ([("T0", "unit:a", 15.0), ("T0", "unit:b", 5.0), ("T1", "unit:a", 15.0), ("T1", "unit:b", 37.0)], 2)


def test_review_after_the_mark_holds_the_pr_and_starts_its_lag_clock() -> None:
    r = simulate([arrive(1, 0.0, "a/x.py", reviewed=60.0)])
    assert mains(r) == {1: 75.0}
    assert r.held == ((60.0, 75.0),)


def test_a_breaking_pr_is_ejected_and_lands_once_its_fix_comes_back() -> None:
    r = simulate([arrive(1, 0.0, "a/x.py", breaks=("unit:a",)), arrive(2, 0.0, "b/x.py")],
                 sim.World(tasks=(A, B), fix=120.0), lq.Policy(batch=2, depth=1))
    rows = {x.pr: x for x in r.landings}
    assert (rows[1].ejects, rows[1].false_ejects, rows[1].head) == (1, 0, "h1+")
    assert rows[1].main == 15.0 + 120.0 + 15.0
    assert rows[2].main is not None and rows[2].ejects == 0
    assert not r.unsound


def test_a_break_must_name_a_listed_task() -> None:
    with pytest.raises(ValueError, match=r"#1@h1 breaks or fixes tasks the world doesn't list: \['unit:z'\]"):
        simulate([arrive(1, 0.0, "a/x.py", breaks=("unit:z",))])


def test_an_off_pool_task_runs_beside_the_nodes() -> None:
    slim = sim.Task("slim", frozenset({"."}), 20.0, off_pool=True)
    r = simulate([arrive(1, 0.0, "a/x.py", "b/x.py")], sim.World(tasks=(A, B, slim), nodes=1))
    assert mains(r) == {1: 2.0 + 3.0 + 10.0 + 30.0}
    assert (r.busy, r.runs) == (40.0, 3)


def test_a_deferred_task_runs_only_when_the_queue_starts_it() -> None:
    slim = sim.Task("slim", frozenset({"."}), 20.0, off_pool=True, deferred=True)
    r = simulate([arrive(1, 0.0, "a/x.py")], sim.World(tasks=(A, slim), nodes=1))
    assert (mains(r), r.runs) == ({1: None}, 1)


def test_infra_errors_retry_and_never_eject() -> None:
    trace = [arrive(i, 5.0 * i, "a/x.py" if i % 2 else "b/x.py") for i in range(1, 21)]
    retries = pages = 0
    for seed in range(5):
        s = sim.summary(simulate(trace, sim.World(tasks=(A, B), infra=0.5, seed=seed), lq.Policy(batch=3, depth=2)))
        assert s["landed"] == 20 and s["ejects"] == 0
        retries, pages = retries + s["retries"], pages + s["pages"]
    assert retries > 5 and pages > 0


def test_false_fails_eject_heads_that_break_nothing_and_every_pr_still_lands() -> None:
    trace = [arrive(i, 5.0 * i, "a/x.py" if i % 2 else "b/x.py") for i in range(1, 21)]
    false = 0
    for seed in range(5):
        r = simulate(trace, sim.World(tasks=(A, B), flaky=0.2, seed=seed), lq.Policy(batch=3, depth=2))
        assert all(x.main is not None for x in r.landings) and not r.unsound
        false += sim.summary(r)["false_ejects"]
    assert false > 0


def test_the_unsound_count_catches_a_queue_that_advances_on_a_failed_tree() -> None:
    trace = [arrive(1, 0.0, "a/x.py", breaks=("unit:a",))]
    assert not simulate(trace).unsound
    odd = types.SimpleNamespace(**{k: getattr(vq, k) for k in dir(vq) if not k.startswith("_")})
    odd.decide = lambda s: ([vq.Advance(s.candidates[0].sha)] if s.candidates and s.candidates[0].verdict.state == vq.FAIL
                            else vq.decide(s))
    assert simulate(trace, q=snapshots.Driver(odd)).unsound


def test_the_same_trace_and_seed_give_the_same_report() -> None:
    trace = [arrive(i, 3.0 * i, "a/x.py" if i % 3 else "b/x.py", breaks=("unit:a",) if i == 4 else ()) for i in range(1, 30)]
    world = sim.World(tasks=(A, B), infra=0.2, flaky=0.1, seed=7)
    assert simulate(trace, world, lq.Policy()) == simulate(trace, world, lq.Policy())


def test_summary_reads_lag_from_held_intervals_and_quantiles_as_the_scorer_does() -> None:
    r = sim.Report(landings=(sim.Landing(1, "h1", 0.0, 0.0, 30.0, 0, 0), sim.Landing(2, "h2", 0.0, 10.0, None, 0, 0)),
                   held=((0.0, 30.0), (10.0, 50.0)), end=50.0, nodes=1)
    s = sim.summary(r, every=10.0, lag=20.0)
    assert (s["prs"], s["landed"], s["p50_h"], s["within"]) == (2, 1, 0.5, 0.5)
    assert s["lag_max_h"] == pytest.approx(0.5)
    assert s["lag_ok"] == pytest.approx(5 / 6)
    assert sim.quantile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0
