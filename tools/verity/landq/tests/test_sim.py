"""The simulator's world on small traces whose outcome is known: timing, the cache by key, capacity, ejects and faults."""
import pytest

from verity_landq import model as lq
from verity_landq import sim

A = sim.Task("unit:a", frozenset({"a/"}), 10.0)
B = sim.Task("unit:b", frozenset({"b/"}), 30.0)
W = sim.World(tasks=(A, B), nodes=2, build=2.0, listing=3.0)
ONE = lq.Policy(batch=1, depth=2)


def arrive(pr: int, t: float, *paths: str, breaks: tuple[str, ...] = (), reviewed: float | None = None) -> sim.Arrival:
    return sim.Arrival(pr, f"h{pr}", t, frozenset(paths), reviewed, frozenset(breaks))


def mains(r: sim.Report) -> dict[int, float | None]:
    return {x.pr: x.main for x in r.landings}


def test_a_pr_lands_after_its_build_its_listing_and_the_tasks_it_touches() -> None:
    r = sim.simulate([arrive(1, 0.0, "a/x.py")], W, ONE)
    assert mains(r) == {1: 15.0}
    assert (r.runs, r.cached, r.candidates, r.advances) == (1, 1, 1, 1)
    assert sim.summary(r)["p50_h"] == pytest.approx(0.25)


def test_a_cold_start_runs_every_task_once_and_later_candidates_read_the_cache() -> None:
    r = sim.simulate([arrive(1, 0.0, "a/x.py"), arrive(2, 100.0, "a/y.py")], sim.World(tasks=(A, B), warm=False), ONE)
    assert mains(r) == {1: 35.0, 2: 115.0}
    assert r.runs == 3


def test_a_second_candidate_waits_for_a_node_and_shares_the_run_below() -> None:
    trace = [arrive(1, 0.0, "a/x.py"), arrive(2, 0.0, "b/x.py")]
    one = sim.simulate(trace, sim.World(tasks=(A, B), nodes=1), ONE)
    two = sim.simulate(trace, sim.World(tasks=(A, B), nodes=2), ONE)
    assert mains(one) == {1: 15.0, 2: 45.0}
    assert mains(two) == {1: 15.0, 2: 37.0}
    assert one.runs == two.runs == 2


def test_review_after_the_mark_holds_the_pr_and_starts_its_lag_clock() -> None:
    r = sim.simulate([arrive(1, 0.0, "a/x.py", reviewed=60.0)], W, ONE)
    assert mains(r) == {1: 75.0}
    assert r.held == ((60.0, 75.0),)


def test_a_breaking_pr_is_ejected_and_lands_once_its_fix_comes_back() -> None:
    r = sim.simulate([arrive(1, 0.0, "a/x.py", breaks=("unit:a",)), arrive(2, 0.0, "b/x.py")],
                     sim.World(tasks=(A, B), fix=120.0), lq.Policy(batch=2, depth=1))
    rows = {x.pr: x for x in r.landings}
    assert (rows[1].ejects, rows[1].false_ejects, rows[1].head) == (1, 0, "h1+")
    assert rows[1].main == 15.0 + 120.0 + 15.0
    assert rows[2].main is not None and rows[2].ejects == 0
    assert not r.unsound


def test_a_break_must_name_a_listed_task() -> None:
    with pytest.raises(ValueError, match=r"#1@h1 breaks tasks the world doesn't list: \['unit:z'\]"):
        sim.simulate([arrive(1, 0.0, "a/x.py", breaks=("unit:z",))], W, ONE)


def test_an_off_pool_task_runs_beside_the_nodes() -> None:
    slim = sim.Task("slim", frozenset({"."}), 20.0, off_pool=True)
    r = sim.simulate([arrive(1, 0.0, "a/x.py", "b/x.py")], sim.World(tasks=(A, B, slim), nodes=1), ONE)
    assert mains(r) == {1: 2.0 + 3.0 + 10.0 + 30.0}
    assert (r.busy, r.runs) == (40.0, 3)


def test_infra_errors_retry_and_never_eject() -> None:
    trace = [arrive(i, 5.0 * i, "a/x.py" if i % 2 else "b/x.py") for i in range(1, 21)]
    retries = pages = 0
    for seed in range(5):
        r = sim.simulate(trace, sim.World(tasks=(A, B), infra=0.5, seed=seed), lq.Policy(batch=3, depth=2, page_after=2))
        s = sim.summary(r)
        assert s["landed"] == 20 and s["ejects"] == 0
        retries, pages = retries + s["retries"], pages + s["pages"]
    assert retries > 5 and pages > 0


def test_false_fails_eject_heads_that_break_nothing_and_every_pr_still_lands() -> None:
    trace = [arrive(i, 5.0 * i, "a/x.py" if i % 2 else "b/x.py") for i in range(1, 21)]
    false = 0
    for seed in range(5):
        r = sim.simulate(trace, sim.World(tasks=(A, B), flaky=0.2, seed=seed), lq.Policy(batch=3, depth=2))
        assert all(x.main is not None for x in r.landings) and not r.unsound
        false += sim.summary(r)["false_ejects"]
    assert false > 0


def test_the_unsound_count_catches_a_queue_that_advances_on_a_failed_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = [arrive(1, 0.0, "a/x.py", breaks=("unit:a",))]
    assert not sim.simulate(trace, W, ONE).unsound
    monkeypatch.setattr(lq, "_verdict", lambda results, c: lq.PASS if c.tree is not None and c.tasks is not None
                        else lq.PENDING)
    assert sim.simulate(trace, W, ONE).unsound


def test_the_same_trace_and_seed_give_the_same_report() -> None:
    trace = [arrive(i, 3.0 * i, "a/x.py" if i % 3 else "b/x.py", breaks=("unit:a",) if i == 4 else ()) for i in range(1, 30)]
    world = sim.World(tasks=(A, B), infra=0.2, flaky=0.1, seed=7)
    assert sim.simulate(trace, world, lq.Policy()) == sim.simulate(trace, world, lq.Policy())


def test_summary_reads_lag_from_held_intervals_and_quantiles_as_the_scorer_does() -> None:
    r = sim.Report(landings=(sim.Landing(1, "h1", 0.0, 0.0, 30.0, 0, 0), sim.Landing(2, "h2", 0.0, 10.0, None, 0, 0)),
                   held=((0.0, 30.0), (10.0, 50.0)), end=50.0, nodes=1)
    s = sim.summary(r, every=10.0, lag=20.0)
    assert (s["prs"], s["landed"], s["p50_h"], s["within"]) == (2, 1, 0.5, 0.5)
    assert s["lag_max_h"] == pytest.approx(0.5)
    assert s["lag_ok"] == pytest.approx(5 / 6)
    assert sim.quantile([1.0, 2.0, 3.0, 4.0], 0.9) == 4.0
