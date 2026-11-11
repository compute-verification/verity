"""The snapshot driver runs a `decide(snapshot)` queue through the simulator and the property harness: a small queue with
the verified queue's contract lands what passes, ejects by inputs, retries infra and reruns a fail, and the driver hands
`decide` each field the contract has.  The property checks over it are `test_properties`'."""
import random
import types
from dataclasses import dataclass

import fake_vqueue as vq
import harness
import pytest

from verity_landq import model as lq
from verity_landq import sim, snapshots

A = sim.Task("unit:a", frozenset({"a/"}), 10.0)
B = sim.Task("unit:b", frozenset({"b/"}), 30.0)
W = sim.World(tasks=(A, B), nodes=2)
D = snapshots.Driver(vq)
TWO = lq.Policy(batch=2, depth=1)


def arrive(pr: int, t: float, *paths: str, breaks: tuple[str, ...] = ()) -> sim.Arrival:
    return sim.Arrival(pr, f"h{pr}", t, frozenset(paths), None, frozenset(breaks))


def run(trace: list[sim.Arrival], world: sim.World = W, policy: lq.Policy = TWO) -> sim.Report:
    return sim.simulate(trace, world, policy, step=D.step, state=D.start("main"))


def test_a_pr_lands_through_decide_after_its_build_its_listing_and_its_task() -> None:
    assert {x.pr: x.main for x in run([arrive(1, 0.0, "a/x.py")]).landings} == {1: 15.0}


def test_a_breaking_pr_is_ejected_by_inputs_and_the_pr_beside_it_lands() -> None:
    r = run([arrive(1, 0.0, "a/x.py", breaks=("unit:a",)), arrive(2, 0.0, "b/x.py")])
    rows = {x.pr: x for x in r.landings}
    assert (rows[1].ejects, rows[2].ejects) == (1, 0)
    assert all(x.main is not None for x in r.landings) and not r.unsound


def test_infra_is_retried_and_never_ejects() -> None:
    trace = [arrive(i, 5.0 * i, "a/x.py" if i % 2 else "b/x.py") for i in range(1, 13)]
    retries = 0
    for seed in range(4):
        s = sim.summary(run(trace, sim.World(tasks=(A, B), infra=0.4, seed=seed), lq.Policy(batch=3, depth=1)))
        assert s["landed"] == 12 and s["ejects"] == 0
        retries += s["retries"]
    assert retries > 4


def test_the_drop_of_a_candidate_now_on_main_records_its_pass() -> None:
    s, p = D.start("m"), lq.Policy(batch=2, depth=1)
    s, (build,) = D.step(s, lq.Ready(1, "h1", frozenset({"a/x.py"})), p)
    for e in [lq.Built(build.cid, "T1"), lq.Listed("T1", frozenset({"unit:a"}))]:
        s, _ = D.step(s, e, p)
    s, out = D.step(s, lq.Result("T1", "unit:a", lq.PASS), p)
    assert out == (lq.Advance("T1", (lq.Member(1, "h1", frozenset({"a/x.py"})),)),)
    assert (s.main, s.stack, s.history) == ("T1", [], [vq.Outcome((1,), ("h1",), vq.PASS)])


def test_an_eject_after_its_candidates_drop_names_that_candidates_tree() -> None:
    s, p = D.start("m"), lq.Policy(batch=2, depth=1)
    s, (build,) = D.step(s, lq.Ready(1, "h1", frozenset({"a/x.py"})), p)
    for e in [lq.Built(build.cid, "T1"), lq.Listed("T1", frozenset({"unit:a"}))]:
        s, _ = D.step(s, e, p)
    s, out = D.step(s, lq.Result("T1", "unit:a", lq.FAIL, frozenset({"a/"}), "1"), p)
    assert out == (lq.Cancel(build.cid, "T1", "fail"), lq.Eject(1, "h1", "T1", "unit:a", "fail"))


def test_a_failed_task_keyed_on_the_whole_tree_reaches_decide_as_a_glob_of_it() -> None:
    c = snapshots.Cand(1, "m", (lq.Member(1, "h1", frozenset({"a/x.py"})),), tree="T1", tasks=frozenset({"lean-suites"}),
                       results={"lean-suites": lq.Result("T1", "lean-suites", lq.FAIL, frozenset({"."}))})
    assert D.verdict(c).failed == (vq.Failed("lean-suites", "lean-suites@T1", frozenset({snapshots.WHOLE})),)


@dataclass(frozen=True)
class Failed:
    task: str
    key: str
    inputs: frozenset[str]
    base_key: str | None = None
    tests: frozenset[str] = frozenset()
    base_tests: frozenset[str] | None = None


@dataclass(frozen=True)
class Verdict:
    state: str = vq.PENDING
    failed: tuple[Failed, ...] = ()
    infra: tuple[Failed, ...] = ()
    ended: bool = False


@dataclass(frozen=True)
class Candidate:
    sha: str
    base: str
    members: tuple[int, ...]
    heads: tuple[str, ...]
    verdict: Verdict = Verdict()
    tries: int = 1
    reruns: int = 0
    enqueued: float | None = None


@dataclass(frozen=True)
class Snapshot:
    main: str
    ready: tuple
    candidates: tuple = ()
    history: tuple = ()
    batch: int = 6
    depth: int = 2
    rings: int = 2
    lean: int = 1
    now: float | None = None


@dataclass(frozen=True)
class Rerun:
    sha: str
    keys: tuple[str, ...]


#: the fake queue with every field the driver fills where the contract has it
FULL = types.SimpleNamespace(**{k: getattr(vq, k) for k in dir(vq) if not k.startswith("_")})
FULL.__name__, FULL.Failed, FULL.Verdict, FULL.Candidate, FULL.Snapshot, FULL.Rerun = (
    "full", Failed, Verdict, Candidate, Snapshot, Rerun)


def lone(results: dict[str, lq.Result], tasks: frozenset[str]) -> snapshots.Cand:
    return snapshots.Cand(1, "m", (lq.Member(1, "h1", frozenset({"a/x.py"})),), tree="T1", tasks=tasks, results=results)


def test_a_failure_carries_its_keys_and_tests_where_the_contract_has_them() -> None:
    c = lone({"unit:b": lq.Result("T1", "unit:b", lq.FAIL, frozenset({"b/"}), "7", "5", frozenset({"red"}),
                                  frozenset({"red"}))}, frozenset({"unit:b"}))
    assert snapshots.Driver(FULL).verdict(c).failed == (
        Failed("unit:b", "7", frozenset({"b/"}), "5", frozenset({"red"}), frozenset({"red"})),)
    assert D.verdict(c).failed == (vq.Failed("unit:b", "7", frozenset({"b/"})),)


def test_a_fail_has_ended_once_every_task_has_an_outcome_with_its_infra_beside_it() -> None:
    fail = lq.Result("T1", "unit:a", lq.FAIL, frozenset({"a/"}), "1", "1", frozenset({"red"}), frozenset({"red"}))
    full = snapshots.Driver(FULL)
    running = full.verdict(lone({"unit:a": fail}, frozenset({"unit:a", "unit:b"})))
    assert (running.state, running.ended, running.infra) == (vq.FAIL, False, ())
    erred = full.verdict(lone({"unit:a": fail, "unit:b": lq.Result("T1", "unit:b", lq.INFRA, frozenset({"b/"}), "2")},
                              frozenset({"unit:a", "unit:b"})))
    assert (erred.ended, [f.task for f in erred.infra]) == (True, ["unit:b"])


def test_infra_reads_once_a_task_ends_in_it_though_others_still_run() -> None:
    erred = D.verdict(lone({"unit:b": lq.Result("T1", "unit:b", lq.INFRA, frozenset({"b/"}), "2")},
                           frozenset({"unit:a", "unit:b"})))
    assert (erred.state, [f.task for f in erred.infra]) == (vq.INFRA, ["unit:b"])


def test_the_snapshot_carries_the_clock_reruns_and_fields_set_by_keyword() -> None:
    full = snapshots.Driver(FULL, lean=3)
    s = full.start("m")
    s.now = 1.0
    s, (build,) = full.step(s, lq.Ready(1, "h1", frozenset({"a/x.py"})), TWO)
    s.now = 3.0
    s, _ = full.step(s, lq.Built(build.cid, "T1"), TWO)
    s.now = 10.0
    snap = full.snapshot(s, TWO)
    assert (snap.lean, snap.now, snap.candidates[0].enqueued, snap.candidates[0].reruns) == (3, 600.0, 180.0, 0)
    with pytest.raises(TypeError, match="has no field lean"):
        snapshots.Driver(vq, lean=3)


def test_a_queue_that_reruns_a_lone_fail_stays_sound_in_the_harness() -> None:
    def decide(s):
        c = s.candidates[0] if s.candidates else None
        if c is not None and c.verdict.state == vq.FAIL and len(c.members) == 1 and not c.reruns:
            return [Rerun(c.sha, tuple(f.key for f in c.verdict.failed))]
        return vq.decide(s)

    q = types.SimpleNamespace(**vars(FULL))
    q.decide = decide
    d, reruns = snapshots.Driver(q), 0
    for seed in range(60):
        r = harness.run(harness.random_scenario(random.Random(seed), flaky=3), step=d.step, state=d.start(harness.MAIN0))
        assert not r.unsound, (seed, r.unsound)
        reruns += sum(isinstance(x, lq.Rerun) for x in r.decisions)
    assert reruns > 10


def test_a_rerun_clears_the_failed_keys_and_asks_the_world_to_run_them_afresh() -> None:
    s = snapshots.QState("m")
    s.stack.append(lone({"unit:a": lq.Result("T1", "unit:a", lq.FAIL, frozenset({"a/"}), "1"),
                         "unit:b": lq.Result("T1", "unit:b", lq.FAIL, frozenset({"b/"}), "2")},
                        frozenset({"unit:a", "unit:b"})))
    assert snapshots.Driver(FULL).do_Rerun(s, Rerun("T1", ("2",))) == [lq.Rerun("T1", "unit:b")]
    assert (s.stack[0].reruns, sorted(s.stack[0].results)) == (1, ["unit:a"])


def test_a_ring_pages_once_a_candidate_and_why_and_a_red_once_a_base_and_task() -> None:
    s = snapshots.QState("m")
    s.stack.append(lone({}, frozenset({"unit:a"})))
    assert len(D.do_Ring(s, vq.Ring("T1", "infra")) + D.do_Ring(s, vq.Ring("T1", "infra"))) == 1
    assert len(D.do_Ring(s, vq.Ring("T1", "pending"))) == 1
    a, b = (vq.Failed(t, "1", frozenset({"a/"})) for t in ("unit:a", "unit:b"))
    red = types.SimpleNamespace(sha="T1", base="m", why="", failed=(a,))
    assert [len(D.do_Red(s, red)), len(D.do_Red(s, red))] == [1, 0]
    assert D.do_Red(s, types.SimpleNamespace(sha="T1", base="m", why="", failed=(a, b))) == [lq.Page("T1", "red: unit:b", 0)]


def test_an_action_naming_a_candidate_that_isnt_live_is_refused() -> None:
    odd = types.SimpleNamespace(**{k: getattr(vq, k) for k in dir(vq) if not k.startswith("_")})
    odd.decide = lambda s: [vq.Advance("nope")]
    with pytest.raises(ValueError, match="isn't live: nope"):
        snapshots.Driver(odd).step(snapshots.QState("m"), lq.Ready(1, "h1", frozenset({"a/x.py"})), lq.Policy())


def test_the_snapshot_holds_the_reviewed_prs_by_first_mark_with_their_lean_flag() -> None:
    s = D.start("m")
    for e in [lq.Ready(2, "h2", frozenset({"S/A.lean"})), lq.Ready(1, "h1", frozenset({"a/x.py"}), reviewed=False),
              lq.Ready(3, "h3", frozenset({"b/y.py"})), lq.Ready(2, "h2", frozenset({"S/A.lean"}))]:
        D.on_Ready(s, e)
    snap = D.snapshot(s, lq.Policy(batch=5, depth=2))
    assert [(r.number, r.lean, r.rank) for r in snap.ready] == [(2, True, 0), (3, False, 2)]
    assert (snap.main, snap.batch, snap.depth) == ("m", 5, 2)


def test_a_deferred_task_starts_once_on_the_lowest_candidate_on_main_once_the_rest_passed() -> None:
    d, tasks = snapshots.Driver(vq, deferred=("slim",)), frozenset({"unit:a", "slim"})
    s = snapshots.QState("m")
    s.stack += [lone({}, tasks), snapshots.Cand(2, "T1", (lq.Member(2, "h2", frozenset({"a/y.py"})),), tree="T2", tasks=tasks,
                                                results={"unit:a": lq.Result("T2", "unit:a", lq.PASS, frozenset({"a/"}))})]
    assert d.begin(s) == []
    s.stack[0].results["unit:a"] = lq.Result("T1", "unit:a", lq.PASS, frozenset({"a/"}))
    assert [d.begin(s), d.begin(s), d.verdict(s.stack[0]).state] == [[lq.Start("T1", "slim")], [], vq.PENDING]


def test_the_queues_reading_says_when_a_deferred_task_starts() -> None:
    q = types.SimpleNamespace(**vars(FULL))
    q._reading = lambda v: vq.PASS
    s = snapshots.QState("m")
    s.stack.append(lone({"unit:a": lq.Result("T1", "unit:a", lq.FAIL, frozenset({"a/"}), "1")}, frozenset({"unit:a", "slim"})))
    assert snapshots.Driver(q, deferred=("slim",)).begin(s) == [lq.Start("T1", "slim")]
    assert snapshots.Driver(FULL, deferred=("slim",)).begin(s) == []


def test_a_pr_lands_once_its_check_and_then_the_run_beside_it_passed() -> None:
    slim = sim.Task("slim", frozenset({"."}), 20.0, off_pool=True, deferred=True)
    d = snapshots.Driver(vq, deferred=("slim",))
    r = sim.simulate([arrive(1, 0.0, "a/x.py")], sim.World(tasks=(A, slim), nodes=1), TWO, step=d.step,
                     state=d.start("main"))
    assert {x.pr: x.main for x in r.landings} == {1: 2.0 + 3.0 + 10.0 + 20.0}


def test_rings_are_the_queues_own_unless_set() -> None:
    s = D.start("m")
    assert D.snapshot(s, lq.Policy()).rings == vq.Snapshot("m", (), (), ()).rings
    assert snapshots.Driver(vq, rings=1).snapshot(s, lq.Policy()).rings == 1
