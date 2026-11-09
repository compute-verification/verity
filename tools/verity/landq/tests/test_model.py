"""Each rule of the queue model on a small, hand-written sequence of events."""
from __future__ import annotations

import pytest

from verity_landq import model as lq

P = lq.Policy(batch=2, depth=2, page_after=2)
TASKS = {"unit:a": frozenset({"a/"}), "unit:b": frozenset({"b/x.py"}), "unit:c": frozenset({"c/"})}


def feed(state, events, policy=P):
    out = []
    for e in events:
        state, ds = lq.step(state, e, policy)
        out += ds
    return state, out


def ready(pr, *paths, head=None, reviewed=True):
    return lq.Ready(pr, head or f"h{pr}", frozenset(paths), reviewed)


def start(*prs, main="m0"):
    """The queue with `prs` ((number, path), ...) all ready before anything is in flight, after the first batch formed: the
    first PR's ready mark again, which changes nothing but lets the queue form."""
    queue = {pr: lq.Queued(pr, f"h{pr}", frozenset({path}), True, i) for i, (pr, path) in enumerate(prs)}
    return feed(lq.State(main, queue=queue, seq=len(queue)), [ready(prs[0][0], prs[0][1])])


def listed(tree):
    return lq.Listed(tree, frozenset(TASKS))


def results(tree, fail=(), infra=()):
    return [lq.Result(tree, t, lq.FAIL if t in fail else lq.INFRA if t in infra else lq.PASS, TASKS[t]) for t in sorted(TASKS)]


def of(kind, ds):
    return [d for d in ds if isinstance(d, kind)]


def test_one_pr_lands_once_every_gating_task_passed_on_its_tree() -> None:
    s, ds = feed(lq.State.start("m0"), [ready(1, "a/f.py")])
    assert ds == [lq.Build(0, "m0", (lq.Member(1, "h1", frozenset({"a/f.py"})),), False)]
    s, ds = feed(s, [lq.Built(0, "T1"), listed("T1"), *results("T1")[:-1]])
    assert not of(lq.Advance, ds) and s.main == "m0"
    s, ds = feed(s, results("T1")[-1:])
    assert ds == [lq.Advance("T1", (lq.Member(1, "h1", frozenset({"a/f.py"})),))]
    assert s.main == "T1" and not s.stack and not s.queue and not s.results


def test_batches_stack_on_the_tree_below_and_main_skips_to_the_highest_that_passed() -> None:
    s, ds = start((1, "a/1.py"), (2, "a/2.py"))
    assert [(d.base, [m.pr for m in d.members]) for d in of(lq.Build, ds)] == [("m0", [1, 2])]
    s, ds = feed(s, [ready(3, "a/3.py"), lq.Built(0, "T1")])
    assert [(d.base, [m.pr for m in d.members]) for d in of(lq.Build, ds)] == [("T1", [3])]
    s, ds = feed(s, [lq.Built(1, "T2"), listed("T2"), *results("T2")])
    assert of(lq.Advance, ds) == [lq.Advance("T2", tuple(lq.Member(n, f"h{n}", frozenset({f"a/{n}.py"})) for n in (1, 2, 3)))]
    s, ds = feed(s, [listed("T1"), *results("T1", fail={"unit:a"})])
    assert ds == [] and s.main == "T2"


def test_nothing_is_built_past_depth_or_on_an_unbuilt_candidate() -> None:
    s, ds = feed(lq.State.start("m0"), [ready(n, "a/x.py") for n in range(1, 8)])
    assert len(of(lq.Build, ds)) == 1
    s, ds = feed(s, [lq.Built(0, "T1")])
    assert len(of(lq.Build, ds)) == 1 and len(s.stack) == P.depth
    s, ds = feed(s, [lq.Built(1, "T2")])
    assert of(lq.Build, ds) == []


def test_the_one_pr_touching_a_failing_tasks_inputs_is_ejected_and_the_rest_retested() -> None:
    s, _ = start((1, "a/f.py"), (2, "b/x.py"))
    s, ds = feed(s, [lq.Built(0, "T1"), listed("T1"), *results("T1", fail={"unit:b"})])
    assert of(lq.Eject, ds) == [lq.Eject(2, "h2", "T1", "unit:b", lq.INPUTS)]
    assert of(lq.Cancel, ds) == [lq.Cancel(0, "T1", lq.EJECTED)]
    assert [(d.base, [m.pr for m in d.members]) for d in of(lq.Build, ds)] == [("m0", [1])]
    assert 2 not in s.queue and s.ejected[2].task == "unit:b"


def test_a_failure_two_prs_touch_bisects_and_the_half_that_fails_alone_is_ejected() -> None:
    s, _ = start((1, "a/f.py"), (2, "a/g.py"))
    s, ds = feed(s, [lq.Built(0, "T1"), listed("T1"), *results("T1", fail={"unit:a"})])
    assert of(lq.Cancel, ds) == [lq.Cancel(0, "T1", lq.BISECTED)] and not of(lq.Eject, ds)
    assert [(d.base, [m.pr for m in d.members], d.bisect) for d in of(lq.Build, ds)] == [("m0", [1], True)]
    s, ds = feed(s, [lq.Built(1, "T2")])
    assert [(d.base, [m.pr for m in d.members], d.bisect) for d in of(lq.Build, ds)] == [("T2", [2], True)]
    s, ds = feed(s, [lq.Built(2, "T3"), listed("T2"), *results("T2")])
    assert of(lq.Advance, ds) == [lq.Advance("T2", (lq.Member(1, "h1", frozenset({"a/f.py"})),))]
    s, ds = feed(s, [listed("T3"), *results("T3", fail={"unit:a"})])
    assert of(lq.Eject, ds) == [lq.Eject(2, "h2", "T3", "unit:a", lq.INPUTS)]


def test_a_lone_suspect_is_ejected_whatever_it_touches() -> None:
    s, ds = feed(lq.State.start("m0"), [ready(1, "docs/x.md"), lq.Built(0, "T1"), listed("T1"), *results("T1", fail={"unit:c"})])
    assert of(lq.Eject, ds) == [lq.Eject(1, "h1", "T1", "unit:c", lq.ALONE)]


def test_a_failure_above_waits_for_the_candidate_below_before_bisecting() -> None:
    s, _ = start((1, "a/1.py"), (2, "a/2.py"))
    s, ds = feed(s, [ready(3, "a/3.py"), ready(4, "a/4.py"), lq.Built(0, "T1"), lq.Built(1, "T2"), listed("T2"),
                     *results("T2", fail={"unit:a"})])
    assert not of(lq.Eject, ds) and not of(lq.Cancel, ds) and [c.tree for c in s.stack] == ["T1", "T2"]
    s, ds = feed(s, [listed("T1"), *results("T1")])
    assert of(lq.Advance, ds)[0].tree == "T1"
    assert of(lq.Cancel, ds) == [lq.Cancel(1, "T2", lq.BISECTED)]
    assert [[m.pr for m in d.members] for d in of(lq.Build, ds)] == [[3]]


def test_a_named_suspect_below_is_ejected_at_once() -> None:
    s, _ = start((1, "b/x.py"), (2, "a/f.py"))
    s, ds = feed(s, [ready(3, "c/f.py"), lq.Built(0, "T1"), lq.Built(1, "T2"), listed("T2"), *results("T2", fail={"unit:b"})])
    assert of(lq.Eject, ds) == [lq.Eject(1, "h1", "T2", "unit:b", lq.INPUTS)]
    assert of(lq.Cancel, ds) == [lq.Cancel(0, "T1", lq.EJECTED), lq.Cancel(1, "T2", lq.EJECTED)]


def test_an_infra_outcome_is_retried_and_paged_and_never_a_verdict() -> None:
    s, ds = feed(lq.State.start("m0"), [ready(1, "a/f.py"), lq.Built(0, "T1"), listed("T1"),
                                        *results("T1", infra={"unit:a"})])
    assert of(lq.Retry, ds) == [lq.Retry("T1", "unit:a", 1)] and not of(lq.Advance, ds) and not of(lq.Eject, ds)
    s, ds = feed(s, [lq.Result("T1", "unit:a", lq.INFRA)])
    assert ds == [lq.Retry("T1", "unit:a", 2), lq.Page("T1", "unit:a", 2)]
    s, ds = feed(s, [lq.Result("T1", "unit:a", lq.PASS)])
    assert of(lq.Advance, ds) == [lq.Advance("T1", (lq.Member(1, "h1", frozenset({"a/f.py"})),))]


def test_an_infra_outcome_after_a_verdict_changes_nothing() -> None:
    s, _ = feed(lq.State.start("m0"), [ready(1, "a/f.py"), ready(2, "c/f.py"), lq.Built(0, "T1"), listed("T1")])
    s, ds = feed(s, [lq.Result("T1", "unit:b", lq.PASS), lq.Result("T1", "unit:b", lq.INFRA)])
    assert ds == [] and s.results[("T1", "unit:b")].status == lq.PASS


def test_a_fail_stands_and_results_for_other_trees_count_for_nothing() -> None:
    s, _ = start((1, "a/f.py"), (2, "b/x.py"))
    s, _ = feed(s, [lq.Built(0, "T1"), listed("T1")])
    s, ds = feed(s, [lq.Result("T9", t, lq.PASS) for t in TASKS] + [lq.Result("m0", t, lq.PASS) for t in TASKS])
    assert ds == [] and not s.results
    s, ds = feed(s, [lq.Result("T1", "unit:b", lq.FAIL, TASKS["unit:b"]), lq.Result("T1", "unit:b", lq.PASS)])
    assert of(lq.Eject, ds)[0].pr == 2


def test_a_new_head_or_a_withdrawal_takes_a_pr_out_of_flight() -> None:
    s, _ = start((1, "a/f.py"), (2, "a/g.py"), (3, "c/f.py"))
    s, _ = feed(s, [lq.Built(0, "T1")])
    s, ds = feed(s, [ready(2, "a/g.py", head="h2b")])
    assert of(lq.Cancel, ds) == [lq.Cancel(0, "T1", lq.NEW_HEAD), lq.Cancel(1, None, lq.NEW_HEAD)]
    assert [[m.pr for m in d.members] for d in of(lq.Build, ds)] == [[1, 3]]
    assert [q.pr for q in lq.queued(s)] == [1, 3, 2]
    s, ds = feed(s, [lq.Withdrawn(3)])
    assert of(lq.Cancel, ds)[0].why == lq.WITHDRAWN and 3 not in s.queue


def test_an_unreviewed_pr_waits_and_a_held_one_leaves_flight() -> None:
    s, ds = feed(lq.State.start("m0"), [ready(1, "a/f.py", reviewed=False)])
    assert ds == [] and 1 in s.queue
    s, ds = feed(s, [ready(1, "a/f.py")])
    assert [d.cid for d in of(lq.Build, ds)] == [0] and s.queue[1].seq == 0
    s, ds = feed(s, [ready(1, "a/f.py", reviewed=False)])
    assert of(lq.Cancel, ds) == [lq.Cancel(0, None, lq.UNREVIEWED)]


def test_a_conflict_sets_the_pr_aside_until_main_moves() -> None:
    s, _ = start((1, "a/f.py"), (2, "a/g.py"))
    s, ds = feed(s, [lq.Conflict(0, 2)])
    assert of(lq.SetAside, ds) == [lq.SetAside(2, "h2", 0)] and s.queue[2].aside
    assert [[m.pr for m in d.members] for d in of(lq.Build, ds)] == [[1]]
    s, ds = feed(s, [lq.Built(1, "T1"), listed("T1"), *results("T1")])
    assert of(lq.Advance, ds)[0].tree == "T1"
    assert [[m.pr for m in d.members] for d in of(lq.Build, ds)] == [[2]] and not s.queue[2].aside


def test_main_moved_outside_lands_up_to_its_tree_or_cancels_everything() -> None:
    s, _ = start((1, "a/1.py"), (2, "a/2.py"))
    s, _ = feed(s, [ready(3, "a/3.py"), lq.Built(0, "T1"), lq.Built(1, "T2")])
    s2, ds = feed(s, [lq.MainMoved("T1")])
    assert s2.main == "T1" and [c.tree for c in s2.stack] == ["T2"] and set(s2.queue) == {3} and not of(lq.Advance, ds)
    s3, ds = feed(s, [lq.MainMoved("elsewhere")])
    assert s3.main == "elsewhere" and [d.why for d in of(lq.Cancel, ds)] == [lq.MAIN_MOVED] * 2
    assert [(d.base, [m.pr for m in d.members]) for d in of(lq.Build, ds)] == [("elsewhere", [1, 2])]


def test_an_ejected_pr_queues_again_at_its_next_ready_mark() -> None:
    s, _ = feed(lq.State.start("m0"), [ready(1, "a/f.py"), lq.Built(0, "T1"), listed("T1"), *results("T1", fail={"unit:a"})])
    assert 1 in s.ejected and 1 not in s.queue
    s, ds = feed(s, [ready(1, "a/f.py")])
    assert 1 not in s.ejected and [[m.pr for m in d.members] for d in of(lq.Build, ds)] == [[1]]


def test_state_is_frozen_and_step_leaves_its_input_alone() -> None:
    s0 = lq.State.start("m0")
    s1, _ = lq.step(s0, ready(1, "a/f.py"), P)
    assert not s0.queue and not s0.stack and s1.queue[1].head == "h1"
    with pytest.raises(TypeError):
        s1.queue[2] = s1.queue[1]  # type: ignore[index]
    with pytest.raises(AttributeError):
        s1.main = "x"  # type: ignore[misc]


def test_the_same_events_give_the_same_decisions() -> None:
    events = [ready(n, f"a/{n}.py") for n in (1, 2, 3)] + [lq.Built(0, "T1"), listed("T1"), *results("T1", fail={"unit:a"})]
    assert feed(lq.State.start("m0"), events) == feed(lq.State.start("m0"), events)


def test_touches_reads_an_input_as_a_file_a_directory_the_tree_or_a_glob() -> None:
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b"}))
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/"}))
    assert lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/c.py"}))
    assert not lq.touches(frozenset({"a/bc/d.py"}), frozenset({"a/b"}))
    assert not lq.touches(frozenset({"a/b/c.py"}), frozenset({"a/b/c.pyi"}))
    assert lq.touches(frozenset({"x.md"}), frozenset({"."}))
    assert not lq.touches(frozenset(), frozenset({"."}))
    assert lq.touches(frozenset({"S/P/A.lean"}), frozenset({"S/**/*.lean"}))
    assert lq.touches(frozenset({"S/A.lean"}), frozenset({"S/*.lean"}))
    assert not lq.touches(frozenset({"S/A.py"}), frozenset({"S/**/*.lean"}))


def test_bad_policies_and_results_are_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Policy.batch"):
        lq.Policy(batch=0)
    with pytest.raises(ValueError, match="Result.status"):
        lq.Result("T1", "unit:a", "error")
    with pytest.raises(TypeError, match="not a queue event"):
        lq.step(lq.State.start("m0"), object(), P)  # type: ignore[arg-type]
