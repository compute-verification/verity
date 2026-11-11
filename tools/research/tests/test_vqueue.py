"""research.vqueue: `decide` rule by rule on small snapshots, `meets`, which ejection rests on, the red team's cases played out
in a small world, and a simulated world that applies `decide`'s actions as the loop does, with verdicts from a scripted oracle,
held to the queue's invariants over a fixed list of seeds."""
from __future__ import annotations

import builtins
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Advance, Build, Candidate, Drop, Eject, Failed, Outcome,
                             Ready, Red, Rerun, Retry, Ring, Snapshot, Verdict, decide, meets)

MAIN = "m0"


def pr(n: int, head: str | None = None, *files: str, rank: int | None = None, lean: bool = False,
       conflict: bool = False) -> Ready:
    return Ready(n, head or f"h{n}", frozenset(files or (f"pkg{n}/mod.py",)), lean, n if rank is None else rank, conflict)


def prs(*ns: int) -> list[Ready]:
    return [pr(n) for n in ns]


def cand(sha: str, base: str, *members: int, state: str = PENDING, failed: tuple[Failed, ...] = (),
         infra: tuple[Failed, ...] = (), ended: bool = True, tries: int = 1, reruns: int = 0, enqueued: float | None = None,
         heads: tuple[str, ...] | None = None) -> Candidate:
    return Candidate(sha, base, members, heads or tuple(f"h{m}" for m in members), Verdict(state, failed, infra, ended), tries,
                     reruns, enqueued)


def task(name: str, *inputs: str) -> Failed:
    return Failed(name, f"key:{name}", frozenset(inputs))


def inherited(name: str, *inputs: str) -> Failed:
    """A failure at its base's key: the base fails it too."""
    return Failed(name, f"key:{name}", frozenset(inputs), f"key:{name}")


def by_tests(name: str, tests: set[str], base_tests: set[str] | None, *inputs: str) -> Failed:
    """A failure at a key a member gave the task, with ``tests`` failing where the base fails ``base_tests`` of it."""
    return Failed(name, f"key:{name}+", frozenset(inputs), f"key:{name}", frozenset(tests),
                  None if base_tests is None else frozenset(base_tests))


def snap(*cands: Candidate, ready=(), history=(), main: str = MAIN, **kw) -> Snapshot:
    return Snapshot(main, tuple(ready), cands, tuple(history), **kw)


def heads(*ns: int) -> tuple[str, ...]:
    return tuple(f"h{n}" for n in ns)


def kinds(acts: list) -> list[type]:
    return [type(a) for a in acts]


# ---- meets ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("files", "inputs", "want"), [
    (["a/b.py"], ["a/b.py"], True),                                       # a path meets itself
    (["a/b.py"], ["a/b"], False),                                         # a prefix of its name is not a directory of it
    (["a/bc/d.py"], ["a/b"], False),
    (["a/b/c/d.py"], ["a/b"], True),                                      # a directory meets everything under it
    (["a/b/c/d.py"], ["a/b/"], True),                                     # with a trailing / too
    (["a/bc/d.py"], ["a/b/"], False),
    (["a.py"], ["a"], False),
    (["a/b.py"], ["a/b.py/"], True),
    (["a.py"], ["*.py"], True),                                           # * stays within a directory
    (["x/a.py"], ["*.py"], False),
    (["a/b.py"], ["a/*.py"], True),
    (["a/b/c.py"], ["a/*.py"], False),
    (["a/b/c.py"], ["a/*/c.py"], True),
    (["a/x/y/c.py"], ["a/*/c.py"], False),
    (["a/bc.py"], ["a/b?.py"], True),                                     # ? is one character, never /
    (["a/b/.py"], ["a/b?.py"], False),
    (["a/bcd.py"], ["a/b?.py"], False),
    (["a/d.py"], ["a/**/d.py"], True),                                    # ** crosses directories, none included
    (["a/b/c/d.py"], ["a/**/d.py"], True),
    (["b/c/d.py"], ["a/**/d.py"], False),
    (["lean-toolchain"], ["**/lean-toolchain"], True),                    # a leading **/ matches at the root
    (["x/y/lean-toolchain"], ["**/lean-toolchain"], True),
    (["x/lean-toolchain.bak"], ["**/lean-toolchain"], False),
    (["tools/c/x/y.py"], ["tools/c/**"], True),
    (["tools/cx/y.py"], ["tools/c/**"], False),
    (["a/b/c.py"], ["a/b/**/"], True),
    (["anything/at/all"], ["**"], True),
    (["v/tests/sub/test_x.py"], ["**/tests/**/test_*.py"], True),         # the shapes graph2's specs use
    (["v/tests/test_x.py"], ["**/tests/**/test_*.py"], True),
    (["v/tests/x_test.py"], ["**/tests/**/test_*.py"], False),
    (["v/ml/flock/flock-a-b684b12.patch"], ["v/ml/flock/flock-*-b684b12.patch"], True),
    (["a+b/c.py"], ["a+b/*.py"], True),                                   # other characters are literal
    (["axb/c.py"], ["a.b/*.py"], False),
    (["a1/c.py"], ["a[1]/*.py"], False),
    (["a[1]/c.py"], ["a[1]/*.py"], True),
    (["a/b.py"], [""], False),                                            # an empty input meets nothing
    (["a/b.py"], ["/"], False),
    ([], ["a"], False),
    (["a"], [], False),
    (["x/y.py", "a/b.py"], ["q", "a"], True),                             # any path, any input
])
def test_meets(files, inputs, want):
    assert meets(files, inputs) is want
    assert meets(iter(files), frozenset(inputs)) is want


# ---- 1. the chain --------------------------------------------------------------------------------------------------------------

def test_the_chain_is_ordered_from_main_and_only_its_lowest_verdict_is_read():
    ready = prs(1, 2, 3, 4)
    assert decide(snap(cand("c2", "c1", 3, state=PASS), cand("c1", MAIN, 1, 2), ready=ready)) == []
    assert decide(snap(cand("c2", "c1", 3, state=FAIL), cand("c1", MAIN, 1, 2), ready=ready)) == []
    assert decide(snap(cand("c2", "c1", 3), cand("c1", MAIN, 1, 2, state=PASS), ready=ready)) == [
        Advance("c1"), Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS)), Build((4,), "c2", 1)]


def test_a_candidate_off_the_chain_is_dropped_with_every_candidate_above_it():
    s = snap(cand("y", "x", 3), cand("c1", MAIN, 1), cand("x", "gone", 2, state=PASS), ready=prs(1, 2, 3), depth=3)
    assert decide(s) == [Drop("x", "not on main's chain: its base gone is neither main nor a chain member"),
                         Drop("y", "its base was dropped"), Build((2, 3), "c1", 2)]


def test_of_two_candidates_on_one_base_the_one_whose_members_come_first_in_rank_stays():
    ready = [pr(2, rank=1), pr(3, rank=2), pr(4, rank=0), pr(5, rank=5)]
    s = snap(cand("a", MAIN, 3, state=PASS), cand("a2", "a", 5), cand("b", MAIN, 2, 4), ready=ready)
    assert decide(s) == [Drop("a", "a second candidate on m0: b's members come first in rank"),
                         Drop("a2", "its base was dropped"), Build((3, 5), "b", 1)]


@pytest.mark.parametrize(("ready", "why"), [
    ([pr(1, "h1b"), pr(2), pr(3)], "#1's head moved from h1 to h1b"),
    ([pr(2), pr(3)], "#1's ready mark was withdrawn"),
])
def test_a_chain_candidate_whose_member_moved_or_was_unmarked_is_dropped_with_everything_above(ready, why):
    s = snap(cand("c1", MAIN, 1, 2, state=PASS), cand("c2", "c1", 3), ready=ready)
    rebuilt = tuple(r.number for r in ready)
    assert decide(s) == [Drop("c1", why), Drop("c2", "its base was dropped"), Build(rebuilt, MAIN, 2)]
    s = snap(cand("c1", MAIN, 3), cand("c2", "c1", 1, 2), ready=ready)
    assert decide(s) == [Drop("c2", why), Build(tuple(n for n in rebuilt if n != 3), "c1", 1)]


def test_a_pr_in_a_candidate_below_makes_the_one_above_stale():
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, 1, heads=heads(2, 1)), ready=prs(1, 2))
    assert decide(s) == [Drop("c2", "#1 is in a candidate below it too"), Build((2,), "c1", 1)]


def test_when_main_moves_elsewhere_every_candidate_is_dropped_and_rebuilt_on_it():
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2), ready=prs(1, 2, 3), main="m1")
    assert decide(s) == [Drop("c1", "not on main's chain: its base m0 is neither main nor a chain member"),
                         Drop("c2", "its base was dropped"), Build((1, 2, 3), "m1", 2)]


def test_a_candidate_main_already_is_on_is_dropped_on_main_and_its_members_stay_out():
    """The loop advanced `main` to c1 and stopped before its drop: the next tick records c1's pass and rebuilds none of it."""
    s = snap(cand("c1", MAIN, 1, 2, state=PASS), cand("c2", "c1", 3), ready=prs(1, 2, 3, 4), main="c1")
    assert decide(s) == [Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS)), Build((4,), "c2", 1)]
    s = snap(cand("c1", MAIN, 1, 2), cand("c2", "c1", 3), ready=prs(1, 2, 3, 4), main="c2")
    assert decide(s) == [Drop("c1", "on main"), Drop("c2", "on main"), Build((4,), "c2", 2)]


def test_a_candidate_on_main_that_failed_only_as_its_base_does_records_a_pass_with_its_reds():
    t = inherited("suite:r", "red/")
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=prs(1, 2, 3), main="c1")
    assert decide(s) == [Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS, (), MAIN, (t,))), Build((3,), "c1", 2)]
    for still in (cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,), ended=False),
                  cand("c1", MAIN, 1, 2, state=FAIL, failed=(t, task("t", "pkg1/")))):
        assert decide(snap(still, ready=prs(1, 2, 3), main="c1"))[0] == Drop("c1", "on main")


def test_a_pr_a_pass_outcome_holds_at_its_head_is_on_main_and_never_rebuilt():
    passed = Outcome((1, 2), heads(1, 2), PASS)
    assert decide(snap(ready=prs(1, 2, 3), history=[passed])) == [Build((3,), MAIN, 2)]


# ---- 2. the lowest candidate ---------------------------------------------------------------------------------------------------

def test_a_pass_advances_main_and_the_candidate_above_sits_on_it():
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2), ready=prs(1, 2, 3))
    assert decide(s) == [Advance("c1"), Drop("c1", "on main", Outcome((1,), heads(1), PASS)), Build((3,), "c2", 1)]
    s = snap(cand("c1", MAIN, 1, state=PASS), ready=prs(1, 2, 3))
    assert decide(s) == [Advance("c1"), Drop("c1", "on main", Outcome((1,), heads(1), PASS)), Build((2, 3), "c1", 2)]


def test_a_failure_whose_inputs_meet_one_members_files_ejects_it():
    t1, t2 = task("suite:a", "pkg2/"), task("lean:b", "elsewhere/x.lean")
    s = snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(t1, t2)), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4))
    assert decide(s) == [
        Eject(2, "h2", "suite:a failed, and its changed files meet their inputs, no other member's", (t1, t2)),
        Drop("c1", "failed: suite:a, lean:b", Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (2,), MAIN)),
        Drop("c2", "its base failed"), Build((1, 3, 4), MAIN, 2)]


def test_a_failure_whose_inputs_meet_several_members_suspects_only_them():
    t = task("t", "pkg1/mod.py", "pkg3")
    s = snap(cand("c1", MAIN, 1, 2, 3, 4, state=FAIL, failed=(t,)), ready=prs(1, 2, 3, 4))
    assert decide(s) == [Drop("c1", "failed: t", Outcome((1, 2, 3, 4), heads(1, 2, 3, 4), FAIL, (1, 3), MAIN)),
                         Build((1,), MAIN, 2)]
    s = snap(cand("c1", MAIN, 1, 2, 3, 4, state=FAIL, failed=(t,)), ready=[pr(1, rank=9), *prs(2, 3, 4)])
    assert decide(s)[-1] == Build((2, 4), MAIN, 2)


def test_a_failure_whose_inputs_meet_no_member_suspects_every_member():
    t, six = task("t", "elsewhere/"), (1, 2, 3, 4, 5, 6)
    s = snap(cand("c1", MAIN, *six, state=FAIL, failed=(t,)), ready=prs(*six, 7))
    assert decide(s) == [Drop("c1", "failed: t", Outcome(six, heads(*six), FAIL, six, MAIN)), Build((1, 2, 3), MAIN, 2)]


def test_a_fail_is_acted_on_once_a_failure_of_its_own_is_known_while_the_rest_still_runs():
    t = task("suite:a", "pkg2/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,), ended=False), ready=prs(1, 2)))
    assert kinds(acts) == [Eject, Drop, Build] and acts[0].number == 2


def test_a_lone_members_own_failure_meeting_none_of_its_files_is_rerun_afresh_once_then_ejected():
    """Built alone again on the same base, its tree would have the same keys, which graph2 answers from its cached fail: so its
    failed keys are rerun on the same candidate, which stays, and a failure after the rerun ejects it as a possible flake."""
    t, u = task("t", "elsewhere/"), task("u", "docs/")
    s = snap(cand("c1", MAIN, 5, state=FAIL, failed=(t, u)), cand("c2", "c1", 6), ready=prs(5, 6, 7), depth=3)
    assert decide(s) == [Rerun("c1", ("key:t", "key:u")), Build((7,), "c2", 1)]
    acts = decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,), reruns=1), ready=prs(5, 6)))
    assert acts == [Eject(5, "h5", acts[0].why, (t,)), Drop("c1", "failed: t", Outcome((5,), heads(5), FAIL, (5,), MAIN)),
                    Build((6,), MAIN, 2)]
    assert acts[0].why.startswith("t failed alone at h5 and again when rerun afresh, and no failing task's inputs meet")
    assert "if a test flaked, delete it (a test that flakes is deleted, not rerun)" in acts[0].why


def test_a_lone_member_reruns_only_its_own_failures_and_is_ejected_at_once_when_one_meets_its_files():
    own, red = task("t", "elsewhere/"), inherited("suite:r", "red/")
    assert decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(red, own)), ready=prs(5))) == [Rerun("c1", ("key:t",))]
    hit = task("suite:5", "pkg5/")
    acts = decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(hit,)), ready=prs(5)))
    assert acts == [Eject(5, "h5", "suite:5 failed, and its changed files meet their inputs, no other member's", (hit,)),
                    Drop("c1", "failed: suite:5", Outcome((5,), heads(5), FAIL, (5,), MAIN))]
    nameless = Failed("t", "", frozenset({"elsewhere/"}))
    acts = decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(nameless,)), ready=prs(5)))
    assert kinds(acts) == [Eject, Drop] and acts[0].why.startswith("t failed alone at h5, and no failing task's inputs meet")


@pytest.mark.parametrize("before", [
    Outcome((5,), heads(5), FAIL, (5,), MAIN),               # failed alone at this head before, on another task
    Outcome((5,), ("h5-old",), FAIL, (5,)),
    Outcome((5, 6), heads(5, 6), FAIL, (5, 6)),
])
def test_what_a_lone_failure_calls_for_reads_its_candidates_reruns_not_the_history(before):
    t = task("t", "elsewhere/")
    assert decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,)), ready=prs(5), history=[before])) == [Rerun("c1", ("key:t",))]
    acts = decide(snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,), reruns=1), ready=prs(5), history=[before]))
    assert kinds(acts) == [Eject, Drop]


@pytest.mark.parametrize("state", [INFRA, LIST_FAILED])
def test_with_tree_ejects_infra_and_list_failed_are_retried_at_every_reading_rung_from_the_second_and_blame_nobody(state):
    """With `tree_ejects`, an infra error graph2 saw on one node, or a listing failure, says nothing about the members: the
    check is enqueued again at every reading, and infra is rung from `rings` enqueues on (2 at most), however often it errs."""
    err = (task("test-file:pkg2/tests/test_reads.py", "pkg2/tests/test_reads.py", "pkg2/mod.py"),) if state == INFRA else ()
    why = "c1's check ended in infra errors: test-file:pkg2/tests/test_reads.py" if state == INFRA else "c1's check failed to list"
    c = Candidate("c1", MAIN, (1, 2, 3), heads(1, 2, 3), Verdict(state, (), err))
    on = {"tree_ejects": True}
    assert decide(snap(c, cand("c2", "c1", 4), ready=prs(1, 2, 3, 4, 5), rings=3, **on)) == [Retry("c1")]
    assert decide(snap(c, ready=prs(1, 2, 3, 4), rings=1, **on)) == [Retry("c1"), Ring("c1", why), Build((4,), "c1", 1)]
    for tries, rings in ((2, 2), (2, 3), (5, 2)):
        acts = decide(snap(replace(c, tries=tries), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4), rings=rings, **on))
        assert acts == [Retry("c1"), Ring("c1", why)], (tries, rings)


@pytest.mark.parametrize("state", [INFRA, LIST_FAILED])
def test_without_tree_ejects_infra_that_persists_is_bisected_on_its_inputs_ejecting_nobody(state):
    """Without `tree_ejects`, a check that errs again once enqueued again (from `rings` readings, 2 at most) is bisected like
    a fail, on the members whose files meet the erring tasks' inputs, else on every member, and nobody is ejected: the
    members without the error land, and the one with it ends up alone.  A first reading is only retried."""
    err = (task("test-file:pkg2/tests/test_reads.py", "pkg2/tests/test_reads.py", "pkg2/mod.py"),) if state == INFRA else ()
    why = "c1's check ended in infra errors: test-file:pkg2/tests/test_reads.py" if state == INFRA else "c1's check failed to list"
    c = Candidate("c1", MAIN, (1, 2, 3), heads(1, 2, 3), Verdict(state, (), err))
    assert decide(snap(c, cand("c2", "c1", 4), ready=prs(1, 2, 3, 4, 5))) == [Retry("c1")]
    for tries, rings in ((2, 2), (2, 3), (5, 2), (1, 1)):
        acts = decide(snap(replace(c, tries=tries), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4), rings=rings))
        suspects, rebuilt = ((2,), (1, 3, 4)) if state == INFRA else ((1, 2, 3), (1,))
        assert acts == [Ring("c1", why), Drop("c1", f"{why}, again after it was enqueued again: bisected on them, ejecting nobody",
                                              Outcome((1, 2, 3), heads(1, 2, 3), FAIL, suspects, MAIN)),
                        Drop("c2", "its base failed"), Build(rebuilt, MAIN, 2)], (tries, rings)


def test_infra_that_persists_is_retried_at_every_reading_and_rings_in_the_same_words_from_the_second():
    """The loop rings each (candidate, why) once, so the why changes with the erring tasks, in any order, and not with the
    enqueues: five readings ring infra once, and another set of erring tasks rings it again."""
    hang, runner = task("test-file:t/test_hang.py", "t/test_hang.py"), task("runner", "tools/")
    whys = []
    for tries in range(1, 6):
        c = Candidate("c1", MAIN, (1, 2), heads(1, 2), Verdict(INFRA, (), (hang, runner) if tries % 2 else (runner, hang)), tries)
        acts = decide(snap(c, cand("c2", "c1", 3), ready=prs(1, 2, 3), tree_ejects=True))
        assert kinds(acts) == [Retry] + [Ring] * (tries >= 2) and acts[0] == Retry("c1"), tries
        whys += [a.why for a in acts if isinstance(a, Ring)]
    assert whys == ["c1's check ended in infra errors: runner, test-file:t/test_hang.py"] * 4
    acts = decide(snap(replace(c, verdict=Verdict(INFRA, (), (hang,))), cand("c2", "c1", 3), ready=prs(1, 2, 3), tree_ejects=True))
    assert acts == [Retry("c1"), Ring("c1", "c1's check ended in infra errors: test-file:t/test_hang.py")]


@pytest.mark.parametrize("state", [INFRA, LIST_FAILED])
def test_a_lone_member_whose_check_errs_again_after_a_retry_is_rung_to_its_owner_and_waits_never_ejected(state):
    """Without `tree_ejects`, a lone member whose check errs again is never ejected: its owner is rung beside infra.  It is
    set aside behind the PRs waiting (an `infra` outcome, which batches it last and alone, and after which setting it aside
    again at that head rings nobody), and retried when none waits.  With `tree_ejects` it is retried and infra alone is rung."""
    err = (task("test-file:t/test_hang.py", "t/test_hang.py"),) if state == INFRA else ()
    c = Candidate("c1", MAIN, (3,), heads(3), Verdict(state, (), err), tries=2)
    infra = Ring("c1", "c1's check failed to list" if state == LIST_FAILED else
                 "c1's check ended in infra errors: test-file:t/test_hang.py")
    acts = decide(snap(c, ready=prs(3)))
    assert acts == [Retry("c1"), infra, Ring("c1", acts[2].why, 3)]
    assert acts[2].why.startswith("#3's check, with only it merged on m0, ") and "Nobody is ejected for it" in acts[2].why
    owner = acts[2]
    aside = Outcome((3,), heads(3), INFRA, (3,), MAIN)
    why = f"{infra.why}, with only #3 merged: it goes behind the PRs waiting, and nobody is ejected for it"
    assert decide(snap(c, ready=prs(3, 4))) == [infra, owner, Drop("c1", why, aside), Build((4,), MAIN, 2)]
    assert decide(snap(c, cand("c2", "c1", 4), ready=prs(3, 4))) == [infra, owner, Drop("c1", why, aside),
                                                                     Drop("c2", "its base failed"), Build((4,), MAIN, 2)]
    assert decide(snap(c, ready=prs(3), history=(aside,))) == [Retry("c1"), infra]
    assert decide(snap(c, ready=prs(3, 4), history=(aside,))) == [Drop("c1", why, aside), Build((4,), MAIN, 2)]
    assert decide(snap(c, ready=prs(3, 4), tree_ejects=True)) == [Retry("c1"), infra, Build((4,), "c1", 1)]


def test_a_pr_set_aside_is_batched_after_every_other_and_alone():
    aside = Outcome((3,), heads(3), INFRA, (3,), MAIN)
    assert decide(snap(ready=prs(1, 3, 4, 5), history=(aside,))) == [Build((1, 4, 5), MAIN, 2)]
    assert decide(snap(ready=prs(3, 4), history=(aside,), batch=1)) == [Build((4,), MAIN, 2)]
    assert decide(snap(ready=prs(3), history=(aside,))) == [Build((3,), MAIN, 2)]
    assert decide(snap(ready=prs(1, 3), history=(aside, Outcome((1,), heads(1), INFRA, (1,), MAIN)))) == [Build((3,), MAIN, 2)]


def test_with_tree_ejects_infra_errors_that_follow_the_tree_are_bisected_at_once_like_a_fail_on_the_members_meeting_their_inputs():
    """graph2 saw the test err on two or more nodes (`tree`): the tree breaks the machines, so the members whose files meet its
    inputs are the suspects, else every member; an infra error beside it that doesn't follow the tree suspects nobody."""
    hang = replace(task("test-file:pkg2/tests/test_hang.py", "pkg2/"), tree=True)
    c = cand("c1", MAIN, 1, 2, 3, state=INFRA, infra=(task("runner", "pkg1/"), hang))
    acts = decide(snap(c, cand("c2", "c1", 4), ready=prs(1, 2, 3, 4), tree_ejects=True))
    why = "c1's check ended in infra errors on two or more nodes, which follow its tree: test-file:pkg2/tests/test_hang.py"
    assert acts == [Drop("c1", why, Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (2,), MAIN)), Drop("c2", "its base failed"),
                    Build((1, 3, 4), MAIN, 2)]
    nowhere = replace(c, verdict=Verdict(INFRA, (), (replace(hang, inputs=frozenset({"elsewhere/"})),)))
    acts = decide(snap(nowhere, ready=prs(1, 2, 3, 4), tree_ejects=True))
    assert acts[0].outcome == Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1, 2, 3), MAIN) and acts[1:] == [Build((1,), MAIN, 2)]


def test_without_tree_ejects_infra_errors_that_follow_the_tree_are_bisected_like_any_other_once_they_persist():
    """graph2 marks `tree` on any error that ran out of attempts on two or more nodes, an outage's too, so by default a `tree`
    error ejects nobody: it is retried, then bisected on every erring task's inputs and rung in the plain infra words, and a
    lone member is set aside behind the rest with its owner rung."""
    hang = replace(task("test-file:pkg2/tests/test_hang.py", "pkg2/"), tree=True)
    why = "c1's check ended in infra errors: runner, test-file:pkg2/tests/test_hang.py"
    for members in ((1, 2, 3), (2,)):
        c = cand("c1", MAIN, *members, state=INFRA, infra=(hang, task("runner", "pkg1/")))
        assert decide(snap(c, cand("c2", "c1", 4), ready=prs(1, 2, 3, 4))) == [Retry("c1")]
        acts = decide(snap(replace(c, tries=2), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4)))
        if len(members) > 1:
            assert acts == [Ring("c1", why), Drop("c1", f"{why}, again after it was enqueued again: bisected on them, ejecting "
                                                        "nobody", Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1, 2), MAIN)),
                            Drop("c2", "its base failed"), Build((1,), MAIN, 2)]
        else:
            assert kinds(acts) == [Ring, Ring, Drop, Drop, Build] and acts[0] == Ring("c1", why) and acts[1].pr == 2
            assert acts[2].outcome == Outcome((2,), heads(2), INFRA, (2,), MAIN) and acts[-1] == Build((1, 3, 4), MAIN, 2)


def test_a_lone_member_whose_check_errs_on_two_or_more_nodes_is_ejected_at_once_with_the_errors_with_tree_ejects():
    hang = replace(task("test-file:t/test_hang.py", "t/test_hang.py"), tree=True)
    acts = decide(snap(cand("c1", MAIN, 3, state=INFRA, infra=(task("runner", "tools/"), hang)), ready=prs(3, 4),
                       tree_ejects=True))
    assert acts == [Eject(3, "h3", acts[0].why, (hang,)), Drop("c1", acts[1].why, Outcome((3,), heads(3), FAIL, (3,), MAIN)),
                    Build((4,), MAIN, 2)]
    assert acts[0].why == ("c1's check ended in infra errors on two or more nodes, which follow its tree: "
                           "test-file:t/test_hang.py, with only it merged on m0: a check that keeps erring on a change (an "
                           "undeclared read, a hang, an untraced program) is the change's to fix")


def test_a_fail_naming_no_failed_task_rings_and_blames_drops_and_bisects_nothing():
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL), cand("c2", "c1", 3), ready=prs(1, 2, 3, 4), depth=3)
    acts = decide(s)
    assert acts == [Ring("c1", acts[0].why), Build((4,), "c2", 1)]
    assert acts[0].why == "c1's verdict is fail but names no failed task: nothing is blamed or dropped until it names one"
    assert kinds(decide(replace(s, candidates=(replace(s.candidates[0], tries=5),)))) == [Ring, Build]


def test_a_pending_lowest_candidate_waits():
    assert decide(snap(cand("c1", MAIN, 1), ready=prs(1, 2))) == [Build((2,), "c1", 1)]


def test_a_lowest_candidate_pending_past_the_bound_is_retried_once_and_rung_past_twice_it():
    c = cand("c1", MAIN, 1, enqueued=1000.0)

    def at(now: float | None, **kw) -> list:
        return decide(snap(replace(c, **kw), ready=prs(1), now=now, pending_s=100))

    assert at(1100) == [] and at(1100.5) == [Retry("c1")] and at(1150, tries=2) == []
    acts = at(1200.5, tries=2)
    assert acts == [Ring("c1", "c1's check is still pending 0.0555556 h after its first enqueue, 2 times enqueued")]
    assert kinds(at(5000)) == [Retry, Ring] and kinds(at(5000, tries=0)) == kinds(at(5000, tries=3)) == [Ring]
    assert at(None) == at(5000, enqueued=None) == []
    waiting = cand("c1", MAIN, 1, state=FAIL, failed=(inherited("suite:r", "red/"),), ended=False, enqueued=1000.0)
    assert decide(snap(waiting, ready=prs(1), now=1100.5, pending_s=100)) == [Retry("c1")]
    assert decide(snap(waiting, ready=prs(1), now=1100.5)) == []
    assert decide(snap(cand("c1", MAIN, 1, state=PASS, enqueued=0.0), ready=prs(1), now=10**9))[0] == Advance("c1")


# ---- inherited failures: a red main ---------------------------------------------------------------------------------------------

FIX = pr(9, None, "red/fix.py")


def test_a_candidate_failing_only_as_its_base_does_advances_and_its_red_rings_once():
    """No new failure: `main` moves on, its reds can only shrink, and batches stay open to every PR."""
    t = inherited("suite:r", "red/")
    s = snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(t,)), cand("c2", "c1", 4), ready=[*prs(1, 2, 3, 4, 5), FIX])
    acts = decide(s)
    passed = Outcome((1, 2, 3), heads(1, 2, 3), PASS, (), MAIN, (t,))
    assert acts == [Red("c1", MAIN, acts[0].why, (t,)), Advance("c1"), Drop("c1", "on main", passed), Build((5, 9), "c2", 1)]
    assert acts[0].why == ("m0 fails suite:r too, as c1 does: `main` stays red on them until a fix lands, while the queue lands "
                           "what fails nothing more")
    s = snap(cand("c2", "c1", 4, state=FAIL, failed=(t,)), ready=prs(4, 5), history=[passed], main="c1")
    assert decide(s) == [Advance("c2"), Drop("c2", "on main", Outcome((4,), heads(4), PASS, (), "c1", (t,))), Build((5,), "c2", 2)]
    assert decide(snap(ready=prs(1, 2, 3, 4, 5), history=[passed])) == [Build((4, 5), MAIN, 2)]


def test_a_fail_whose_failures_are_all_inherited_waits_while_a_gating_task_still_runs():
    """graph2 answers an inherited key from its cache, long before the rest of the check ends: a new failure may still come."""
    t = inherited("suite:r", "red/")
    assert decide(snap(cand("c1", MAIN, 1, state=FAIL, failed=(t,), ended=False), ready=prs(1, 2))) == [Build((2,), "c1", 1)]


def test_a_fail_whose_failures_are_all_inherited_beside_infra_errors_reads_as_infra():
    t, runner = inherited("suite:r", "red/"), task("runner", "tools/x")
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,), infra=(runner,)), ready=prs(1, 2, 3))
    assert decide(s) == [Retry("c1"), Build((3,), "c1", 1)]
    again = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,), infra=(runner,), tries=2), ready=prs(1, 2, 3))
    acts = decide(replace(again, tree_ejects=True))
    assert acts == [Retry("c1"), Ring("c1", "c1's check ended in infra errors: runner"), Build((3,), "c1", 1)]
    acts = decide(again)
    assert kinds(acts) == [Ring, Red, Drop, Build] and acts[-1] == Build((1,), MAIN, 2)
    assert acts[2].outcome == Outcome((1, 2), heads(1, 2), FAIL, (1, 2), MAIN, (t,))
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,), infra=(replace(runner, tree=True),)), ready=prs(1, 2, 3))
    assert decide(s) == [Retry("c1"), Build((3,), "c1", 1)]
    acts = decide(replace(s, tree_ejects=True))
    why = "c1's check ended in infra errors on two or more nodes, which follow its tree: runner"
    assert acts == [Red("c1", MAIN, acts[0].why, (t,)), Drop("c1", why, Outcome((1, 2), heads(1, 2), FAIL, (1, 2), MAIN, (t,))),
                    Build((1,), MAIN, 2)]


def test_a_mixed_failure_ejects_or_bisects_on_its_own_failures_alone_and_rings_red_on_the_rest():
    """An inherited failure suspects nobody even where its inputs meet a member's files (here #3's): only the batch's own
    failures do."""
    own, red = task("suite:a", "pkg2/"), inherited("suite:r", "red/", "pkg3/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(own, red)), ready=[*prs(1, 2, 3), FIX]))
    assert acts == [Red("c1", MAIN, acts[0].why, (red,)),
                    Eject(2, "h2", "suite:a failed, and its changed files meet their inputs, no other member's", (own,)),
                    Drop("c1", "failed: suite:a, suite:r", Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (2,), MAIN, (red,))),
                    Build((1, 3, 9), MAIN, 2)]
    nobody = task("suite:x", "elsewhere/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(red, nobody)), ready=prs(1, 2, 4)))
    assert acts == [Red("c1", MAIN, acts[0].why, (red,)),
                    Drop("c1", "failed: suite:r, suite:x", Outcome((1, 2), heads(1, 2), FAIL, (1, 2), MAIN, (red,))),
                    Build((1,), MAIN, 2)]


@pytest.mark.parametrize("base_key", [None, "key:t-on-the-base"])
def test_a_failure_not_at_its_bases_key_is_the_batchs_own(base_key):
    t = Failed("t", "key:t", frozenset({"pkg2/"}), base_key)
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=prs(1, 2, 3))
    assert decide(s) == [Eject(2, "h2", "t failed, and its changed files meet their inputs, no other member's", (t,)),
                         Drop("c1", "failed: t", Outcome((1, 2), heads(1, 2), FAIL, (2,), MAIN)), Build((1, 3), MAIN, 2)]


@pytest.mark.parametrize(("tests", "base_tests", "inherits"), [
    ({"t::a"}, {"t::a", "t::b"}, True),                     # its failing tests are among the base's
    ({"t::a", "t::b"}, {"t::a", "t::b"}, True),
    ({"t::a", "t::c"}, {"t::a", "t::b"}, False),            # a test the base passes fails: a new failure
    (set(), {"t::a"}, False),                               # its failing tests unknown
    ({"t::a"}, set(), False),                               # the base's failing tests of it unknown
    ({"t::a"}, None, False),                                # the base passed it, or that is unknown
    ({"undeclared:a", "undeclared:new"}, {"undeclared:a", "undeclared:b"}, True),  # a fault the base has too, whatever files
    ({"undeclared:new"}, {"t::a"}, False),                  # the base fails the task, but not by a fault
    ({"undeclared:a", "t::c"}, {"undeclared:a", "undeclared:b"}, False),  # a test failing beside the fault is new
    ({"undeclared:a", "t::c"}, {"undeclared:b", "t::c"}, True),   # the base's run fails that test beside its fault too
    ({"undeclared:a", "fault:x"}, {"undeclared:b", "fault:x"}, True),  # and its runner's error
    ({"undeclared:a", "fault:exit -9"}, {"undeclared:a"}, False),  # a crash beside the reads, which the base's run hasn't
    ({"undeclared:a"}, None, False),                        # the base passed it
])
def test_a_failure_at_a_key_a_member_changed_is_inherited_when_its_failing_tests_are_among_the_bases(tests, base_tests,
                                                                                                        inherits):
    """A PR that edits what a red test reads changes its key, and is still not to blame for the tests `main` fails already."""
    f = by_tests("suite:r", tests, base_tests, "red/", "src/x.py")
    acts = decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(f,)), ready=[pr(1), pr(2, None, "src/x.py")]))
    if inherits:
        assert acts[:3] == [Red("c1", MAIN, acts[0].why, (f,)), Advance("c1"),
                            Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS, (), MAIN, (f,)))]
    else:
        assert kinds(acts) == [Eject, Drop, Build] and acts[0].number == 2


def test_a_red_rings_once_for_each_task_and_key_an_outcome_records_as_inherited():
    """The history's outcomes and this tick's count; a failure's base key is the same red as the key it was recorded at."""
    t = inherited("suite:r", "red/")
    seen = Outcome((1,), heads(1), FAIL, (1,), MAIN, (t,))
    s = snap(cand("c1", MAIN, 2, state=FAIL, failed=(t,)), ready=prs(2), history=[seen])
    assert kinds(decide(s)) == [Advance, Drop]
    other = replace(t, key="key:other", base_key="key:other")
    assert kinds(decide(replace(s, candidates=(cand("c1", MAIN, 2, state=FAIL, failed=(other,)),)))) == [Red, Advance, Drop]
    touched = by_tests("suite:r", {"r::a"}, {"r::a"}, "red/")
    assert kinds(decide(replace(s, candidates=(cand("c1", MAIN, 2, state=FAIL, failed=(touched,)),)))) == [Advance, Drop]
    assert kinds(decide(snap(cand("c1", MAIN, 2, state=FAIL, failed=(t,)), ready=prs(2),
                             history=[Outcome((1,), heads(1), PASS, (), MAIN, (touched,))]))) == [Advance, Drop]
    on_main = snap(cand("c0", MAIN, 1, state=FAIL, failed=(t,)), cand("c1", "c0", 2, state=FAIL, failed=(t,)), ready=prs(1, 2),
                   main="c0")
    assert kinds(decide(on_main)) == [Drop, Advance, Drop]


def test_a_red_main_lands_what_fails_nothing_more_and_its_reds_only_shrink():
    """`main` fails r1 and r2; a batch holding #5, which fixes r1, fails only r2 as `main` does, so it advances, and `main` is
    red on r2 alone until #6's fix lands."""
    r1, r2 = inherited("suite:r1", "red1/"), inherited("suite:r2", "red2/")
    five, six = pr(5, None, "red1/fix.py"), pr(6, None, "red2/fix.py")
    acts = decide(snap(cand("c1", MAIN, 1, 5, state=FAIL, failed=(r2,)), ready=[pr(1), five, six]))
    assert acts == [Red("c1", MAIN, acts[0].why, (r2,)), Advance("c1"),
                    Drop("c1", "on main", Outcome((1, 5), heads(1, 5), PASS, (), MAIN, (r2,))), Build((6,), "c1", 2)]
    history = [Outcome((1, 2), heads(1, 2), PASS, (), MAIN, (r1, r2))]
    assert decide(snap(ready=[*prs(1, 2, 3), five, six], history=history)) == [Build((3, 5, 6), MAIN, 2)]
    assert decide(snap(cand("c2", "c1", 6, state=PASS), ready=[six], main="c1")) == [
        Advance("c2"), Drop("c2", "on main", Outcome((6,), heads(6), PASS))]


# ---- 3. fill --------------------------------------------------------------------------------------------------------------------

def test_a_batch_goes_on_main_at_priority_depth_or_on_the_chains_top_at_depth_minus_its_position():
    assert decide(snap(ready=prs(1, 2), depth=3)) == [Build((1, 2), MAIN, 3)]
    assert decide(snap(cand("c1", MAIN, 1), ready=prs(1, 2), depth=3)) == [Build((2,), "c1", 2)]
    assert decide(snap(cand("c1", MAIN, 1), cand("c2", "c1", 2), ready=prs(1, 2, 3), depth=3)) == [Build((3,), "c2", 1)]
    assert decide(snap(cand("c1", MAIN, 1), cand("c2", "c1", 2), ready=prs(1, 2, 3))) == []
    assert decide(snap(ready=prs(1), depth=0)) == []
    assert decide(snap(ready=[pr(1, conflict=True)])) == []


def test_the_pool_is_in_rank_order_ties_by_number_up_to_batch():
    ready = [pr(5, rank=0), pr(3, rank=1), pr(1, rank=1), pr(2, rank=2), pr(4, rank=0)]
    assert decide(snap(ready=ready, batch=4)) == [Build((4, 5, 1, 3), MAIN, 2)]


def test_the_pool_leaves_out_chain_members_conflicts_and_the_ejected():
    t = task("t", "pkg2/")
    ready = [pr(1), pr(2), pr(3, conflict=True), pr(4)]
    assert decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=ready))[-1] == Build((1, 4), MAIN, 2)
    assert decide(snap(cand("c1", MAIN, 1), ready=ready)) == [Build((2, 4), "c1", 1)]


@pytest.mark.parametrize(("lean", "first", "then"), [(1, (1, 3, 5), (2,)), (3, (1, 2, 3, 4, 5), (6,))])
def test_a_batch_holds_at_most_lean_lean_prs_and_the_next_waits_for_a_later_batch(lean, first, then):
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True), pr(5), pr(6, lean=True)]
    assert decide(snap(ready=ready, lean=lean, batch=5)) == [Build(first, MAIN, 2)]
    assert decide(snap(cand("q1", MAIN, *first), ready=ready, lean=lean, batch=5)) == [Build(then, "q1", 1)]


def test_a_failed_batch_of_six_with_no_culprit_comes_back_as_threes_then_ones():
    ready = prs(1, 2, 3, 4, 5, 6, 7, 8)
    six = Outcome((1, 2, 3, 4, 5, 6), heads(1, 2, 3, 4, 5, 6), FAIL, (1, 2, 3, 4, 5, 6))
    assert decide(snap(ready=ready, history=[six])) == [Build((1, 2, 3), MAIN, 2)]
    assert decide(snap(cand("q1", MAIN, 1, 2, 3), ready=ready, history=[six])) == [Build((4, 5, 6), "q1", 1)]
    three = Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1, 2, 3))
    assert decide(snap(ready=ready, history=[six, three])) == [Build((1,), MAIN, 2)]
    assert decide(snap(cand("q2", MAIN, 1), ready=ready, history=[six, three])) == [Build((2,), "q2", 1)]


def test_a_suspect_with_cap_one_waits_for_a_batch_it_heads_and_goes_alone():
    lone = Outcome((3,), heads(3), FAIL, (3,))
    ready = prs(1, 3, 4, 5)
    assert decide(snap(ready=ready, history=[lone])) == [Build((1, 4, 5), MAIN, 2)]
    assert decide(snap(cand("q1", MAIN, 1, 4, 5), ready=ready, history=[lone])) == [Build((3,), "q1", 1)]


def test_a_cap_comes_from_the_newest_outcome_holding_the_pr_at_its_head():
    pair = Outcome((1, 3), heads(1, 3), FAIL, (1, 3))
    assert decide(snap(ready=prs(1, 2, 3), history=[pair])) == [Build((1,), MAIN, 2)]
    assert decide(snap(ready=[pr(1, "h1b"), pr(2), pr(3)], history=[pair])) == [Build((1, 2), MAIN, 2)]
    later = Outcome((1, 9), heads(1, 9), FAIL, (9,))
    assert decide(snap(ready=prs(1, 2, 3), history=[pair, later])) == [Build((1, 2), MAIN, 2)]


@pytest.mark.parametrize("red", [(), (inherited("r", "pkg10/", "pkg8/"),), (by_tests("r", {"a"}, {"a", "b"}, "pkg8/"),)])
def test_decide_depends_on_what_the_snapshot_holds_not_its_order(red):
    t = task("t", "pkg9/")
    cands = [cand("c1", MAIN, 1, 2, state=FAIL, failed=(t, *red)), cand("c2", "c1", 3), cand("a", "gone", 4), cand("b", "a", 5),
             cand("d", "a", 6), cand("e", "gone2", 7), cand("f", MAIN, 8)]
    ready = [*prs(1, 2, 3, 4, 5, 6, 7), pr(8, rank=0), pr(9, lean=True), pr(10, lean=True)]
    want = decide(snap(*cands, ready=ready))
    rng = random.Random(0)
    for _ in range(50):
        rng.shuffle(cands)
        rng.shuffle(ready)
        assert decide(snap(*cands, ready=ready)) == want


def test_decide_reads_no_clock_files_or_randomness(monkeypatch):
    t, r, u = task("t", "pkg2/", "**/x.py"), inherited("r", "pkg4/"), by_tests("u", {"a"}, {"a"}, "pkg3/")
    snaps = [snap(cand("c1", MAIN, 1, 2, state=state, failed=(t, r) if state == FAIL else (), enqueued=0.0),
                  cand("c2", "c1", 3), cand("z", "gone", 4), ready=[*prs(1, 2, 3, 4), pr(5, lean=True)],
                  history=[Outcome((5,), heads(5), FAIL, (5,)), Outcome((4,), heads(4), FAIL, (), MAIN, (r,))], now=10**6)
             for state in (PENDING, PASS, FAIL, INFRA, LIST_FAILED)]
    snaps += [snap(cand("c1", MAIN, 1, state=FAIL, failed=fs, reruns=n, tries=n + 1), ready=prs(1, 2))
              for fs in ((t,), (r, u), (task("x", "elsewhere/"),), ()) for n in (0, 1)]

    def refuse(*a, **k):
        raise AssertionError("decide is pure: no I/O, no clock, no randomness")

    with monkeypatch.context() as m:
        for mod, name in ((builtins, "open"), (time, "time"), (time, "monotonic"), (time, "perf_counter"),
                          (random, "random"), (random, "randrange"), (random, "shuffle")):
            m.setattr(mod, name, refuse)
        got = [decide(s) for s in snaps]
    assert got == [decide(s) for s in snaps]


# ---- the red team's cases ----------------------------------------------------------------------------------------------------

HAWKEYE = "benchmarks/numerical/tests/reference/test_hawkeye_close.py"


@dataclass
class Play:
    """A small world for the red team's cases: `decide`'s actions applied as the loop does, each pending candidate's verdict
    from ``oracle(tree, candidate)``, and a clock that moves ``step_s`` a tick."""
    ready: list[Ready]
    oracle: Callable[[frozenset[int], Candidate], Verdict]
    kw: dict = field(default_factory=dict)
    step_s: float = 600.0
    main: str = MAIN
    cands: dict[str, Candidate] = field(default_factory=dict)
    trees: dict[str, frozenset[int]] = field(default_factory=lambda: {MAIN: frozenset()})
    history: list[Outcome] = field(default_factory=list)
    landed: list[int] = field(default_factory=list)
    ejected: list[Eject] = field(default_factory=list)
    acts: list[list] = field(default_factory=list)
    built: int = 0

    def tick(self) -> list:
        now = len(self.acts) * self.step_s
        gone = {*self.landed, *(e.number for e in self.ejected)}
        acts = decide(Snapshot(self.main, tuple(r for r in self.ready if r.number not in gone), tuple(self.cands.values()),
                               tuple(self.history), now=now, **self.kw))
        self.acts.append(acts)
        for a in acts:
            if isinstance(a, Advance):
                self.main = a.sha
                self.landed += self.cands[a.sha].members
            elif isinstance(a, Drop):
                self.cands.pop(a.sha)
                if a.outcome is not None:
                    self.history.append(a.outcome)
            elif isinstance(a, Eject):
                self.ejected.append(a)
            elif isinstance(a, Retry):
                self.cands[a.sha] = replace(self.cands[a.sha], verdict=Verdict(), tries=self.cands[a.sha].tries + 1)
            elif isinstance(a, Rerun):
                self.cands[a.sha] = replace(self.cands[a.sha], verdict=Verdict(), reruns=self.cands[a.sha].reruns + 1)
            elif isinstance(a, Build):
                self.built += 1
                sha = f"c{self.built}"
                self.trees[sha] = self.trees[a.base] | set(a.members)
                self.cands[sha] = Candidate(sha, a.base, a.members, tuple(f"h{m}" for m in a.members), enqueued=now)
        for sha, c in self.cands.items():
            if c.verdict.state == PENDING:
                self.cands[sha] = replace(c, verdict=self.oracle(self.trees[sha], c))
        return acts

    def run(self, ticks: int = 60) -> Play:
        for _ in range(ticks):
            self.tick()
        return self

    def count(self, kind: type) -> int:
        return sum(isinstance(a, kind) for acts in self.acts for a in acts)


def test_a_red_main_no_pr_touches_lands_every_pr_and_rings_its_red_once():
    """`main` fails a test no PR touches, at the key every candidate gives it: nobody is ejected, all eight land, Red rings
    once.  Without the base's key the failure is each batch's own, and bisecting would eject every PR as a possible flake."""
    def oracle(tree, c):
        return Verdict(FAIL, (Failed(f"test-file:{HAWKEYE}", "sha512:red", frozenset({HAWKEYE}), "sha512:red"),), ended=True)

    w = Play(prs(*range(1, 9)), oracle).run()
    assert sorted(w.landed) == list(range(1, 9)) and w.ejected == []
    assert w.count(Red) == 1


def test_a_fix_for_one_of_two_reds_lands_and_main_stays_red_on_the_other():
    a_in, b_in = {"a/tests/test_a.py", "a/mod.py"}, {HAWKEYE}

    def oracle(tree, c):
        b = Failed(f"test-file:{HAWKEYE}", "sha512:b", frozenset(b_in), "sha512:b")
        a = Failed("test-file:a/tests/test_a.py", "sha512:a", frozenset(a_in), "sha512:a")
        return Verdict(FAIL, (b,) if 9 in tree else (a, b), ended=True)

    w = Play([*prs(*range(1, 9)), pr(9, None, "a/mod.py")], oracle).run()
    assert sorted(w.landed) == list(range(1, 10)) and w.ejected == []
    assert [o.inherited for o in w.history][-1] == (oracle(frozenset({9}), None).failed[0],)
    assert w.count(Red) == 1 and len(next(a for acts in w.acts for a in acts if isinstance(a, Red)).failed) == 2


def test_a_fail_naming_no_failed_task_ejects_and_drops_nobody():
    w = Play(prs(1, 2, 3, 4), lambda tree, c: Verdict(FAIL)).run(10)
    assert w.landed == [] and w.ejected == [] and w.count(Drop) == 0
    assert all(kinds(acts)[:1] == [Ring] for acts in w.acts[1:])


@pytest.mark.parametrize("tree_ejects", [True, False])
@pytest.mark.parametrize("tree", [True, False])
def test_an_infra_error_one_pr_causes_ejects_it_only_once_graph2_says_it_follows_the_tree_and_tree_ejects_is_set(tree, tree_ejects):
    """#3's tree makes a test end in errors every time.  Seen on two or more nodes (`tree`), with `tree_ejects`, its batch is
    bisected at once on the erring test's inputs, #3 alone errs again and is ejected, and every other PR lands.  Without
    `tree_ejects` the batch is bisected once the errors persist, every other PR lands, and #3 waits alone, retried, with its
    owner rung once.  With `tree_ejects` and no `tree`, the check is enqueued again at every reading, infra is rung in the
    same words each time, and the queue blames nobody."""
    def oracle(members, c):
        if 3 in members:
            return Verdict(INFRA, infra=(Failed("test-file:pkg3/tests/test_reads.py", "sha512:k3",
                                                frozenset({"pkg3/tests/test_reads.py", "pkg3/mod.py"}), tree=tree),))
        return Verdict(PASS)

    w = Play(prs(*range(1, 13)), oracle, {"tree_ejects": tree_ejects}).run()
    rings = [a for acts in w.acts for a in acts if isinstance(a, Ring)]
    if tree and tree_ejects:
        assert sorted(w.landed) == [n for n in range(1, 13) if n != 3] and [e.number for e in w.ejected] == [3]
        assert "test-file:pkg3/tests/test_reads.py" in w.ejected[0].why
        assert w.count(Retry) == w.count(Ring) == 0 and w.acts[-1] == []
    elif not tree_ejects:
        assert sorted(w.landed) == [n for n in range(1, 13) if n != 3] and w.ejected == []
        owner = [a for a in rings if a.pr is not None]
        assert {a.pr for a in owner} == {3} and owner[0].why.startswith("#3's check, with only it merged on ")
        assert kinds(w.acts[-1])[:2] == [Retry, Ring] and w.cands[w.acts[-1][0].sha].members == (3,)
    else:
        assert w.landed == [] and w.ejected == [] and w.count(Drop) == 0
        assert all(kinds(acts) == [Retry, Ring] for acts in w.acts[2:])
        assert len({a.why for a in rings}) == 1


@pytest.mark.parametrize("meets", [True, False])
@pytest.mark.parametrize("erring", [1, 3, 20])
def test_without_tree_ejects_a_pr_whose_check_errs_on_every_node_holds_up_nobody_and_is_rung_to_its_owner(erring, meets):
    """One PR of twenty makes a task err on every node (a hang, an out-of-memory kill), at a batch of 16 and a depth of 3,
    the erring task's inputs meeting its files or not: its batch is bisected once the errors persist, ejecting nobody, the
    other nineteen land within 20 ticks, and it waits alone, enqueued again, its owner rung."""
    def oracle(members, c):
        if erring in members:
            return Verdict(INFRA, infra=(Failed(f"test-file:pkg{erring}/tests/test_big.py", "sha512:big",
                                                frozenset({f"pkg{erring}/mod.py" if meets else "elsewhere/"}), tree=True),),
                           ended=True)
        return Verdict(PASS)

    w = Play(prs(*range(1, 21)), oracle, {"batch": 16, "depth": 3, "lean": 3}).run(20)
    assert sorted(w.landed) == [n for n in range(1, 21) if n != erring] and w.ejected == []
    w.run(10)
    assert w.ejected == [] and [c.members for c in w.cands.values()] == [(erring,)] and kinds(w.acts[-1])[:2] == [Retry, Ring]
    assert {a.pr for acts in w.acts for a in acts if isinstance(a, Ring) and a.pr is not None} == {erring}


@pytest.mark.parametrize("tree_ejects", [False, True])
def test_an_outage_graph2_marks_tree_ejects_nobody_unless_tree_ejects_is_set(tree_ejects):
    """In an outage every check ends in the same infra error, which graph2 marks `tree` since it ran out of attempts on two
    or more nodes.  By default nobody is ejected: the batch is bisected down to lone members, each set aside with its owner
    rung once, then enqueued alone in turn, ringing nobody, and nothing lands.  With `tree_ejects` each lone member is
    ejected for the outage."""
    store = replace(task("test-file:t/test_store.py", "t/test_store.py"), tree=True)
    w = Play(prs(*range(1, 7)), lambda members, c: Verdict(INFRA, infra=(store,), ended=True), {"tree_ejects": tree_ejects}).run(120)
    assert w.landed == []
    if tree_ejects:
        assert sorted(e.number for e in w.ejected) == list(range(1, 7)) and all(e.failed == (store,) for e in w.ejected)
    else:
        assert w.ejected == [] and w.count(Build) > 60 and not any(isinstance(a, Ring) for acts in w.acts[-30:] for a in acts)
        assert {a.pr for acts in w.acts for a in acts if isinstance(a, Ring) and a.pr is not None} == set(range(1, 7))


def test_a_candidate_pending_forever_is_retried_once_and_then_rung():
    w = Play(prs(*range(1, 13)), lambda tree, c: Verdict(PENDING), {"pending_s": 3600.0}).run(20)
    assert len(w.cands) == 2 and w.count(Retry) == 1
    assert [i for i, acts in enumerate(w.acts) if Retry in kinds(acts)] == [7]
    assert [i for i, acts in enumerate(w.acts) if Ring in kinds(acts)] == list(range(13, 20))


def test_a_lone_retest_reruns_its_failed_keys_instead_of_building_the_same_tree_again():
    """The same tree has the same keys, which graph2 answers from its cached fail: a flake would eject at its first run."""
    failed = Failed("test-file:t/test_x.py", "sha512:x", frozenset({"t/test_x.py"}))
    c1 = Candidate("c1", MAIN, (7,), ("h7",), Verdict(FAIL, (failed,)))
    assert decide(Snapshot(MAIN, (pr(7),), (c1,))) == [Rerun("c1", ("sha512:x",))]

    def flaky(tree, c):
        return Verdict(FAIL, (failed,), ended=True) if c.sha == "c1" and not c.reruns else Verdict(PASS)

    w = Play(prs(7, 8), flaky, {"batch": 1}).run(10)
    assert sorted(w.landed) == [7, 8] and w.ejected == [] and w.count(Rerun) == 1
    w = Play(prs(7, 8), lambda tree, c: Verdict(FAIL, (failed,), ended=True) if 7 in tree else Verdict(PASS), {"batch": 1}).run(10)
    assert w.landed == [8] and [e.number for e in w.ejected] == [7] and "if a test flaked, delete it" in w.ejected[0].why


def test_an_earlier_lone_failure_of_another_task_ejects_nobody_without_a_rerun():
    earlier = Outcome((5,), ("h5",), FAIL, (5,))
    other = Failed("test-file:t/test_other.py", "sha512:o", frozenset({"t/test_other.py"}))
    c1 = Candidate("c1", MAIN, (5,), ("h5",), Verdict(FAIL, (other,)))
    assert decide(Snapshot(MAIN, (pr(5),), (c1,), (earlier,))) == [Rerun("c1", ("sha512:o",))]


def test_a_pr_that_touches_a_red_tasks_inputs_is_not_ejected_for_the_tests_main_fails():
    tests = frozenset({"t/test_x.py::test_red"})
    red = Failed("test-file:t/test_x.py", "sha512:changed-by-the-pr", frozenset({"t/test_x.py", "src/x.py"}),
                 "sha512:red-on-main", tests, tests)
    c1 = Candidate("c1", MAIN, (1, 2, 3), ("h1", "h2", "h3"), Verdict(FAIL, (red,), ended=True))
    acts = decide(Snapshot(MAIN, (pr(1), pr(2, None, "src/x.py"), pr(3)), (c1,)))
    assert kinds(acts) == [Red, Advance, Drop]
    worse = replace(red, tests=tests | {"t/test_x.py::test_new"})
    acts = decide(Snapshot(MAIN, (pr(1), pr(2, None, "src/x.py"), pr(3)), (replace(c1, verdict=Verdict(FAIL, (worse,))),)))
    assert [a.number for a in acts if isinstance(a, Eject)] == [2]


@pytest.mark.parametrize("env", [False, True])
def test_an_env_bump_is_ejected_by_its_inputs_once_they_hold_the_envs_spec_and_after_bisecting_and_a_rerun_else(env):
    def oracle(tree, c):
        if 4 in tree:
            return Verdict(FAIL, (Failed("test-file:t/test_np.py", "sha512:np", frozenset({"t/test_np.py"} | ({"uv.lock"} if env
                                                                                                                 else set()))),),
                           ended=True)
        return Verdict(PASS)

    ready = prs(*range(1, 7))
    ready[3] = pr(4, None, "uv.lock")
    w = Play(ready, oracle).run()
    assert sorted(w.landed) == [1, 2, 3, 5, 6] and [e.number for e in w.ejected] == [4]
    assert ("meet their inputs" if env else "if a test flaked, delete it") in w.ejected[0].why
    assert (w.built, w.count(Rerun)) == ((2, 0) if env else (7, 1))


# ---- the simulated world -------------------------------------------------------------------------------------------------------

TICK_S, PENDING_S = 600.0, 1800.0


def inherits(f: Failed) -> bool:
    """The world's statement of an inherited failure, apart from `decide`'s."""
    return f.key == f.base_key or bool(f.tests) and f.base_tests is not None and f.tests <= f.base_tests


def reads(v: Verdict) -> str:
    """What the world expects ``v`` to call for: a fail whose every failure is inherited is a pass once ended, infra beside an
    infra error, pending before."""
    if v.state == LIST_FAILED:
        return INFRA
    if v.state == FAIL and v.failed and all(inherits(f) for f in v.failed):
        return INFRA if v.infra else PASS if v.ended else PENDING
    return v.state


class World:
    """The loop's world, scripted by a seed: `main`, the ready marks, the live candidates with graph2's verdicts, and the outcomes
    the loop recorded.  A bad PR fails every candidate whose tree holds it, on a task whose inputs meet its files (and
    `shared/`'s), or for an interaction nobody's, at a key only such trees give the task, or, in a red world, on a red task's
    test of its own.  An erring PR makes a task end in infra errors on every node (`tree`) in every such tree; it is ejected
    with `tree_ejects` (every other seed), and without it waits alone, its owner rung, while an outage's errors are marked
    `tree` too, as graph2 marks them.  In a
    red world `main` fails one or two red tasks in every tree without their fix, a good PR, at a key each PR touching their inputs
    changes (some good PRs touch them, the same tests failing); a red task whose failing tests graph2 doesn't report is
    inherited at its key alone, and no good PR but its fix touches it.  PRs arrive, push, conflict and are unmarked for a while;
    graph2 answers late, sends the cached inherited failures before the rest, flakes on a first run, errs on a first
    enqueue, errs or fails to list at every enqueue until infra, rung, reruns the check, or wedges until retried; someone
    else moves `main`, the loop stops a tick early, and a merged PR stays marked for a while; `step` checks every invariant
    on the way."""

    def __init__(self, seed: int, red: bool | None = None, bad: bool = True,
                 policy: Callable[[Snapshot], list] = decide):
        self.seed, self.policy, self.tree_ejects = seed, policy, seed % 2 == 0
        rng = self.rng = random.Random(seed)
        numbers = sorted(rng.sample(range(10, 90), rng.randint(3, 12)))
        self.batch, self.depth, self.rings = rng.choice((1, 2, 3, 4, 6)), rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        self.lean = rng.choice((1, 3))
        roll = {m: rng.random() for m in numbers}
        self.bad = {m: "own" if rng.random() < 0.5 else "interaction" for m in numbers if bad and roll[m] < 0.25}
        self.erring = {m: rng.choice(("meets", "nobody")) for m in numbers if bad and 0.25 <= roll[m] < 0.32}
        self.good = set(numbers) - set(self.bad) - set(self.erring)
        red = rng.random() < 0.3 if red is None else red
        self.fixes = rng.sample(sorted(self.good), min(len(self.good), rng.choice((1, 2)))) if red else []
        self.detail = [rng.random() < 0.6 for _ in self.fixes]   # graph2 reports the red task's failing tests
        files = {m: {f"pkg{m}/mod{m}.py"} | ({"shared/x.py"} if rng.random() < 0.3 else set()) for m in numbers}
        self.breaks: dict[int, int] = {}
        for i, fix in enumerate(self.fixes):
            files[fix].add(f"red{i}/fix.py")
            for m in sorted(self.good - set(self.fixes)):
                if self.detail[i] and rng.random() < 0.2:
                    files[m].add(f"red{i}/other{m}.py")
            for m in sorted(self.bad):
                if m not in self.breaks and rng.random() < 0.4:
                    self.breaks[m] = i
                    files[m].add(f"red{i}/break{m}.py")
        self.marks = {m: Ready(m, f"h{m}", frozenset(files[m]), rng.random() < 0.25, rng.randint(0, 4), rng.random() < 0.1)
                      for m in numbers}
        self.arrive = {m: rng.randint(0, 6) for m in numbers}
        self.lag = rng.choice((0, 1, None))           # ticks a merged PR stays marked; None: for good
        self.pushes: dict[int, list[int]] = {}
        for m in numbers:
            if self.marks[m].conflict:
                self.pushes.setdefault(self.arrive[m] + rng.randint(1, 8), []).append(m)
        for _ in range(rng.randint(0, 3)):
            self.pushes.setdefault(rng.randint(1, 30), []).append(rng.choice(numbers))
        self.unmarks = {rng.randint(1, 30): (rng.choice(numbers), rng.randint(1, 6)) for _ in range(rng.randint(0, 2))}
        self.elsewhere = set() if self.lag is None else {rng.randint(2, 40) for _ in range(rng.randint(0, 2))}
        self.crashes = rng.randint(0, 4)
        self.main, self.tick, self.made = MAIN, 0, 0
        self.cands: dict[str, Candidate] = {}
        self.tree: dict[str, frozenset[int]] = {MAIN: frozenset()}
        self.due: dict[str, int | None] = {}
        self.early: set[str] = set()
        self.outage: dict[str, str] = {}              # sha -> the state its check ends in until infra reruns it
        self.history: list[Outcome] = []
        self.merged: dict[int, int] = {}              # number -> the tick it reached main
        self.ejected: set[int] = set()
        self.unmarked: dict[int, int] = {}            # number -> the tick it is marked again
        self.rung: set[tuple[str, str]] = set()
        self.owned: set[tuple[int, str]] = set()      # (number, head) whose owner was rung
        self.again: set[tuple[str, str]] = set()      # rung by a tick that stopped before its outcome was recorded
        self.seen: Counter[str] = Counter()

    def ready(self) -> tuple[Ready, ...]:
        return tuple(r for m, r in self.marks.items() if self.arrive[m] <= self.tick and m not in self.ejected
                     and m not in self.unmarked and (m not in self.merged or self.lag is None or self.tick < self.merged[m] + self.lag))

    def snapshot(self) -> Snapshot:
        cands = list(self.cands.values())
        self.rng.shuffle(cands)
        return Snapshot(self.main, self.ready(), tuple(cands), tuple(self.history), self.batch, self.depth, self.rings, self.lean,
                        self.tick * TICK_S, PENDING_S, self.tree_ejects)

    def done(self) -> bool:
        if self.tree_ejects:
            return set(self.merged) == self.good and self.ejected == set(self.bad) | set(self.erring) and not self.cands
        return (set(self.merged) == self.good and self.ejected == set(self.bad)
                and all(set(c.members) <= set(self.erring) for c in self.cands.values())
                and all((m, self.marks[m].head) in self.owned for m in self.erring))

    def key(self, name: str, m: int, sha: str) -> str:
        return f"{name}@{'with' if m in self.tree[sha] else 'without'} #{m}"

    def touches(self, m: int, i: int) -> bool:
        return meets(self.marks[m].files, (f"red{i}/",))

    def red_tests(self, i: int, sha: str) -> frozenset[str]:
        tree = self.tree[sha]
        return frozenset(({f"red{i}::test_broken"} if self.fixes[i] not in tree else set())
                         | {f"red{i}::test_{m}" for m in tree if self.breaks.get(m) == i})

    def red_key(self, i: int, sha: str) -> str:
        return f"suite:red{i}@" + ",".join(str(m) for m in sorted(self.tree[sha]) if self.touches(m, i))

    def reds(self, sha: str) -> frozenset[tuple[str, str]]:
        """What ``sha``'s tree fails, by task and test: its bad members' tasks and the red tasks' tests."""
        own = {(f"suite:{m}" if self.bad[m] == "own" else f"interaction:{m}", "") for m in self.tree[sha] & set(self.bad)
               if m not in self.breaks}
        return frozenset(own | {(f"suite:red{i}", t) for i in range(len(self.fixes)) for t in self.red_tests(i, sha)})

    def failures(self, c: Candidate) -> tuple[Failed, ...]:
        """``c``'s failures: its bad members' and the red tasks', each with its key on ``c``'s base and, where graph2 reports
        them, its failing tests and the base's."""
        out = []
        for m in sorted(self.tree[c.sha] & set(self.bad)):
            if m in self.breaks:
                continue
            if self.bad[m] == "interaction":
                name = f"interaction:{m}"
                out.append(Failed(name, self.key(name, m, c.sha), frozenset({"elsewhere/untouched.py", "docs/**"}),
                                  None if m % 2 else self.key(name, m, c.base)))
            else:
                name, own = f"suite:{m}", (f"pkg{m}/", f"pkg{m}", f"pkg{m}/**", f"pkg{m}/*.py", f"**/mod{m}.py")[m % 5]
                out.append(Failed(name, self.key(name, m, c.sha), frozenset({own, "shared/x.py"}), self.key(name, m, c.base)))
        for i in range(len(self.fixes)):
            if tests := self.red_tests(i, c.sha):
                out.append(Failed(f"suite:red{i}", self.red_key(i, c.sha), frozenset({f"red{i}/"}), self.red_key(i, c.base),
                                  *((tests, self.red_tests(i, c.base) or None) if self.detail[i] else ())))
        return tuple(out)

    def verdict(self, c: Candidate) -> Verdict:
        """Its tree's failures and infra errors as the loop derives the state from them; on a first enqueue, now and then a
        listing or infra error, and on a first run a flake; before the rest of the check ends, now and then only the failures
        graph2 answers from its cache (at the base's key)."""
        rng = self.rng
        if self.outage.get(c.sha) == INFRA:
            return Verdict(INFRA, (), (Failed("node", f"node@{c.sha}", frozenset({"tools/runner.py"}), tree=not self.tree_ejects),))
        if self.outage.get(c.sha) == LIST_FAILED or (c.tries == 1 and rng.random() < 0.06):
            return Verdict(LIST_FAILED)
        failed = self.failures(c)
        if c.tries == 1 and not c.reruns and rng.random() < 0.08:
            failed += (Failed("flaky:x", f"flaky@{c.sha}", frozenset({"elsewhere/flaky.py"}), f"flaky@{c.base}",
                              frozenset({"flaky::test_x"})),)
        infra = tuple(Failed(f"erring:{m}", f"erring@{c.sha}", frozenset({f"pkg{m}/" if self.erring[m] == "meets" else "nowhere/"}),
                             tree=True) for m in sorted(self.tree[c.sha] & set(self.erring)))
        if c.tries == 1 and rng.random() < 0.08:
            infra += (Failed("runner", f"runner@{c.sha}", frozenset({"tools/runner.py"})),)
        cached = tuple(f for f in failed if f.key == f.base_key)
        if cached and c.sha not in self.early and rng.random() < 0.3:
            self.early.add(c.sha)
            self.due[c.sha] = self.tick + rng.randint(1, 2)
            return Verdict(FAIL, cached)
        if failed:
            return Verdict(FAIL, failed, infra, ended=True)
        return Verdict(INFRA, (), infra) if infra else Verdict(PASS)

    def events(self) -> str | None:
        """What happens before the tick's snapshot; the new `main` when someone else moved it."""
        moved = None
        if self.tick in self.elsewhere:
            moved = f"x{self.tick}"
            self.tree[moved], self.main = self.tree[self.main], moved
            self.seen["main moved elsewhere"] += 1
        for m in self.pushes.get(self.tick, ()):
            if m not in self.merged and m not in self.ejected:
                self.marks[m] = replace(self.marks[m], head=self.marks[m].head + "+", conflict=False)
        if self.tick in self.unmarks:
            m, back = self.unmarks[self.tick]
            if m not in self.merged and m not in self.ejected:
                self.unmarked[m] = self.tick + back
        self.unmarked = {m: t for m, t in self.unmarked.items() if t > self.tick}
        for sha, c in self.cands.items():
            v = c.verdict
            if (v.state == PENDING or v.state == FAIL and not v.ended) and self.due[sha] is not None and self.due[sha] <= self.tick:
                self.cands[sha] = replace(c, verdict=self.verdict(c))
        return moved

    def step(self) -> None:
        self.tick += 1
        moved = self.events()
        s = self.snapshot()
        acts = self.policy(s)
        self.check(s, acts, moved)
        n = len(acts)
        if acts and self.crashes and self.rng.random() < 0.15:
            self.crashes -= 1
            n = self.rng.randrange(len(acts))
            self.seen["the loop stopped early"] += 1
        for a in acts[:n]:
            self.apply(a)
        recorded = {a.sha for a in acts[:n] if isinstance(a, Drop) and a.outcome is not None}
        self.again |= {(f.task, f.key) for a in acts[:n] if isinstance(a, Red) and a.sha not in recorded for f in a.failed}
        live = [m for c in self.cands.values() for m in c.members]
        assert len(live) == len(set(live)), f"seed {self.seed} tick {self.tick}: a PR in two live candidates: {self.cands}"
        again = self.policy(self.snapshot())
        dropped = {a.sha for a in acts[:n] if isinstance(a, Drop)}
        repeated = [a for a in set(again) & set(acts[:n]) if not isinstance(a, Ring) and not (isinstance(a, Red) and a.sha not in dropped)]
        assert not repeated, f"seed {self.seed} tick {self.tick}: a restart repeats {repeated}"

    def check(self, s: Snapshot, acts: list, moved: str | None) -> None:
        where = f"seed {self.seed} tick {self.tick}: {acts}"
        by = {c.sha: c for c in s.candidates}
        ready = {r.number: r for r in s.ready}
        decided = {a.sha for a in acts if isinstance(a, Drop) and a.outcome is not None}
        for a in acts:
            if isinstance(a, Advance):
                c = by[a.sha]
                assert c.base == s.main and reads(c.verdict) == PASS, where
                assert all(m in ready and ready[m].head == h for m, h in zip(c.members, c.heads)), where
                self.seen["advanced on a red main"] += bool(self.reds(s.main))
                self.seen["inherited by tests"] += any(f.key != f.base_key for f in c.verdict.failed)
                self.seen["inherited at its key alone"] += any(not f.tests for f in c.verdict.failed)
            elif isinstance(a, Eject):
                d = next(d for d in acts if isinstance(d, Drop) and d.outcome is not None and d.outcome.state == FAIL
                         and a.number in d.outcome.members)
                assert not [f for f in a.failed if inherits(f)], where
                assert s.tree_ejects or set(a.failed) <= set(by[d.sha].verdict.failed), f"{where}: ejected for infra errors"
                self.seen["ejected by inputs" if "meet their inputs" in a.why else "ejected after a rerun"
                          if "rerun afresh" in a.why else "ejected for erring"] += 1
            elif isinstance(a, Red):
                c = by[a.sha]
                assert a.base == c.base == s.main and a.sha in decided and a.failed and all(map(inherits, a.failed)), where
                self.seen["a red rung"] += 1
            elif isinstance(a, Build):
                assert 1 <= len(a.members) <= s.batch and sum(ready[m].lean for m in a.members) <= s.lean, where
                assert not any(ready[m].conflict for m in a.members), where
                self.seen["a lean PR built"] += any(ready[m].lean for m in a.members)
                self.seen["two lean PRs built together"] += sum(ready[m].lean for m in a.members) > 1
                self.seen["built on a red main"] += bool(self.reds(s.main))
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == FAIL:
                c = by[a.sha]
                own = [f for f in c.verdict.failed if not inherits(f)]
                persists = reads(c.verdict) == INFRA and c.tries >= min(s.rings, 2) and len(c.members) > 1
                assert own or (reads(c.verdict) == INFRA and s.tree_ejects and any(f.tree for f in c.verdict.infra)) or (
                    persists and not s.tree_ejects), where
                self.seen["bisected"] += len(a.outcome.suspects) > 1
                self.seen["bisected on infra errors"] += not own and len(a.outcome.suspects) > 1
                self.seen["bisected on infra errors, ejecting nobody"] += not own and not s.tree_ejects
                self.seen["a mixed failure"] += bool(own and a.outcome.inherited)
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == INFRA:
                c = by[a.sha]
                assert a.outcome == Outcome(c.members, c.heads, INFRA, c.members, c.base) and not s.tree_ejects, where
                assert reads(c.verdict) == INFRA and c.tries >= min(s.rings, 2) and len(c.members) == 1, where
                self.seen["set aside"] += 1
            elif isinstance(a, Retry):
                c = by[a.sha]
                late = c.enqueued is not None and s.now - c.enqueued > s.pending_s
                assert ((reads(c.verdict) == INFRA and not (s.tree_ejects and any(f.tree for f in c.verdict.infra)))
                        or (reads(c.verdict) == PENDING and late and c.tries == 1)), where
                self.seen["retried late" if reads(c.verdict) == PENDING else "retried"] += 1
                self.seen["retried while the rest ran"] += c.verdict.state == FAIL
            elif isinstance(a, Rerun):
                c = by[a.sha]
                own = [f for f in c.verdict.failed if not inherits(f)]
                assert len(c.members) == 1 and not c.reruns and own and a.keys == tuple(f.key for f in own), where
                assert not any(meets(ready[c.members[0]].files, f.inputs) for f in own), where
                self.seen["rerun"] += 1
            elif isinstance(a, Ring):
                assert a.pr is None or (not s.tree_ejects and by[a.sha].members == (a.pr,)), where
                self.seen["rung late" if "still pending" in a.why else "an owner rung" if a.pr else "rung"] += 1
        for c in by.values():
            fresh = all(m in ready and ready[m].head == h for m, h in zip(c.members, c.heads))
            stale = any(isinstance(a, Drop) and a.sha == c.sha and a.outcome is None for a in acts)
            if c.base != s.main or not fresh or stale:
                continue
            if reads(c.verdict) == INFRA:
                tree = s.tree_ejects and any(f.tree for f in c.verdict.infra)
                persists = not s.tree_ejects and c.tries >= min(s.rings, 2)
                bisect, lone = tree or (persists and len(c.members) > 1), persists and len(c.members) == 1
                aside = any(isinstance(a, Drop) and a.sha == c.sha and a.outcome is not None and a.outcome.state == INFRA
                            for a in acts)
                self.seen["a tree error retried"] += not s.tree_ejects and any(f.tree for f in c.verdict.infra) and not persists
                assert (Retry(c.sha) in acts) == (not bisect and not aside) and (c.sha in decided) == (bisect or aside), where
                assert (not aside or lone) and Advance(c.sha) not in acts, where
                assert tree or not any(isinstance(a, Eject) for a in acts), where
                told = lone and any(o.state == INFRA and (c.members[0], c.heads[0]) in zip(o.members, o.heads) for o in s.history)
                rung = any(isinstance(a, Ring) and a.sha == c.sha and a.pr is None for a in acts)
                assert rung == (not tree and c.tries >= min(s.rings, 2) and not (aside and told)), where
                owners = [a.pr for a in acts if isinstance(a, Ring) and a.sha == c.sha and a.pr is not None]
                assert owners == (list(c.members) if lone and not told else []), where
                self.seen["infra rung and retried"] += rung and c.tries >= 2 and not bisect and not aside
                self.seen["a lone member retried, its owner rung"] += lone and not aside
            elif reads(c.verdict) == PENDING:
                waited = s.now - c.enqueued
                assert (Retry(c.sha) in acts) == (waited > s.pending_s and c.tries == 1), where
                assert any(isinstance(a, Ring) and a.sha == c.sha for a in acts) == (waited > 2 * s.pending_s), where
                self.seen["waited for the rest of a check"] += c.verdict.state == FAIL
        if moved:
            drops = [a for a in acts if isinstance(a, Drop)]
            assert sorted(d.sha for d in drops) == sorted(by) and all(d.outcome is None for d in drops), where
            pool = [r for r in s.ready if not r.conflict and r.number not in self.merged]
            assert [a.base for a in acts if isinstance(a, Build)] == ([moved] if pool else []), where

    def apply(self, a) -> None:
        where = f"seed {self.seed} tick {self.tick}: {a}"
        rng = self.rng
        if isinstance(a, Build):
            assert a.base == self.main or a.base in self.cands, where
            sha, self.made = f"q{self.made}", self.made + 1
            self.cands[sha] = Candidate(sha, a.base, a.members, tuple(self.marks[m].head for m in a.members),
                                        enqueued=self.tick * TICK_S)
            self.tree[sha] = self.tree[a.base] | set(a.members)
            self.due[sha] = None if rng.random() < 0.06 else self.tick + rng.randint(1, 3)
            self.seen["wedged"] += self.due[sha] is None
            if rng.random() < 0.04:
                self.outage[sha] = rng.choice((INFRA, LIST_FAILED))
        elif isinstance(a, Advance):
            c = self.cands[a.sha]
            assert c.base == self.main and reads(c.verdict) == PASS, where
            assert not set(c.members) & set(self.merged), f"{where}: a PR reaches main twice"
            assert not set(c.members) & (set(self.bad) | set(self.erring)), f"{where}: a bad PR reaches main"
            assert self.reds(a.sha) <= self.reds(self.main), f"{where}: main advances to a new failure"
            self.seen["a fix cleared a red"] += self.reds(a.sha) < self.reds(self.main)
            self.main = a.sha
            self.merged |= dict.fromkeys(c.members, self.tick)
        elif isinstance(a, Drop):
            del self.cands[a.sha]
            if a.outcome is not None:
                self.history.append(a.outcome)
        elif isinstance(a, Eject):
            assert self.marks[a.number].head == a.head and a.number in set(self.bad) | set(self.erring), f"{where}: a good PR ejected"
            self.ejected.add(a.number)
        elif isinstance(a, (Retry, Rerun)):
            c = self.cands[a.sha]
            self.cands[a.sha] = replace(c, verdict=Verdict(), tries=c.tries + isinstance(a, Retry),
                                        reruns=c.reruns + isinstance(a, Rerun))
            self.due[a.sha] = self.tick + (rng.randint(4, 7) if rng.random() < 0.3 else rng.randint(1, 2))
        elif isinstance(a, Red):
            assert a.base == self.main and a.sha in self.cands, where
            pairs = {(f.task, f.key) for f in a.failed}
            assert not pairs & (self.rung - self.again), f"{where}: a red rung twice"
            self.rung |= pairs
            self.again -= pairs
        else:
            assert isinstance(a, Ring), where
            c = self.cands.get(a.sha)
            if c is not None and a.pr is not None:
                self.owned.add((a.pr, c.heads[0]))
            if c is not None and a.sha in self.outage and c.tries >= 2:
                del self.outage[a.sha]
                self.cands[a.sha] = replace(c, verdict=Verdict())
                self.due[a.sha] = self.tick + rng.randint(2, 5)
                self.seen["rung, and infra reran it"] += 1


SEEDS = range(500)


def ticks(w: World) -> int:
    """The ticks within which a scenario settles: every good PR on main, every bad PR ejected, no candidate left (without
    `tree_ejects`, only erring PRs left, each with its owner rung).  Seeds 0 to 4999 settled within 88 ticks and 15 per PR,
    none more than 21 past 8 a PR, and with a broken main whose fixes are ranked last and late, within 72 and 10.7 per PR,
    none more than 8 past 8 a PR."""
    return 40 + 8 * len(w.marks)


def test_the_simulated_queue_keeps_every_invariant_and_settles():
    """Over every seed: main advances only to a candidate on it with no new failure (each failure inherited, nothing running,
    no infra error), at its members' marked heads, so its reds never grow; no PR is in two live candidates or reaches main
    twice; with `tree_ejects`, an infra error that follows the tree (`tree`) is bisected like a fail at once, and any other
    infra or listing error is retried at every reading, rung from `rings` enqueues (2 at most), and blames nobody; without
    it, any infra or listing error, a `tree` one too, is retried, then rung and bisected from `rings` enqueues, ejecting
    nobody, and a lone member's is rung to its owner too and set aside behind the PRs waiting, else retried; a candidate
    pending past the bound is retried once and rung past twice it; a lone member's own failure is rerun once
    before it is ejected; a batch holds at most `lean` Lean PRs; a move of main by someone else drops the stale candidates
    and rebuilds on it; an inherited failure ejects nobody, and its red rings once; a restart repeats nothing but a ring, or
    a red whose drop it didn't apply; and every good PR reaches main and every bad one is ejected within `ticks`, every
    erring one too with `tree_ejects`, and without it only erring PRs stay live, each with its owner rung.  Each situation
    the invariants are about happens in some seed."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        w = World(seed)
        while not w.done():
            assert w.tick < ticks(w), (f"seed {seed}: not settled after {ticks(w)} ticks: merged {sorted(w.merged)} of "
                                       f"{sorted(w.good)}, ejected {sorted(w.ejected)} of {sorted(w.bad)} and "
                                       f"{sorted(w.erring)}, live {list(w.cands.values())}")
            w.step()
        assert not w.reds(w.main), f"seed {seed}: main still fails {sorted(w.reds(w.main))} with every fix on it"
        seen.update(w.seen)
    for what in ("main moved elsewhere", "the loop stopped early", "ejected by inputs", "ejected after a rerun",
                 "ejected for erring", "a tree error retried", "a lean PR built", "two lean PRs built together", "bisected",
                 "bisected on infra errors", "bisected on infra errors, ejecting nobody", "set aside", "an owner rung",
                 "a lone member retried, its owner rung",
                 "retried", "retried late", "retried while the rest ran", "rerun", "rung", "rung late", "wedged", "a red rung",
                 "infra rung and retried", "rung, and infra reran it",
                 "advanced on a red main", "built on a red main", "inherited by tests", "inherited at its key alone",
                 "a mixed failure", "waited for the rest of a check", "a fix cleared a red"):
        assert seen[what], f"no seed exercised: {what}"


def test_a_red_main_keeps_landing_good_prs_its_reds_never_grow_and_its_fixes_clear_them():
    """`main` fails one or two red tasks in every tree without their fixes, which are ranked last and arrive late: no PR is
    ejected, good PRs land on the red main before the fixes do, main's reds never grow (checked at each advance), they are
    gone once the fixes are on main, and every PR is on main within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in range(100):
        w = World(seed, red=True, bad=False)
        for fix in w.fixes:
            w.marks[fix], w.arrive[fix] = replace(w.marks[fix], rank=9), seed % 10
        while not w.done():
            assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks: merged {sorted(w.merged)}"
            w.step()
        assert not w.ejected and not w.reds(w.main), f"seed {seed}: {w.ejected} {w.reds(w.main)}"
        seen["a PR landed before a fix"] += min(w.merged.values()) < min(w.merged[f] for f in w.fixes)
        seen.update(w.seen)
    for what in ("a red rung", "advanced on a red main", "a fix cleared a red", "inherited by tests",
                 "inherited at its key alone", "a PR landed before a fix"):
        assert seen[what], f"no seed exercised: {what}"


def _candidates(s: Snapshot, f: Callable[[Candidate], Candidate]) -> Snapshot:
    return replace(s, candidates=tuple(map(f, s.candidates)))


def _verdicts(s: Snapshot, **kw) -> Snapshot:
    return _candidates(s, lambda c: replace(c, verdict=replace(c.verdict, **kw)))


def _failures(s: Snapshot, f: Callable[[Failed], Failed], field: str = "failed") -> Snapshot:
    return _candidates(s, lambda c: replace(c, verdict=replace(c.verdict, **{field: tuple(map(f, getattr(c.verdict, field)))})))


MISREADS: dict[str, Callable[[Snapshot], Snapshot]] = {
    "tries as one": lambda s: _candidates(s, lambda c: replace(c, tries=1)),
    "tree ejects flipped": lambda s: replace(s, tree_ejects=not s.tree_ejects),
    "reruns as none": lambda s: _candidates(s, lambda c: replace(c, reruns=0)),
    "reruns as one": lambda s: _candidates(s, lambda c: replace(c, reruns=1)),
    "every check as ended": lambda s: _verdicts(s, ended=True),
    "no infra errors": lambda s: _verdicts(s, infra=()),
    "no infra error following the tree": lambda s: _failures(s, lambda f: replace(f, tree=False), "infra"),
    "every infra error following the tree": lambda s: _failures(s, lambda f: replace(f, tree=True), "infra"),
    "no clock": lambda s: replace(s, now=None),
    "no base keys": lambda s: _failures(s, lambda f: replace(f, base_key=None)),
    "no failing tests": lambda s: _failures(s, lambda f: replace(f, tests=frozenset())),
    "the base failing what the candidate fails": lambda s: _failures(
        s, lambda f: f if f.base_tests is None else replace(f, base_tests=f.base_tests | f.tests)),
    "no history": lambda s: replace(s, history=()),
    "no Lean cap": lambda s: replace(s, lean=s.batch),
}


@pytest.mark.parametrize("misread", MISREADS)
def test_decide_misreading_a_field_of_the_snapshot_breaks_an_invariant_of_the_simulated_queue(misread):
    """The invariants hold `decide` to every field it reads: handed snapshots with one field misread, it breaks one of them,
    or doesn't settle, in some seed."""
    with pytest.raises(AssertionError):
        for seed in SEEDS:
            w = World(seed, policy=lambda s: decide(MISREADS[misread](s)))
            while not w.done():
                assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks"
                w.step()
