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
from research.vqueue import (BASE_DROPPED, FAIL, INFRA, INFRA_SPLIT, LIST_FAILED, ON_MAIN, PASS, PENDING, Advance, Build,
                             Candidate, Commit,
                             Drop, Eject, Failed, Outcome, Ready, RecordsRun, Red, Regenerate, Rerun, Retry, Ring, Snapshot,
                             Verdict, Wait, decide, meets)
from research.vqueue_slim import Read, joined, read_step, read_suite, unnamed

MAIN = "m0"


def pr(n: int, head: str | None = None, *files: str, rank: int | None = None, lean: bool = False,
       conflict: bool = False, lean_input: bool = False) -> Ready:
    return Ready(n, head or f"h{n}", frozenset(files or (f"pkg{n}/mod.py",)), lean, n if rank is None else rank, conflict,
                 lean_input)


def prs(*ns: int) -> list[Ready]:
    return [pr(n) for n in ns]


def cand(sha: str, base: str, *members: int, state: str = PENDING, failed: tuple[Failed, ...] = (),
         infra: tuple[Failed, ...] = (), ended: bool = True, tries: int = 1, reruns: int = 0, enqueued: float | None = None,
         heads: tuple[str, ...] | None = None) -> Candidate:
    return Candidate(sha, base, members, heads or tuple(f"h{m}" for m in members), Verdict(state, failed, infra, ended), tries,
                     reruns, enqueued)


def task(name: str, *inputs: str, kind: str | None = None) -> Failed:
    return Failed(name, f"key:{name}", frozenset(inputs), kind=kind)


def inherited(name: str, *inputs: str) -> Failed:
    """A failure at its base's key: the base fails it too."""
    return Failed(name, f"key:{name}", frozenset(inputs), f"key:{name}")


def by_tests(name: str, tests: set[str], base_tests: set[str] | None, *inputs: str) -> Failed:
    """A failure at a key a member gave the task, with ``tests`` failing where the base fails ``base_tests`` of it."""
    return Failed(name, f"key:{name}+", frozenset(inputs), f"key:{name}", frozenset(tests),
                  None if base_tests is None else frozenset(base_tests))


def snap(*cands: Candidate, ready=(), history=(), main: str = MAIN, wide=(), pinned=(), **kw) -> Snapshot:
    return Snapshot(main, tuple(ready), cands, tuple(history), wide=frozenset(wide), pinned=frozenset(pinned), **kw)


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

def test_the_chain_is_ordered_from_main_and_only_its_lowest_verdict_is_read_whole():
    ready = prs(1, 2, 3, 4)
    assert decide(snap(cand("c2", "c1", 3, state=PASS), cand("c1", MAIN, 1, 2), ready=ready)) == []
    assert decide(snap(cand("c2", "c1", 3, state=FAIL), cand("c1", MAIN, 1, 2), ready=ready)) == []
    assert decide(snap(cand("c2", "c1", 3), cand("c1", MAIN, 1, 2, state=PASS), ready=ready)) == [
        Advance("c1"), Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS)), Build((4,), "c2", 1)]


@pytest.mark.parametrize("lowest", [PENDING, PASS])
def test_a_fail_above_the_lowest_candidate_is_decided_at_once_and_nothing_is_built_on_it(lowest):
    """A candidate above the lowest whose check failed, still running or not, on tests its base passed, is decided as the
    lowest's fail is: its lone suspect ejected, the candidates above it dropped as a base failure and built again in the same
    tick on the candidate below it (`main`, once the lowest passed), never on it, and the next batch on them.  While its base
    hasn't said, it waits, and nothing is built above it."""
    ready = [pr(1), pr(2, None, "pkg/x.py"), pr(3), pr(4)]
    t = by_tests("step pytest", {"pkg::t"}, set(), "pkg/x.py")
    s = snap(cand("c1", MAIN, 1, state=lowest), cand("c2", "c1", 2, state=FAIL, failed=(t,), ended=False),
             cand("c3", "c2", 3), ready=ready, depth=3)
    passed = [Advance("c1"), Drop("c1", "on main", Outcome((1,), heads(1), PASS))] if lowest == PASS else []
    deep = 3 if lowest == PASS else 2
    assert decide(s) == [*passed, Eject(2, "h2", "step pytest failed, and its changed files meet their inputs, no other "
                                                "member's", (t,)),
                         Drop("c2", "failed: step pytest", Outcome((2,), heads(2), FAIL, (2,), "c1")),
                         Drop("c3", "its base failed"), Build((3,), "c1", deep, replaces="c3"), Build((4,), "c3", deep - 1)]
    unsaid = by_tests("step pytest", {"pkg::t"}, None, "pkg/x.py")
    s = snap(cand("c1", MAIN, 1, state=lowest), cand("c2", "c1", 2, state=FAIL, failed=(unsaid,), ended=False),
             ready=ready, depth=3)
    assert decide(s) == passed


def test_a_fail_above_the_lowest_candidate_bisects_its_suspects_reruns_a_lone_one_and_waits_on_what_is_inherited():
    """Its suspects are bisected; a lone member whose failures meet none of its files is rerun afresh first; one its base
    fails too waits for its check's end."""
    ready = [pr(1), pr(2, None, "pkg/x.py"), pr(3, None, "pkg/y.py"), pr(4)]
    t = by_tests("step pytest", {"pkg::t"}, set(), "pkg/")
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, 3, state=FAIL, failed=(t,), ended=False), ready=ready, depth=3)
    assert decide(s) == [Drop("c2", "failed: step pytest", Outcome((2, 3), heads(2, 3), FAIL, (2, 3), "c1")),
                         Build((2,), "c1", 2)]
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 4, state=FAIL, failed=(t,), ended=False), ready=ready, depth=3)
    assert decide(s) == [Rerun("c2", (t.key,)), Build((2, 3), "c2", 1)]
    red = by_tests("step pytest", {"pkg::t"}, {"pkg::t"}, "pkg/")
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, 3, state=FAIL, failed=(red,), ended=False), ready=ready, depth=3)
    assert decide(s) == [Build((4,), "c2", 1)]


def by_main(name: str, tests: set[str], main_tests: set[str] | None, *inputs: str, base_tests: set[str] | None = None
            ) -> Failed:
    """A failure of a candidate above another, with ``tests`` failing where `main` fails ``main_tests`` of it and its base
    ``base_tests`` (None: not said yet)."""
    return replace(by_tests(name, tests, base_tests, *inputs), main_tests=None if main_tests is None else frozenset(main_tests))


def test_a_fail_above_the_lowest_on_tests_main_fails_too_is_inherited_on_main_s_word():
    """Only its own member meets the failure, and `main` fails the same tests: its base, which leaves them as `main` has
    them, fails them too, so it fails nothing of its own and the next batch goes on it."""
    t = by_main("step pytest", {"pkg::t"}, {"pkg::t"}, "pkg2/")
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, state=FAIL, failed=(t,), ended=False), ready=prs(1, 2, 3), depth=3)
    assert decide(s) == [Build((3,), "c2", 1)]


def test_a_fail_above_the_lowest_decided_on_main_s_word_suspects_only_its_own_failures():
    """Two failures meet only its own members, one `main` passes and one `main` fails too: only the first suspects anyone,
    so #2 is ejected and #3 built again at once on the candidate below; a lone member whose own failure (its base passed it)
    meets none of its files is rerun, the one `main` fails too set aside."""
    a, b = by_main("step a", {"a::x"}, set(), "pkg2/"), by_main("step b", {"b::y"}, {"b::y"}, "pkg3/")
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, 3, state=FAIL, failed=(a, b)), ready=prs(1, 2, 3, 4), depth=3)
    assert decide(s) == [Eject(2, "h2", "step a failed, and its changed files meet their inputs, no other member's",
                               (replace(a, base_tests=frozenset()),)),
                         Drop("c2", "failed: step a, step b", Outcome((2, 3), heads(2, 3), FAIL, (2,), "c1")),
                         Build((3,), "c1", 2, replaces="c2"), Build((4,), "c2", 1)]
    lone = by_main("step a", {"a::x"}, None, "elsewhere/", base_tests=set())
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, state=FAIL, failed=(lone, replace(b, inputs=frozenset({"pkg2/"})))),
             ready=prs(1, 2, 3), depth=3)
    assert decide(s) == [Rerun("c2", (lone.key,)), Build((3,), "c2", 1)]


def test_a_rebuild_whose_records_run_goes_first_is_staged_off_the_chain_and_those_above_it_go_on_the_chain_s_top():
    """A Lean member's candidate rebuilt on a new base runs its records again, staged off main's chain: nothing is built on
    it, so the candidates above it are built again on the chain's top, and the next batch on them takes no Lean PR.  A
    candidate on a staged batch is off main's chain, and is dropped, not built again."""
    s = snap(cand("c1", MAIN, 1, state=FAIL, failed=(task("suite:1", "pkg1/"),)), cand("c2", "c1", 2), cand("c3", "c2", 3),
             ready=[pr(1), lean(2), pr(3), pr(4)], depth=3, records=True)
    assert decide(s)[-6:] == [Drop("c1", "failed: suite:1", Outcome((1,), heads(1), FAIL, (1,), MAIN)),
                              Drop("c2", "its base failed"), Drop("c3", "its base failed"),
                              Build((2,), MAIN, 3, records=True, replaces="c2"), Build((3,), MAIN, 3, replaces="c3"),
                              Build((4,), "c3", 2)]
    s = snap(waiting("c1", MAIN, 1, records=RecordsRun(PENDING, "r1 runs")), cand("c2", "c1", 2, 5), ready=[lean(1), pr(5)],
             records=True, left=((2, "h2", "held by top (no reason)"),), pulled=frozenset({2}))
    assert decide(s) == [Drop("c2", "not on main's chain: its base c1 is neither main nor a chain member"),
                         Ring("c1", "c1's records run: r1 runs"), Build((5,), MAIN, 2)]


def test_a_rebuild_whose_records_run_goes_first_waits_for_a_lean_batch_while_another_batch_stays_staged():
    """One batch is staged at a time: while another stays staged, a Lean member's candidate a red dropped is not built
    again, and its members wait for a Lean batch as any Lean PR does; the candidates above it go on the chain's top."""
    s = snap(cand("c1", MAIN, 1, state=FAIL, failed=(task("suite:1", "pkg1/"),)), cand("c2", "c1", 2), cand("c3", "c2", 3),
             waiting("c4", MAIN, 5, records=RecordsRun(PENDING, "r4 runs")),
             ready=[pr(1), lean(2), pr(3), pr(4), lean(5)], depth=3, records=True)
    assert decide(s)[-5:] == [Drop("c2", "its base failed"), Drop("c3", "its base failed"),
                              Ring("c4", "c4's records run: r4 runs"), Build((3,), MAIN, 3, replaces="c3"),
                              Build((4,), "c3", 2)]


def test_a_rebuild_leaves_out_a_member_on_main_or_in_a_candidate_below():
    """Built again, a candidate holding a member that landed below it, or one a rebuild below it holds, leaves it out."""
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2, 1, heads=heads(2, 1)), ready=prs(1, 2, 3))
    assert decide(s) == [Drop("c2", "#1 is in a candidate below it too"), Advance("c1"),
                         Drop("c1", "on main", Outcome((1,), heads(1), PASS)), Build((2,), "c1", 2, replaces="c2"),
                         Build((3,), "c2", 1)]
    s = snap(cand("c1", MAIN, 1, state=FAIL, failed=(task("suite:1", "pkg1/"),)), cand("c2", "c1", 2),
             cand("c3", "c2", 3, 2, heads=heads(3, 2)), ready=prs(1, 2, 3, 4), depth=3)
    assert decide(s)[-3:] == [Build((2,), MAIN, 3, replaces="c2"), Build((3,), "c2", 2, replaces="c3"), Build((4,), "c3", 1)]


def test_a_candidate_off_the_chain_is_dropped_with_every_candidate_above_it():
    s = snap(cand("y", "x", 3), cand("c1", MAIN, 1), cand("x", "gone", 2, state=PASS), ready=prs(1, 2, 3), depth=3)
    assert decide(s) == [Drop("x", "not on main's chain: its base gone is neither main nor a chain member"),
                         Drop("y", "its base was dropped"), Build((2, 3), "c1", 2)]


def test_of_two_candidates_on_one_base_the_one_whose_members_come_first_in_rank_stays():
    ready = [pr(2, rank=1), pr(3, rank=2), pr(4, rank=0), pr(5, rank=5)]
    s = snap(cand("a", MAIN, 3, state=PASS), cand("a2", "a", 5), cand("b", MAIN, 2, 4), ready=ready)
    assert decide(s) == [Drop("a", "a second candidate on m0: b's members come first in rank"),
                         Drop("a2", "its base was dropped"), Build((3, 5), "b", 1)]


@pytest.mark.parametrize(("left", "why"), [
    (((1, "h1", "held by top (no reason)"), (4, "h4", "no land verdict")), "#1 isn't taken now: held by top (no reason)"),
    (((1, "h1", "its ready mark was withdrawn by ci"),), "#1 isn't taken now: its ready mark was withdrawn by ci"),
    (((1, "h1", "ejected by top at h1, the head it was built at: broke main"),),
     "#1 isn't taken now: ejected by top at h1, the head it was built at: broke main"),
])
def test_a_chain_candidate_holding_a_member_others_took_from_the_queue_is_rebuilt_without_it_with_everything_above(left, why):
    """A hold or a withdrawal by someone else takes out that PR alone: its candidate and those above it are dropped saying
    why and, in the same tick, built again in order without it, each keeping its other members."""
    ready = prs(2, 3, 4)
    s = snap(cand("c1", MAIN, 1, 2, state=PASS), cand("c2", "c1", 3), ready=ready, left=left, pulled=frozenset({1}),
             depth=3)
    assert decide(s) == [Drop("c1", why), Drop("c2", "its base was dropped"), Build((2,), MAIN, 3, replaces="c1"),
                         Build((3,), "c1", 2, replaces="c2"), Build((4,), "c2", 1)]
    s = snap(cand("c1", MAIN, 3), cand("c2", "c1", 1, 2), ready=ready, left=left, pulled=frozenset({1}))
    assert decide(s) == [Drop("c2", why), Build((2,), "c1", 1, replaces="c2")]


@pytest.mark.parametrize(("ready", "left"), [
    ([pr(1, "h1b"), pr(2), pr(3)], ()),
    ([pr(1, "h1b", conflict=True), pr(2), pr(3)], ()),
    ([pr(2), pr(3)], ((1, "h1b", "it isn't marked ready"),)),
    ([pr(2), pr(3)], ((1, "h1", "its ready mark was withdrawn by lane-1"),)),
    ([pr(2), pr(3)], ()),
])
def test_a_chain_candidate_whose_member_its_owner_pushed_unmarked_or_closed_lands_as_built(ready, left):
    """Its new head, if any, waits for the next batch."""
    s = snap(cand("c1", MAIN, 1, 2, state=PASS), cand("c2", "c1", 3), ready=ready, left=left)
    assert decide(s) == [Advance("c1"), Drop("c1", "on main", Outcome((1, 2), heads(1, 2), PASS))]
    s = snap(cand("c1", MAIN, 3), cand("c2", "c1", 1, 2), ready=ready, left=left)
    assert decide(s) == []


def test_a_fail_blames_a_member_its_owner_took_out_by_its_files_as_built_and_ejects_nobody():
    t = task("suite:pkg1", "pkg1/")
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=prs(2, 3), flying=(pr(1),))
    assert decide(s) == [Drop("c1", "failed: suite:pkg1", Outcome((1, 2), heads(1, 2), FAIL, (1,), MAIN)),
                         Build((2,), MAIN, 2, replaces="c1"), Build((3,), "c1", 1)]
    s = snap(cand("c1", MAIN, 1, state=FAIL, failed=(t,)), ready=prs(2, 3), flying=(pr(1),))
    assert decide(s) == [Drop("c1", "failed: suite:pkg1", Outcome((1,), heads(1), FAIL, (1,), MAIN)), Build((2, 3), MAIN, 2)]


def test_a_pr_in_a_candidate_below_makes_the_one_above_stale():
    s = snap(cand("c1", MAIN, 1), cand("c2", "c1", 2, 1, heads=heads(2, 1)), ready=prs(1, 2))
    assert decide(s) == [Drop("c2", "#1 is in a candidate below it too"), Build((2,), "c1", 1, replaces="c2")]


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


def lands(sha: str, *members: int, base: str | None = None, inherited: tuple[Failed, ...] = ()) -> list:
    return [Advance(sha), Drop(sha, ON_MAIN, Outcome(members, heads(*members), PASS, (), base, inherited))]


def test_a_pass_advances_main_through_each_passed_candidate_above_it_in_the_same_tick():
    """A candidate above the lowest that passed on the one below counts once that one is on `main`, so the tick that advances
    to it goes on through it: an advance and a drop each, in order, and the next batch on the last."""
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2, state=PASS), ready=prs(1, 2, 3))
    assert decide(s) == [*lands("c1", 1), *lands("c2", 2), Build((3,), "c2", 2)]
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2, state=PASS), cand("c3", "c2", 3, state=PASS),
             ready=prs(1, 2, 3, 4), depth=3)
    assert decide(s) == [*lands("c1", 1), *lands("c2", 2), *lands("c3", 3), Build((4,), "c3", 3)]


def test_a_tick_advances_only_through_the_passed_candidates_at_the_bottom_of_the_chain():
    """It stops at the first candidate that hasn't passed: one that passed above it waits for it; and a batch staged on the
    last one, its records run first, is off the chain, whatever its verdict says."""
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2, state=PASS), cand("c3", "c2", 3),
             cand("c4", "c3", 4, state=PASS), ready=prs(1, 2, 3, 4, 5), depth=4)
    assert decide(s) == [*lands("c1", 1), *lands("c2", 2), Build((5,), "c4", 2)]
    s = snap(cand("c1", MAIN, 1, state=PASS), cand("c2", "c1", 2, state=PASS),
             replace(waiting("c3", "c2", 3), verdict=Verdict(PASS)), ready=[pr(1), pr(2), lean(3)], depth=3, records=True)
    assert decide(s) == [*lands("c1", 1), *lands("c2", 2)]


def test_candidates_advanced_in_one_tick_ring_a_red_they_share_once():
    """The one above inherits the lowest's red from it (at its key, or by its tests), which the lowest's outcome records."""
    t = inherited("suite:r", "red/")
    touched = Failed(t.task, "key:suite:r+", t.inputs, t.key, frozenset({"r::a"}), frozenset({"r::a"}))
    for above in (t, touched):
        s = snap(cand("c1", MAIN, 1, state=FAIL, failed=(t,)), cand("c2", "c1", 2, state=FAIL, failed=(above,)),
                 ready=prs(1, 2, 3))
        acts = decide(s)
        assert acts == [Red("c1", MAIN, acts[0].why, (t,)), *lands("c1", 1, base=MAIN, inherited=(t,)),
                        *lands("c2", 2, base="c1", inherited=(above,)), Build((3,), "c2", 2)]


def test_a_failure_whose_inputs_meet_one_members_files_ejects_it():
    t1, t2 = task("suite:a", "pkg2/"), task("lean:b", "elsewhere/x.lean")
    s = snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(t1, t2)), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4))
    assert decide(s) == [
        Eject(2, "h2", "suite:a failed, and its changed files meet their inputs, no other member's", (t1, t2)),
        Drop("c1", "failed: suite:a, lean:b", Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (2,), MAIN)),
        Drop("c2", "its base failed"), Build((1, 3), MAIN, 2, replaces="c1"), Build((4,), "c1", 1, replaces="c2")]


def test_a_failure_whose_inputs_meet_several_members_suspects_only_them():
    """The others are built again at once in their order, and the suspects bisected in the batch after them."""
    t = task("t", "pkg1/mod.py", "pkg3")
    s = snap(cand("c1", MAIN, 1, 2, 3, 4, state=FAIL, failed=(t,)), ready=prs(1, 2, 3, 4))
    assert decide(s) == [Drop("c1", "failed: t", Outcome((1, 2, 3, 4), heads(1, 2, 3, 4), FAIL, (1, 3), MAIN)),
                         Build((2, 4), MAIN, 2, replaces="c1"), Build((1,), "c1", 1)]
    s = snap(cand("c1", MAIN, 1, 2, 3, 4, state=FAIL, failed=(t,)), ready=[pr(1, rank=9), *prs(2, 3, 4)])
    assert decide(s)[1:] == [Build((2, 4), MAIN, 2, replaces="c1"), Build((3,), "c1", 1)]


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
def test_without_tree_ejects_infra_that_persists_is_split_on_its_inputs_as_infra_ejecting_nobody(state):
    """Without `tree_ejects`, a check that errs again once enqueued again (from `rings` readings, 2 at most) is split like a
    fail, on the members whose files meet the erring tasks' inputs, else on every member, and nobody is ejected: the
    members without the error land, and the one with it ends up alone.  The split is recorded as infra (`INFRA_SPLIT`),
    never as a fail.  A first reading is only retried."""
    err = (task("test-file:pkg2/tests/test_reads.py", "pkg2/tests/test_reads.py", "pkg2/mod.py"),) if state == INFRA else ()
    why = "c1's check ended in infra errors: test-file:pkg2/tests/test_reads.py" if state == INFRA else "c1's check failed to list"
    c = Candidate("c1", MAIN, (1, 2, 3), heads(1, 2, 3), Verdict(state, (), err))
    assert decide(snap(c, cand("c2", "c1", 4), ready=prs(1, 2, 3, 4, 5))) == [Retry("c1")]
    for tries, rings in ((2, 2), (2, 3), (5, 2), (1, 1)):
        acts = decide(snap(replace(c, tries=tries), cand("c2", "c1", 4), ready=prs(1, 2, 3, 4), rings=rings))
        suspects, rebuilt = ((2,), [Build((1, 3), MAIN, 2, replaces="c1"), Build((4,), "c1", 1, replaces="c2")]) if (
            state == INFRA) else ((1, 2, 3), [Build((4,), MAIN, 2, replaces="c2"), Build((1,), "c2", 1)])
        assert acts == [Ring("c1", why), Drop("c1", f"{why}, again after it was enqueued again: bisected on them as infra, "
                                                    "ejecting nobody", Outcome((1, 2, 3), heads(1, 2, 3), INFRA_SPLIT, suspects, MAIN)),
                        Drop("c2", "its base failed"), *rebuilt], (tries, rings)


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
    again at that head rings nobody), and retried when none waits, unless it isn't marked at the head it was built at: then
    it is set aside, and a new head is built.  With `tree_ejects` it is retried and infra alone is rung."""
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
    assert decide(snap(c, cand("c2", "c1", 4), ready=prs(3, 4))) == [
        infra, owner, Drop("c1", why, aside), Drop("c2", "its base failed"), Build((4,), MAIN, 2, replaces="c2"),
        Build((3,), "c2", 1)]
    assert decide(snap(c, ready=prs(3), history=(aside,))) == [Retry("c1"), infra]
    assert decide(snap(c, ready=prs(3, 4), history=(aside,))) == [Drop("c1", why, aside), Build((4,), MAIN, 2)]
    assert decide(snap(c, ready=prs(3, 4), tree_ejects=True)) == [Retry("c1"), infra, Build((4,), "c1", 1)]
    moved = why.replace("merged:", "merged, which isn't marked at h3 now:")     # its owner pushed: the new head is built
    assert decide(snap(c, ready=[pr(3, "h3b")])) == [infra, owner, Drop("c1", moved, aside), Build((3,), MAIN, 2)]
    assert decide(snap(c, ready=())) == [infra, owner, Drop("c1", moved, aside)]


def test_infra_from_launches_that_started_no_step_is_retried_never_split_and_sets_a_lone_member_aside_after_launch_tries():
    """Infra from launches that started none of the candidate's steps (`Failed.launch`: no run registered, a launcher killed
    by a signal, check.py refusing its own plan) points at the machines or the loop's configuration: its candidate is
    enqueued again and infra is rung from `rings` readings on, however often, and a candidate of several is never split for
    it; a lone member is set aside behind the PRs waiting only after `launch_tries` readings.  Infra from runs that started
    the candidate's steps (no `launch`) splits a candidate of several at its second reading, as infra."""
    err = (Failed("the run beside its check registered no run: shard 1/5: train: c1 on m1,m2 shard 1/5\nTerminated",
                  "c1:no run", frozenset(), launch=True),)
    ran = (replace(err[0], task="the run beside its check (r1) was killed (failed rc=137)", launch=False),)
    infra = Ring("c1", f"c1's check ended in infra errors: {err[0].task}")
    pair = Candidate("c1", MAIN, (1, 2), heads(1, 2), Verdict(INFRA, (), err))
    lone = Candidate("c1", MAIN, (3,), heads(3), Verdict(INFRA, (), err))
    assert decide(snap(pair, cand("c2", "c1", 4), ready=prs(1, 2, 4))) == [Retry("c1")]
    for tries in (2, 3, 10):
        assert decide(snap(replace(pair, tries=tries), cand("c2", "c1", 4), ready=prs(1, 2, 4))) == [Retry("c1"), infra]
    for tries, on in ((2, {}), (4, {"launch_tries": 5})):
        assert decide(snap(replace(lone, tries=tries), ready=prs(3, 4), **on)) == [Retry("c1"), infra, Build((4,), "c1", 1)]
    acts = decide(snap(replace(pair, verdict=Verdict(INFRA, (), ran), tries=2), cand("c2", "c1", 4), ready=prs(1, 2, 4)))
    assert kinds(acts) == [Ring, Drop, Drop, Build, Build] and acts[1].outcome.state == INFRA_SPLIT, acts
    assert not any(isinstance(a, Eject) for a in acts)
    for c in (replace(lone, tries=3), replace(lone, verdict=Verdict(INFRA, (), ran), tries=2)):
        acts = decide(snap(c, ready=prs(3, 4)))
        assert kinds(acts) == [Ring, Ring, Drop, Build] and acts[2].outcome.state == INFRA, (c, acts)


@pytest.mark.parametrize("kind", ["timeout", "killed", "infra", "store-state"])
def test_a_failure_its_records_say_the_machine_made_is_infra_never_a_red(kind):
    """A failed step whose records say a timeout, a kill, a full disk or no memory, or a shared state's fault
    (`MACHINE_KINDS`) is an infra error: retried at once, rung from the second reading, then split on its inputs as infra,
    ejecting nobody; a lone member is never ejected for it.  Beside a failure of the change's own, only that one blames; one
    the base fails too is inherited, as any failure is; and while other tasks run it waits, never failing early."""
    oom = task("step pytest", "pkg2/", kind=kind)
    why = "c1's check ended in infra errors: step pytest"
    c = cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(oom,))
    assert decide(snap(c, ready=prs(1, 2, 3, 4))) == [Retry("c1"), Build((4,), "c1", 1)]
    acts = decide(snap(replace(c, tries=2), ready=prs(1, 2, 3, 4)))
    assert acts == [Ring("c1", why), Drop("c1", f"{why}, again after it was enqueued again: bisected on them as infra, ejecting "
                                                "nobody", Outcome((1, 2, 3), heads(1, 2, 3), INFRA_SPLIT, (2,), MAIN)),
                    Build((1, 3), MAIN, 2, replaces="c1"), Build((2,), "c1", 1)]
    lone = cand("c1", MAIN, 2, state=FAIL, failed=(oom,), tries=2)
    acts = decide(snap(lone, ready=prs(2, 4)))
    assert kinds(acts) == [Ring, Ring, Drop, Build] and acts[2].outcome == Outcome((2,), heads(2), INFRA, (2,), MAIN), acts
    own = task("step lean-build", "pkg1/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(own, oom)), ready=prs(1, 2, 3)))
    assert acts[0] == Eject(1, "h1", "step lean-build failed, and its changed files meet their inputs, no other member's", (own,))
    assert acts[1].outcome == Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1,), MAIN)
    red = replace(oom, base_key=oom.key)
    assert kinds(decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(red,)), ready=prs(1, 2)))) == [Red, Advance, Drop]
    assert decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(oom,), ended=False), ready=prs(1, 2, 4))) == [
        Build((4,), "c1", 1)]


def test_a_flaky_or_own_failure_is_still_a_fail():
    for kind in ("flaky", "own", None):
        t = task("step pytest", "pkg2/", kind=kind)
        acts = decide(snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(t,)), ready=prs(1, 2, 3)))
        assert kinds(acts) == [Eject, Drop, Build] and acts[1].outcome.state == FAIL, (kind, acts)


def test_an_infra_split_caps_its_suspects_as_a_fail_does_and_sets_none_aside():
    """An `infra-split` outcome bounds its suspects' batches at half its suspects, as a fail's does, and, unlike a PR set aside
    (`infra`), puts none of them last or alone."""
    split = Outcome((1, 2, 3, 4), heads(1, 2, 3, 4), INFRA_SPLIT, (1, 2, 3, 4), MAIN)
    assert decide(snap(ready=prs(1, 2, 3, 4, 5), history=(split,))) == [Build((1, 2), MAIN, 2)]
    assert decide(snap(ready=[pr(5, rank=0), pr(1, rank=1)], history=(split,))) == [Build((5, 1), MAIN, 2)]


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
                    Build((1, 3), MAIN, 2, replaces="c1"), Build((4,), "c1", 1, replaces="c2")]
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
            assert acts == [Ring("c1", why), Drop("c1", f"{why}, again after it was enqueued again: bisected on them as infra, "
                                                        "ejecting nobody", Outcome((1, 2, 3), heads(1, 2, 3), INFRA_SPLIT, (1, 2),
                                                                                   MAIN)),
                            Drop("c2", "its base failed"), Build((3,), MAIN, 2, replaces="c1"),
                            Build((4,), "c1", 1, replaces="c2")]
        else:
            assert kinds(acts) == [Ring, Ring, Drop, Drop, Build, Build] and acts[0] == Ring("c1", why) and acts[1].pr == 2
            assert acts[2].outcome == Outcome((2,), heads(2), INFRA, (2,), MAIN)
            assert acts[-2:] == [Build((4,), MAIN, 2, replaces="c2"), Build((1, 3), "c2", 1)]


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
    assert acts[2].outcome == Outcome((1, 2), heads(1, 2), INFRA_SPLIT, (1, 2), MAIN, (t,))
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
                    Build((1, 3), MAIN, 2, replaces="c1"), Build((9,), "c1", 1)]
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
                         Drop("c1", "failed: t", Outcome((1, 2), heads(1, 2), FAIL, (2,), MAIN)),
                         Build((1,), MAIN, 2, replaces="c1"), Build((3,), "c1", 1)]


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


# ---- the records run before a Lean candidate's check ---------------------------------------------------------------------------

def stale(pkg: str = "security", *inputs: str) -> Failed:
    """graph2's lean-lock fail of ``pkg``: its lock records don't match its tree."""
    return Failed(f"lean-lock:{pkg}", f"key:lean-lock:{pkg}", frozenset(inputs or (f"{pkg}/",)))


def lean(n: int, *files: str) -> Ready:
    return pr(n, None, *(files or (f"security/A{n}.lean",)), lean=True)


#: a records run that runs
RUNNING = RecordsRun()


def waiting(sha: str, base: str, *members: int, records: RecordsRun | None = RUNNING, held: float | None = None,
            ) -> Candidate:
    """A candidate built with its records run first (`Build.records`), whose check waits for it."""
    return replace(cand(sha, base, *members, tries=0), records=records, waits=True, held=held)


def test_a_lean_batch_is_staged_its_records_run_first_and_its_tick_builds_the_prs_with_no_lean_change_beside_it():
    """`Build.records`: a batch whose members change Lean, a lock or a spec (`Ready.lean`) has its records regenerated before
    its check.  Staged, it takes no place on main's chain and no build from the PRs with no Lean change: the tick that
    stages it builds theirs too, on the same base, their check started at once, while the chain has room, and stages it
    alone on a full chain.  With the step off (`Snapshot.records`) the Lean batch's check starts at once, and it is the
    tick's one build."""
    ready = [pr(1), lean(2), pr(3)]
    assert decide(snap(ready=ready, records=True)) == [Build((2,), MAIN, 2, records=True), Build((1, 3), MAIN, 2)]
    assert decide(snap(cand("c1", MAIN, 1), ready=ready, depth=3, records=True)) == [
        Build((2,), "c1", 2, records=True), Build((3,), "c1", 2)]
    assert decide(snap(cand("c1", MAIN, 1), cand("c2", "c1", 3), ready=[*ready, pr(4)], records=True)) == [
        Build((2,), "c2", 0, records=True)]
    assert decide(snap(ready=[lean(2), lean(4)], records=True, lean=2)) == [Build((2, 4), MAIN, 2, records=True)]
    assert decide(snap(ready=ready)) == [Build((2,), MAIN, 2)]
    assert decide(snap(ready=[lean(2), lean(4)], lean=2)) == [Build((2, 4), MAIN, 2)]


def test_the_batch_built_beside_the_lean_batch_its_tick_stages_takes_no_pr_set_aside_whether_or_not_it_changes_a_lean_input():
    """As beside a batch staged by an earlier tick: a lone PR set aside goes behind it as behind any PR waiting, and one
    changing a Lean input (`Ready.lean_input`) would change the Lean inputs of the top the staged batch goes onto.  Set
    aside, such a PR keeps no Lean batch waiting either."""
    aside = Outcome((5,), heads(5), INFRA, (5,), MAIN)
    s = snap(ready=[lean(1), pr(5)], history=[aside], records=True)
    assert decide(s) == [Build((1,), MAIN, 2, records=True)]
    touching = pr(5, None, "tools/verity/check/check.py", lean_input=True)
    assert decide(replace(s, ready=(lean(1), touching))) == [Build((1,), MAIN, 2, records=True)]
    assert decide(replace(s, ready=(lean(1), touching, pr(6)))) == [Build((1,), MAIN, 2, records=True), Build((6,), MAIN, 2)]


@pytest.mark.parametrize("pins", [dict(), dict(pinned=frozenset({2})), dict(pinned=frozenset({2}), lean=0)],
                         ids=["no pin", "pinned", "pinned at lean 0"])
def test_no_lean_batch_is_staged_pinned_or_not_while_a_pr_changing_a_lean_input_that_a_batch_may_take_waits(pins):
    """Staged, a Lean batch keeps a PR changing a Lean input out of every build for its whole records run, and the batch
    staged after it would again.  So such a PR with no Lean change that a batch may take goes first, whatever its rank and
    whether or not a pin names the Lean batch's PRs, at `lean` 0 too: the tick builds it with the PRs with no Lean change,
    and the Lean batch is staged once it is in a candidate, on it; with the chain full, both wait, and one the tick's batch
    has no place for keeps the Lean batch waiting as well.  One set aside goes behind the PRs waiting, the Lean batch's
    among them, and keeps none waiting; nor does any with no records run, where no batch keeps it out."""
    touching = pr(1, None, "tools/verity/check/check.py", lean_input=True)
    s = snap(ready=[touching, lean(2), pr(3)], records=True, **pins)
    assert decide(s) == [Build((1, 3), MAIN, 2)]
    assert decide(replace(s, ready=(replace(touching, rank=9), lean(2), pr(3)))) == [Build((3, 1), MAIN, 2)]
    assert decide(replace(s, candidates=(cand("c1", MAIN, 1, 3),))) == [Build((2,), "c1", 1, records=True)]
    assert decide(replace(s, candidates=(cand("c8", MAIN, 8), cand("c9", "c8", 9)))) == []
    assert decide(replace(s, ready=(pr(0), touching, lean(2)), batch=1)) == [Build((0,), MAIN, 2)]
    aside = Outcome((1,), heads(1), INFRA, (1,), MAIN)
    assert decide(replace(s, history=(aside,))) == [Build((2,), MAIN, 2, records=True), Build((3,), MAIN, 2)]
    assert decide(replace(s, records=False)) == [Build((2,), MAIN, 2)]


def test_with_a_records_run_no_lean_batch_is_staged_on_a_chain_holding_a_pr_set_aside_before_that_changes_a_lean_input():
    """A lone PR set aside (an `infra` outcome) is built again in its turn, and leaves main's chain when its check errs again:
    a Lean batch staged on it would be staged again there, its records run lost.  While the chain holds one at the head it
    was set aside at, with a Lean change or changing a Lean input, no Lean batch is staged: the PRs with no Lean change are
    built, with none to take `Wait` says why, and the Lean PR waiting sets it aside again when its check errs, as any PR
    waiting does.  One set aside that changes no Lean input, or at a new head, keeps none out; nor does any with no records
    run, where the top candidate's Lean change does."""
    aside = Outcome((1,), heads(1), INFRA, (1,), MAIN)
    s = snap(cand("c1", MAIN, 1), ready=[lean(1), lean(2)], history=[aside], records=True, depth=3)
    why = ("nothing built: with a Lean change, #2 waits while the chain holds #1, set aside before, which changes a Lean "
           "input, and no PR with no Lean change is ready to take")
    assert decide(s) == [Wait(why)]
    assert decide(replace(s, ready=(*s.ready, pr(3)))) == [Build((3,), "c1", 2)]
    assert decide(replace(s, ready=(pr(1, None, "tools/verity/check/check.py", lean_input=True), lean(2)))) == [Wait(why)]
    err = (Failed("node", "key:node", frozenset({"tools/runner.py"})),)
    acts = decide(replace(s, candidates=(cand("c1", MAIN, 1, state=INFRA, infra=err, tries=2),)))
    assert kinds(acts) == [Drop, Build] and acts[0].outcome.state == INFRA and acts[1] == Build((2,), MAIN, 3, records=True)
    on = [Build((2,), "c1", 2, records=True)]
    assert decide(replace(s, ready=(pr(1), lean(2)))) == on
    assert decide(replace(s, candidates=(cand("c1", MAIN, 1, heads=("h1b",)),), ready=(pr(1, "h1b", lean=True), lean(2)))) == on
    assert decide(replace(s, records=False)) == [Wait(
        "nothing built: with a Lean change, #2 waits while the newest candidate, c1, holds #1's Lean change, and no PR with "
        "no Lean change is ready to take")]


def test_a_staged_batch_reads_no_verdict_and_the_chain_fills_beside_it_with_no_second_lean_batch():
    """Whatever its verdict says (graph2 knows no check of it), it is never read.  It is off main's chain: nothing is built on
    it and it counts toward no depth, so batches of PRs with no Lean change are built on the chain's top up to the depth
    while its records run runs, and no other Lean PR is taken meanwhile.  The candidate below it is decided as ever."""
    c1 = replace(waiting("c1", MAIN, 1), verdict=Verdict(FAIL, (task("t", "security/"),), ended=True))
    assert decide(snap(c1, ready=[lean(1), *prs(2, 3)], depth=3, records=True)) == [Build((2, 3), MAIN, 3)]
    assert decide(snap(c1, ready=[lean(1), lean(4)], depth=3, records=True)) == []
    s = snap(cand("c0", MAIN, 3, state=PASS), waiting("c1", "c0", 1), ready=[lean(1), *prs(2, 3), lean(4)], depth=3,
             records=True)
    assert decide(s) == [Advance("c0"), Drop("c0", "on main", Outcome((3,), heads(3), PASS)), Build((2,), "c0", 3)]
    s = replace(s, candidates=(cand("c0", MAIN, 3), waiting("c1", "c0", 1), cand("c2", "c0", 2)))
    assert decide(s) == decide(replace(s, depth=2)) == []


@pytest.mark.parametrize("files", [("tools/verity/check/check.py",), ("verity/protocols/pous/lean/scripts/gen.py",)],
                         ids=["check's code", "a Python file of a Lake package"])
def test_a_pr_changing_a_lean_input_waits_while_a_batch_is_staged_and_is_built_once_that_batch_is_on_the_chain(files):
    """A PR with no Lean change whose files are Lean inputs (`Ready.lean_input`), built beside a staged batch, would change the
    Lean inputs of the chain's top that batch goes onto, so its records would be regenerated there again: no batch takes it
    while the batch is staged, a PR touching no Lean input is built beside it meanwhile, and it is built in rank order once
    the batch is on the chain (committed as staged, or merged again on the top as its records commit), as with none staged."""
    touching = pr(2, None, *files, lean_input=True)
    s = snap(waiting("c1", MAIN, 1), ready=[lean(1), touching], depth=3, records=True)
    assert decide(s) == []
    assert decide(replace(s, ready=(*s.ready, pr(3)))) == [Build((3,), MAIN, 3)]
    assert decide(replace(s, candidates=(cand("c1", MAIN, 1),), ready=(*s.ready, pr(3)))) == [Build((2, 3), "c1", 2)]
    fixed = (cand("c3", MAIN, 3), replace(cand("c1r", "c3", 1), repairs="c1"))
    assert decide(replace(s, candidates=fixed)) == [Build((2,), "c1r", 1)]
    assert decide(replace(s, candidates=(), ready=(touching,))) == [Build((2,), MAIN, 3)]
    assert decide(replace(s, ready=(lean(1), replace(touching, lean_input=False)))) == [Build((2,), MAIN, 3)]


def test_a_pr_set_aside_waits_behind_a_staged_batch_as_behind_any_pr_waiting():
    """A lone PR whose check kept erring goes behind the PRs waiting: with a staged batch, it isn't built until that batch is
    on main's chain, so the batch never goes onto a candidate the next erring check drops."""
    aside = Outcome((5,), heads(5), INFRA, (5,), MAIN)
    s = snap(waiting("c1", MAIN, 1), ready=[lean(1), pr(5)], history=[aside], records=True)
    assert decide(s) == []
    assert decide(replace(s, ready=(*s.ready, pr(2)))) == [Build((2,), MAIN, 2)]
    assert decide(replace(s, candidates=())) == [Build((1,), MAIN, 2, records=True)]
    assert decide(replace(s, candidates=(), ready=(pr(5),))) == [Build((5,), MAIN, 2)]


def test_a_staged_batch_whose_base_left_main_s_chain_stays_staged_and_a_candidate_on_it_is_dropped():
    """Its records were regenerated on its tree, whatever becomes of its base: `main` moving elsewhere or its base failing
    drops the chain and leaves it staged.  None is built on it, but should a log hold one, it is off main's chain."""
    s = snap(cand("c0", MAIN, 3, state=FAIL, failed=(task("t", "pkg3/"),)), waiting("c1", "c0", 1), ready=[lean(1), *prs(2, 3)],
             depth=3, records=True)
    assert decide(s) == [Eject(3, "h3", "t failed, and its changed files meet their inputs, no other member's", (task("t", "pkg3/"),)),
                         Drop("c0", "failed: t", Outcome((3,), heads(3), FAIL, (3,), MAIN)), Build((2,), MAIN, 3)]
    assert decide(replace(s, main="m1")) == [Drop("c0", "not on main's chain: its base m0 is neither main nor a chain member"),
                                             Build((2, 3), "m1", 3)]
    s = snap(waiting("c1", MAIN, 1), cand("c2", "c1", 2), ready=[lean(1), pr(2)], records=True)
    assert decide(s) == [Drop("c2", "not on main's chain: its base c1 is neither main nor a chain member"), Build((2,), MAIN, 2)]


def test_a_staged_batch_holding_a_member_of_the_chain_or_of_a_staged_batch_first_in_rank_is_dropped():
    s = snap(cand("c0", MAIN, 1), waiting("c1", "c0", 1, 2), ready=[lean(1), lean(2)], records=True, lean=2)
    why = "#1 is in a candidate on main's chain, on `main` or staged first too"
    assert decide(s) == [Drop("c1", why), Build((2,), "c0", 1, records=True, replaces="c1")]
    s = snap(waiting("c1", MAIN, 1, 2), waiting("c2", MAIN, 2), ready=[lean(1), lean(2)], records=True, lean=2)
    assert decide(s) == [Drop("c2", "#2 is in a candidate on main's chain, on `main` or staged first too")]


def test_a_records_run_is_launched_once_more_after_infra_and_its_pending_rings_infra_only_with_why():
    """No record of a start (a tick cut off after its build line) launches it; one that ended in infra is launched again,
    once; pending, nothing happens; pending with why (its passed run's records can't be read yet), infra is rung, in words a
    ring says once."""
    s = snap(waiting("c1", MAIN, 1, records=None), ready=[lean(1), lean(2)], records=True)
    assert decide(s) == [Regenerate("c1")]
    assert decide(replace(s, candidates=(waiting("c1", MAIN, 1, records=RecordsRun(INFRA, "it registered no run")),))) == [
        Regenerate("c1")]
    assert decide(replace(s, candidates=(waiting("c1", MAIN, 1),))) == []
    why = "r1 passed, but its Lean report (lean-audit) can't be read yet"
    assert decide(replace(s, candidates=(waiting("c1", MAIN, 1, records=RecordsRun(PENDING, why)),))) == [
        Ring("c1", f"c1's records run: {why}")]


def test_a_passed_records_run_puts_its_batch_on_the_chain_s_top_and_a_commit_held_past_pending_s_drops_it():
    """`Commit` onto the chain's top (as staged, on it as built or on its records commit, or merged again: the loop's to
    tell), each tick while it is held, `pending_s` from the first hold; then it is dropped, no outcome, and its members are
    built again."""
    s = snap(waiting("c1", MAIN, 1, 2, records=RecordsRun(PASS)), ready=[lean(1), lean(2)], records=True, lean=2, now=9000.0)
    assert decide(s) == [Commit("c1", MAIN, 2)]
    held = replace(s, candidates=(waiting("c1", MAIN, 1, 2, records=RecordsRun(PASS), held=1000.0),))
    assert decide(replace(held, now=1000.0 + 4 * 3600)) == [Commit("c1", MAIN, 2)]
    assert decide(replace(held, now=1001.0 + 4 * 3600)) == [
        Drop("c1", "its records commit is still held 4 h after it was first: its members are built again"),
        Build((1, 2), MAIN, 2, records=True)]


def test_a_passed_staged_batch_goes_onto_the_chain_s_top_once_it_has_room_and_its_tick_builds_on_it():
    """On the batches built beside it, at the priority its place gives it, ahead of any other batch; and its tick builds on
    it there (`Build.after`), as the next tick would: the PRs with no Lean change while the chain has room after it, and
    the next Lean batch, staged, whether it has or not.  With the chain full it waits until a candidate lands, and nothing
    is built; above a candidate that failed, it goes once that fail is decided."""
    passed = waiting("c1", MAIN, 1, records=RecordsRun(PASS))
    s = snap(passed, cand("c2", MAIN, 2), ready=[lean(1), *prs(2, 3)], depth=3, records=True)
    assert decide(s) == [Commit("c1", "c2", 2), Build((3,), "c2", 1, after="c1")]
    assert decide(replace(s, depth=2)) == [Commit("c1", "c2", 1)]
    more = replace(s, ready=(*s.ready, lean(4)))
    assert decide(more) == [Commit("c1", "c2", 2), Build((4,), "c2", 1, records=True, after="c1"),
                            Build((3,), "c2", 1, after="c1")]
    assert decide(replace(more, depth=2)) == [Commit("c1", "c2", 1), Build((4,), "c2", 0, records=True, after="c1")]
    assert decide(replace(s, depth=1)) == []
    assert decide(replace(s, depth=1, candidates=(passed, cand("c2", MAIN, 2, state=PASS)))) == [
        Advance("c2"), Drop("c2", "on main", Outcome((2,), heads(2), PASS)), Commit("c1", "c2", 1)]
    failed = (task("t", "pkg2/"),)
    acts = decide(replace(s, candidates=(passed, cand("c2", MAIN, 2, state=FAIL, failed=failed))))
    assert kinds(acts) == [Eject, Drop, Commit, Build] and acts[2:] == [Commit("c1", MAIN, 3), Build((3,), MAIN, 2, after="c1")]


def test_a_tick_whose_staged_batch_is_dropped_rather_than_committed_builds_on_the_chain_s_top_as_it_is():
    """No `Commit`, no `Build.after`: the batch whose member moved on, dropped as the top is no longer its base, leaves the
    tick's builds on that top."""
    passed = waiting("c1", MAIN, 1, records=RecordsRun(PASS))
    s = snap(passed, cand("c2", MAIN, 3), ready=[pr(1, "h1b", lean=True), pr(3), pr(4)], depth=3, records=True)
    acts = decide(s)
    assert kinds(acts) == [Drop, Build, Build] and acts[1:] == [Build((1,), "c2", 2, records=True), Build((4,), "c2", 2)]


def test_a_passed_staged_batch_whose_member_moved_on_goes_only_onto_its_own_base_as_built():
    """A member its owner withdrew, pushed or closed lands as built, which only its own base holds: on another top it isn't
    merged again, so the batch is dropped, no outcome, and built again at once without it, staged; its new head waits.  One
    left with no Lean member goes on the chain's top, and the tick's batch goes on it, if the chain has room."""
    passed = waiting("c1", MAIN, 1, 2, records=RecordsRun(PASS))
    s = snap(passed, ready=[pr(1, "h1b", lean=True), lean(2)], records=True, lean=2)
    assert decide(s) == [Commit("c1", MAIN, 2)]
    why = "#1 isn't marked at the head it was staged at, so it isn't merged again onto c2, main's chain's top now: its " \
          "other members are built again"
    assert decide(replace(s, candidates=(passed, cand("c2", MAIN, 3)), ready=(*s.ready, pr(3)))) == [
        Drop("c1", why), Build((2,), "c2", 1, records=True, replaces="c1")]
    plain = replace(s, candidates=(passed, cand("c2", MAIN, 3)), ready=(pr(1, "h1b", lean=True), pr(2), pr(3), pr(4)))
    assert decide(plain) == [Drop("c1", why), Build((2,), "c2", 1, replaces="c1"), Build((1,), "c1", 0, records=True)]
    assert decide(replace(plain, depth=3)) == [Drop("c1", why), Build((2,), "c2", 2, replaces="c1"),
                                               Build((1,), "c1", 1, records=True), Build((4,), "c1", 1)]
    assert decide(replace(plain, ready=(pr(1, "h1b"), *plain.ready[1:]), depth=3)) == [
        Drop("c1", why), Build((2,), "c2", 2, replaces="c1"), Build((1, 4), "c1", 1)]


def test_a_passed_staged_batch_whose_member_isnt_taken_at_the_head_it_was_staged_at_says_why_and_unmarked_only_for_its_mark():
    """A member out of `ready` at the very head it was staged at is dropped with the loop's cause there (`Snapshot.left`: a
    move it lacks, `queue.lacked_moves`); one whose mark was withdrawn or is missing there still isn't marked."""
    passed = waiting("c1", MAIN, 1, 2, records=RecordsRun(PASS))
    s = snap(passed, cand("c2", MAIN, 3), ready=[lean(2), pr(3)], records=True, lean=2)
    tail = ", so it isn't merged again onto c2, main's chain's top now: its other members are built again"
    lacks = "it lacks the move m1 `main` landed (tools/move/x_map.toml), whose retired names its own change names (`c.txt`)"
    for why, cause in ((lacks, f"#1 isn't taken at the head it was staged at ({lacks})"),
                       ("its ready mark was withdrawn by lane-1", "#1 isn't marked at the head it was staged at"),
                       ("it isn't marked ready", "#1 isn't marked at the head it was staged at")):
        assert decide(replace(s, left=((1, "h1", why),)))[0] == Drop("c1", cause + tail), why
    assert decide(replace(s, left=((1, "h1b", lacks),)))[0] == Drop("c1", "#1 isn't marked at the head it was staged at" + tail)


def test_a_records_run_that_failed_drops_its_candidate_ejects_its_lean_members_and_builds_the_rest():
    """Each Lean member is ejected with the run's reason, out of the next batch until it is marked again; the others are built
    again at once without them, with no Lean member left, so its check starts at once; the outcome suspects the Lean
    members."""
    r = RecordsRun(FAIL, "it failed: r1 failed rc=1")
    acts = decide(snap(waiting("c1", MAIN, 1, 2, 3, records=r), ready=[lean(1), pr(2), lean(3), pr(4)], records=True, lean=2))
    why = "its records run refused its records: it failed: r1 failed rc=1"
    assert acts == [Eject(1, "h1", why), Eject(3, "h3", why), Drop("c1", why, Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1, 3), MAIN)),
                    Build((2,), MAIN, 2, replaces="c1"), Build((4,), "c1", 1)]


def test_a_records_run_that_ended_in_infra_twice_refuses_and_rings_infra():
    r = RecordsRun(INFRA, "it never ran or was stopped (r1)", starts=2)
    acts = decide(snap(waiting("c1", MAIN, 1, 2, records=r), ready=[lean(1), pr(2)], records=True))
    refused = "its records run ended in infra twice: it never ran or was stopped (r1)"
    assert acts == [Ring("c1", "c1's records run ended in infra twice: it never ran or was stopped (r1)"), Eject(1, "h1", refused),
                    Drop("c1", refused, Outcome((1, 2), heads(1, 2), FAIL, (1,), MAIN)), Build((2,), MAIN, 2, replaces="c1")]


def test_a_refusal_ejects_no_member_its_owner_moved_on_from_and_suspects_it_whatever_its_change():
    """A member whose owner withdrew its mark or pushed a new head stays in its candidate in flight: the refusal suspects it,
    as its change as built may be the Lean one, and ejects only the Lean members still marked at the heads they were built
    at.  With the only Lean member withdrawn, nobody is ejected.  Neither is built again in its place: a new head goes into
    the next batch."""
    r = RecordsRun(FAIL, "it failed: r1 failed rc=1")
    why = "its records run refused its records: it failed: r1 failed rc=1"
    assert decide(snap(waiting("c1", MAIN, 1, 2, records=r), ready=[pr(2)], records=True)) == [
        Drop("c1", why, Outcome((1, 2), heads(1, 2), FAIL, (1,), MAIN)), Build((2,), MAIN, 2, replaces="c1")]
    assert decide(snap(waiting("c1", MAIN, 1, 2, 3, records=r), ready=[pr(1, "h1b"), pr(2), lean(3)], records=True)) == [
        Eject(3, "h3", why), Drop("c1", why, Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (1, 3), MAIN)),
        Build((2,), MAIN, 2, replaces="c1"), Build((1,), "c1", 1)]


def test_a_refused_candidate_above_another_is_rebuilt_without_its_lean_members_on_the_one_below():
    r = RecordsRun(FAIL, "it failed: r2 failed rc=1")
    s = snap(cand("c1", MAIN, 1), waiting("c2", "c1", 2, 3, records=r), ready=[pr(1), lean(2), pr(3)], depth=3, records=True)
    why = "its records run refused its records: it failed: r2 failed rc=1"
    assert decide(s) == [Eject(2, "h2", why), Drop("c2", why, Outcome((2, 3), heads(2, 3), FAIL, (2,), "c1")),
                         Build((3,), "c1", 2, replaces="c2")]


def test_a_candidate_on_a_refused_staged_batch_is_off_main_s_chain():
    """None is built on a staged batch, but should a log hold one, it is dropped as off main's chain before the refusal."""
    r = RecordsRun(FAIL, "it failed: r1 failed rc=1")
    s = snap(waiting("c1", MAIN, 1, records=r), cand("c2", "c1", 2), ready=[lean(1), pr(2)], records=True)
    why = "its records run refused its records: it failed: r1 failed rc=1"
    assert decide(s) == [Drop("c2", "not on main's chain: its base c1 is neither main nor a chain member"),
                         Eject(1, "h1", why), Drop("c1", why, Outcome((1,), heads(1), FAIL, (1,), MAIN)), Build((2,), MAIN, 2)]


def test_a_records_commit_is_checked_in_place_of_the_staged_batch_it_repairs_with_every_candidate_above_that_dropped():
    """The records commit wins whatever the shas' order and whatever its base, and the chain goes on from it (the loop
    pushes the records commit and leaves the staged batch for the next tick to drop here, so a tick cut off after the push
    still finds it and adopts the commit).  A batch staged again on another tree replaces it alike."""
    for name in ("a", "z"):
        fixed = replace(cand(name, MAIN, 1), repairs="c1")
        s = snap(waiting("c1", MAIN, 1, records=RecordsRun(PASS)), fixed, cand("c2", "c1", 2), ready=[lean(1), *prs(2, 3)],
                 records=True)
        assert decide(s) == [Drop("c1", f"its records were regenerated in {name}, which is checked in its place"),
                             Drop("c2", "its base was dropped"), Build((2, 3), name, 1)]
    s = snap(cand("c0", MAIN, 3), waiting("c1", MAIN, 1, records=RecordsRun(PASS)), replace(cand("c1r", "c0", 1), repairs="c1"),
             ready=[lean(1), *prs(2, 3)], depth=3, records=True)
    assert decide(s) == [Drop("c1", "its records were regenerated in c1r, which is checked in its place"), Build((2,), "c1r", 1)]
    again = replace(waiting("c1s", "c0", 1), repairs="c1")
    assert decide(replace(s, candidates=(s.candidates[0], s.candidates[1], again))) == [
        Drop("c1", "it is staged again in c1s, on c0, whose Lean inputs aren't those its records were regenerated on: its "
                   "records run first again"), Build((2,), "c0", 2)]


def test_a_records_commit_whose_records_are_stale_still_fails_as_any_candidate_with_no_second_records_run():
    """Its check found what its records run didn't settle: an ordinary fail, ejected as any is."""
    c = replace(cand("c1r", MAIN, 1, state=FAIL, failed=(stale(),)), repairs="c1")
    assert decide(snap(c, ready=[lean(1)], records=True)) == [
        Eject(1, "h1", "lean-lock:security failed, and its changed files meet their inputs, no other member's", (stale(),)),
        Drop("c1r", "failed: lean-lock:security", Outcome((1,), heads(1), FAIL, (1,), MAIN))]


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
    assert decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=ready))[-2:] == [
        Build((1,), MAIN, 2, replaces="c1"), Build((4,), "c1", 1)]
    assert decide(snap(cand("c1", MAIN, 1), ready=ready)) == [Build((2, 4), "c1", 1)]


@pytest.mark.parametrize("wide", [frozenset(), frozenset({3, 5, 99})])
@pytest.mark.parametrize(("lean", "first", "then"), [(1, (1,), (2,)), (3, (1, 2, 4), (6,))])
def test_a_lean_batch_opens_whether_or_not_prs_with_no_lean_change_wait_and_holds_at_most_lean_lean_prs(lean, first, then, wide):
    """Admitted with no pin: the Lean PRs first in rank, up to `lean`, open a batch of their own though PRs with no Lean
    change wait: staged beside theirs with a records run; with none, the tick's one build, theirs the next, and the next
    Lean batch on that.  With no wide PR (`Snapshot.wide` empty, as a missing or empty `--wide-file` reads, or naming only
    PRs the cap doesn't count or that aren't ready), the cap alone makes every Lean batch."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True), pr(5), pr(6, lean=True)]
    s = snap(ready=ready, lean=lean, batch=5, depth=3, wide=wide)
    assert decide(replace(s, records=True)) == [Build(first, MAIN, 3, records=True), Build((3, 5), MAIN, 3)]
    assert decide(s) == [Build(first, MAIN, 3)]
    on = (cand("q1", MAIN, *first),)
    assert decide(replace(s, candidates=on)) == [Build((3, 5), "q1", 2)]
    on += (cand("q2", "q1", 3, 5),)
    assert decide(replace(s, candidates=on)) == [Build(then, "q2", 1)]


def test_a_lean_batch_and_a_batch_of_prs_with_no_lean_change_share_no_member_and_a_suspect_s_cap_bounds_its_own_kind():
    """No batch mixes the two, whatever the cap, so no Lean failure drops a PR with no Lean change: a suspect of a failed
    outcome bounds the batch of its kind, and the other is built whole beside it."""
    ready = [pr(1, lean=True), pr(2), pr(3, lean=True), pr(4)]
    s = snap(ready=ready, lean=3, history=[Outcome((2, 4), heads(2, 4), FAIL, (2, 4), MAIN)], records=True)
    assert decide(s) == [Build((1, 3), MAIN, 2, records=True), Build((2,), MAIN, 2)]
    assert decide(replace(s, history=(Outcome((1, 3), heads(1, 3), FAIL, (1, 3), MAIN),))) == [
        Build((1,), MAIN, 2, records=True), Build((2, 4), MAIN, 2)]


def test_a_lean_pr_set_aside_opens_a_batch_only_in_its_turn_behind_every_pr_a_batch_may_take_and_those_set_aside_longer():
    """A lone Lean PR whose check kept erring (an `infra` outcome) goes behind the PRs waiting, whatever their change, as
    one with no Lean change does: after every PR a batch may take, alone, the longest set aside first, though it be a PR
    with no Lean change; a PR set aside later isn't built beside it."""
    aside = Outcome((1,), heads(1), INFRA, (1,), MAIN)
    ready = [pr(1, lean=True), pr(2), pr(3, lean=True)]
    s = snap(ready=ready, lean=3, history=[aside], records=True)
    assert decide(s) == [Build((3,), MAIN, 2, records=True), Build((2,), MAIN, 2)]
    assert decide(replace(s, ready=tuple(ready[:2]))) == [Build((2,), MAIN, 2)]
    assert decide(replace(s, ready=(ready[0],))) == [Build((1,), MAIN, 2, records=True)]
    later = Outcome((4,), heads(4), INFRA, (4,), MAIN)
    assert decide(replace(s, ready=(ready[0], pr(4)), history=(later, aside))) == [Build((4,), MAIN, 2)]
    assert decide(replace(s, ready=(ready[0], pr(4)), history=(aside, later))) == [Build((1,), MAIN, 2, records=True)]


def test_a_batch_taking_a_wide_pr_takes_no_other_lean_pr():
    """A: a wide PR (`Snapshot.wide`, one the `lean` cap counts) fills that cap.  A Lean batch taking it takes no other PR
    the cap counts, and one that took such a PR first leaves the wide one for a later batch."""
    ready = [pr(1, lean=True), pr(2), pr(3, lean=True), pr(4), pr(5, lean=True)]
    s = snap(ready=ready, lean=3, batch=5, depth=3)
    assert decide(replace(s, wide=frozenset({1}))) == [Build((1,), MAIN, 3)]
    assert decide(replace(s, wide=frozenset({3}))) == [Build((1, 5), MAIN, 3)]
    on = (cand("q1", MAIN, 1, 5), cand("q2", "q1", 2, 4))
    assert decide(replace(s, candidates=on, wide=frozenset({3}))) == [Build((3,), "q2", 1)]
    assert decide(replace(s, wide=frozenset({3}), records=True)) == [Build((1, 5), MAIN, 3, records=True), Build((2, 4), MAIN, 3)]


def test_a_batch_on_a_candidate_holding_a_wide_pr_takes_no_lean_pr_and_prs_with_no_lean_change_still_go():
    """B: while a candidate on main's chain holds a wide PR, no Lean batch opens on it, or on a candidate above it, staged or
    not, wide or not, and the PRs with no Lean change are built as ever; with only Lean PRs waiting, none is built, and
    `Wait` says why.  A candidate holding Lean PRs none of which is wide keeps nothing out: a wide PR goes on the chain it
    is in, filling its batch's cap."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True), pr(5), pr(6)]
    for records in (False, True):
        assert decide(snap(cand("q1", MAIN, 1), ready=ready, lean=3, depth=3, wide={1, 4}, records=records)) == [
            Build((3, 5, 6), "q1", 2)]
    assert decide(snap(cand("q1", MAIN, 1), cand("q2", "q1", 3), ready=ready, lean=3, depth=3, wide={1})) == [
        Build((5, 6), "q2", 1)]
    assert decide(snap(cand("q1", MAIN, 1, 3, 5, 6), ready=ready, lean=3, wide={1})) == [Wait(
        "nothing built: with a Lean change, #2, #4 wait while the chain holds #1's wide Lean change, and no PR with no Lean "
        "change is ready to take")]
    assert decide(snap(cand("q1", MAIN, 2), ready=ready, lean=3, depth=3, wide={1})) == [Build((3, 5, 6), "q1", 2)]
    assert decide(snap(cand("q1", MAIN, 2), cand("q2", "q1", 3, 5, 6), ready=ready, lean=3, depth=3, wide={1})) == [
        Build((1,), "q2", 1)]


def test_once_the_wide_pr_is_on_main_a_batch_takes_lean_prs_again():
    """B lifts once the wide PR is on `main`: in the tick that advances `main` to its candidate, the batch built on the chain's
    top takes Lean PRs again, as it does once `main` holds that candidate."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True)]
    on = Drop("q1", "on main", Outcome((1,), heads(1), PASS))
    chain = (cand("q1", MAIN, 1, state=PASS), cand("q2", "q1", 3))
    assert decide(snap(*chain, ready=ready, lean=3, depth=3, wide={1})) == [Advance("q1"), on, Build((2, 4), "q2", 2)]
    assert decide(snap(*chain, main="q1", ready=ready, lean=3, depth=3, wide={1})) == [on, Build((2, 4), "q2", 2)]


def test_a_lean_pr_b_keeps_out_still_waits_behind_the_wide_pr_so_a_wide_lone_member_that_keeps_erring_is_set_aside():
    """B keeps Lean PRs out of a batch on the wide PR's candidate, not out of the queue: they still wait behind it, so a wide
    lone member whose check keeps erring goes behind them, as behind any PR waiting, and they are built on `main`."""
    err = (Failed("node", "key:node", frozenset({"tools/runner.py"})),)
    acts = decide(snap(cand("q1", MAIN, 1, state=INFRA, infra=err, tries=2), ready=[pr(1, lean=True), pr(2, lean=True)],
                       lean=3, wide={1}))
    assert kinds(acts) == [Ring, Ring, Drop, Build] and acts[2].outcome.state == INFRA and acts[3] == Build((2,), MAIN, 2)


def test_at_lean_0_a_lean_pr_opens_no_batch_whatever_its_rank_unless_pinned_and_a_pinned_batch_takes_only_pinned_lean_prs():
    """`lean` 0 admits no Lean PR without a pin: one ranked first (rank is drive's priority file, then mark time, so a stale
    or empty file ranks a Lean PR marked long ago first) goes in no batch, the PRs with no Lean change are built, and then
    nothing, with no `Wait`.  Pinned (`Snapshot.pinned`), it opens a batch of pinned Lean PRs only, staged beside the PRs
    with no Lean change with a records run, ahead of them with none.  Under a cap above 0 it opens one with no pin."""
    ready = [pr(10, rank=0), pr(20, rank=1, lean=True), pr(30, rank=2)]
    on = snap(cand("q1", MAIN, 10), ready=ready, lean=0)
    assert decide(on) == [Build((30,), "q1", 1)]
    assert decide(replace(on, pinned=frozenset({20}))) == decide(replace(on, lean=3)) == [Build((20,), "q1", 1)]
    first = [pr(20, rank=0, lean=True), pr(30, rank=1), pr(40, rank=2, lean=True)]
    for records in (False, True):
        s = snap(ready=first, lean=0, records=records)
        assert decide(s) == [Build((30,), MAIN, 2)]
        assert decide(replace(s, candidates=(cand("q1", MAIN, 30),))) == []
        assert decide(replace(s, pinned=frozenset({20}))) == (
            [Build((20,), MAIN, 2, records=True), Build((30,), MAIN, 2)] if records else [Build((20,), MAIN, 2)])
    assert decide(snap(ready=first, lean=3)) == [Build((20, 40), MAIN, 2)]


def test_a_pinned_lean_batch_takes_only_pinned_lean_prs_in_rank_order_up_to_batch_under_no_lean_cap():
    """A batch of pinned Lean PRs takes no unpinned Lean PR and no PR with no Lean change, pinned or not, so its Lean failure
    drops none of them; the `lean` cap doesn't count its members, at 0 either, and `batch` bounds it."""
    ready = [pr(1, lean=True), pr(2), pr(3, lean=True), pr(4, lean=True), pr(5, lean=True), pr(6, lean=True)]
    for lean in (0, 1, 3):
        assert decide(snap(ready=ready, lean=lean, batch=6, pinned={1, 2, 4, 5, 6})) == [Build((1, 4, 5, 6), MAIN, 2)]
    assert decide(snap(ready=ready, lean=1, batch=3, pinned={1, 2, 4, 5, 6})) == [Build((1, 4, 5), MAIN, 2)]


def test_with_no_records_run_a_lean_batch_goes_on_main_or_on_a_top_candidate_holding_no_lean_change_so_at_most_every_second_build():
    """With no records run a Lean batch takes a place on main's chain and the tick's build.  On a top candidate holding a Lean
    change, pinned or not, the tick's batch is of PRs with no Lean change; on one holding none, or on `main` (the chain
    empty, or advanced to this tick), it is the Lean one, here the pinned PRs'.  With a records run it is staged on either,
    beside the PRs with no Lean change."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True), pr(5)]
    assert decide(snap(cand("q1", MAIN, 4), ready=ready, lean=3, depth=3, pinned={1, 2})) == [Build((3, 5), "q1", 2)]
    assert decide(snap(cand("q1", MAIN, 4), ready=ready, lean=3, depth=3, pinned={1, 2}, records=True)) == [
        Build((1, 2), "q1", 2, records=True), Build((3, 5), "q1", 2)]
    on = (cand("q1", MAIN, 4), cand("q2", "q1", 3, 5))
    assert decide(snap(*on, ready=ready, lean=3, depth=3, pinned={1, 2})) == [Build((1, 2), "q2", 1)]
    assert decide(snap(ready=ready, lean=3, pinned={1, 2})) == [Build((1, 2), MAIN, 2)]
    assert decide(snap(cand("q1", MAIN, 4, state=PASS), ready=ready, lean=3, pinned={1, 2})) == [
        Advance("q1"), Drop("q1", ON_MAIN, Outcome((4,), heads(4), PASS)), Build((1, 2), "q1", 2)]


def test_with_no_records_run_a_lean_pr_waiting_for_a_top_with_no_lean_change_and_nothing_else_to_take_builds_nothing_and_wait_says_why():
    """While a pinned Lean PR waits so, no unpinned one goes, under any cap: the tick builds nothing, and `Wait` says why,
    naming the batch that waits, the pinned PRs' or, with none pinned, the first in rank under the cap.  With a records run
    nothing waits for a top: the batch is staged on it."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(6, lean=True)]
    s = snap(cand("q1", MAIN, 1), ready=ready, lean=3)
    assert decide(replace(s, pinned=frozenset({1, 2}))) == [
        Wait("nothing built: with a Lean change, #2 waits while the newest candidate, q1, holds #1's Lean change, and no PR "
             "with no Lean change is ready to take")]
    assert decide(s) == [
        Wait("nothing built: with a Lean change, #2, #6 wait while the newest candidate, q1, holds #1's Lean change, and no "
             "PR with no Lean change is ready to take")]
    assert decide(replace(s, records=True)) == [Build((2, 6), "q1", 1, records=True)]


def test_a_pinned_wide_pr_goes_alone_and_a_chain_holding_a_wide_pr_opens_no_pinned_batch():
    """A: a pinned wide PR fills its batch alone, and one ranked behind a pinned Lean PR waits for a later batch.  B: on a
    chain holding a wide PR, even on a top holding no Lean change, the tick's batch is of PRs with no Lean change, and
    with none to take, nothing is built."""
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4), pr(9, lean=True)]
    assert decide(snap(ready=ready, lean=3, pinned={1, 2}, wide={1})) == [Build((1,), MAIN, 2)]
    assert decide(snap(ready=ready, lean=3, pinned={1, 2}, wide={2})) == [Build((1,), MAIN, 2)]
    on = (cand("q1", MAIN, 9), cand("q2", "q1", 4))
    assert decide(snap(*on, ready=ready, lean=3, depth=3, pinned={1, 2}, wide={9})) == [Build((3,), "q2", 1)]
    assert decide(snap(*on, ready=[r for r in ready if r.number != 3], lean=3, depth=3, pinned={1, 2}, wide={9})) == [
        Wait("nothing built: with a Lean change, #1, #2 wait while the chain holds #9's wide Lean change, and no PR with no "
             "Lean change is ready to take")]


def test_a_pinned_lean_pr_set_aside_or_on_main_at_its_head_opens_no_pinned_batch_and_a_suspect_keeps_its_cap():
    """Only a pinned Lean PR that a batch may take makes a Lean batch the pinned PRs' alone, as `_batch` reads PRs with no
    Lean change: one set aside (`infra`) or on `main` (`pass`) at its head doesn't, so at `lean` 0 none opens ahead of the
    PRs with no Lean change, and under a cap the Lean PR with no pin goes; at a new head it does.  Set aside, it is built
    again in its turn, at `lean` 0 too.  A suspect's cap bounds a pinned batch it heads."""
    ready = [pr(1, lean=True), pr(2), pr(3, lean=True)]
    for state in (INFRA, PASS):
        held = Outcome((1,), heads(1), state, (1,) if state == INFRA else (), MAIN)
        s = snap(ready=ready, lean=0, history=[held], pinned={1})
        assert decide(s) == [Build((2,), MAIN, 2)]
        assert decide(replace(s, lean=3)) == [Build((3,), MAIN, 2)]
        assert decide(replace(s, ready=(pr(1, "h1b", lean=True), *ready[1:]))) == [Build((1,), MAIN, 2)]
        assert decide(replace(s, candidates=(cand("q1", MAIN, 2),))) == ([Build((1,), "q1", 1)] if state == INFRA else [])
    fail = Outcome((1, 3), heads(1, 3), FAIL, (1, 3), MAIN)
    assert decide(snap(ready=ready, lean=3, history=[fail], pinned={1, 3})) == [Build((1,), MAIN, 2)]
    assert decide(snap(ready=ready, lean=3, pinned={1, 3})) == [Build((1, 3), MAIN, 2)]


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


#: a tree's member that stands for the records a records commit regenerated (`Play`)
RECORDS = 0


@dataclass
class Play:
    """A small world for the red team's cases: `decide`'s actions applied as the loop does, each pending candidate's verdict
    from ``oracle(tree, candidate)`` once its check started, each pending records run's from ``records(tree, candidate)``, a
    passed run's staged batch put on the chain's top: checked as built there, or, when ``regenerates(tree, candidate)``, its
    records commit (its tree with `RECORDS`) checked in its place, a build after it (`Build.after`) on what it put there,
    and a clock that moves ``step_s`` a tick.  Its trees hold no Lean input but the members', so records go onto any top."""
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
    records: Callable[[frozenset[int], Candidate], RecordsRun] | None = None
    regenerates: Callable[[frozenset[int], Candidate], bool] | None = None
    #: the ticks that built a batch of PRs with no Lean change while a batch stayed staged, its records run first, or
    #: beside the one they staged
    beside_staged: list[int] = field(default_factory=list)

    def tick(self) -> list:
        now = len(self.acts) * self.step_s
        gone = {*self.landed, *(e.number for e in self.ejected)}
        acts = decide(Snapshot(self.main, tuple(r for r in self.ready if r.number not in gone), tuple(self.cands.values()),
                               tuple(self.history), now=now, **self.kw))
        self.acts.append(acts)
        gone = {a.sha for a in acts if isinstance(a, (Drop, Commit))}
        staged = any(c.waits and c.sha not in gone for c in self.cands.values()) or any(
            isinstance(a, Build) and a.records for a in acts)
        if staged and any(isinstance(a, Build) and not a.records for a in acts):
            self.beside_staged.append(len(self.acts))
        assert not any(isinstance(a, (Build, Commit)) and a.base in self.cands and self.cands[a.base].waits for a in acts), (
            f"built on a staged batch: {acts}")
        put: dict[str, str] = {}
        rebuilds: dict[str, str] = {}
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
                base = put[a.after] if a.after is not None else rebuilds.get(a.base, a.base)
                if a.replaces is not None:
                    rebuilds[a.replaces] = sha
                self.trees[sha] = self.trees[base] | set(a.members)
                c = Candidate(sha, base, a.members, tuple(f"h{m}" for m in a.members), enqueued=now)
                self.cands[sha] = replace(c, tries=0, enqueued=None, records=RecordsRun(), waits=True) if a.records else c
            elif isinstance(a, Regenerate):
                c = self.cands[a.sha]
                self.cands[a.sha] = replace(c, records=RecordsRun(starts=c.records.starts + 1 if c.records else 1))
            elif isinstance(a, Commit):
                c = self.cands[a.sha]
                regen = self.regenerates is not None and self.regenerates(self.trees[a.sha], c)
                if a.base == c.base and not regen:
                    self.cands[a.sha], put[a.sha] = replace(c, waits=False, tries=1, enqueued=now), a.sha
                    continue
                sha = put[a.sha] = f"{a.sha}r"
                self.trees[sha] = self.trees[a.base] | set(c.members) | ({RECORDS} if regen else set())
                self.cands[sha] = Candidate(sha, a.base, c.members, c.heads, enqueued=now, repairs=a.sha)
        for sha, c in self.cands.items():
            if c.verdict.state == PENDING and not c.waits:
                self.cands[sha] = replace(c, verdict=self.oracle(self.trees[sha], c))
            if c.records is not None and c.records.state == PENDING and self.records is not None:
                got = self.records(self.trees[sha], c)
                self.cands[sha] = replace(self.cands[sha], records=replace(got, starts=c.records.starts))
        return acts

    def run(self, ticks: int = 60) -> Play:
        for _ in range(ticks):
            self.tick()
        return self

    def count(self, kind: type) -> int:
        return sum(isinstance(a, kind) for acts in self.acts for a in acts)


def test_two_lean_prs_whose_records_go_stale_only_together_land_together_with_their_records_commit():
    """Each carries its own records, and merged they leave a lock's records stale, as two Lean PRs meeting do: their
    candidate's records are regenerated on its tree before its check and committed on top of it, and that records commit,
    checked in its place, lands them on #3, which has no Lean change and is built beside them by the tick that stages them,
    in a batch of its own: all three land, nobody ejected or bisected, two batches built."""
    def oracle(tree, c):
        return Verdict(FAIL, (stale(),), ended=True) if {1, 2} <= tree and RECORDS not in tree else Verdict(PASS)

    p = Play([lean(1), lean(2), pr(3)], oracle, {"records": True, "lean": 2}, records=lambda tree, c: RecordsRun(PASS),
             regenerates=lambda tree, c: {1, 2} <= tree).run(12)
    assert kinds(p.acts[0]) == [Build, Build] and p.beside_staged == [1] and p.trees["c2"] == {3}
    assert sorted(p.landed) == [1, 2, 3] and not p.ejected and p.main == "c1r" and p.trees["c1r"] == {RECORDS, 1, 2, 3}
    assert p.built == 2 and p.count(Commit) == 1 and p.count(Regenerate) == 0 and not any(o.state == FAIL for o in p.history)


def test_a_lean_pr_whose_records_run_regenerates_nothing_lands_as_built_and_a_slow_records_run_holds_no_batch():
    """No records commit: its check starts on its batch as built, merged again onto the chain's top when that moved.  A slow
    records run holds no batch: those with no Lean change are built and land beside it, and none is built on it; the tick
    that puts it on the chain builds on it there."""
    slow = Counter()

    def records(tree, c):
        slow[c.sha] += 1
        return RecordsRun(PASS) if slow[c.sha] > 3 else RecordsRun()

    p = Play([lean(1)], lambda tree, c: Verdict(PASS), {"records": True, "batch": 1, "depth": 3}, records=records,
             regenerates=lambda tree, c: False)
    assert kinds(p.tick()) == [Build] and p.cands["c1"].waits
    p.ready += prs(2, 3, 4)
    p.run(20)
    assert sorted(p.landed) == [1, 2, 3, 4] and not p.ejected and p.count(Commit) == 1 and p.built == 4
    assert p.landed[:3] == [2, 3, 4] and p.beside_staged == [2, 3, 4] and p.main == "c1r" and RECORDS not in p.trees["c1r"]


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


#: the tree a check of the world ran in, which its logs' frames name
ROOT = "/workspace/research/scratch/r-world/tree"


@dataclass(frozen=True)
class Records:
    """What a check's records say of one failure (`World.records`), as `vqueue_slim.failing` reads them: its step; of a suite,
    the report its row is in (``at``: suites.json once the check ended, suites.failed.json while it runs, `-lean` of
    lean-suites), the row, its log, its guard's report and its children's reads, and what it declares; of another step, its
    log and what it declares (None: the whole tree)."""
    step: str
    log: str
    at: str | None = None
    row: dict | None = None
    report: dict | None = None
    trace: str = ""
    declared: frozenset[str] | None = None
    dirs: frozenset[str] = frozenset()


def read_records(r: Records, rest: str) -> Read:
    """``r`` read as `vqueue_slim.failing` reads a failed step's records: a suite's row by `read_suite`, ``rest`` alone when no
    row says it failed (`joined` of none), another step's log by `read_step`."""
    if r.at is None:
        return read_step(r.step, {"step": r.step, "rc": 1}, r.log, rest, ROOT, r.declared)
    return joined([read_suite(r.row["name"], r.row, r.log, rest, r.declared, r.dirs, ROOT, r.report, r.trace)] if r.row else [],
                  rest)


class World:
    """The loop's world, scripted by a seed: `main`, the ready marks, the live candidates with graph2's verdicts, and the outcomes
    the loop recorded.  A bad PR fails every candidate whose tree holds it, on a task whose inputs meet its files (and
    `shared/`'s), or for an interaction nobody's, at a key only such trees give the task, or, in a red world, on a red task's
    test of its own.  An erring PR makes a task end in infra errors on every node (`tree`) in every such tree; it is ejected
    with `tree_ejects` (every other seed), and without it waits alone, its owner rung, while an outage's errors are marked
    `tree` too, as graph2 marks them.  In a
    red world `main` fails one or two red tasks in every tree without their fix, a good PR, at a key each PR touching their inputs
    changes (some good PRs touch them, the same tests failing); a red task whose failing tests graph2 doesn't report is
    inherited at its key alone, and no good PR but its fix touches it.  PRs arrive, push, conflict, are unmarked by their
    owners for a while (a candidate in flight lands them as built) and are held by a reviewer for a while (`Snapshot.pulled`);
    graph2 answers late, sends the cached inherited failures before the rest, flakes on a first run, errs on a first
    enqueue, errs or fails to list at every enqueue until infra, rung, reruns the check, or wedges until retried; someone
    else moves `main`, the loop stops a tick early, and a merged PR stays marked for a while; `step` checks every invariant
    on the way.  Each failure a verdict holds is read from the records a check writes of it (`records`), as ``fetch``
    hands them to the loop's reader (`read`).

    With ``records``, a Lean batch's records run comes first (`Snapshot.records`): it takes a few ticks, ends in infra now and
    then on a first start, and fails only a batch of bad PRs alone.  A tree's Lean inputs are its Lean members and the PRs
    with no Lean change that touch a Lean input (`touch`); its lock records are a function of them (`lock`, which only some
    Lean PRs change), and are what the tree below it holds until a records run regenerates them.  Commit is applied as the
    loop applies it: on the staged batch's own base, its records commit or the batch as built; on another, its members merged
    again there, its records carried onto the merge only when the two trees' Lean inputs are the same, else the merge staged
    again, never for a Lean input a PR built while a batch was staged changed (`Ready.lean_input`); and a build after it
    (`Build.after`) goes on what it put on the chain, or nowhere when it staged the batch again.  No check reads stale
    records, so only the invariants catch them.  A staged batch keeps the PRs changing a Lean input out of the builds
    beside it, each behind one staged batch at most (`waits`).  ``lean`` is the `lean` cap, in place of the seed's (1 or
    3): at 0 no batch takes a Lean PR the pins file doesn't name, and the world is done without them."""

    def __init__(self, seed: int, red: bool | None = None, bad: bool = True,
                 policy: Callable[[Snapshot], list] = decide, wide: bool = False, pinned: bool = False,
                 fetch: Callable[[Records], Records] | None = None, records: bool = False, lean: int | None = None):
        self.seed, self.policy, self.tree_ejects = seed, policy, seed % 2 == 0
        self.fetch = fetch or (lambda r: r)
        rng = self.rng = random.Random(seed)
        numbers = sorted(rng.sample(range(10, 90), rng.randint(3, 12)))
        self.batch, self.depth, self.rings = rng.choice((1, 2, 3, 4, 6)), rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        self.lean = rng.choice((1, 3)) if lean is None else (rng.choice((1, 3)), lean)[1]
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
        self.universe = frozenset({p for fs in files.values() for p in fs} | {
            "shared/x.py", "elsewhere/untouched.py", "elsewhere/flaky.py", "docs/index.md", "pyproject.toml"})
        # the wide file: about half the Lean PRs, and now and then a PR with no Lean change, which isn't wide
        self.wide = frozenset(m for m in numbers if rng.random() < (0.5 if self.marks[m].lean else 0.05)) if wide else frozenset()
        # the pins file: about half the Lean PRs, and now and then a PR with no Lean change, which goes as any other
        self.pinned = frozenset(m for m in numbers if rng.random() < (0.5 if self.marks[m].lean else 0.1)) if pinned else frozenset()
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
        self.holds = {rng.randint(1, 30): (rng.choice(numbers), rng.randint(1, 6)) for _ in range(rng.randint(0, 2))}
        self.main, self.tick, self.made = MAIN, 0, 0
        self.main_since = 0                           # the tick `main` last moved
        self.rebuilds: dict[str, str] = {}            # this tick's: each candidate a build rebuilt -> what that build made
        self.cands: dict[str, Candidate] = {}
        self.tree: dict[str, frozenset[int]] = {MAIN: frozenset()}
        self.due: dict[str, int | None] = {}
        self.early: set[str] = set()
        self.outage: dict[str, str] = {}              # sha -> the state its check ends in until infra reruns it
        self.history: list[Outcome] = []
        self.merged: dict[int, int] = {}              # number -> the tick it reached main
        self.ejected: set[int] = set()
        self.unmarked: dict[int, int] = {}            # number -> the tick it is marked again
        self.held: dict[int, int] = {}                # number -> the tick a reviewer's hold on it is lifted
        self.landed: set[tuple[int, str]] = set()     # (number, head) that reached main
        self.rung: set[tuple[str, str]] = set()
        self.owned: set[tuple[int, str]] = set()      # (number, head) whose owner was rung
        self.again: set[tuple[str, str]] = set()      # rung by a tick that stopped before its outcome was recorded
        self.seen: Counter[str] = Counter()
        self.records_first = records
        self.touch = frozenset(m for m in numbers if not self.marks[m].lean and rng.random() < 0.2) if records else frozenset()
        self.marks = {m: replace(r, lean_input=r.lean or m in self.touch) for m, r in self.marks.items()}
        self.beside_staged: set[int] = set()     # the PRs built while a batch was staged, not as a rebuild (`Build.replaces`)
        self.put: dict[str, str] = {}            # this tick's Commits: the staged batch -> what it put on main's chain
        self.opened: dict[str, str] = {}         # a staged batch -> the batch it was first staged as, itself or one it repairs
        self.behind: dict[int, set[str]] = {}    # a PR changing a Lean input, waiting -> the staged batches it waited behind
        self.lockers = frozenset(m for m in numbers if self.marks[m].lean and rng.random() < 0.7) if records else frozenset()
        self.locks: dict[str, frozenset[int]] = {MAIN: frozenset()}      # sha -> the lock records its tree holds
        self.regenerated_on: dict[str, frozenset[int]] = {}             # sha -> the Lean inputs its records were regenerated on
        self.records_due: dict[str, int] = {}

    @staticmethod
    def same(a: frozenset[int], b: frozenset[int]) -> bool:
        """Whether records regenerated on Lean inputs ``b`` are carried onto a tree with ``a`` (`vqueue_records.same_inputs`)."""
        return a == b

    def lean_in(self, sha: str) -> frozenset[int]:
        """The Lean inputs of ``sha``'s tree, as the PRs that brought them."""
        return frozenset(m for m in self.tree[sha] if self.marks[m].lean or m in self.touch)

    def lock(self, inputs: frozenset[int]) -> frozenset[int]:
        """The lock records a records run regenerates on a tree with Lean inputs ``inputs``."""
        return inputs & self.lockers

    def ready(self) -> tuple[Ready, ...]:
        return tuple(r for m, r in self.marks.items() if self.arrive[m] <= self.tick and m not in self.ejected
                     and m not in self.unmarked and m not in self.held
                     and (m not in self.merged or self.lag is None or self.tick < self.merged[m] + self.lag))

    def snapshot(self) -> Snapshot:
        cands = [c if c.base == self.main else replace(c, verdict=replace(c.verdict, failed=tuple(
            replace(f, main_tests=self.word(f)) for f in c.verdict.failed))) for c in self.cands.values()]
        self.rng.shuffle(cands)
        ready = self.ready()
        at = {(r.number, r.head) for r in ready}
        flying = {(m, h): replace(self.marks[m], head=h) for c in cands for m, h in zip(c.members, c.heads) if (m, h) not in at}
        return Snapshot(self.main, ready, tuple(cands), tuple(self.history), self.batch, self.depth, self.rings, self.lean,
                        self.tick * TICK_S, PENDING_S, self.tree_ejects, wide=self.wide, pinned=self.pinned,
                        pulled=frozenset(self.held), flying=tuple(flying.values()), records=self.records_first)

    def done(self) -> bool:
        out = {m for m, r in self.marks.items() if r.lean and m not in self.pinned} if self.lean == 0 else set()
        good, bad, erring = self.good - out, set(self.bad) - out, set(self.erring) - out
        if self.tree_ejects:
            return set(self.merged) == good and self.ejected == bad | erring and not self.cands
        return (set(self.merged) == good and self.ejected == bad
                and all(set(c.members) <= erring for c in self.cands.values())
                and all((m, self.marks[m].head) in self.owned for m in erring))

    def key(self, name: str, m: int, sha: str) -> str:
        return f"{name}@{'with' if m in self.tree[sha] else 'without'} #{m}"

    def word(self, f: Failed) -> frozenset[str] | None:
        """`main`'s word on ``f``'s task, as the run beside its check gives it to a candidate above another: its failing tests,
        none where it passes it; None in the tick `main` moved and the next, before that run has said, and of a red task
        whose failing tests graph2 doesn't report."""
        if self.tick <= self.main_since + 1:
            return None
        if f.task.startswith("suite:red"):
            i = int(f.task.removeprefix("suite:red"))
            return self.red_tests(i, self.main) if self.detail[i] else None
        return frozenset()

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
            self.read(cached, ended=False)
            return Verdict(FAIL, cached)
        if failed:
            self.read(failed, ended=True)
            return Verdict(FAIL, failed, infra, ended=True)
        return Verdict(INFRA, (), infra) if infra else Verdict(PASS)

    def records(self, f: Failed, ended: bool) -> Records:
        """What a check writes of ``f``, by coins of its own (so the world's are the same with and without its records): of a
        suite (``suite:M``, lean-suites' when M's change is Lean; a red task's; the flake's), its row naming its failing tests
        (now and then none, failed beyond them), its log's frames through the check's tree and out of it, what it declares
        (its directory, `shared/`, `docs/` and the workspace's file) and what its run traced: each file it read, but now and
        then the one a program no tracer sees into read, its module `untraced`, and now and then nothing traced; of an
        interaction, a step's log naming nothing."""
        rng = random.Random(f"{self.seed} {f.task} {f.key}")
        kind, _, name = f.task.partition(":")
        if kind == "interaction":
            return Records("lean-agreement", "lean-agreement: the verifiers disagree on 3 of 40 vectors\n")
        suite = f"pkg{name}" if name.isdigit() else "elsewhere" if kind == "flaky" else name
        step = "lean-suites" if name.isdigit() and self.marks[int(name)].lean else "pytest"
        tests = sorted(f"tests/test_{suite}.py::{t.rpartition('::')[2]}" for t in f.tests) or (
            [f"tests/test_{suite}.py::test_it"] if rng.random() < 0.85 else [])
        row = {"name": suite, "suite": suite, "status": "failed", "step": step, "failed_tests": tests,
               "beyond_tests": not tests, "nondeterministic": tests if kind == "flaky" else []}
        read = sorted(p for p in self.universe if meets((p,), f.inputs))
        log = ("=" * 20 + " FAILURES " + "=" * 20 + "\n" + "".join(
            f"tests/test_{suite}.py:4: in {t.rpartition('::')[2]}\n" for t in tests)
            + "".join(f"{ROOT}/{p}:3: in f\n" for p in read) + "/usr/lib/python3.12/runpy.py:9: in g\nE   AssertionError\n"
            + "=" * 20 + " short test summary info " + "=" * 20 + "\n" + "".join(f"FAILED {t} - AssertionError\n" for t in tests))
        blind = bool(read) and bool(tests) and rng.random() < 0.2
        report = None if rng.random() < 0.1 else {
            "traced": [f"F\t{p}" for p in (read[:-1] if blind else read)],
            "untraced": {f"{suite}/{tests[0].partition('::')[0]}": ["git"]} if blind else {}}
        at = f"suites{'-lean' if step == 'lean-suites' else ''}{'' if ended else '.failed'}.json"
        return Records(step, log, at, row, report, "", frozenset({suite, "shared", "docs", "pyproject.toml"}),
                       frozenset({suite, "shared", "docs"}))

    def read(self, failed: tuple[Failed, ...], ended: bool) -> None:
        """Each of ``failed`` read from its records (`records`) as the loop's reader reads them, through ``fetch``: one whose
        report names its failing tests, failed for nothing more, is read as those tests and never as unnamed, and what it
        read is among what any one's records bound its reads to."""
        for f in failed:
            r, rest = self.records(f, ended), unnamed(f.key)
            got = read_records(self.fetch(r), rest)
            where = f"seed {self.seed} tick {self.tick}: {f.task}"
            if r.row and r.row["failed_tests"] and not r.row["beyond_tests"]:
                named = {f"{r.row['name']}::{t}" for t in r.row["failed_tests"]}
                assert named <= got.tests and not {rest, f"{r.row['name']}::{rest}"} & got.tests, (
                    f"{where}: its report names {sorted(named)}, read as {sorted(got.tests)}")
                self.seen["a failure read as the tests its report names"] += 1
            if got.reads is None:
                self.seen["a failure read with its reads unbounded"] += 1
                continue
            missed = sorted(p for p in self.universe if meets((p,), f.inputs) and not meets((p,), got.reads))
            assert not missed, f"{where}: it read {missed}, which its reads {sorted(got.reads)} miss"
            self.seen["a suite's reads narrowed by what its run traced" if got.reads != r.declared
                      else "a suite's reads its declared inputs"] += 1
            self.seen["an untraced program's module read as its declared inputs"] += bool(r.report and r.report["untraced"])

    def events(self) -> str | None:
        """What happens before the tick's snapshot; the new `main` when someone else moved it."""
        moved = None
        if self.tick in self.elsewhere:
            moved = f"x{self.tick}"
            self.tree[moved], self.locks[moved] = self.tree[self.main], self.locks[self.main]
            self.main, self.main_since = moved, self.tick
            self.seen["main moved elsewhere"] += 1
        for m in self.pushes.get(self.tick, ()):
            if m not in self.merged and m not in self.ejected:
                self.marks[m] = replace(self.marks[m], head=self.marks[m].head + "+", conflict=False)
        if self.tick in self.unmarks:
            m, back = self.unmarks[self.tick]
            if m not in self.merged and m not in self.ejected:
                self.unmarked[m] = self.tick + back
        self.unmarked = {m: t for m, t in self.unmarked.items() if t > self.tick}
        if self.tick in self.holds:
            m, back = self.holds[self.tick]
            if m not in self.merged and m not in self.ejected:
                self.held[m] = self.tick + back
        self.held = {m: t for m, t in self.held.items() if t > self.tick}
        for sha, c in self.cands.items():
            v = c.verdict
            if (v.state == PENDING or v.state == FAIL and not v.ended) and self.due[sha] is not None and self.due[sha] <= self.tick:
                self.cands[sha] = replace(c, verdict=self.verdict(c))
            r = c.records
            if c.waits and r is not None and r.state == PENDING and self.records_due[sha] <= self.tick:
                state = INFRA if r.starts == 1 and self.rng.random() < 0.1 else FAIL if all(
                    m in self.bad for m in c.members) and self.rng.random() < 0.5 else PASS
                self.cands[sha] = replace(c, records=replace(r, state=state, why="" if state == PASS else f"it {state}ed"))
                self.seen[f"a records run ended in {state}"] += 1
        return moved

    def step(self) -> None:
        self.tick += 1
        self.put = {}
        moved = self.events()
        s = self.snapshot()
        acts = self.policy(s)
        self.check(s, acts, moved)
        n = len(acts)
        if acts and self.crashes and self.rng.random() < 0.15:
            self.crashes -= 1
            n = self.rng.randrange(len(acts))
            self.seen["the loop stopped early"] += 1
        self.rebuilds = {}
        for a in acts[:n]:
            self.apply(a)
        if n == len(acts):
            self.restacked(s, acts)
        recorded = {a.sha for a in acts[:n] if isinstance(a, Drop) and a.outcome is not None}
        self.again |= {(f.task, f.key) for a in acts[:n] if isinstance(a, Red) and a.sha not in recorded for f in a.failed}
        repaired = {c.repairs for c in self.cands.values()}
        live = [m for c in self.cands.values() if c.sha not in repaired for m in c.members]
        assert len(live) == len(set(live)), f"seed {self.seed} tick {self.tick}: a PR in two live candidates: {self.cands}"
        after = self.snapshot()
        self.waits(after)
        again = self.policy(after)
        dropped = {a.sha for a in acts[:n] if isinstance(a, Drop)}
        repeated = [a for a in set(again) & set(acts[:n]) if not isinstance(a, (Ring, Wait))
                    and not (isinstance(a, Red) and a.sha not in dropped)]
        assert not repeated, f"seed {self.seed} tick {self.tick}: a restart repeats {repeated}"

    def restacked(self, s: Snapshot, acts: list) -> None:
        """Once a tick's actions are all applied, every member of a candidate it dropped for a red, a member set aside or a
        leave, that is neither a suspect of the tick's outcomes nor itself left (not ready at the head it was built at: a
        reviewer's hold, an unmark, a push), is in a live candidate: a red takes out only its suspects, and a leaving member
        takes no other PR out of the chain.  A drop on `main` lands its members; one off main's chain, a second candidate on
        a base, a candidate a records commit replaces or stages again, and every candidate above them, are no red or leave.
        One with a Lean member, with records runs, waits whole in the pool while another batch stays staged."""
        by, ready = {c.sha: c for c in s.candidates}, {r.number: r.head for r in s.ready}
        live = {m for c in self.cands.values() for m in c.members}
        out = {m for a in acts if isinstance(a, Drop) and a.outcome is not None for m in a.outcome.suspects}
        excused = set()
        for a in acts:
            if isinstance(a, Drop) and (a.why == ON_MAIN or a.why.startswith((
                    "not on main's chain", "a second candidate on", "its records were regenerated", "it is staged again")) or (
                    a.why == BASE_DROPPED and by[a.sha].base in excused)):
                excused.add(a.sha)
        for a in acts:
            if not isinstance(a, Drop) or a.sha in excused:
                continue
            c = by[a.sha]
            kept = {m for m, h in zip(c.members, c.heads) if m not in out and ready.get(m) == h}
            if s.records and any(self.marks[m].lean for m in kept) and any(d.waits for d in self.cands.values()) and not any(
                    isinstance(b, Build) and b.replaces == a.sha for b in acts):
                assert not kept & live, f"seed {self.seed} tick {self.tick}: {a.sha}'s members split: {acts}"
                self.seen["a Lean rebuild waited for the batch staged"] += 1
                continue
            assert kept <= live, (f"seed {self.seed} tick {self.tick}: {a.sha}, dropped ({a.why}), takes "
                                  f"{sorted(kept - live)} out of the chain: {acts}")
            left = set(c.members) - kept - out
            self.seen["a red's others rebuilt in its tick" if a.outcome is not None else
                      "a leaver's others rebuilt in its tick" if left else "others rebuilt above a drop in its tick"] += bool(kept)
            self.seen["a leaver subtracted"] += bool(left & s.pulled)

    def check(self, s: Snapshot, acts: list, moved: str | None) -> None:
        where = f"seed {self.seed} tick {self.tick}: {acts}"
        by = {c.sha: c for c in s.candidates}
        ready = {r.number: r for r in s.ready}
        decided = {a.sha for a in acts if isinstance(a, Drop) and a.outcome is not None}
        built = {**ready, **{r.number: r for r in s.flying}}
        staged, repaired = {c.sha for c in s.candidates if c.waits}, {c.repairs for c in s.candidates if c.repairs}
        assert not [c for c in s.candidates if c.base in staged | repaired], f"{where}: a candidate on a staged batch"
        assert not [a for a in acts if isinstance(a, (Build, Commit)) and a.base in staged | repaired], (
            f"{where}: built on a staged batch")
        assert not [a for a in acts if isinstance(a, Drop) and a.outcome is None and a.sha in by
                    and by[a.sha].base in repaired], f"{where}: dropped as its records commit replaced the candidate below it"
        self.beside(s, acts, where)
        self.admits(s, acts, where)
        main = s.main
        for a in acts:
            if isinstance(a, Advance):
                c = by[a.sha]
                assert c.base == main and reads(c.verdict) == PASS and not c.waits, where
                self.seen["advanced through two candidates in a tick"] += main != s.main
                main = a.sha
                assert all(self.marks[m].head.startswith(h) for m, h in zip(c.members, c.heads)), where
                assert not set(c.members) & s.pulled, f"{where}: a PR a reviewer holds lands"
                self.seen["landed as built"] += not all(m in ready and ready[m].head == h for m, h in zip(c.members, c.heads))
                self.seen["advanced on a red main"] += bool(self.reds(s.main))
                self.seen["inherited by tests"] += any(f.key != f.base_key for f in c.verdict.failed)
                self.seen["inherited at its key alone"] += any(not f.tests for f in c.verdict.failed)
            elif isinstance(a, Eject):
                d = next(d for d in acts if isinstance(d, Drop) and d.outcome is not None and d.outcome.state == FAIL
                         and a.number in d.outcome.members)
                assert not [f for f in a.failed if inherits(f)], where
                assert s.tree_ejects or {(f.task, f.key) for f in a.failed} <= {(f.task, f.key) for f in by[d.sha].verdict.failed}, (
                    f"{where}: ejected for infra errors")
                self.seen["ejected by inputs" if "meet their inputs" in a.why else "ejected after a rerun"
                          if "rerun afresh" in a.why else "ejected for erring"] += 1
            elif isinstance(a, Red):
                c = by[a.sha]
                assert a.base == c.base == main and a.sha in decided and a.failed and all(map(inherits, a.failed)), where
                self.seen["a red rung"] += 1
            elif isinstance(a, Build) and a.replaces is not None:
                c, out = by[a.replaces], {m for d in acts if isinstance(d, Drop) and d.outcome is not None
                                          for m in d.outcome.suspects}
                assert any(isinstance(d, Drop) and d.sha == c.sha for d in acts), f"{where}: a rebuild of a live candidate"
                assert a.members == tuple(m for m in c.members if m in a.members), f"{where}: a rebuild takes another PR"
                assert all(ready[m].head == h for m, h in zip(c.members, c.heads) if m in a.members), (
                    f"{where}: a rebuild takes a PR at another head than it was built at")
                assert not set(a.members) & (out | s.pulled), f"{where}: a rebuild keeps a suspect or a held PR"
                self.seen["rebuilt without a suspect" if set(c.members) & out else "rebuilt without a leaver"
                          if len(a.members) < len(c.members) else "rebuilt whole above a drop"] += 1
                self.seen["a rebuild staged off the chain"] += a.records
                self.seen["a staged batch rebuilt without a leaver"] += c.waits
            elif isinstance(a, Build):
                self.batched(s, a, acts, where)
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == FAIL and by[a.sha].waits:
                r = by[a.sha].records
                assert r is not None and (r.state == FAIL or r.state == INFRA and r.starts >= 2), where
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == FAIL:
                c = by[a.sha]
                own = [f for f in c.verdict.failed if not inherits(f)]
                assert own or (reads(c.verdict) == INFRA and s.tree_ejects and any(f.tree for f in c.verdict.infra)), where
                if c.base != s.main:
                    under = [p for m in self.under(s, c) for p in built[m].files]
                    unsaid = [f for f in own if f.base_tests is None]
                    assert not [f for f in unsaid if f.main_tests is None or meets(under, f.inputs)], (
                        f"{where}: a fail above decided before its base's word on a failure a member below it meets")
                    self.seen["a fail above decided at once on main's word"] += bool(unsaid)
                self.seen["bisected"] += len(a.outcome.suspects) > 1
                self.seen["bisected on infra errors"] += not own and len(a.outcome.suspects) > 1
                self.seen["a mixed failure"] += bool(own and a.outcome.inherited)
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == INFRA_SPLIT:
                c = by[a.sha]
                assert reads(c.verdict) == INFRA and c.tries >= min(s.rings, 2) and len(c.members) > 1, where
                assert not s.tree_ejects and not (c.verdict.infra and all(f.launch for f in c.verdict.infra)), where
                assert not any(isinstance(e, Eject) and e.number in c.members for e in acts), f"{where}: an infra split ejects"
                self.seen["bisected"] += len(a.outcome.suspects) > 1
                self.seen["bisected on infra errors"] += len(a.outcome.suspects) > 1
                self.seen["bisected on infra errors, ejecting nobody"] += 1
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
                assert not any(meets(self.marks[c.members[0]].files, f.inputs) for f in own), where
                self.seen["rerun"] += 1
            elif isinstance(a, Ring):
                assert a.pr is None or (not s.tree_ejects and by[a.sha].members == (a.pr,)), where
                self.seen["rung late" if "still pending" in a.why else "an owner rung" if a.pr else "rung"] += 1
            elif isinstance(a, Drop) and a.outcome is None and "isn't taken now" in a.why:
                assert set(by[a.sha].members) & s.pulled, where
                self.seen["dropped for a reviewer's hold"] += 1
        for c in by.values():
            stale = any(isinstance(a, Drop) and a.sha == c.sha and a.outcome is None for a in acts)
            if c.base != s.main or set(c.members) & s.pulled or stale or c.waits:
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
                assert tree or not any(isinstance(a, Eject) and a.number in c.members for a in acts), where
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
        for c in by.values():
            unsaid = [f for f in c.verdict.failed if not inherits(f) and f.base_tests is None]
            if c.base != s.main and reads(c.verdict) == FAIL and unsaid and c.sha not in decided:
                under = [p for m in self.under(s, c) for p in built[m].files]
                self.seen["a fail above waited for the word of the one below"] += any(meets(under, f.inputs) for f in unsaid)
        if moved:               # a staged batch's records are its own tree's: its run's refusal is decided wherever `main` is
            drops = [a for a in acts if isinstance(a, Drop)]
            assert {d.sha for d in drops} >= {c for c in by if c not in staged} and all(
                d.outcome is None for d in drops if d.sha not in staged), where
            out = self.merged.keys() | {a.number for a in acts if isinstance(a, Eject)}
            pool = [r for r in s.ready if not r.conflict and r.number not in out and (
                s.lean or not r.lean or r.number in s.pinned)]
            bases = {a.base for a in acts if isinstance(a, (Build, Commit))}
            if staged - {d.sha for d in drops}:                 # beside a staged batch: what `beside` holds it to
                assert bases <= {moved}, where
            else:
                assert bases == ({moved} if pool else set()), where

    def beside(self, s: Snapshot, acts: list, where: str) -> None:
        """While every staged batch's records run is pending, and none is dropped, the builds are the plain ones `decide` makes
        with no staged batch and none of their members: plain batches up to the depth and the rebuilds of what a red or a
        leave dropped (one with a Lean member staged again), and no Lean batch beside them, nor a PR changing a Lean input,
        nor a PR set aside, which goes behind the staged batch as behind any PR waiting."""
        staged = [c for c in s.candidates if c.waits]
        if not staged or any(c.records is None or c.records.state != PENDING for c in staged) or any(
                isinstance(a, (Drop, Commit)) and a.sha in {c.sha for c in staged} for a in acts):
            return
        out = {m for c in staged for m in c.members}
        alone = replace(s, candidates=tuple(c for c in s.candidates if not c.waits), ready=tuple(
            replace(r, conflict=r.conflict or r.lean or r.lean_input or r.number in out) for r in s.ready))
        self.seen["a PR changing a Lean input waited behind a staged batch"] += any(
            r.lean_input and not r.lean for r in self.pool(s, acts))
        now = [a.outcome for a in acts if isinstance(a, Drop) and a.outcome is not None]
        aside = {(m, h) for o in [*s.history, *now] for m, h in zip(o.members, o.heads)
                 if o is [p for p in [*s.history, *now] if (m, h) in zip(p.members, p.heads)][-1] and o.state == INFRA}
        heads = {r.number: r.head for r in s.ready}
        builds = [a for a in acts if isinstance(a, Build) and not (a.records and a.replaces is not None)]
        assert builds == [a for a in decide(alone) if isinstance(a, Build) and not a.records and not (
            a.replaces is None and len(a.members) == 1 and (a.members[0], heads[a.members[0]]) in aside)], (
            f"{where}: built otherwise beside a staged batch than with none")
        self.seen["a PR set aside waited behind a staged batch"] += any(
            isinstance(a, Build) and a.replaces is None and len(a.members) == 1 and (a.members[0], heads[a.members[0]]) in aside
            for a in decide(alone))
        self.seen["a plain batch built beside a staged batch"] += bool(builds)
        self.seen["a plain batch built beside a staged batch, the chain above one"] += bool(builds) and builds[0].base != s.main

    def batched(self, s: Snapshot, a: Build, acts: list, where: str) -> None:
        """What a batch holds and where it goes: Lean PRs only or none, in rank order from the first of its kind a batch may
        take, a move last (`vqueue._moves_last`), on the chain's top as the tick's rebuilds leave it or, in a tick that commits a staged batch, on that batch there
        (`Build.after`); a rebuild staged again stays staged.  A Lean batch
        holds the pinned Lean PRs alone while one waits, else at most `lean`, a wide one alone, on no chain holding a wide one
        in flight; with records runs it is staged, only while no other batch stays staged, on no chain holding a PR set aside
        before that changes a Lean input (`erred`), and, pinned or not, only while no PR with no Lean change that changes a
        Lean input waits, whatever its rank: it would keep that PR out; with none, it goes on `main` or on a top holding no
        Lean change.  A batch of PRs with no Lean change, beside a staged batch, holds none changing a Lean input and no PR
        set aside; a Lean PR set aside goes behind the PRs with no Lean change waiting."""
        by = {c.sha: c for c in s.candidates}
        ready = {r.number: r for r in s.ready}
        flown = {**ready, **{r.number: r for r in s.flying}}
        waiting = self.pool(s, acts)
        commit = next((b for b in acts if isinstance(b, Commit)), None)
        assert (a.after, a.base) == ((commit.sha, commit.base) if commit else (None, a.base)) and (
            commit is None or acts.index(commit) < acts.index(a)), f"{where}: built elsewhere than on the batch this tick commits"
        on = [c.sha for c in self.chain(s, acts) if c.sha != a.after]
        assert commit or a.base == (on[-1] if on else next(
            (b.sha for b in reversed(acts) if isinstance(b, Advance)), s.main)), f"{where}: built below main's chain's top"
        below = self.below(s, acts, a.base) | set(by[a.after].members if a.after else ())
        top = by[a.after].members if a.after else self.top(s, acts, a.base)
        leans = [m for m in a.members if ready[m].lean]
        assert 1 <= len(a.members) <= s.batch and not any(ready[m].conflict for m in a.members), where
        ranked = sorted(a.members, key=lambda m: (ready[m].rank, m))

        def opens(first: list[int]) -> set[int]:
            """What may open the batch: the PR first in rank, or, that one a move, which goes last or waits for a batch with
            no other move and no change to the move scripts, what the batch takes after it."""
            return {first[0], *a.members[:1]} if ready[first[0]].move else {first[0]}

        assert list(a.members) == [*(m for m in ranked if not ready[m].move), *(m for m in ranked if ready[m].move)], (
            f"{where}: a batch out of rank order, but for its move last")
        assert not leans or len(leans) == len(a.members), f"{where}: a batch mixes Lean PRs and PRs with no Lean"
        assert a.records == (s.records and bool(leans)), f"{where}: a Lean batch with no records run first, or another with one"
        gone = {b.sha for b in acts if isinstance(b, (Drop, Commit))}
        staged = [c.sha for c in s.candidates if c.waits and c.sha not in gone]
        staged += [c for c, b in self.rebuilt(acts).items() if b.records]
        if leans:
            pinned = [r.number for r in waiting if r.lean and r.number in s.pinned]
            first = pinned or [r.number for r in waiting if r.lean and s.lean]
            assert not pinned or set(leans) <= s.pinned, f"{where}: a Lean PR built while a pinned Lean PR waited"
            assert pinned or len(leans) <= s.lean or not first and set(leans) <= s.pinned, (
                f"{where}: a Lean batch over the `lean` cap")
            assert not first or a.members[0] in opens(first), f"{where}: a Lean batch not opened by the Lean PR first in rank"
            plain = [r for r in waiting if not r.lean]
            assert first or not plain, f"{where}: a Lean PR set aside built while a PR with no Lean change waited"
            ahead = [r.number for r in plain if r.lean_input] if s.records else []
            assert not ahead, f"{where}: a Lean batch staged while #{ahead[0]}, which changes a Lean input, waits: it goes first"
            wide = [m for m in leans if m in s.wide]
            assert not wide or len(leans) == 1, f"{where}: a wide PR built with another Lean PR"
            assert not [m for m in below if flown[m].lean and m in s.wide], f"{where}: a Lean PR built on a wide PR in flight"
            assert not staged, f"{where}: a Lean batch built while another is staged"
            assert not self.erred(s, acts), (
                f"{where}: a Lean batch staged on a chain holding #{self.erred(s, acts)[:1]}, set aside before, a Lean input")
            assert s.records or not [m for m in top if flown[m].lean], (
                f"{where}: a Lean batch built on a candidate holding a Lean change, with no records run")
            self.seen["a pinned Lean batch built"] += bool(pinned)
            self.seen["a pinned Lean batch built on a candidate"] += bool(pinned and top)
            self.seen["an unpinned Lean PR built beside pins"] += bool(s.pinned and not pinned)
            self.seen["a Lean batch opened while a PR with no Lean change waited"] += any(not r.lean for r in waiting)
            self.seen["a Lean batch staged on a full chain"] += a.records and len(self.chain(s, acts)) >= s.depth
            self.seen["a Lean batch staged on the batch its tick committed"] += a.after is not None
            self.seen["two lean PRs built together"] += len(leans) > 1
            self.seen["a wide PR built"] += bool(wide)
        else:
            beside = bool(staged) or any(b.records for b in acts if isinstance(b, Build))
            first = [r.number for r in waiting if not r.lean and not (beside and r.lean_input)]
            assert not first or a.members[0] in opens(first), f"{where}: a batch not opened by the PR first in rank"
            assert not beside or not any(ready[m].lean_input for m in a.members), (
                f"{where}: a PR changing a Lean input built beside a staged batch")
            assert first or not beside, f"{where}: a PR set aside built beside a staged batch"
            self.seen["a batch built beside the Lean batch its tick staged"] += any(
                b.records for b in acts if isinstance(b, Build))
            self.seen["a batch built on the batch its tick committed"] += a.after is not None
            self.seen["a Lean PR waited for a top with no Lean change"] += bool(
                not s.records and [m for m in top if flown[m].lean] and [r for r in waiting if r.lean])
            self.seen["a pinned PR with no Lean change built"] += any(m in s.pinned for m in a.members)
            self.seen["a PR the wide file names with no Lean change built"] += any(m in s.wide for m in a.members)
        self.seen["a lean PR built"] += bool(leans)
        self.seen["built on a wide PR in flight"] += any(flown[m].lean and m in s.wide for m in below)
        self.seen["built on a red main"] += bool(self.reds(s.main))

    def admits(self, s: Snapshot, acts: list, where: str) -> None:
        """What a tick builds: at most one Lean batch and one batch of PRs with no Lean change, both only with records runs,
        the Lean one staged beside the other, and nothing above a fail that waits for its base's word (`fail_waits`).  Else it
        opens a Lean batch whenever a Lean PR could be admitted (a pinned one, or any under a `lean` above 0) and nothing keeps
        it out: no batch stays staged, the chain holds no wide Lean change, nor, with records runs, a PR set aside before that
        changes a Lean input (`erred`), and, with no records run, it has room and a top holding no Lean change; with records
        runs it waits, pinned or not, while a PR with no Lean change that changes a Lean input waits, which goes first: the
        tick builds it when the chain has room.  And it builds a batch of PRs with no Lean change whenever one that a batch may
        take waits and the chain has room, on the batch the tick commits too: beside a staged batch, one changing no Lean
        input; with no records run, unless the Lean batch took the tick's build.  It says it builds nothing (`Wait`) only while
        a Lean PR waits for such a chain and no PR with no Lean change is ready to take.  The tick's rebuilds
        (`Build.replaces`) are none of these builds, and one staged again keeps a Lean batch out as a batch kept staged."""
        ready = {r.number: r for r in s.ready}
        flown = {**ready, **{r.number: r for r in s.flying}}
        builds = [a for a in acts if isinstance(a, Build) and a.replaces is None]
        lean = [a for a in builds if any(ready[m].lean for m in a.members)]
        assert len(lean) <= 1, f"{where}: a tick builds two Lean batches"
        assert len(builds) - len(lean) <= 1, f"{where}: a tick builds two batches of PRs with no Lean change"
        assert s.records or len(builds) <= 1, f"{where}: a tick builds two batches on the chain"
        if self.fail_waits(s, acts):
            assert not builds and not any(isinstance(a, (Commit, Wait)) for a in acts), (
                f"{where}: built above a fail that waits for its base's word")
            return
        chain, waiting = self.chain(s, acts), self.pool(s, acts)
        gone = {a.sha for a in acts if isinstance(a, (Drop, Commit))}
        staged = [c.sha for c in s.candidates if c.waits and c.sha not in gone]
        staged += [c for c, a in self.rebuilt(acts).items() if a.records]
        room = len(chain) < s.depth
        wide = [m for c in chain for m in c.members if flown[m].lean and m in s.wide]
        top = [m for m in chain[-1].members if flown[m].lean] if chain and not s.records else []
        admit = [r for r in waiting if r.lean and (s.lean or r.number in s.pinned)]
        turn = [r.number for r in waiting if r.lean_input and not r.lean] if admit and s.records else []
        erred = self.erred(s, acts)
        assert lean or not admit or staged or wide or erred or top or turn or not (s.records or room), (
            f"{where}: no Lean batch opened while #{admit[0].number} could be admitted and nothing is staged")
        first = bool(turn and not lean and not staged and not wide and not erred)
        self.seen["a Lean batch waited for a PR changing a Lean input to go first"] += first
        self.seen["a pinned Lean batch waited for a PR changing a Lean input to go first"] += first and any(
            r.number in s.pinned for r in admit)
        self.seen["a PR changing a Lean input that waited behind a staged batch went ahead of the next"] += first and any(
            self.behind.get(m) for m in turn)
        self.seen["a Lean PR waited for a PR set aside before to leave the chain"] += bool(
            erred and admit and not staged and not wide)
        beside = bool(staged or lean)
        takes = [r.number for r in waiting if not r.lean and not (beside and r.lean_input)]
        assert len(builds) > len(lean) or not takes or not room or (lean and not s.records), (
            f"{where}: no batch of PRs with no Lean change built while #{takes[0]} waited and the chain had room")
        if any(isinstance(a, Wait) for a in acts):
            assert not builds and room and not takes, f"{where}: a wait beside a build, or while a PR with no Lean change waited"
            assert [r for r in self.pool(s, acts, aside=True) if r.lean], f"{where}: a wait with no Lean PR waiting"
            assert wide or erred or top, f"{where}: a Lean PR waited on a chain it could be built on"
            self.seen["nothing built while a Lean PR waited"] += 1

    def waits(self, s: Snapshot) -> None:
        """Within one staging: a ready PR with no Lean change that changes a Lean input, which a batch may take, is kept out
        of the builds beside a staged batch, and it waits so behind one staged batch at most, the one staged before it was
        ready (the same batch staged again is that one still).  None is staged while it waits, so it is in a build before
        the next one is; ``s``: the snapshot once the tick's actions are applied."""
        replaced = {c.repairs for c in s.candidates}
        staged = {self.opened[c.sha] for c in s.candidates if c.waits and c.sha not in replaced}
        waiting = {r.number for r in self.pool(s, []) if r.lean_input and not r.lean}
        self.behind = {m: self.behind.get(m, set()) | staged for m in waiting}
        for m, batches in self.behind.items():
            assert len(batches) <= 1, (f"seed {self.seed} tick {self.tick}: #{m}, which changes a Lean input, waits behind a "
                                       f"second staged batch, {sorted(batches)}: it was in no build within one staging")

    def erred(self, s: Snapshot, acts: list) -> list[int]:
        """With records runs, the lone members of main's chain set aside before at the heads they are built at (an infra
        outcome) that change a Lean input: one's check erring again takes its candidate off the chain, and a batch staged
        on it would be staged again."""
        aside = {(m, h) for o in s.history if o.state == INFRA for m, h in zip(o.members, o.heads)}
        flown = {**{r.number: r for r in s.ready}, **{r.number: r for r in s.flying}}
        return [c.members[0] for c in self.chain(s, acts) if s.records and len(c.members) == 1
                and (c.members[0], c.heads[0]) in aside and (flown[c.members[0]].lean or flown[c.members[0]].lean_input)]

    def fail_waits(self, s: Snapshot, acts: list) -> bool:
        """Whether a fail above the lowest candidate holds the tick's builds: the lowest one ``acts`` leaves on main's chain,
        neither dropped nor rerun, waits for its base's word."""
        fail = next((c for c in self.chain(s, acts) if c.base != s.main and not c.waits and reads(c.verdict) == FAIL
                     and c.verdict.failed), None)
        return fail is not None and Rerun(fail.sha, tuple(f.key for f in fail.verdict.failed if not inherits(f))) not in acts

    def pool(self, s: Snapshot, acts: list, aside: bool = False) -> list[Ready]:
        """The ready PRs a build in ``acts`` may take, in rank order: in no candidate left on main's chain, staged or on `main`,
        not ejected, not conflicting, and neither on `main` nor set aside at its head (a pass or infra outcome, this tick's
        among them); with ``aside``, those set aside too."""
        dropped = {a.sha: a.why for a in acts if isinstance(a, Drop)}
        busy = {m for c in s.candidates if c.sha not in dropped or dropped[c.sha] == ON_MAIN for m in c.members}
        busy |= {a.number for a in acts if isinstance(a, Eject)}
        busy |= {m for a in acts if isinstance(a, Build) and a.replaces is not None for m in a.members}
        now = [a.outcome for a in acts if isinstance(a, Drop) and a.outcome is not None]
        newest = {(m, h): o.state for o in [*s.history, *now] for m, h in zip(o.members, o.heads)}
        return sorted((r for r in s.ready if not r.conflict and r.number not in busy and newest.get((r.number, r.head)) not in (
            (PASS,) if aside else (PASS, INFRA))), key=lambda r: (r.rank, r.number))

    @staticmethod
    def rebuilt(acts: list) -> dict[str, Build]:
        """The builds of ``acts`` that rebuild a candidate, by the candidate they rebuild: a build on it goes on them."""
        return {a.replaces: a for a in acts if isinstance(a, Build) and a.replaces is not None}

    def below(self, s: Snapshot, acts: list, sha: str) -> set[int]:
        """The members of the candidates a build on ``sha`` sits on that aren't on `main`: down main's chain to `main`, or to
        the candidate ``acts`` advances it to, through what ``acts`` rebuilds."""
        by, advanced, out = {c.sha: c for c in s.candidates}, {a.sha for a in acts if isinstance(a, Advance)}, set()
        again = self.rebuilt(acts)
        while sha in again or sha in by and sha != s.main and sha not in advanced:
            c = again.get(sha) or by[sha]
            out |= set(c.members)
            sha = c.base
        return out

    def under(self, s: Snapshot, c: Candidate) -> set[int]:
        """The members of the candidates of ``s`` below ``c``, down to `main`."""
        by, out, sha = {d.sha: d for d in s.candidates}, set(), c.base
        while sha in by and sha != s.main:
            out |= set(by[sha].members)
            sha = by[sha].base
        return out

    def top(self, s: Snapshot, acts: list, sha: str) -> tuple[int, ...]:
        """The members of candidate ``sha``, which a build on it sits on directly: none when ``sha`` is `main` or the candidate
        ``acts`` advances it to; what ``acts`` rebuilds it as, when it does."""
        if (again := self.rebuilt(acts).get(sha)) is not None:
            return again.members
        c = next((c for c in s.candidates if c.sha == sha), None)
        return () if c is None or sha == s.main or Advance(sha) in acts else c.members

    def chain(self, s: Snapshot, acts: list) -> list[Candidate]:
        """Main's chain once ``acts`` drops, advances, rebuilds and commits what it does, lowest first: what a build in
        ``acts`` sits on, each candidate rebuilt on the chain at the sha it replaces, and the staged batch a `Commit` of
        ``acts`` puts on its top last."""
        dropped = {a.sha for a in acts if isinstance(a, Drop)}
        on = {c.base: c for c in s.candidates if c.sha not in dropped and not c.waits}
        sha, out = next((a.sha for a in reversed(acts) if isinstance(a, Advance)), s.main), []
        while sha in on:
            out.append(on[sha])
            sha = on[sha].sha
        heads = {r.number: r.head for r in s.ready}
        for c, a in self.rebuilt(acts).items():
            if not a.records:
                out.append(Candidate(c, a.base, a.members, tuple(heads[m] for m in a.members)))
                sha = c
        return [*out, *(c for c in s.candidates if Commit(c.sha, sha, s.depth - len(out)) in acts)]

    def apply(self, a) -> None:
        where = f"seed {self.seed} tick {self.tick}: {a}"
        rng = self.rng
        if isinstance(a, Build):
            base = self.rebuilds.get(a.base, a.base) if a.after is None else self.put.get(a.after)
            if base is None:                            # its Commit staged the batch again: the next tick decides
                self.seen["nothing built on a batch its Commit staged again"] += 1
                return
            assert base == self.main or base in self.cands and not self.cands[base].waits, where
            assert not a.records or self.records_first, where
            replaced = {c.repairs for c in self.cands.values()}
            if a.replaces is None and any(c.waits and c.sha not in replaced for c in self.cands.values()):
                self.beside_staged |= set(a.members)
            sha = self._built(base, a.members, tuple(self.marks[m].head for m in a.members), staged=a.records)
            if a.replaces is not None:
                self.rebuilds[a.replaces] = sha
                if a.records:
                    self.opened[sha] = self.opened.get(a.replaces, a.replaces)
            if not a.records and rng.random() < 0.04:
                self.outage[sha] = rng.choice((INFRA, LIST_FAILED))
        elif isinstance(a, Regenerate):
            c = self.cands[a.sha]
            assert c.waits, where
            self.cands[a.sha] = replace(c, records=RecordsRun(starts=c.records.starts + 1 if c.records else 1))
            self.records_due[a.sha] = self.tick + rng.randint(1, 4)
        elif isinstance(a, Commit):
            self._commit(a, where)
        elif isinstance(a, Advance):
            c = self.cands[a.sha]
            assert c.base == self.main and reads(c.verdict) == PASS and not c.waits, where
            assert not set(zip(c.members, c.heads)) & self.landed, f"{where}: a PR reaches main twice at one head"
            assert not set(c.members) & (set(self.bad) | set(self.erring)), f"{where}: a bad PR reaches main"
            assert self.reds(a.sha) <= self.reds(self.main), f"{where}: main advances to a new failure"
            if self.records_first:
                inputs = self.lean_in(a.sha)
                assert self.locks[a.sha] == self.lock(inputs), f"{where}: main advances to stale lock records"
                assert not any(self.marks[m].lean for m in c.members) or self.regenerated_on.get(a.sha) == inputs, (
                    f"{where}: a Lean batch lands with records not regenerated on its tree's Lean inputs")
                self.seen["a Lean batch landed"] += any(self.marks[m].lean for m in c.members)
            self.seen["a fix cleared a red"] += self.reds(a.sha) < self.reds(self.main)
            self.main, self.main_since = a.sha, self.tick
            self.merged |= dict.fromkeys(c.members, self.tick)
            self.landed |= set(zip(c.members, c.heads))
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
        elif not isinstance(a, Wait):
            assert isinstance(a, Ring), where
            c = self.cands.get(a.sha)
            if c is not None and a.pr is not None:
                self.owned.add((a.pr, c.heads[0]))
            if c is not None and a.sha in self.outage and c.tries >= 2:
                del self.outage[a.sha]
                self.cands[a.sha] = replace(c, verdict=Verdict())
                self.due[a.sha] = self.tick + rng.randint(2, 5)
                self.seen["rung, and infra reran it"] += 1

    def _built(self, base: str, members: tuple[int, ...], heads: tuple[str, ...], staged: bool = False,
               repairs: str | None = None, locks: frozenset[int] | None = None) -> str:
        """A new candidate of ``members`` merged onto ``base``: its check enqueued, or ``staged``, its records run first; its
        tree's lock records ``locks``, else ``base``'s."""
        sha, self.made = f"q{self.made}", self.made + 1
        self.tree[sha] = self.tree[base] | set(members)
        self.locks[sha] = self.locks[base] if locks is None else locks
        if staged:
            self.opened[sha] = self.opened.get(repairs, sha)
            self.cands[sha] = Candidate(sha, base, members, heads, tries=0, records=RecordsRun(), waits=True, repairs=repairs)
            self.due[sha], self.records_due[sha] = None, self.tick + self.rng.randint(1, 8)
            return sha
        self.cands[sha] = Candidate(sha, base, members, heads, enqueued=self.tick * TICK_S, repairs=repairs)
        self.due[sha] = None if self.rng.random() < 0.06 else self.tick + self.rng.randint(1, 3)
        self.seen["wedged"] += self.due[sha] is None
        return sha

    def _commit(self, a: Commit, where: str) -> None:
        """Staged batch ``a.sha`` onto ``a.base``, as `vqueue_loop.Loop._commit` puts it there."""
        c = self.cands[a.sha]
        assert c.waits and c.records is not None and c.records.state == PASS, f"{where}: committed before its records passed"
        assert a.base == self.main or a.base in self.cands and not self.cands[a.base].waits, where
        inputs = self.lean_in(a.sha)
        regenerated = self.lock(inputs)
        same = regenerated == self.locks[a.sha]
        if a.base == c.base and same:
            self.cands[a.sha] = replace(c, waits=False, tries=1, enqueued=self.tick * TICK_S)
            self.due[a.sha], self.regenerated_on[a.sha], self.put[a.sha] = self.tick + self.rng.randint(1, 3), inputs, a.sha
            self.seen["a staged batch checked as built"] += 1
            return
        top = frozenset(m for m in self.tree[a.base] | set(c.members) if self.marks[m].lean or m in self.touch)
        if a.base != c.base and not self.same(top, inputs):
            assert not (top ^ inputs) & self.beside_staged, (
                f"{where}: staged again for a Lean input #{min((top ^ inputs) & self.beside_staged)} changed, built while a "
                "batch was staged")
            self._built(a.base, c.members, c.heads, staged=True, repairs=a.sha)
            self.seen["a staged batch staged again on other Lean inputs"] += 1
            return
        sha = self.put[a.sha] = self._built(a.base, c.members, c.heads, repairs=a.sha, locks=None if same else regenerated)
        self.regenerated_on[sha], self.opened[sha] = inputs, self.opened.get(a.sha, a.sha)
        self.seen["its records committed on it" if a.base == c.base else "its records carried onto another base" if not same
                  else "checked as built on another base"] += 1


SEEDS = range(500)


def ticks(w: World) -> int:
    """The ticks within which a scenario settles: every good PR on main, every bad PR ejected, no candidate left (without
    `tree_ejects`, only erring PRs left, each with its owner rung).  Seeds 0 to 4999 settled within 81 ticks and 13.2 per PR,
    none more than 21 past 8 a PR; with a broken main whose fixes are ranked last and late, within 62 and 11.7 per PR, none
    more than 11 past 8 a PR; with wide PRs, within 87 and 15.7 per PR, none more than 23 past 8 a PR; and with records
    runs, within 102 and 15.7 per PR, none more than 38 past 8 a PR."""
    return 40 + 8 * len(w.marks)


def test_the_simulated_queue_keeps_every_invariant_and_settles():
    """Over every seed: main advances only to a candidate on it, or on the one the tick advanced it to before, with no new
    failure (each failure inherited, nothing running, no infra error), at heads its members were marked at (one its owner unmarked or pushed in flight lands as built) and
    holding none a reviewer holds, so its reds never grow; no PR is in two live candidates or reaches main twice at one head;
    with `tree_ejects`, an infra error that follows the tree (`tree`) is bisected like a fail at once, and any other
    infra or listing error is retried at every reading, rung from `rings` enqueues (2 at most), and blames nobody; without
    it, any infra or listing error, a `tree` one too, is retried, then rung and bisected from `rings` enqueues, ejecting
    nobody, and a lone member's is rung to its owner too and set aside behind the PRs waiting, else retried; a candidate
    pending past the bound is retried once and rung past twice it; a lone member's own failure is rerun once
    before it is ejected; a batch holds Lean PRs only, at most `lean`, or none, in rank order, and a Lean one opens whenever
    its chain allows, whether or not a PR with no Lean change waits; a move of main by someone else drops the stale candidates
    and rebuilds on it; a red or a leave takes out only its suspects or that member, every other member of what it dropped
    built again in its tick, in order, and a fail above the lowest is decided before its base's word only on `main`'s, where
    no member below meets it (`World.restacked`);
    an inherited failure ejects nobody, and its red rings once; a restart repeats nothing but a ring, or
    a red whose drop it didn't apply; every failure is read from its check's records as the loop reads them, never as
    unnamed where its report names its tests, its reads covering every file it read; and every good PR reaches main and
    every bad one is ejected within `ticks`, every erring one too with `tree_ejects`, and without it only erring PRs stay
    live, each with its owner rung.  Each situation the invariants are about happens in some seed."""
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
                 "a mixed failure", "waited for the rest of a check", "a fix cleared a red", "landed as built",
                 "dropped for a reviewer's hold", "a Lean batch opened while a PR with no Lean change waited",
                 "a Lean PR waited for a top with no Lean change", "nothing built while a Lean PR waited",
                 "a failure read as the tests its report names", "a failure read with its reads unbounded",
                 "a suite's reads narrowed by what its run traced", "a suite's reads its declared inputs",
                 "an untraced program's module read as its declared inputs", "rebuilt without a suspect",
                 "rebuilt without a leaver", "rebuilt whole above a drop", "a red's others rebuilt in its tick",
                 "a leaver's others rebuilt in its tick", "a leaver subtracted", "a fail above decided at once on main's word",
                 "a fail above waited for the word of the one below", "advanced through two candidates in a tick"):
        assert seen[what], f"no seed exercised: {what}"


def _unstacked(s: Snapshot) -> list:
    """`decide` building nothing again in the tick that drops a red or a leaver's candidate, as before the subtraction: the
    rebuilds and what it builds on them go, and the rest waits for the pool."""
    acts = decide(s)
    again = {a.replaces for a in acts if isinstance(a, Build) and a.replaces is not None}
    return [a for a in acts if not (isinstance(a, Build) and (a.replaces is not None or a.base in again))]


def test_decide_rebuilding_nothing_a_red_or_a_leave_dropped_breaks_the_subtraction_invariant():
    """The subtraction invariant (`World.restacked`) holds `decide` to building again, in the same tick, every member of a
    candidate a red or a leave dropped that isn't a suspect or the leaver, with the invariants on a tick's builds
    (`World.admits`), which a tick building nothing breaks first, off."""
    with pytest.raises(AssertionError, match="out of the chain"):
        for seed in SEEDS:
            w = World(seed, policy=_unstacked)
            w.admits = lambda *a: None
            _settle(w)


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


def _settle(w: World) -> World:
    while not w.done():
        assert w.tick < ticks(w), f"seed {w.seed}: not settled after {ticks(w)} ticks"
        w.step()
    return w


def test_the_simulated_queue_with_records_runs_builds_plain_batches_beside_a_staged_one_and_lands_only_fresh_records():
    """With each Lean batch's records run first (`Snapshot.records`), wide PRs in every third seed and pins in every fourth:
    every invariant holds, among them that while every staged batch's records run is pending the builds are those `decide`
    makes with no staged batch, no Lean PR and no PR changing a Lean input, plain batches up to the depth; that nothing is
    built on a staged batch, so no candidate is dropped because a records commit replaced the one below it; that no batch is
    staged again for a Lean input a PR built beside a staged batch changed; that every tree `main` advances to holds the
    lock records of its Lean inputs, a Lean batch's regenerated on that tree's Lean inputs; and that a red or a leave takes
    out only its suspects or that member, a Lean candidate it drops and a staged batch holding a leaver staged again in its
    tick (`World.restacked`).  And those of a tick's builds (`World.admits`, `World.batched`): a Lean batch is staged
    whenever a Lean PR could be admitted and none stays staged, on a full chain too, but, pinned or not, only once no PR
    changing a Lean input waits, and on no chain holding one set aside before; the tick that stages it builds the PRs with
    no Lean change beside it, and the tick that puts one on main's chain builds on it there; never two Lean batches a tick,
    nor two of the others, beside the tick's rebuilds.  And a PR changing a Lean input waits behind one staged batch at
    most (`World.waits`).  Every scenario settles within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        seen.update(_settle(World(seed, wide=seed % 3 == 0, pinned=seed % 4 == 0, records=True)).seen)
    for what in ("a plain batch built beside a staged batch", "a plain batch built beside a staged batch, the chain above one",
                 "a PR changing a Lean input waited behind a staged batch",
                 "a staged batch checked as built", "its records committed on it", "its records carried onto another base",
                 "checked as built on another base", "a staged batch staged again on other Lean inputs", "a Lean batch landed",
                 "a records run ended in infra", "a records run ended in fail", "main moved elsewhere",
                 "the loop stopped early", "two lean PRs built together",
                 "a batch built beside the Lean batch its tick staged", "a Lean batch staged on a full chain",
                 "a Lean batch opened while a PR with no Lean change waited",
                 "a batch built on the batch its tick committed", "a Lean batch staged on the batch its tick committed",
                 "nothing built on a batch its Commit staged again",
                 "a Lean batch waited for a PR changing a Lean input to go first",
                 "a pinned Lean batch waited for a PR changing a Lean input to go first",
                 "a PR changing a Lean input that waited behind a staged batch went ahead of the next",
                 "a Lean PR waited for a PR set aside before to leave the chain",
                 "a PR set aside waited behind a staged batch", "a pinned Lean batch built",
                 "an unpinned Lean PR built beside pins",
                 "a rebuild staged off the chain", "a staged batch rebuilt without a leaver",
                 "a Lean rebuild waited for the batch staged"):
        assert seen[what], f"no seed exercised: {what}"


RECORDS_MISREADS: dict[str, Callable[[Snapshot], Snapshot]] = {
    "no records runs": lambda s: replace(s, records=False),
    "nothing staged": lambda s: _candidates(s, lambda c: replace(c, waits=False)),
    "every records run as passed": lambda s: _candidates(
        s, lambda c: replace(c, records=replace(c.records, state=PASS)) if c.records else c),
    "no records run as passed": lambda s: _candidates(
        s, lambda c: replace(c, records=replace(c.records, state=PENDING)) if c.records else c),
    "nothing repaired": lambda s: _candidates(s, lambda c: replace(c, repairs=None)),
    "no PR changing a Lean input": lambda s: replace(s, ready=tuple(replace(r, lean_input=False) for r in s.ready)),
}


def test_records_carried_onto_other_lean_inputs_land_stale_in_the_simulated_queue():
    """The records invariant holds the loop's carry to `vqueue_records.same_inputs`: a world carrying a staged batch's records
    onto any tree lands stale records in some seed, though no check reads them."""
    with pytest.raises(AssertionError, match="stale lock records|records not regenerated on its tree's Lean inputs"):
        for seed in SEEDS:
            w = World(seed, records=True)
            w.same = lambda a, b: True
            _settle(w)


def test_a_pr_changing_a_lean_input_built_beside_a_staged_batch_stages_it_again_in_the_simulated_queue():
    """The restage invariant holds `decide` to `Ready.lean_input` alone: handed snapshots in which no PR changes a Lean input,
    and with the invariants on what is built beside a staged batch and staged ahead of such a PR (`World.beside`,
    `World.batched`, `World.waits`) off, it stages a batch again for a Lean input a PR it built beside a staged batch
    changed, in some seed."""
    with pytest.raises(AssertionError, match="staged again for a Lean input"):
        for seed in SEEDS:
            w = World(seed, records=True, policy=lambda s: decide(RECORDS_MISREADS["no PR changing a Lean input"](s)))
            w.beside = w.batched = w.waits = lambda *a: None
            _settle(w)


def _without(drop: Callable[[Build, list], bool], read: Callable[[Snapshot], Snapshot] = lambda s: s,
             ) -> Callable[[Snapshot], list]:
    """`decide` (of ``read``'s snapshot) with the builds ``drop`` names left out of its actions: the policy without one part
    of what a tick builds."""
    def policy(s: Snapshot) -> list:
        acts = decide(read(s))
        return [a for a in acts if not (isinstance(a, Build) and drop(a, acts))]
    return policy


def _ahead(pinned: bool = False) -> Callable[[Snapshot], list]:
    """`decide` staging a Lean batch (with ``pinned``, only one a pin names) ahead of the PRs with no Lean change that change
    a Lean input and wait: where `decide` stages none, what it decides with those PRs out of the pool, when that stages
    one."""
    def policy(s: Snapshot) -> list:
        acts = decide(s)
        if any(isinstance(a, Build) and a.records for a in acts):
            return acts
        busy = {m for c in s.candidates for m in c.members}
        bare = decide(replace(s, ready=tuple(replace(r, conflict=True) if r.lean_input and not r.lean and r.number not in busy
                                             else r for r in s.ready)))
        staged = [a for a in bare if isinstance(a, Build) and a.records and (not pinned or a.members[0] in s.pinned)]
        return bare if staged else acts
    return policy


#: the policy without each part of what a tick builds, and the invariant of the simulated queue it then breaks
PARTS: dict[str, tuple[Callable[[Snapshot], list], str]] = {
    "the tick that stages a Lean batch builds no other": (
        _without(lambda a, acts: not a.records and any(isinstance(b, Build) and b.records for b in acts)),
        "no batch of PRs with no Lean change built while"),
    "the tick that commits a staged batch builds nothing on it": (
        _without(lambda a, acts: a.after is not None),
        "no batch of PRs with no Lean change built while|no Lean batch opened while"),
    "a Lean batch waits for the PRs with no Lean change": (
        _without(lambda a, acts: a.records and any(isinstance(b, Build) and not b.records for b in acts)),
        "no Lean batch opened while"),
    "no Lean PR is admitted without a pin": (_without(lambda a, acts: False, lambda s: replace(s, lean=0)),
                                             "no Lean batch opened while"),
    "a Lean batch is staged ahead of a PR changing a Lean input": (_ahead(), "a Lean batch staged while"),
}


@pytest.mark.parametrize("part", PARTS)
def test_decide_without_a_part_of_what_a_tick_builds_breaks_its_invariant_of_the_simulated_queue(part):
    """The invariants on a tick's builds (`World.admits`) hold `decide` to each part of the rule: a policy that builds no
    batch of PRs with no Lean change in the tick that stages a Lean one, or nothing in the tick that puts a staged batch on
    main's chain, or stages no Lean batch while PRs with no Lean change are built, or admits no Lean PR without a pin
    under a `lean` cap above 0, or stages a Lean batch while a PR changing a Lean input waits, breaks the invariant that
    names it in some seed."""
    policy, breaks = PARTS[part]
    with pytest.raises(AssertionError, match=breaks):
        for seed in SEEDS:
            _settle(World(seed, records=True, policy=policy))


def test_the_simulated_queue_at_lean_0_admits_only_pinned_lean_prs_and_settles_without_the_others():
    """`lean` 0, pins only: with about half the Lean PRs pinned, records runs in every other seed and wide PRs in every third,
    every invariant holds, no batch takes a Lean PR the pins file doesn't name, and every other PR is settled within
    `ticks`: the pinned Lean PRs through batches of their own, one set aside built again in its turn."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        w = _settle(World(seed, wide=seed % 3 == 0, pinned=True, records=seed % 2 == 0, lean=0))
        out = {m for m, r in w.marks.items() if r.lean and m not in w.pinned}
        assert not out & (set(w.merged) | w.ejected), f"seed {seed}: a Lean PR with no pin landed or was ejected"
        seen.update(w.seen)
        seen["a Lean PR with no pin waited"] += bool(out)
    for what in ("a pinned Lean batch built", "a Lean batch landed", "a Lean PR with no pin waited",
                 "a batch built beside the Lean batch its tick staged", "a batch built on the batch its tick committed"):
        assert seen[what], f"no seed exercised: {what}"


def _coming(seed: int, policy: Callable[[Snapshot], list] = decide) -> World:
    """A world at `lean` 0 where pinned Lean batches keep coming: of its PRs in number order every third has a Lean change
    and a pin, the next changes a Lean input with no Lean change, and they arrive in that order, a tick apart.  A records
    run takes several ticks, so pinned Lean PRs wait whenever a staged batch goes onto main's chain, and PRs changing a
    Lean input become ready beside each staged batch."""
    w = World(seed, pinned=True, records=True, lean=0, policy=policy)
    for i, m in enumerate(sorted(w.marks)):
        w.marks[m] = replace(w.marks[m], lean=i % 3 == 0, lean_input=i % 3 < 2)
        w.arrive[m] = i
    w.pinned = frozenset(m for m, r in w.marks.items() if r.lean or m in w.pinned)
    w.touch = frozenset(m for m, r in w.marks.items() if r.lean_input and not r.lean)
    w.lockers = frozenset(m for m, r in w.marks.items() if r.lean and m % 3)
    return w


def test_a_pr_changing_a_lean_input_is_in_a_build_within_one_staging_while_pinned_lean_batches_keep_coming_at_lean_0():
    """Pins only order the Lean PRs: at `lean` 0, with a pinned Lean PR waiting whenever a staged batch goes onto main's
    chain, a ready PR with no Lean change that changes a Lean input, which a batch may take, waits behind one staged
    batch at most, the one staged before it was ready (`World.waits`).  No pinned batch is staged while it waits
    (`World.batched`): it goes ahead of the next into a build, as soon as the chain has room, and that batch is staged
    once it is in a candidate, back to back with the last where no such PR waits.  Every invariant holds and every world
    settles within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        seen.update(_settle(_coming(seed)).seen)
    for what in ("a PR changing a Lean input waited behind a staged batch",
                 "a pinned Lean batch waited for a PR changing a Lean input to go first",
                 "a PR changing a Lean input that waited behind a staged batch went ahead of the next",
                 "a Lean batch staged on the batch its tick committed", "a pinned Lean batch built", "a Lean batch landed",
                 "a batch built on the batch its tick committed"):
        assert seen[what], f"no seed exercised: {what}"


def test_decide_staging_a_pinned_lean_batch_ahead_of_a_pr_changing_a_lean_input_keeps_it_out_past_one_staging():
    """`World.waits` holds `decide` to it: a policy under which a pin puts a Lean batch ahead of the PRs changing a Lean
    input leaves one of them behind a second staged batch in some seed, with the invariant on the staging itself
    (`World.batched`) off; with it on, that invariant names the batch staged ahead."""
    with pytest.raises(AssertionError, match="waits behind a second staged batch"):
        for seed in SEEDS:
            w = _coming(seed, _ahead(pinned=True))
            w.batched = lambda *a: None
            _settle(w)
    with pytest.raises(AssertionError, match="a Lean batch staged while"):
        for seed in SEEDS:
            _settle(_coming(seed, _ahead(pinned=True)))


@pytest.mark.parametrize("misread", RECORDS_MISREADS)
def test_decide_misreading_the_records_fields_breaks_an_invariant_of_the_simulated_queue_with_records_runs(misread):
    """The records invariants hold `decide` to the fields staging reads: handed snapshots with one misread, it lands stale
    records, builds on a staged batch, commits one before its records passed, or doesn't settle, in some seed."""
    with pytest.raises(AssertionError):
        for seed in SEEDS:
            _settle(World(seed, records=True, policy=lambda s: decide(RECORDS_MISREADS[misread](s))))


def test_the_simulated_queue_with_wide_prs_builds_none_beside_or_above_another_lean_pr_and_settles():
    """With about half the Lean PRs wide (`Snapshot.wide`), and the file naming a PR with no Lean change now and then: every
    invariant holds, among them that a batch taking a wide PR takes no other Lean PR (A) and that one built on a candidate
    holding a wide PR not on `main` takes no Lean PR (B), and every scenario settles within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        seen.update(_settle(World(seed, wide=True)).seen)
    for what in ("a wide PR built", "built on a wide PR in flight", "a PR the wide file names with no Lean change built",
                 "two lean PRs built together"):
        assert seen[what], f"no seed exercised: {what}"


def test_decide_reading_no_wide_prs_builds_a_lean_pr_beside_or_above_a_wide_one_in_the_simulated_queue():
    """The wide invariants hold `decide` to `Snapshot.wide`: handed snapshots naming no wide PR, it builds a Lean PR beside a
    wide one or above one in flight in some seed."""
    with pytest.raises(AssertionError, match="a wide PR built with another Lean PR|a Lean PR built on a wide PR in flight"):
        for seed in SEEDS:
            _settle(World(seed, wide=True, policy=lambda s: decide(replace(s, wide=frozenset()))))


def test_the_simulated_queue_with_pins_builds_pinned_lean_batches_alone_at_most_every_second_build_and_settles():
    """With about half the Lean PRs pinned (`Snapshot.pinned`), the pins file naming a PR with no Lean change now and then, and
    wide PRs in every third seed, with no records run: every invariant holds, among them that a batch holding a pinned Lean
    PR it may take holds only pinned Lean PRs, under no `lean` cap; that no other Lean PR is built while a pinned one
    waits; that a Lean batch goes on `main` or on a top candidate holding no Lean change; and that a tick builds nothing,
    saying why (`Wait`), only while a Lean PR waits for such a top, or for a wide one to land, with nothing else to take.
    Every scenario settles within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        seen.update(_settle(World(seed, wide=seed % 3 == 0, pinned=True)).seen)
    for what in ("a pinned Lean batch built", "a pinned Lean batch built on a candidate",
                 "a Lean PR waited for a top with no Lean change", "nothing built while a Lean PR waited",
                 "a pinned PR with no Lean change built", "an unpinned Lean PR built beside pins", "a wide PR built",
                 "two lean PRs built together"):
        assert seen[what], f"no seed exercised: {what}"


@pytest.mark.parametrize(("misread", "pinned"), [("no pins", lambda s: frozenset()),
                                                 ("every PR pinned", lambda s: frozenset(r.number for r in s.ready))])
def test_decide_misreading_the_pins_breaks_an_invariant_of_the_simulated_queue(misread, pinned):
    """The pins invariants hold `decide` to `Snapshot.pinned`: handed snapshots pinning nothing, or every PR, it builds a Lean
    PR while a pinned one waits, or a Lean batch over the `lean` cap, in some seed."""
    with pytest.raises(AssertionError, match="a Lean PR built while a pinned Lean PR waited|a Lean batch over the `lean` cap"):
        for seed in SEEDS:
            _settle(World(seed, wide=seed % 3 == 0, pinned=True, policy=lambda s: decide(replace(s, pinned=pinned(s)))))


def _candidates(s: Snapshot, f: Callable[[Candidate], Candidate]) -> Snapshot:
    return replace(s, candidates=tuple(map(f, s.candidates)))


def _verdicts(s: Snapshot, **kw) -> Snapshot:
    return _candidates(s, lambda c: replace(c, verdict=replace(c.verdict, **kw)))


def _failures(s: Snapshot, f: Callable[[Failed], Failed], field: str = "failed") -> Snapshot:
    return _candidates(s, lambda c: replace(c, verdict=replace(c.verdict, **{field: tuple(map(f, getattr(c.verdict, field)))})))


@dataclass(frozen=True)
class Fetch:
    """A misread of the records the loop's reader gets (`World.fetch`), where the others misread the snapshot `decide` gets."""
    f: Callable[[Records], Records]


def _cut(*reports: str) -> Callable[[Records], Records]:
    """A suite's records with its row's report never fetched (one of ``reports``), and so nothing the reader fetches beside a
    failed suite's row: its log, its guard's report, its children's reads."""
    return lambda r: replace(r, row=None, log="", report=None, trace="") if r.at in reports else r


MISREADS: dict[str, Callable[[Snapshot], Snapshot] | Fetch] = {
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
    "main passing what a candidate above another fails": lambda s: _failures(s, lambda f: replace(f, main_tests=frozenset())),
    "no history": lambda s: replace(s, history=()),
    "no Lean cap": lambda s: replace(s, lean=s.batch),
    "every change as Lean": lambda s: replace(s, ready=tuple(replace(r, lean=True) for r in s.ready)),
    "nothing pulled": lambda s: replace(s, pulled=frozenset()),
    "no files as built": lambda s: replace(s, flying=()),
    "an untraced program's module as traced": Fetch(
        lambda r: replace(r, report={**r.report, "untraced": {}}) if r.report else r),
    # records as a fetch gave them when it named a file twice, which cut off every file sorted after it: a check that ended
    # with lean-suites failing one parametrized test of the vLLM suite (and lean-audit one module), each read as unnamed
    "an ended check's suite reports, cut off": Fetch(_cut("suites.json", "suites-lean.json")),
    # a check that ended with lean-suites failing 16 tests under one directory of the vLLM suite, read as every member's
    "an ended check's lean-suites report, cut off": Fetch(_cut("suites-lean.json")),
    # a shard that failed the repository's own suite on one boundary test while the check ran: its failures so far never
    # reached the loop, so its early red never read them
    "a running check's failures so far, cut off": Fetch(_cut("suites.failed.json", "suites-lean.failed.json")),
}


@pytest.mark.parametrize("misread", MISREADS)
def test_decide_misreading_a_field_of_the_snapshot_breaks_an_invariant_of_the_simulated_queue(misread):
    """The invariants hold `decide` to every field it reads, and the loop's reader to every record it reads: handed snapshots
    with one field misread, or records misread (`Fetch`), it breaks one of them, or doesn't settle, in some seed."""
    m = MISREADS[misread]
    with pytest.raises(AssertionError):
        for seed in SEEDS:
            w = World(seed, fetch=m.f) if isinstance(m, Fetch) else World(seed, policy=lambda s: decide(m(s)))
            while not w.done():
                assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks"
                w.step()
