"""research.vqueue: `decide` rule by rule on small snapshots, `meets`, which ejection rests on, and a simulated world that applies
`decide`'s actions as the loop does, with verdicts from a scripted oracle, held to the queue's invariants over a fixed list of
seeds."""
from __future__ import annotations

import builtins
import random
import time
from collections import Counter
from dataclasses import replace

import pytest
from research.vqueue import (FAIL, INFRA, LIST_FAILED, PASS, PENDING, Advance, Build, Candidate, Drop, Eject, Failed, Outcome,
                             Ready, Red, Retry, Ring, Snapshot, Verdict, decide, meets)

MAIN = "m0"


def pr(n: int, head: str | None = None, *files: str, rank: int | None = None, lean: bool = False,
       conflict: bool = False) -> Ready:
    return Ready(n, head or f"h{n}", frozenset(files or (f"pkg{n}/mod.py",)), lean, n if rank is None else rank, conflict)


def prs(*ns: int) -> list[Ready]:
    return [pr(n) for n in ns]


def cand(sha: str, base: str, *members: int, state: str = PENDING, failed: tuple[Failed, ...] = (), tries: int = 1,
         heads: tuple[str, ...] | None = None) -> Candidate:
    return Candidate(sha, base, members, heads or tuple(f"h{m}" for m in members), Verdict(state, failed), tries)


def task(name: str, *inputs: str) -> Failed:
    return Failed(name, f"key:{name}", frozenset(inputs))


def inherited(name: str, *inputs: str) -> Failed:
    """A failure at its base's key: the base fails it too."""
    return Failed(name, f"key:{name}", frozenset(inputs), f"key:{name}")


def snap(*cands: Candidate, ready=(), history=(), main: str = MAIN, **kw) -> Snapshot:
    return Snapshot(main, tuple(ready), cands, tuple(history), **kw)


def heads(*ns: int) -> tuple[str, ...]:
    return tuple(f"h{n}" for n in ns)


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


def test_a_member_failing_alone_is_checked_alone_once_more_then_ejected():
    t = task("t", "elsewhere/")
    lone = Outcome((5,), heads(5), FAIL, (5,), MAIN)
    s = snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,)), ready=prs(5, 6))
    assert decide(s) == [Drop("c1", "failed: t", lone), Build((5,), MAIN, 2)]
    s = snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,)), ready=prs(5, 6), history=[lone])
    acts = decide(s)
    assert acts == [Eject(5, "h5", acts[0].why, (t,)), Drop("c1", "failed: t", lone), Build((6,), MAIN, 2)]
    assert "failed alone twice" in acts[0].why and "no failing task's inputs meet its files" in acts[0].why
    assert "a test that flakes is deleted" in acts[0].why


@pytest.mark.parametrize("before", [
    Outcome((5,), ("h5-old",), FAIL, (5,)),                  # at another head
    Outcome((5, 6), heads(5, 6), FAIL, (5, 6)),              # not alone
    Outcome((5,), heads(5), PASS),                           # not a failure
    Outcome((5,), heads(5), FAIL, (), MAIN, (inherited("r", "red/"),)),  # not its own: every failure inherited
])
def test_a_lone_failure_ejects_only_after_a_lone_failure_at_the_same_head(before):
    s = snap(cand("c1", MAIN, 5, state=FAIL, failed=(task("t", "elsewhere/"),)), ready=prs(5), history=[before])
    assert not [a for a in decide(s) if isinstance(a, Eject)]


@pytest.mark.parametrize("state", [INFRA, LIST_FAILED])
def test_infra_and_list_failed_retry_and_ring_and_never_eject_or_advance(state):
    c = Candidate("c1", MAIN, (1,), heads(1), Verdict(state, (task("t", "pkg1/"),), (task("runner", "tools/x"),)), tries=2)
    assert decide(snap(c, ready=prs(1, 2), rings=3)) == [Retry("c1"), Build((2,), "c1", 1)]
    acts = decide(snap(replace(c, tries=3), ready=prs(1, 2), rings=3))
    assert acts == [Retry("c1"), Ring("c1", acts[1].why), Build((2,), "c1", 1)]
    assert acts[1].why.startswith("c1's check") and "3 times" in acts[1].why
    assert [type(a) for a in decide(snap(replace(c, tries=4), cand("c2", "c1", 2), ready=prs(1, 2), rings=3))] == [Retry, Ring]


def test_a_pending_lowest_candidate_waits():
    assert decide(snap(cand("c1", MAIN, 1), ready=prs(1, 2))) == [Build((2,), "c1", 1)]


# ---- inherited failures: a red base --------------------------------------------------------------------------------------------

FIX = pr(9, None, "red/fix.py")


def test_failures_all_at_their_bases_keys_blame_nobody_ring_red_and_leave_the_base_to_fixes():
    t = inherited("suite:r", "red/")
    s = snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(t,)), cand("c2", "c1", 4), ready=[*prs(1, 2, 3, 4), FIX])
    acts = decide(s)
    out = Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (), MAIN, (t,))
    assert acts == [Red("c1", MAIN, acts[0].why, (t,)), Drop("c1", "failed: suite:r", out), Drop("c2", "its base failed"),
                    Build((9,), MAIN, 2)]
    assert acts[0].why.startswith("m0 fails suite:r with no member of c1 touching them: a fix must touch their inputs")
    assert decide(snap(ready=[*prs(1, 2, 3, 4), FIX], history=[out])) == [Build((9,), MAIN, 2)]
    assert decide(snap(ready=prs(1, 2, 3, 4), history=[out])) == []
    assert [type(a) for a in decide(replace(s, ready=tuple(prs(1, 2, 3, 4))))] == [Red, Drop, Drop]


def test_a_lone_member_failing_only_at_its_bases_keys_is_not_ejected_even_after_failing_alone():
    t = inherited("suite:r", "red/")
    s = snap(cand("c1", MAIN, 5, state=FAIL, failed=(t,)), ready=prs(5), history=[Outcome((5,), heads(5), FAIL, (5,), MAIN)])
    acts = decide(s)
    assert acts == [Red("c1", MAIN, acts[0].why, (t,)),
                    Drop("c1", "failed: suite:r", Outcome((5,), heads(5), FAIL, (), MAIN, (t,)))]


def test_a_mixed_failure_ejects_or_bisects_on_its_own_failures_alone_and_rings_red_on_the_rest():
    """An inherited failure suspects nobody even where its inputs meet a member's files (here #3's): only the batch's own
    failures do.  The base is red on the inherited part, so only its fixes are batched on it."""
    own, red = task("suite:a", "pkg2/"), inherited("suite:r", "red/", "pkg3/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, 3, state=FAIL, failed=(own, red)), ready=[*prs(1, 2, 3), FIX]))
    assert acts == [Red("c1", MAIN, acts[0].why, (red,)),
                    Eject(2, "h2", "suite:a failed, and its changed files meet their inputs, no other member's", (own,)),
                    Drop("c1", "failed: suite:a, suite:r", Outcome((1, 2, 3), heads(1, 2, 3), FAIL, (2,), MAIN, (red,))),
                    Build((3, 9), MAIN, 2)]
    nobody = task("suite:x", "elsewhere/")
    acts = decide(snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(red, nobody)), ready=prs(1, 2, 4)))
    assert acts == [Red("c1", MAIN, acts[0].why, (red,)),
                    Drop("c1", "failed: suite:r, suite:x", Outcome((1, 2), heads(1, 2), FAIL, (1, 2), MAIN, (red,)))]


@pytest.mark.parametrize("base_key", [None, "key:t-on-the-base"])
def test_a_failure_not_at_its_bases_key_is_the_batchs_own(base_key):
    t = Failed("t", "key:t", frozenset({"pkg2/"}), base_key)
    s = snap(cand("c1", MAIN, 1, 2, state=FAIL, failed=(t,)), ready=prs(1, 2, 3))
    assert decide(s) == [Eject(2, "h2", "t failed, and its changed files meet their inputs, no other member's", (t,)),
                         Drop("c1", "failed: t", Outcome((1, 2), heads(1, 2), FAIL, (2,), MAIN)), Build((1, 3), MAIN, 2)]


def test_an_inherited_failure_caps_nobody_and_the_red_ends_once_main_moves_past_its_base():
    """A batch on a pending fix sits on another base, so it is not held to fixes; nor is anything once the fix is on main."""
    t, six = inherited("suite:r", "red/"), (1, 2, 3, 4, 5, 6)
    out, ready = Outcome(six, heads(*six), FAIL, (), MAIN, (t,)), [*prs(*six, 7), FIX]
    assert decide(snap(cand("f", MAIN, 9), ready=ready, history=[out])) == [Build(six, "f", 1)]
    fixed = Outcome((9,), heads(9), PASS)
    assert decide(snap(cand("f", MAIN, 9, state=PASS), ready=ready, history=[out])) == [
        Advance("f"), Drop("f", "on main", fixed), Build(six, "f", 2)]
    assert decide(snap(ready=ready, history=[out, fixed], main="f")) == [Build(six, "f", 2)]


def test_fixes_on_a_red_base_keep_the_lean_rule_the_caps_and_the_batch_size():
    t = inherited("suite:r", "red/")
    fixes = [pr(5, None, "red/a.py", lean=True), pr(6, None, "red/b.py", lean=True), pr(7, None, "red/c.py"),
             pr(8, None, "red/d")]
    history = [Outcome((1,), heads(1), FAIL, (), MAIN, (t,)), Outcome((7, 3), heads(7, 3), FAIL, (7, 3), "m-old")]
    assert decide(snap(ready=[*prs(1, 2), *fixes], history=history, batch=2)) == [Build((5, 8), MAIN, 2)]


def test_a_base_stays_red_on_every_inherited_failure_recorded_on_it():
    """A base never changes and a fail stands.  Here #5 fixes suite:r1 and #6 suite:r2, and #5 alone failed on suite:r2: both
    go in one batch (were the base red only on the newest outcome's suite:r2, #6 alone would fail on suite:r1, and the two
    fixes would take turns forever), #5 alone isn't rebuilt to fail the same way, and a fix failing for its own reasons
    leaves the base red."""
    r1, r2 = inherited("suite:r1", "red1/"), inherited("suite:r2", "red2/")
    history = [Outcome((1, 2), heads(1, 2), FAIL, (), MAIN, (r1, r2)), Outcome((5,), heads(5), FAIL, (), MAIN, (r2,))]
    five, six = pr(5, None, "red1/fix.py"), pr(6, None, "red2/fix.py")
    assert decide(snap(ready=[*prs(1, 2), five, six], history=history)) == [Build((5, 6), MAIN, 2)]
    assert decide(snap(ready=[*prs(1, 2), five], history=history)) == []
    assert decide(snap(ready=[*prs(1, 2), pr(5, "h5b", "red1/fix.py")], history=history)) == [Build((5,), MAIN, 2)]
    own = [Outcome((1,), heads(1), FAIL, (), MAIN, (r1,)), Outcome((5,), heads(5), FAIL, (5,), MAIN)]
    assert decide(snap(ready=[*prs(1, 2), five], history=own)) == [Build((5,), MAIN, 2)]


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


def test_a_batch_holds_at_most_one_lean_pr_and_the_next_waits_for_a_later_batch():
    ready = [pr(1, lean=True), pr(2, lean=True), pr(3), pr(4, lean=True), pr(5)]
    assert decide(snap(ready=ready)) == [Build((1, 3, 5), MAIN, 2)]
    assert decide(snap(cand("q1", MAIN, 1, 3, 5), ready=ready)) == [Build((2,), "q1", 1)]


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


@pytest.mark.parametrize("red", [(), (inherited("r", "pkg10/", "pkg8/"),)])
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
    t, r = task("t", "pkg2/", "**/x.py"), inherited("r", "pkg4/")
    snaps = [snap(cand("c1", MAIN, 1, 2, state=state, failed=(t, r)), cand("c2", "c1", 3), cand("z", "gone", 4),
                  ready=[*prs(1, 2, 3, 4), pr(5, lean=True)],
                  history=[Outcome((5,), heads(5), FAIL, (5,)), Outcome((4,), heads(4), FAIL, (), MAIN, (r,))])
             for state in (PENDING, PASS, FAIL, INFRA, LIST_FAILED)]

    def refuse(*a, **k):
        raise AssertionError("decide is pure: no I/O, no clock, no randomness")

    with monkeypatch.context() as m:
        for mod, name in ((builtins, "open"), (time, "time"), (time, "monotonic"), (time, "perf_counter"),
                          (random, "random"), (random, "randrange"), (random, "shuffle")):
            m.setattr(mod, name, refuse)
        got = [decide(s) for s in snaps]
    assert got == [decide(s) for s in snaps]


# ---- the simulated world -------------------------------------------------------------------------------------------------------

class World:
    """The loop's world, scripted by a seed: `main`, the ready marks, the live candidates with graph2's verdicts, and the outcomes
    the loop recorded.  A bad PR fails every candidate whose tree holds it, on a task whose inputs meet its files (and
    `shared/`'s) or, for an interaction, nobody's, at a key only such trees give the task.  In a red world `main` fails
    `suite:red` at the same key in every tree without the fix, a good PR that touches its inputs.  PRs arrive, push, conflict
    and are unmarked for a while, graph2 answers late or with infra errors, someone else moves `main`, the loop stops a tick
    early, and a merged PR stays marked for a while; `step` checks every invariant on the way."""

    def __init__(self, seed: int, red: bool | None = None, bad: bool = True):
        self.seed = seed
        rng = self.rng = random.Random(seed)
        numbers = sorted(rng.sample(range(10, 90), rng.randint(3, 12)))
        self.batch, self.depth, self.rings = rng.choice((1, 2, 3, 4, 6)), rng.choice((1, 2, 3)), rng.choice((1, 2, 3))
        self.bad = {m: rng.random() < 0.5 for m in numbers if rng.random() < 0.3} if bad else {}  # m -> it meets its files
        self.good = set(numbers) - set(self.bad)
        red = rng.random() < 0.3 if red is None else red
        self.fix = rng.choice(sorted(self.good)) if red and self.good else None
        self.marks = {m: Ready(m, f"h{m}", frozenset({f"pkg{m}/mod{m}.py"} | ({"shared/x.py"} if rng.random() < 0.3 else set())
                                                     | ({"red/fix.py"} if m == self.fix else set())),
                               rng.random() < 0.25, rng.randint(0, 4), rng.random() < 0.1) for m in numbers}
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
        self.due: dict[str, int] = {}
        self.infra_left: dict[str, int] = {}
        self.history: list[Outcome] = []
        self.merged: dict[int, int] = {}              # number -> the tick it reached main
        self.ejected: set[int] = set()
        self.unmarked: dict[int, int] = {}            # number -> the tick it is marked again
        self.seen: Counter[str] = Counter()

    def ready(self) -> tuple[Ready, ...]:
        return tuple(r for m, r in self.marks.items() if self.arrive[m] <= self.tick and m not in self.ejected
                     and m not in self.unmarked and (m not in self.merged or self.lag is None or self.tick < self.merged[m] + self.lag))

    def snapshot(self) -> Snapshot:
        cands = list(self.cands.values())
        self.rng.shuffle(cands)
        return Snapshot(self.main, self.ready(), tuple(cands), tuple(self.history), self.batch, self.depth, self.rings)

    def done(self) -> bool:
        return set(self.merged) == self.good and self.ejected == set(self.bad) and not self.cands

    def key(self, name: str, m: int, sha: str) -> str:
        return f"{name}@{'with' if m in self.tree[sha] else 'without'} #{m}"

    def failing(self, m: int, c: Candidate) -> Failed:
        """Bad #m's failure in ``c`` at its key there, with its key on ``c``'s base (now and then unknown for an interaction)."""
        if not self.bad[m]:
            name = f"interaction:{m}"
            return Failed(name, self.key(name, m, c.sha), frozenset({"elsewhere/untouched.py", "docs/**"}),
                          None if m % 2 else self.key(name, m, c.base))
        name, own = f"suite:{m}", (f"pkg{m}/", f"pkg{m}", f"pkg{m}/**", f"pkg{m}/*.py", f"**/mod{m}.py")[m % 5]
        return Failed(name, self.key(name, m, c.sha), frozenset({own, "shared/x.py"}), self.key(name, m, c.base))

    def verdict(self, c: Candidate) -> Verdict:
        """Its tree's failures; or, while it has some left, now and then an infra or listing error beside them."""
        failed = tuple(self.failing(m, c) for m in sorted(set(self.tree[c.sha]) & set(self.bad)))
        if self.fix is not None and self.fix not in self.tree[c.sha]:
            failed += (Failed("suite:red", self.key("suite:red", self.fix, c.sha), frozenset({"red/"}),
                              self.key("suite:red", self.fix, c.base)),)
        if self.infra_left[c.sha] and self.rng.random() < 0.2:
            self.infra_left[c.sha] -= 1
            state = self.rng.choice((INFRA, LIST_FAILED))
            return Verdict(state, failed, (task("runner", "tools/runner.py"),) if state == INFRA else ())
        return Verdict(FAIL, failed) if failed else Verdict(PASS)

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
            if c.verdict.state == PENDING and self.due[sha] <= self.tick:
                self.cands[sha] = replace(c, verdict=self.verdict(c))
        return moved

    def step(self) -> None:
        self.tick += 1
        moved = self.events()
        s = self.snapshot()
        acts = decide(s)
        self.check(s, acts, moved)
        n = len(acts)
        if acts and self.crashes and self.rng.random() < 0.15:
            self.crashes -= 1
            n = self.rng.randrange(len(acts))
            self.seen["the loop stopped early"] += 1
        for a in acts[:n]:
            self.apply(a)
        live = [m for c in self.cands.values() for m in c.members]
        assert len(live) == len(set(live)), f"seed {self.seed} tick {self.tick}: a PR in two live candidates: {self.cands}"
        again = decide(self.snapshot())
        dropped = {a.sha for a in acts[:n] if isinstance(a, Drop)}
        repeated = [a for a in set(again) & set(acts[:n]) if not (isinstance(a, Red) and a.sha not in dropped)]
        assert not repeated, f"seed {self.seed} tick {self.tick}: a restart repeats {repeated}"

    def check(self, s: Snapshot, acts: list, moved: str | None) -> None:
        where = f"seed {self.seed} tick {self.tick}: {acts}"
        by = {c.sha: c for c in s.candidates}
        ready = {r.number: r for r in s.ready}
        outcomes = [*s.history, *(a.outcome for a in acts if isinstance(a, Drop) and a.outcome is not None)]
        for a in acts:
            if isinstance(a, Advance):
                c = by[a.sha]
                assert c.verdict.state == PASS and c.base == s.main, where
                assert all(m in ready and ready[m].head == h for m, h in zip(c.members, c.heads)), where
            elif isinstance(a, Eject):
                assert any(isinstance(d, Drop) and d.outcome is not None and d.outcome.state == FAIL and a.number in d.outcome.members
                           and by[d.sha].verdict.state == FAIL for d in acts), where
                assert not [f for f in a.failed if f.key == f.base_key], where
                self.seen["ejected by inputs" if "meet their inputs" in a.why else "ejected after failing alone twice"] += 1
            elif isinstance(a, Red):
                c = by[a.sha]
                assert c.verdict.state == FAIL and a.base == c.base == s.main, where
                assert a.failed == tuple(f for f in c.verdict.failed if f.key == f.base_key), where
                self.seen["a red base"] += 1
            elif isinstance(a, Build):
                assert 1 <= len(a.members) <= s.batch and sum(ready[m].lean for m in a.members) <= 1, where
                assert not any(ready[m].conflict for m in a.members), where
                self.seen["a lean PR built"] += any(ready[m].lean for m in a.members)
                if red := [f for o in outcomes if o.base == a.base for f in o.inherited]:
                    assert all(any(meets(ready[m].files, f.inputs) for f in red) for m in a.members), f"{where}: not a fix"
                    self.seen["a fix built on a red base"] += 1
            elif isinstance(a, Drop) and a.outcome is not None and a.outcome.state == FAIL:
                failed = by[a.sha].verdict.failed
                assert not (failed and all(f.key == f.base_key for f in failed)) or a.outcome.suspects == (), where
                self.seen["bisected"] += len(a.outcome.suspects) > 1
                self.seen["suspected alone"] += len(a.outcome.members) == 1 and not [e for e in acts if isinstance(e, Eject)]
                self.seen["a mixed failure"] += bool(a.outcome.suspects and a.outcome.inherited)
            elif isinstance(a, Retry):
                self.seen["retried"] += 1
        left = [c for c in by if not any(isinstance(a, Drop) and a.sha == c for a in acts)]
        if not left and not [a for a in acts if isinstance(a, (Advance, Build))]:
            self.seen["stalled on a red base"] += any(o.base == s.main and o.inherited for o in outcomes)
        for c in by.values():
            fresh = all(m in ready and ready[m].head == h for m, h in zip(c.members, c.heads))
            if c.base == s.main and fresh and c.verdict.state in (INFRA, LIST_FAILED):
                assert Retry(c.sha) in acts and not [a for a in acts if isinstance(a, (Advance, Eject))], where
                assert any(isinstance(a, Ring) and a.sha == c.sha for a in acts) == (c.tries >= s.rings), where
        if moved:
            drops = [a for a in acts if isinstance(a, Drop)]
            assert sorted(d.sha for d in drops) == sorted(by) and all(d.outcome is None for d in drops), where
            pool = [r for r in s.ready if not r.conflict and r.number not in self.merged]
            assert [a.base for a in acts if isinstance(a, Build)] == ([moved] if pool else []), where

    def apply(self, a) -> None:
        where = f"seed {self.seed} tick {self.tick}: {a}"
        if isinstance(a, Build):
            assert a.base == self.main or a.base in self.cands, where
            sha, self.made = f"q{self.made}", self.made + 1
            self.cands[sha] = Candidate(sha, a.base, a.members, tuple(self.marks[m].head for m in a.members))
            self.tree[sha] = self.tree[a.base] | set(a.members)
            self.due[sha], self.infra_left[sha] = self.tick + self.rng.randint(1, 3), 2
        elif isinstance(a, Advance):
            c = self.cands[a.sha]
            assert c.base == self.main and c.verdict.state == PASS, where
            assert not set(c.members) & set(self.merged), f"{where}: a PR reaches main twice"
            assert not set(c.members) & set(self.bad), f"{where}: a bad PR reaches main"
            assert self.fix is None or self.fix in self.merged or self.fix in c.members, f"{where}: before the fix"
            self.main = a.sha
            self.merged |= dict.fromkeys(c.members, self.tick)
        elif isinstance(a, Drop):
            del self.cands[a.sha]
            if a.outcome is not None:
                self.history.append(a.outcome)
        elif isinstance(a, Eject):
            assert self.marks[a.number].head == a.head and a.number in self.bad, f"{where}: a good PR ejected"
            self.ejected.add(a.number)
        elif isinstance(a, Retry):
            c = self.cands[a.sha]
            assert c.verdict.state in (INFRA, LIST_FAILED), where
            self.cands[a.sha] = replace(c, verdict=Verdict(), tries=c.tries + 1)
            self.due[a.sha] = self.tick + self.rng.randint(1, 2)
        elif isinstance(a, Red):
            assert a.base == self.main and a.sha in self.cands, where
        else:
            assert isinstance(a, Ring), where


SEEDS = range(500)


def ticks(w: World) -> int:
    """The ticks within which a scenario settles: every good PR on main, every bad PR ejected, no candidate left.  Seeds 0 to
    4999 settled within 22 ticks and 4.8 per PR, and with a broken main whose fix is ranked last and late, within 13 and 4.8
    per PR."""
    return 40 + 8 * len(w.marks)


def test_the_simulated_queue_keeps_every_invariant_and_settles():
    """Over every seed: main advances only to a passing candidate on it, at its members' marked heads; no PR is in two live
    candidates or reaches main twice; infra and list-failed verdicts are retried and rung, never ejected or advanced; a batch
    holds at most one Lean PR; a move of main by someone else drops the stale candidates and rebuilds on it; an inherited
    failure ejects nobody and rings red, and a red base gets only fixes; a restart repeats nothing but a red whose drop it
    didn't apply; and every good PR reaches main and every bad one is ejected within `ticks`.  Each situation the invariants
    are about happens in some seed."""
    seen: Counter[str] = Counter()
    for seed in SEEDS:
        w = World(seed)
        while not w.done():
            assert w.tick < ticks(w), (f"seed {seed}: not settled after {ticks(w)} ticks: merged {sorted(w.merged)} of "
                                       f"{sorted(w.good)}, ejected {sorted(w.ejected)} of {sorted(w.bad)}, live {list(w.cands.values())}")
            w.step()
        seen.update(w.seen)
    for what in ("main moved elsewhere", "the loop stopped early", "ejected by inputs", "ejected after failing alone twice",
                 "a lean PR built", "bisected", "suspected alone", "retried", "a red base", "a fix built on a red base",
                 "a mixed failure", "stalled on a red base"):
        assert seen[what], f"no seed exercised: {what}"


def test_a_broken_main_holds_the_queue_for_its_fix_ejects_nobody_and_then_lets_the_rest_through():
    """`main` fails `suite:red` in every tree without the fix, which is ranked last and arrives late: no good PR is ejected,
    nothing but the fix goes on the red base, the fix reaches main first, and every PR is on main within `ticks`."""
    seen: Counter[str] = Counter()
    for seed in range(100):
        w = World(seed, red=True, bad=False)
        w.marks[w.fix], w.arrive[w.fix] = replace(w.marks[w.fix], rank=9), seed % 10
        while not w.done():
            assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks: merged {sorted(w.merged)}"
            w.step()
        assert not w.ejected and w.merged[w.fix] == min(w.merged.values()), f"seed {seed}: {w.merged}"
        seen.update(w.seen)
    for what in ("a red base", "a fix built on a red base", "stalled on a red base"):
        assert seen[what], f"no seed exercised: {what}"
